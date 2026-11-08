#!/usr/bin/env python3
"""Apply paired background-plane patches using distance, viewpoint, and frame progress while protecting the target mask.

Use apply_paired_patches(...) for moving-camera placement and apply_static_paired_patches(...) for fixed cameras.
Requires NumPy and OpenCV. See README.md and ../docs/optimization.md for details.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import json
import math
import os
from pathlib import Path
import re
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
VERSION = 4


def natural_key(value):
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', str(value))]


def atomic_write(path: Path, content: bytes | str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        temporary.write_bytes(content.encode('utf-8') if isinstance(content, str) else content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_write(Path(path), json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_png(path, image):
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise OSError(f'Cannot encode PNG: {path}')
    atomic_write(Path(path), encoded.tobytes())


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8:
        raise ValueError(f'Cannot read an 8-bit image: {path}')
    if image.ndim != 2 and not (image.ndim == 3 and image.shape[2] == 3):
        raise ValueError(f'Sequence images must have one or three channels: {path}: {image.shape}')
    return image


def load_patch(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8:
        raise ValueError(f'Cannot read an 8-bit patch: {path}')
    if min(image.shape[:2]) < 2:
        raise ValueError(f'Patch width and height must be at least 2: {path}')
    if image.ndim == 2:
        color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        alpha = np.ones(image.shape, dtype=np.float32)
    elif image.ndim == 3 and image.shape[2] in (3, 4):
        color = image[:, :, :3]
        alpha = image[:, :, 3].astype(np.float32) / 255.0 if image.shape[2] == 4 else np.ones(image.shape[:2], np.float32)
    else:
        raise ValueError(f'Patch must be grayscale, RGB, or RGBA: {path}')
    if not alpha.any():
        raise ValueError(f'The patch is fully transparent: {path}')
    # Map the entire patch as a square texture regardless of its source aspect ratio.
    return color, alpha


@dataclass(frozen=True)
class PatchGeometry:
    corners_xy: list
    nominal_side_pixels: float
    center_xy: list
    progress: float
    distance_scale: float
    temporal_scale: float
    angle_deg: float


def patch_geometry(width, height, angle_deg, distance_m, nearest_distance_m,
                   frame_index, frame_count, *, initial_side_ratio=0.5,
                   recession=1.0, drift_ratio=0.30, upward_ratio=0.25, fov_deg=90.0):
    """Return corners in top-left, top-right, bottom-right, bottom-left order; both modalities share the geometry.

    Initial side length = shorter image dimension * 0.5 * nearest distance / current distance.
    t = frame index / (frame count - 1); depth growth factor = 1 + recession * t.
    A 90-degree view is frontal; below 90 the right edge is nearer/longer, above 90 the left edge is nearer/longer.
    Upward displacement = upward_ratio * image height * cos(90 degrees - angle) * t, maximal at 90 degrees.
    """
    if width < 2 or height < 2 or distance_m <= 0 or nearest_distance_m <= 0:
        raise ValueError('Image dimensions and distances must be positive')
    if not 0 < angle_deg < 180 or not 0 < fov_deg < 179:
        raise ValueError('Angle must be in (0,180) degrees and FOV in (0,179) degrees')
    if frame_count < 1 or not 0 <= frame_index < frame_count:
        raise ValueError('Invalid frame index or frame count')
    if not math.isfinite(upward_ratio) or not 0 <= upward_ratio <= 1:
        raise ValueError('upward_ratio must be in [0,1]')
    t = frame_index / max(1, frame_count - 1)
    temporal_scale = 1.0 / (1.0 + recession * t)
    distance_scale = nearest_distance_m / distance_m
    side = min(width, height) * initial_side_ratio * distance_scale * temporal_scale
    yaw = math.radians(90.0 - angle_deg)
    focal_pixels = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    center = np.array([(width - 1) / 2.0 + drift_ratio * width * math.sin(yaw) * t,
                       (height - 1) / 2.0 - upward_ratio * height * math.cos(yaw) * t], dtype=np.float64)
    # Rotate the square plane around its vertical axis and apply pinhole projection.
    # Use side-1 so a frontal integer-side patch spans approximately side raster pixels.
    extent = max(side - 1.0, 0.25)
    local = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    denominators = 1.0 - (extent / focal_pixels) * local[:, 0] * math.sin(yaw)
    if np.any(denominators <= 0.05):
        raise ValueError('The projected patch crosses the camera plane; reduce initial_side_ratio or FOV')
    projected = np.column_stack([extent * local[:, 0] * math.cos(yaw) / denominators,
                                 extent * local[:, 1] / denominators]) + center
    return PatchGeometry(projected.tolist(), float(side), center.tolist(), float(t),
                         float(distance_scale), float(temporal_scale), float(angle_deg))


def composite_patch(image, patch_color, patch_alpha, target_mask, geometry):
    """Composite only where mask==0 and preserve original pixels where mask>0.

    Use premultiplied alpha for perspective interpolation to prevent dark edge artifacts.
    Preserve the target in front of the background patch plane without shifting the patch around it.
    """
    height, width = image.shape[:2]
    if target_mask.shape != (height, width) or target_mask.ndim != 2:
        raise ValueError('Mask must be single-channel and match the image dimensions')
    color = patch_color
    if image.ndim == 2:
        color = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    patch_h, patch_w = color.shape[:2]
    source = np.float32([[0, 0], [patch_w - 1, 0], [patch_w - 1, patch_h - 1], [0, patch_h - 1]])
    transform = cv2.getPerspectiveTransform(source, np.float32(geometry.corners_xy))
    alpha = patch_alpha.astype(np.float32)
    weighted = color.astype(np.float32) * (alpha if color.ndim == 2 else alpha[:, :, None])
    warped_alpha = cv2.warpPerspective(alpha, transform, (width, height), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_color = cv2.warpPerspective(weighted, transform, (width, height), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    protected = target_mask > 0
    visible_alpha = np.clip(warped_alpha, 0.0, 1.0)
    visible_alpha[protected] = 0.0
    # Zero premultiplied colors in protected areas to prevent floating-point changes.
    warped_color[protected] = 0.0
    factor = visible_alpha if image.ndim == 2 else visible_alpha[:, :, None]
    output = np.clip(np.rint(image.astype(np.float32) * (1.0 - factor) + warped_color), 0, 255).astype(np.uint8)
    output[protected] = image[protected]
    return output, dict(projected_patch_pixels=int(np.count_nonzero(warped_alpha > 0)),
                        visible_patch_pixels=int(np.count_nonzero(visible_alpha > 0)),
                        protected_overlap_pixels=int(np.count_nonzero((warped_alpha > 0) & protected)),
                        changed_pixels=int(np.count_nonzero(np.any(output != image, axis=2) if image.ndim == 3 else output != image)))


def files_in(directory):
    files = sorted([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS], key=lambda p: natural_key(p.name))
    if len({p.stem for p in files}) != len(files):
        raise ValueError(f'Duplicate frame stems: {directory}')
    return files


def discover_pairs(dataset_root, mask_root, angles, distances, sequence_names):
    pairs, all_distances = [], []
    angles = None if angles is None else {float(v) for v in angles}
    distances = None if distances is None else {float(v) for v in distances}
    sequence_names = None if sequence_names is None else set(sequence_names)
    for view in sorted(dataset_root.iterdir(), key=lambda p: natural_key(p.name)):
        match = re.fullmatch(r'vi_camera_view_(\d+(?:\.\d+)?)', view.name)
        if not view.is_dir() or not match:
            continue
        angle = float(match[1])
        for directory in sorted(view.iterdir(), key=lambda p: natural_key(p.name)):
            dist_match = re.fullmatch(r'distance_(\d+(?:\.\d+)?)m', directory.name)
            if not directory.is_dir() or not dist_match:
                continue
            distance = float(dist_match[1])
            if distance <= 0:
                raise ValueError(f'Distance must be positive: {directory}')
            all_distances.append(distance)
            if angles is not None and angle not in angles or distances is not None and distance not in distances:
                continue
            for sequence in sorted(directory.iterdir(), key=lambda p: natural_key(p.name)):
                if not sequence.is_dir() or sequence_names is not None and sequence.name not in sequence_names:
                    continue
                vi_rel = sequence.relative_to(dataset_root)
                ir_rel = Path('ir' + vi_rel.parts[0][2:]) / Path(*vi_rel.parts[1:])
                vi_files = files_in(sequence)
                ir_dir = dataset_root / ir_rel
                if not ir_dir.is_dir():
                    raise ValueError(f'Missing paired grayscale directory: {ir_dir}')
                if [p.name for p in vi_files] != [p.name for p in files_in(ir_dir)]:
                    raise ValueError(f'Frame names differ between modalities: {vi_rel}')
                if not vi_files:
                    continue
                for rel in (vi_rel, ir_rel):
                    for frame in vi_files:
                        if not (mask_root / rel / f'{frame.stem}_mask.png').is_file():
                            raise FileNotFoundError(f'Missing mask: {mask_root / rel / (frame.stem + "_mask.png")}')
                pairs.append(dict(vi=vi_rel, ir=ir_rel, angle=angle, distance=distance, files=vi_files))
    if not pairs:
        raise ValueError('No paired sequences match the filters')
    return pairs, min(all_distances)


def make_preview(output_root, pairs):
    """Save first/middle/last paired-frame comparisons and an overview."""
    previews = []
    preview_dir = output_root / 'previews'
    for pair in pairs:
        count = len(pair['files'])
        indices = [0, (count - 1) // 2, count - 1]
        grid = np.zeros((2 * 265, 3 * 320, 3), np.uint8)
        for row, mode in enumerate(('vi', 'ir')):
            for col, index in enumerate(indices):
                name = pair['files'][index].stem + '.png'
                image = cv2.imread(str(output_root / pair[mode] / name), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f'Cannot read preview image: {pair[mode] / name}')
                # Fit the preview within 320x240 while preserving aspect ratio and full-resolution outputs.
                scale = min(320 / image.shape[1], 240 / image.shape[0])
                resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
                tile = grid[row * 265:(row + 1) * 265, col * 320:(col + 1) * 320]
                oy, ox = (240 - resized.shape[0]) // 2, (320 - resized.shape[1]) // 2
                tile[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized
                label = f'{mode} {pair["angle"]:g}deg {pair["distance"]:g}m f={index + 1}/{count}'
                cv2.putText(tile, label, (4, 257), cv2.FONT_HERSHEY_SIMPLEX, .46, (255, 255, 255), 1, cv2.LINE_AA)
        name = f'{pair["vi"].name}_angle_{pair["angle"]:g}_distance_{pair["distance"]:g}m.png'
        write_png(preview_dir / name, grid)
        previews.append(grid)
    # Arrange the nine demonstration sequences in a 3x3 overview; retain larger per-sequence previews.
    cols = min(3, len(previews))
    overview = np.zeros((math.ceil(len(previews) / cols) * 265, cols * 480, 3), np.uint8)
    for index, grid in enumerate(previews):
        row, col = divmod(index, cols)
        overview[row * 265:(row + 1) * 265, col * 480:(col + 1) * 480] = cv2.resize(grid, (480, 265), interpolation=cv2.INTER_AREA)
    write_png(preview_dir / 'overview.png', overview)


def _apply_paired_patches(dataset_root, patch_dir, mask_root, output_root, *,
                         angles: Iterable[float] | None = None,
                         distances: Iterable[float] | None = None,
                         sequence_names: Iterable[str] | None = None,
                         nearest_distance_m: float | None = None,
                         initial_side_ratio: float = 0.5, recession: float = 1.0,
                         drift_ratio: float = 0.30, upward_ratio: float = 0.25, fov_deg: float = 90.0,
                         empty_mask_policy: str = 'skip', overwrite: bool = False,
                         save_previews: bool = True, static_camera: bool = False):
    """Read paired frames, write patched PNG sequences, and return a run-report dictionary.

    Required paths: dataset root, directory containing visible.png/infrared.png, mask root,
    and a separate output root. Preserve the modality/viewpoint/distance/target hierarchy.

    Geometry: nearest-distance initial side = shorter dimension * initial_side_ratio (default 1/2).
    Scale inversely with distance. The final side defaults to 1/(1+recession)=1/2 of the initial side.
    Final horizontal displacement = drift_ratio * image width * sin(90 degrees - angle).
    Final upward displacement = upward_ratio * image height * cos(90 degrees - angle), at most one-quarter height by default.
    Center the initial frame vertically; upward motion is largest at 90 degrees and decreases with angular deviation.
    fov_deg controls perspective strength and defaults to 90 degrees; angle=90 denotes a frontal view.

    With empty_mask_policy='skip', preserve and report empty-mask frames; 'error' raises an exception.
    Do not treat missed detections as confirmed background. Use the union of both masks for consistent occlusion.
    Preserve 8-bit grayscale or three-channel inputs; convert color patches to grayscale for infrared images.
    """
    dataset_root, patch_dir, mask_root, output_root = [Path(p).expanduser().resolve() for p in (dataset_root, patch_dir, mask_root, output_root)]
    for name, path in [('dataset_root', dataset_root), ('patch_dir', patch_dir), ('mask_root', mask_root)]:
        if not path.is_dir():
            raise NotADirectoryError(f'{name}: {path}')
        if output_root == path or output_root in path.parents or path in output_root.parents:
            raise ValueError(f'Output and input directories must be separate and must not contain one another: {output_root}, {path}')
    if not 0 < initial_side_ratio <= 1 or not math.isfinite(recession) or recession < 0:
        raise ValueError('initial_side_ratio must be in (0,1] and recession must be finite and nonnegative')
    if not math.isfinite(drift_ratio) or not 0 <= drift_ratio <= 1 or not 0 < fov_deg < 179:
        raise ValueError('drift_ratio must be in [0,1] and fov_deg must be in (0,179)')
    if not math.isfinite(upward_ratio) or not 0 <= upward_ratio <= 1:
        raise ValueError('upward_ratio must be in [0,1]')
    if empty_mask_policy not in (('skip', 'error', 'bbox_guard') if static_camera else ('skip', 'error')):
        raise ValueError('empty_mask_policy must be skip/error; fixed-camera mode also supports bbox_guard')
    pairs, inferred_nearest = discover_pairs(dataset_root, mask_root, angles, distances, sequence_names)
    if static_camera:
        for pair in pairs:
            pair['frame_metadata'] = fixed_frame_metadata(dataset_root, pair)
            pair['confirmed_absent_frames'] = {name for name, f in pair['frame_metadata'].items()
                                              if f['semantic_target_pixels'] == 0 and f['target_intersects_frustum'] is False}
    frames_by_sequence = {p['vi'].as_posix(): [f.name for f in p['files']] for p in pairs}
    nearest_distance_m = inferred_nearest if nearest_distance_m is None else float(nearest_distance_m)
    if not math.isfinite(nearest_distance_m) or nearest_distance_m <= 0:
        raise ValueError('nearest_distance_m must be finite and positive')
    patches = {'vi': load_patch(patch_dir / 'visible.png'), 'ir': load_patch(patch_dir / 'infrared.png')}
    manifest_path = output_root / 'patch_run.json'
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite or not manifest_path.is_file():
            raise FileExistsError(f'Output directory is not empty; choose another directory or use overwrite=True for compatible results: {output_root}')
        previous = json.loads(manifest_path.read_text())
        expected = [p['vi'].as_posix() for p in pairs]
        if previous.get('source_sequences') != expected:
            raise ValueError('Sequence set changed; use a new directory to avoid stale sequences')
        if previous.get('source_frames') != frames_by_sequence:
            raise ValueError('Frame set changed; use a new directory to avoid stale frames')
    report = dict(version=VERSION, completed=False, dataset_root=str(dataset_root), patch_dir=str(patch_dir),
                  mask_root=str(mask_root), output_root=str(output_root),
                  config=dict(static_camera=static_camera, nearest_distance_m=nearest_distance_m, initial_side_ratio=initial_side_ratio,
                              recession=recession, drift_ratio=drift_ratio, upward_ratio=upward_ratio, fov_deg=fov_deg,
                              empty_mask_policy=empty_mask_policy),
                  source_sequences=[p['vi'].as_posix() for p in pairs], source_frames=frames_by_sequence, sequences=[],
                  totals=dict(sequence_pairs=len(pairs), frame_pairs=0, output_images=0,
                              skipped_empty_mask_pairs=0, confirmed_absent_pairs=0, bbox_guard_pairs=0, protected_pixels=0))
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, report)
    try:
        for number, pair in enumerate(pairs, 1):
            print(f'[{number}/{len(pairs)}] {pair["vi"]}', flush=True)
            records = []
            shape = None
            for index, source_path in enumerate(pair['files']):
                images = {mode: read_image(dataset_root / pair[mode] / source_path.name) for mode in ('vi', 'ir')}
                current_shape = images['vi'].shape[:2]
                if current_shape != images['ir'].shape[:2] or shape is not None and current_shape != shape:
                    raise ValueError(f'Modality or frame dimensions do not match: {pair["vi"] / source_path.name}')
                shape = current_shape
                masks = {mode: read_image(mask_root / pair[mode] / f'{source_path.stem}_mask.png') for mode in ('vi', 'ir')}
                if any(mask.ndim != 2 or mask.shape != shape or not np.all((mask == 0) | (mask == 255)) for mask in masks.values()):
                    raise ValueError(f'Mask must be a full-resolution single-channel 0/255 image: {pair["vi"] / source_path.name}')
                empty = any(not mask.any() for mask in masks.values())
                absent = static_camera and source_path.name in pair['confirmed_absent_frames']
                skip = empty and not absent
                protected = (masks['vi'] > 0) | (masks['ir'] > 0)
                bbox_guard = False
                if skip and static_camera and empty_mask_policy == 'bbox_guard':
                    meta = pair['frame_metadata'].get(source_path.name)
                    guard = projected_bbox_guard(meta, *shape)
                    if guard is not None:
                        protected |= guard
                        bbox_guard, skip = True, False
                if skip and empty_mask_policy == 'error':
                    raise ValueError(f'Mask is empty; the target cannot be protected: {pair["vi"] / source_path.name}')
                height, width = shape
                geometry = patch_geometry(width, height, pair['angle'], pair['distance'], nearest_distance_m,
                                          0 if static_camera else index, len(pair['files']), initial_side_ratio=initial_side_ratio,
                                          recession=recession, drift_ratio=drift_ratio, upward_ratio=upward_ratio, fov_deg=fov_deg)
                record = dict(source_frame=source_path.name, output_frame=source_path.stem + '.png',
                              frame_index=index, status='skipped_empty_mask' if skip else ('patched_bbox_guard' if bbox_guard else 'patched_out_of_frame' if absent else 'patched'),
                              geometry=asdict(geometry), protected_pixels=int(protected.sum()), modalities={})
                for mode in ('vi', 'ir'):
                    image = images[mode]
                    if skip:
                        output = image.copy()
                        stats = dict(projected_patch_pixels=0, visible_patch_pixels=0, protected_overlap_pixels=0, changed_pixels=0)
                    else:
                        output, stats = composite_patch(image, *patches[mode], protected.astype(np.uint8) * 255, geometry)
                    if not np.array_equal(output[protected], image[protected]):
                        raise AssertionError('Internal error: pixels inside the target mask were modified')
                    stats['protected_pixels_unchanged'] = True
                    destination = output_root / pair[mode] / record['output_frame']
                    write_png(destination, output)
                    record['modalities'][mode] = dict(output_path=destination.relative_to(output_root).as_posix(), **stats)
                records.append(record)
                report['totals']['frame_pairs'] += 1
                report['totals']['output_images'] += 2
                report['totals']['skipped_empty_mask_pairs'] += int(skip)
                report['totals']['confirmed_absent_pairs'] += int(absent)
                report['totals']['bbox_guard_pairs'] += int(bbox_guard)
                report['totals']['protected_pixels'] += int(protected.sum())
            records_path = output_root / 'placement_records' / pair['vi'] / 'frames.jsonl'
            atomic_write(records_path, ''.join(json.dumps(record, ensure_ascii=False) + '\n' for record in records))
            report['sequences'].append(dict(sequence=pair['vi'].as_posix(), frame_pairs=len(records),
                                             records=records_path.relative_to(output_root).as_posix(),
                                             skipped_empty_mask_pairs=sum(r['status'] == 'skipped_empty_mask' for r in records)))
            write_json(manifest_path, report)
        if save_previews:
            make_preview(output_root, pairs)
        report['completed'] = True
    except BaseException as exc:
        report['error'] = str(exc)
        raise
    finally:
        write_json(manifest_path, report)
    print(json.dumps(report['totals'], ensure_ascii=False), flush=True)
    return report



def fixed_frame_metadata(dataset_root, pair):
    """Allow patching with an empty mask only if acquisition metadata confirms zero target pixels and absence from the view frustum.

    Without metadata, do not infer target absence; retain the empty-mask skip/error policy.
    """
    directory = dataset_root / 'metadata' / 'sequences'
    matches = []
    for path in directory.glob(pair['vi'].name + '_angle_*.json'):
        data = json.loads(path.read_text())
        if data.get('rgb_directory') == pair['vi'].as_posix():
            matches.append(data)
    if not matches:
        return {}
    if len(matches) != 1 or matches[0].get('camera_motion') != 'fixed_world_transform':
        raise ValueError(f'Fixed-camera metadata is ambiguous or has an incompatible mode: {pair["vi"]}')
    frames = {int(f['frame']): f for f in matches[0]['frames']}
    by_name = {}
    for path in pair['files']:
        by_name[path.name] = frames[int(path.stem)]
    return by_name


def confirmed_absent_frames(dataset_root, pair):
    return {name for name, frame in fixed_frame_metadata(dataset_root, pair).items()
            if frame['semantic_target_pixels'] == 0 and frame['target_intersects_frustum'] is False}


def projected_bbox_guard(meta, height, width):
    """Add a conservative patch-protection region after a missed detection without creating segmentation masks or labels."""
    if meta is None or meta.get('bbox_xyxy') is None:
        return None
    ref = np.asarray(meta['bbox_xyxy'], dtype=float)
    if ref.shape != (4,) or not np.isfinite(ref).all():
        return None
    ref[[0, 2]] = np.clip(ref[[0, 2]], 0, width)
    ref[[1, 3]] = np.clip(ref[[1, 3]], 0, height)
    if np.any(ref[2:] <= ref[:2]):
        return None
    pad = max(8, int(.05 * max(ref[2:] - ref[:2])))
    x0, y0 = np.maximum(np.floor(ref[:2]).astype(int) - pad, 0)
    x2, y2 = np.minimum(np.ceil(ref[2:]).astype(int) + pad, [width, height])
    guard = np.zeros((height, width), dtype=bool)
    guard[y0:y2, x0:x2] = True
    return guard


def apply_paired_patches(dataset_root, patch_dir, mask_root, output_root, *,
                         angles=None, distances=None, sequence_names=None,
                         nearest_distance_m=None, initial_side_ratio=.5, recession=1.,
                         drift_ratio=.30, upward_ratio=.25, fov_deg=90.,
                         empty_mask_policy='skip', overwrite=False, save_previews=True):
    """Moving-camera mode: shrink the patch over time and move upward/left/right according to viewpoint."""
    return _apply_paired_patches(
        dataset_root, patch_dir, mask_root, output_root, angles=angles, distances=distances,
        sequence_names=sequence_names, nearest_distance_m=nearest_distance_m,
        initial_side_ratio=initial_side_ratio, recession=recession, drift_ratio=drift_ratio,
        upward_ratio=upward_ratio, fov_deg=fov_deg, empty_mask_policy=empty_mask_policy,
        overwrite=overwrite, save_previews=save_previews)


def apply_static_paired_patches(dataset_root, patch_dir, mask_root, output_root, *,
                                angles=None, distances=None, sequence_names=None,
                                nearest_distance_m=None, initial_side_ratio=.5, fov_deg=90.,
                                empty_mask_policy='bbox_guard', overwrite=False, save_previews=True):
    """Fixed-camera mode with the same four required path arguments.

    Reuse the initial position, side length, and perspective throughout each sequence using directory distance/viewpoint values.
    Protect the moving target with the union of the current masks; keep the patch visible after the target exits.
    Patch confirmed-absent frames normally; otherwise use the metadata projection as a conservative guard by default.
    This fallback preserves masks and labels. Skip if no reliable projection exists, or select skip/error explicitly.
    See README.md for annotation and placement details.
    """
    return _apply_paired_patches(
        dataset_root, patch_dir, mask_root, output_root, angles=angles, distances=distances,
        sequence_names=sequence_names, nearest_distance_m=nearest_distance_m,
        initial_side_ratio=initial_side_ratio, recession=0., drift_ratio=0., upward_ratio=0.,
        fov_deg=fov_deg, empty_mask_policy=empty_mask_policy,
        overwrite=overwrite, save_previews=save_previews, static_camera=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--patch-dir', required=True)
    parser.add_argument('--mask-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--angles', type=float, nargs='+')
    parser.add_argument('--distances', type=float, nargs='+')
    parser.add_argument('--sequence-names', nargs='+')
    parser.add_argument('--static-camera', action='store_true', help='Keep patch position, size, and perspective fixed for dataset_400_2')
    parser.add_argument('--demo', action='store_true', help='Process all paired frames of 01_jeep at 30/90/105 degrees and 9/15/21 m')
    parser.add_argument('--nearest-distance-m', type=float)
    parser.add_argument('--initial-side-ratio', type=float, default=.5)
    parser.add_argument('--recession', type=float, default=1.0)
    parser.add_argument('--drift-ratio', type=float, default=.30)
    parser.add_argument('--upward-ratio', type=float, default=.25, help='Final upward displacement as a fraction of image height at 90 degrees; other views use cosine attenuation')
    parser.add_argument('--fov-deg', type=float, default=90.0)
    parser.add_argument('--empty-mask-policy', choices=['skip', 'error', 'bbox_guard'], default=None,
                        help='Defaults: skip for moving-camera mode, bbox_guard for fixed-camera mode')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--no-previews', action='store_true')
    args = vars(parser.parse_args())
    if args.pop('demo'):
        args.update(angles=[30, 90, 105], distances=[9, 15, 21], sequence_names=['01_jeep'])
    args['save_previews'] = not args.pop('no_previews')
    static = args.pop('static_camera')
    if args['empty_mask_policy'] is None:
        args['empty_mask_policy'] = 'bbox_guard' if static else 'skip'
    if static:
        for name in ('recession', 'drift_ratio', 'upward_ratio'):
            args.pop(name)
    (apply_static_paired_patches if static else apply_paired_patches)(**args)


if __name__ == '__main__':
    main()
