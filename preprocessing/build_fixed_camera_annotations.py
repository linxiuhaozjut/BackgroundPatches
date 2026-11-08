#!/usr/bin/env python3
"""Generate YOLO labels, tracking boxes, and binary masks for moving targets in fixed-camera sequences.

Load the configured segmentation weights, infer visible frames, and mirror annotations to paired infrared frames.
See README.md for usage. Input images and acquisition metadata are preserved.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import cv2
import numpy as np

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / 'datasets/carla/dataset_400_2'
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
VERSION = 2


def natural_key(value):
    return [int(v) if v.isdigit() else v for v in re.split(r'(\d+)', str(value))]


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    try:
        temp.write_bytes(data.encode('utf-8') if isinstance(data, str) else data)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def write_json(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def image_files(directory):
    return sorted((p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), key=lambda p: natural_key(p.name))


def discover(args):
    """Discover images under modality_view/distance/sequence directories, excluding output directories."""
    found = []
    for view in sorted(args.dataset_root.iterdir(), key=lambda p: natural_key(p.name)):
        match = re.fullmatch(r'(vi|ir)_camera_view_(\d+)', view.name)
        if not view.is_dir() or not match or match[1] != args.source_modality:
            continue
        if args.angles and int(match[2]) not in args.angles:
            continue
        for distance in sorted(view.iterdir(), key=lambda p: natural_key(p.name)):
            if not distance.is_dir() or not distance.name.startswith('distance_'):
                continue
            for sequence in sorted(distance.iterdir(), key=lambda p: natural_key(p.name)):
                if not sequence.is_dir():
                    continue
                rel = sequence.relative_to(args.dataset_root)
                if args.sequence and not any(fnmatch.fnmatchcase(rel.as_posix(), p) for p in args.sequence):
                    continue
                files = image_files(sequence)
                if files:
                    if len({p.stem for p in files}) != len(files):
                        raise ValueError(f'Duplicate frame stems: {sequence}')
                    destinations = [rel]
                    if not args.no_mirror:
                        other = 'ir' if args.source_modality == 'vi' else 'vi'
                        paired = Path(other + view.name[2:]) / distance.name / sequence.name
                        pair_dir = args.dataset_root / paired
                        if not pair_dir.is_dir() or [p.name for p in image_files(pair_dir)] != [p.name for p in files]:
                            raise ValueError(f'Missing paired modality or mismatched frame names: {pair_dir}')
                        destinations.append(paired)
                    found.append((rel, files[:args.limit_frames] if args.limit_frames else files, destinations))
    return found[:args.limit_sequences] if args.limit_sequences else found


def expected_coco_classes(sequence_name):
    """Filter by broad COCO categories; dataset class IDs are assigned from sequence identities."""
    if 'pedestrian' in sequence_name:
        return {0}
    if 'bicycle' in sequence_name or 'motorcycle' in sequence_name:
        return {1, 3}
    if re.match(r'0[1-6]_', sequence_name):
        return {2, 5, 7}
    return None


def load_frame_metadata(dataset_root, rel, files):
    """Associate the moving target using projected boxes; metadata determines location and absence, not segmentation masks."""
    paths = list((dataset_root / 'metadata' / 'sequences').glob('*.json'))
    matches = []
    for path in paths:
        # Filter by filename prefix before parsing to avoid reading every metadata file for each sequence.
        if not path.name.startswith(rel.name + '_angle_'):
            continue
        data = json.loads(path.read_text())
        if data.get('rgb_directory') == rel.as_posix():
            matches.append((path, data))
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one matching fixed-camera metadata file: {rel}, found={len(matches)}')
    path, data = matches[0]
    if data.get('camera_motion') != 'fixed_world_transform':
        raise ValueError(f'Metadata does not describe a fixed camera: {path}')
    by_id = {int(f['frame']): f for f in data['frames']}
    if len(by_id) != len(data['frames']):
        raise ValueError(f'Duplicate frame identifiers in metadata: {path}')
    frames = {}
    for image in files:
        frame = by_id[int(image.stem)]
        box = frame['bbox_xyxy']
        if not confirmed_absent(frame) and (box is None or len(box) != 4 or not np.isfinite(box).all()):
            raise ValueError(f'Invalid projected bounding box: {path}, {image.name}')
        if frame['semantic_target_pixels'] < 0:
            raise ValueError(f'Invalid pixel count: {path}, {image.name}')
        frames[image.name] = frame
    return frames, hashlib.sha256(path.read_bytes()).hexdigest()


def confirmed_absent(meta):
    return meta['semantic_target_pixels'] == 0 and meta['target_intersects_frustum'] is False


def annotation_from_result(result, image, class_id, args, sequence_name, meta):
    height, width = image.shape[:2]
    row = dict(status='out_of_frame' if confirmed_absent(meta) else 'missed_visible_target',
               confidence='', coco_class_id='', coco_class_name='', x=0, y=0,
               width=0, height=0, foreground_pixels=0,
               semantic_target_pixels=meta['semantic_target_pixels'],
               target_intersects_frustum=meta['target_intersects_frustum'])
    empty = np.zeros((height, width), np.uint8)
    if confirmed_absent(meta) or result is None or result.masks is None or result.boxes is None:
        return empty, '', '', row
    # Clip projected boxes to the image before instance association; targets may be partially outside the frame.
    ref = np.asarray(meta['bbox_xyxy'], dtype=float)
    ref[[0, 2]] = np.clip(ref[[0, 2]], 0, width)
    ref[[1, 3]] = np.clip(ref[[1, 3]], 0, height)
    if ref[2] <= ref[0] or ref[3] <= ref[1]:
        return empty, '', '', row
    masks = result.masks.data.detach().cpu().numpy()
    if masks.shape[1:] != (height, width):
        raise ValueError('Expected full-resolution masks with retina_masks=True')
    allowed = None if args.no_class_filter else expected_coco_classes(sequence_name)
    candidates = []
    for i, (box, coco, confidence, raw) in enumerate(zip(
            result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int),
            result.boxes.conf.cpu().numpy(), masks)):
        if allowed is not None and coco not in allowed:
            continue
        lo, hi = np.maximum(ref[:2], box[:2]), np.minimum(ref[2:], box[2:])
        intersection = float(np.maximum(hi - lo, 0).prod())
        ref_area = float((ref[2:] - ref[:2]).prod())
        area = float(np.maximum(box[2:] - box[:2], 0).prod())
        iou = intersection / max(ref_area + area - intersection, 1)
        if iou < .05 or intersection / max(area, 1) < .4:
            continue
        mask = (raw > .5).astype(np.uint8) * 255
        # Crop with a slightly expanded projected box to remove adjacent background instances.
        pad = max(3, int(.03 * max(ref[2:] - ref[:2])))
        x1, y1 = np.maximum(np.floor(ref[:2]).astype(int) - pad, 0)
        x2, y2 = np.minimum(np.ceil(ref[2:]).astype(int) + pad, [width, height])
        mask[:y1] = 0
        mask[y2:] = 0
        mask[:, :x1] = 0
        mask[:, x2:] = 0
        if np.count_nonzero(mask) >= 3:
            candidates.append((iou, float(confidence), i, mask))
    if not candidates:
        return empty, '', '', row
    _, _, index, mask = max(candidates, key=lambda c: c[:2])
    yy, xx = np.nonzero(mask)
    x, y = int(xx.min()), int(yy.min())
    box_w, box_h = int(xx.max()) + 1 - x, int(yy.max()) + 1 - y
    bbox = f'{class_id} {(x + box_w / 2) / width:.8f} {(y + box_h / 2) / height:.8f} {box_w / width:.8f} {box_h / height:.8f}\n'
    # Match the final cropped mask; Ultralytics merges disconnected contours into one instance polygon.
    from ultralytics.data.converter import merge_multi_segment
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    parts = [c.reshape(-1, 2) for c in contours if len(c) >= 3]
    polygon = (np.concatenate(merge_multi_segment(parts)) if len(parts) > 1
               else parts[0] if parts else np.empty((0, 2)))
    segment = ''
    if len(polygon) >= 3:
        normalized = np.clip(polygon / np.array([width, height]), 0, 1).ravel()
        segment = str(class_id) + ' ' + ' '.join(f'{v:.8f}' for v in normalized) + '\n'
    coco_id = int(result.boxes.cls[index].item())
    row.update(status='ok', confidence=float(result.boxes.conf[index].item()),
               coco_class_id=coco_id, coco_class_name=result.names[coco_id],
               x=x, y=y, width=box_w, height=box_h, foreground_pixels=len(xx))
    return mask, bbox, segment, row


def profile_for(args, files, destinations, class_id, metadata_hash, version=VERSION):
    weight = args.weights.stat()
    value = dict(version=version, metadata_hash=metadata_hash, weights=str(args.weights), weight_size=weight.st_size,
                 weight_mtime=weight.st_mtime_ns, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                 max_center_distance=args.max_center_distance, no_class_filter=args.no_class_filter,
                 retry_imgsz=args.retry_imgsz, retry_conf=args.retry_conf,
                 retry_roi=args.retry_roi,
                 source_modality=args.source_modality, class_id=class_id,
                 destinations=[p.as_posix() for p in destinations],
                 frames=[(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in files])
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def output_paths(root, rel, stem):
    return (root / 'label' / rel / f'{stem}.txt',
            root / 'label' / rel / 'segments' / f'{stem}.txt',
            root / 'mask' / rel / f'{stem}_mask.png')


def write_tracking_labels(label_dir, rows):
    """Save tracking labels only for valid targets, with a separate filename index for exact frame alignment."""
    valid = [row for row in rows if row['status'] == 'ok']
    atomic_write(label_dir / 'groundtruth.txt', ''.join(
        ','.join(str(row[k]) for k in ('x', 'y', 'width', 'height')) + '\n' for row in valid))
    atomic_write(label_dir / 'groundtruth_frames.txt', ''.join(row['frame'] + '\n' for row in valid))
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=['frame', 'x', 'y', 'width', 'height'])
    writer.writeheader()
    writer.writerows({k: row[k] for k in writer.fieldnames} for row in valid)
    atomic_write(label_dir / 'tracking_boxes.csv', buffer.getvalue())


def outputs_complete(root, rel, files):
    label_dir = root / 'label' / rel
    if not all((label_dir / n).is_file() for n in
               ('annotations.csv', 'groundtruth.txt', 'groundtruth_frames.txt', 'tracking_boxes.csv')):
        return False
    rows = list(csv.DictReader((label_dir / 'annotations.csv').open()))
    if [row['frame'] for row in rows] != [p.name for p in files]:
        return False
    for row, frame in zip(rows, files):
        box, segment, mask = output_paths(root, rel, frame.stem)
        if not mask.is_file():
            return False
        if row['status'] == 'ok':
            if not box.is_file() or not segment.is_file():
                return False
        elif box.exists() or segment.exists():
            return False
    valid_frames = [row['frame'] for row in rows if row['status'] == 'ok']
    return ((label_dir / 'groundtruth_frames.txt').read_text().splitlines() == valid_frames
            and len((label_dir / 'groundtruth.txt').read_text().splitlines()) == len(valid_frames))


def migrate_legacy_labels(args, files, destinations, markers, signature, legacy_signature):
    """Upgrade completed results only when inference signatures match; reuse masks without repeating inference."""
    if not all(p.is_file() for p in markers):
        return
    states = [json.loads(p.read_text()) for p in markers]
    if not all(s.get('signature') in (legacy_signature, signature) for s in states):
        return
    if all(s.get('signature') == signature for s in states):
        return
    tables = []
    for rel in destinations:
        label_dir = args.output_root / 'label' / rel
        rows = list(csv.DictReader((label_dir / 'annotations.csv').open()))
        if [r['frame'] for r in rows] != [p.name for p in files]:
            raise ValueError(f'Existing results contain a different frame set: {rel}')
        for row, frame in zip(rows, files):
            box, segment, mask = output_paths(args.output_root, rel, frame.stem)
            if not mask.is_file() or (row['status'] == 'ok' and (not box.is_file() or not segment.is_file())):
                raise ValueError(f'Existing results are incomplete; regenerate them: {rel / frame.name}')
        tables.append(rows)
    if any(rows != tables[0] for rows in tables[1:]):
        raise ValueError('Existing per-frame records differ between modalities')
    for rel, rows, marker, state in zip(destinations, tables, markers, states):
        for row, frame in zip(rows, files):
            if row['status'] != 'ok':
                for path in output_paths(args.output_root, rel, frame.stem)[:2]:
                    path.unlink(missing_ok=True)
        label_dir = args.output_root / 'label' / rel
        write_tracking_labels(label_dir, rows)
        write_json(label_dir / '_STATE.json', dict(signature=signature, frames=[p.name for p in files]))
        write_json(marker, dict(signature=signature, counts=state['counts'], label_layout='target_only'))


def process_sequence(model, args, rel, files, destinations, class_id):
    metadata, metadata_hash = load_frame_metadata(args.dataset_root, rel, files)
    signature = profile_for(args, files, destinations, class_id, metadata_hash)
    markers = [args.output_root / 'label' / d / '_SUCCESS.json' for d in destinations]
    if not args.overwrite:
        migrate_legacy_labels(args, files, destinations, markers, signature,
                              profile_for(args, files, destinations, class_id, metadata_hash, version=1))
    if not args.overwrite and all(p.exists() for p in markers):
        states = [json.loads(p.read_text()) for p in markers]
        complete = all(s.get('signature') == signature for s in states)
        complete = complete and all(outputs_complete(args.output_root, d, files) for d in destinations)
        if complete:
            return dict(sequence=rel.as_posix(), **states[0]['counts'], skipped=True)
    # An interrupted job can resume; changed inputs or settings require explicit --overwrite.
    for d in destinations:
        state_path = args.output_root / 'label' / d / '_STATE.json'
        label_dir, mask_dir = state_path.parent, args.output_root / 'mask' / d
        occupied = any(p.exists() and any(p.iterdir()) for p in (label_dir, mask_dir))
        if occupied and not args.overwrite:
            if not state_path.exists() or json.loads(state_path.read_text()).get('signature') != signature:
                raise ValueError(f'Output exists with a different configuration: {label_dir}; choose another --output-root or use --overwrite')
    # A changed frame set requires a new output directory to avoid stale files, even with --overwrite.
    for d in destinations:
        state_path = args.output_root / 'label' / d / '_STATE.json'
        if state_path.exists():
            old = json.loads(state_path.read_text())
            if old.get('frames') != [p.name for p in files]:
                raise ValueError(f'The output frame set changed; choose another --output-root: {state_path}')
        write_json(state_path, dict(signature=signature, frames=[p.name for p in files]))
    for marker in markers:
        if marker.exists():
            marker.unlink()
    rows, groundtruth = [], []
    for offset in range(0, len(files), args.batch_size):
        batch_files = files[offset:offset + args.batch_size]
        images = []
        for path in batch_files:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f'Cannot read image: {path}')
            for d in destinations[1:]:
                pair = cv2.imread(str(args.dataset_root / d / path.name), cv2.IMREAD_UNCHANGED)
                if pair is None or pair.shape[:2] != image.shape[:2]:
                    raise ValueError(f'Paired images are unreadable or have different dimensions: {d / path.name}')
            images.append(image)
        visible_indices = [i for i, p in enumerate(batch_files) if not confirmed_absent(metadata[p.name])]
        predictions = model.predict(source=[images[i] for i in visible_indices], conf=args.conf,
                                    iou=args.iou, imgsz=args.imgsz, device=args.device,
                                    retina_masks=True, verbose=False, save=False) if visible_indices else []
        if len(predictions) != len(visible_indices):
            raise ValueError('Inference returned an unexpected number of frames')
        results = [None] * len(images)
        for i, prediction in zip(visible_indices, predictions):
            results[i] = prediction
        for path, image, result in zip(batch_files, images, results):
            mask, bbox, segment, row = annotation_from_result(result, image, class_id, args, rel.name, metadata[path.name])
            row['inference_pass'] = 'metadata_absent' if confirmed_absent(metadata[path.name]) else 'primary'
            if row['status'] == 'missed_visible_target' and args.retry_imgsz:
                # Retry missed edge or small targets using the projected position for the current frame.
                retry = model.predict(source=image, conf=args.retry_conf, iou=args.iou,
                                      imgsz=args.retry_imgsz, device=args.device,
                                      retina_masks=True, verbose=False, save=False)[0]
                mask, bbox, segment, row = annotation_from_result(retry, image, class_id, args, rel.name, metadata[path.name])
                row['inference_pass'] = 'retry'
            if row['status'] == 'missed_visible_target' and args.retry_roi:
                # Expand the target ROI only after a missed detection to improve recovery of small or edge targets.
                h, w = image.shape[:2]
                ref = np.array(metadata[path.name]['bbox_xyxy'], dtype=float)
                ref[[0, 2]] = np.clip(ref[[0, 2]], 0, w)
                ref[[1, 3]] = np.clip(ref[[1, 3]], 0, h)
                padding = np.maximum((ref[2:] - ref[:2]) * .35, 32)
                x0, y0 = np.maximum(np.floor(ref[:2] - padding).astype(int), 0)
                x2, y2 = np.minimum(np.ceil(ref[2:] + padding).astype(int), [w, h])
                if x2 > x0 and y2 > y0:
                    crop = image[y0:y2, x0:x2]
                    roi_meta = dict(metadata[path.name])
                    roi_meta['bbox_xyxy'] = (np.array(roi_meta['bbox_xyxy']) - [x0, y0, x0, y0]).tolist()
                    prediction = model.predict(source=crop, conf=args.retry_conf, iou=args.iou,
                                               imgsz=args.imgsz, device=args.device, retina_masks=True,
                                               verbose=False, save=False)[0]
                    roi_mask, roi_bbox, roi_segment, roi_row = annotation_from_result(
                        prediction, crop, class_id, args, rel.name, roi_meta)
                    if roi_row['status'] == 'ok':
                        mask = np.zeros((h, w), np.uint8)
                        mask[y0:y2, x0:x2] = roi_mask
                        row = roi_row
                        row.update(x=row['x'] + int(x0), y=row['y'] + int(y0), inference_pass='roi_retry')
                        bx, by, bw, bh = [row[k] for k in ('x', 'y', 'width', 'height')]
                        bbox = f'{class_id} {(bx+bw/2)/w:.8f} {(by+bh/2)/h:.8f} {bw/w:.8f} {bh/h:.8f}\n'
                        if roi_segment:
                            polygon = np.array(list(map(float, roi_segment.split()[1:]))).reshape(-1, 2)
                            polygon = (polygon * [x2-x0, y2-y0] + [x0, y0]) / [w, h]
                            segment = str(class_id) + ' ' + ' '.join(f'{v:.8f}' for v in polygon.ravel()) + '\n'
                        else:
                            segment = ''
            ok, encoded = cv2.imencode('.png', mask)
            if not ok:
                raise ValueError(f'PNG encoding failed: {path}')
            png_bytes = encoded.tobytes()
            for d in destinations:
                box_path, segment_path, mask_path = output_paths(args.output_root, d, path.stem)
                if row['status'] == 'ok':
                    atomic_write(box_path, bbox)
                    atomic_write(segment_path, segment)
                else:
                    box_path.unlink(missing_ok=True)
                    segment_path.unlink(missing_ok=True)
                atomic_write(mask_path, png_bytes)
            rows.append(dict(frame=path.name, dataset_class_id=class_id, **row))
        print(f'  {rel}: {min(offset + args.batch_size, len(files))}/{len(files)}', flush=True)
    csv_buffer = io.StringIO(newline='')
    writer = csv.DictWriter(csv_buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    counts = dict(frames=len(rows), detected=sum(r['status'] == 'ok' for r in rows),
                  missed=sum(r['status'] == 'missed_visible_target' for r in rows),
                  out_of_frame=sum(r['status'] == 'out_of_frame' for r in rows), modalities=len(destinations))
    for d, marker in zip(destinations, markers):
        label_dir = args.output_root / 'label' / d
        write_tracking_labels(label_dir, rows)
        atomic_write(label_dir / 'annotations.csv', csv_buffer.getvalue())
        write_json(marker, dict(signature=signature, counts=counts, label_layout='target_only'))
    return dict(sequence=rel.as_posix(), **counts, skipped=False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output-root', type=Path, help='Create label/ and mask/ under the dataset root by default')
    parser.add_argument('--weights', type=Path, default=Path(__file__).resolve().parents[1]/'weights/segmentation/yolov8n-seg.pt')
    parser.add_argument('--source-modality', choices=['vi'], default='vi', help='Infer RGB frames only when the paired grayscale modality is derived from RGB')
    parser.add_argument('--no-mirror', action='store_true', help='Write the source modality only; paired output is enabled by default')
    parser.add_argument('--angles', type=int, nargs='+')
    parser.add_argument('--sequence', action='append', help='Repeatable relative sequence glob, for example */distance_09m/01_jeep')
    parser.add_argument('--limit-sequences', type=int, default=0)
    parser.add_argument('--limit-frames', type=int, default=0)
    parser.add_argument('--conf', type=float, default=0.15)
    parser.add_argument('--iou', type=float, default=0.7)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--retry-imgsz', type=int, default=1280, help='Retry input resolution after a missed detection; 0 disables retries')
    parser.add_argument('--retry-conf', type=float, default=0.05)
    parser.add_argument('--no-retry-roi', dest='retry_roi', action='store_false', help='Disable target-ROI retries after missed detections')
    parser.add_argument('--device', default='0', help='GPU index or cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-center-distance', type=float, default=0.18, help='Candidate-center distance from the image center divided by the image diagonal')
    parser.add_argument('--no-class-filter', action='store_true', help='Disable sequence-name-based COCO category filtering')
    parser.add_argument('--overwrite', action='store_true', help='Regenerate existing outputs; the frame set must remain identical')
    parser.add_argument('--dry-run', action='store_true', help='Check pairing and print jobs without loading a model or writing outputs')
    args = parser.parse_args()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output_root = (args.output_root or args.dataset_root).expanduser().resolve()
    args.weights = args.weights.expanduser().resolve()
    if not args.dataset_root.is_dir() or not args.weights.is_file():
        parser.error('Dataset directory or model weights not found; weights are not downloaded automatically')
    if (args.batch_size < 1 or args.imgsz < 32 or args.limit_frames < 0 or args.limit_sequences < 0
            or (args.retry_imgsz != 0 and args.retry_imgsz < 32) or not 0 < args.retry_conf <= 1
            or not 0 < args.conf <= 1 or not 0 < args.iou <= 1 or not 0 < args.max_center_distance <= 0.5):
        parser.error('Invalid batch size, image size, limits, or thresholds')
    return args


def main():
    args = parse_args()
    sequences = discover(args)
    if not sequences:
        raise ValueError('No matching image sequences found')
    # Build stable class IDs from all sequence directories so subset selection does not change them.
    names = sorted({s.name for v in args.dataset_root.glob('*_camera_view_*') if v.is_dir()
                    for d in v.glob('distance_*') if d.is_dir() for s in d.iterdir() if s.is_dir()}, key=natural_key)
    class_map = {name: i for i, name in enumerate(names)}
    total = sum(len(files) for _, files, _ in sequences)
    print(f'Found {len(sequences)} source sequences and {total} frames; mirror paired modality: {not args.no_mirror}', flush=True)
    if args.dry_run:
        for rel, files, destinations in sequences:
            load_frame_metadata(args.dataset_root, rel, files)
            print(f'{rel}: {len(files)} frames -> {[str(d) for d in destinations]}')
        return 0
    from ultralytics import YOLO
    model = YOLO(str(args.weights), task='segment')
    if model.task != 'segment':
        raise ValueError('The checkpoint is not a segmentation model and cannot produce instance masks')
    args.output_root.mkdir(parents=True, exist_ok=True)
    class_map_path = args.output_root / 'label' / 'classes.json'
    if class_map_path.exists() and json.loads(class_map_path.read_text()) != class_map:
        raise ValueError('Existing classes.json differs from the dataset classes; use another output directory')
    write_json(class_map_path, class_map)
    started = time.time()
    report = dict(version=VERSION, dataset_root=str(args.dataset_root), output_root=str(args.output_root),
                  weights=str(args.weights), config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  sequences=[], errors=[], completed=False)
    # Preserve the existing dataset validation_report.json.
    report_path = args.output_root / 'label' / 'annotation_report.json'
    try:
        for index, (rel, files, destinations) in enumerate(sequences, 1):
            print(f'[{index}/{len(sequences)}] {rel}', flush=True)
            try:
                record = process_sequence(model, args, rel, files, destinations, class_map[rel.name])
                report['sequences'].append(record)
                print(f"  detected={record['detected']}, missed={record['missed']}, skipped={record['skipped']}", flush=True)
            except Exception as exc:
                report['errors'].append(dict(sequence=rel.as_posix(), error=str(exc)))
                print(f'[ERROR] {rel}: {exc}', file=sys.stderr, flush=True)
        report['completed'] = not report['errors']
    finally:
        report['elapsed_seconds'] = round(time.time() - started, 3)
        report['totals'] = {key: sum(s[key] for s in report['sequences']) for key in ('frames', 'detected', 'missed', 'out_of_frame')}
        write_json(report_path, report)
    print(json.dumps(dict(report=str(report_path), **report['totals'], errors=len(report['errors'])), ensure_ascii=False))
    return 0 if report['completed'] else 2


if __name__ == '__main__':
    sys.exit(main())
