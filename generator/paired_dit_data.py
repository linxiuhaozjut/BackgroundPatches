"""Foreground crops, independent keyframes, frozen VAE descriptors, and paired prototype clustering."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

PROJECT = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT.parent / 'datasets/generator_train/dataset_400_2'
MODES = ('visible', 'infrared')
LATENT_SCALE = 0.18215
SCHEMA = 1


def natural_key(value):
    return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', str(value))]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def torch_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    torch.save(value, temp)
    temp.replace(path)


def file_identity(path):
    path = Path(path).resolve()
    s = path.stat()
    return [str(path), s.st_size, s.st_mtime_ns]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_vae(path, device):
    from diffusers import AutoencoderKL
    path = Path(path).resolve()
    if not (path / 'config.json').is_file():
        raise FileNotFoundError(f'VAE directory not found: {path}')
    vae = AutoencoderKL.from_pretrained(str(path), local_files_only=True).to(device).eval()
    vae.requires_grad_(False)
    return vae


def frame_mse(a, b, scale='unit'):
    """Compute MSE in floating point to avoid uint8 overflow; unit uses [0,1] and byte uses [0,255]."""
    a, b = a.astype(np.float32), b.astype(np.float32)
    if scale == 'unit':
        a, b = a / 255., b / 255.
    elif scale != 'byte':
        raise ValueError(scale)
    if a.shape != b.shape:
        raise ValueError('MSE input dimensions differ')
    return float(np.mean((a - b) ** 2, dtype=np.float64))


def foreground_crop(image_path, label_path, mask_path, size, mode, resize_mode):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR if mode == 'visible' else cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if image is None or mask is None or mask.ndim != 2 or image.shape[:2] != mask.shape:
        raise ValueError(f'Image or mask is unreadable or has incompatible dimensions: {image_path}')
    if not np.all((mask == 0) | (mask == 255)) or not mask.any():
        raise ValueError(f'Masks for valid labeled frames must be nonempty binary 0/255 images: {mask_path}')
    values = [float(v) for v in label_path.read_text().split()]
    if len(values) != 5 or not all(math.isfinite(v) for v in values):
        raise ValueError(f'Expected one valid YOLO target box: {label_path}')
    h, w = mask.shape
    yy, xx = np.nonzero(mask)
    box = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
    _, cx, cy, bw, bh = values
    label_box = np.array([(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h])
    if not np.allclose(box, label_box, atol=.01):
        raise ValueError(f'Label and mask bounding boxes do not match: {label_path}')
    x0, y0, x1, y1 = box
    crop, alpha = image[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1]
    crop[alpha == 0] = 0
    if resize_mode == 'letterbox':
        side = max(crop.shape[:2])
        top, left = (side-crop.shape[0])//2, (side-crop.shape[1])//2
        crop = cv2.copyMakeBorder(crop, top, side-crop.shape[0]-top,
                                 left, side-crop.shape[1]-left, cv2.BORDER_CONSTANT, value=0)
        alpha = cv2.copyMakeBorder(alpha, top, side-alpha.shape[0]-top,
                                  left, side-alpha.shape[1]-left, cv2.BORDER_CONSTANT, value=0)
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (size, size), interpolation=cv2.INTER_NEAREST)
    crop[alpha == 0] = 0
    return crop, alpha, box, int(values[0])


def discover(args):
    data, labels, masks = args.dataset_root, args.label_root, args.mask_root
    sequences = []
    for view in sorted(data.glob('vi_camera_view_*'), key=natural_key):
        angle = float(view.name.split('_')[-1])
        if args.angles and angle not in args.angles:
            continue
        for distance in sorted(view.glob('distance_*m'), key=natural_key):
            meters = float(distance.name.removeprefix('distance_').removesuffix('m'))
            if args.distances and meters not in args.distances:
                continue
            for target in sorted(distance.iterdir(), key=natural_key):
                if not target.is_dir() or args.sequence_names and target.name not in args.sequence_names:
                    continue
                rel = target.relative_to(data)
                paired = Path('ir'+rel.parts[0][2:]) / Path(*rel.parts[1:])
                tables = {}
                for mode, mr in zip(MODES, (rel, paired)):
                    tables[mode] = list(csv.DictReader((labels / mr / 'annotations.csv').open()))
                if [r['frame'] for r in tables['visible']] != [r['frame'] for r in tables['infrared']]:
                    raise ValueError(f'Paired modality frame identifiers differ: {rel}')
                valid, skipped = [], 0
                for vi, ir in zip(tables['visible'], tables['infrared']):
                    if vi['status'] != 'ok' or ir['status'] != 'ok':
                        skipped += 1
                        continue
                    frame = vi['frame']
                    paths = {}
                    for mode, mr in zip(MODES, (rel, paired)):
                        paths[mode] = dict(image=data/mr/frame, label=labels/mr/(Path(frame).stem+'.txt'),
                                           mask=masks/mr/(Path(frame).stem+'_mask.png'))
                        if not all(p.is_file() for p in paths[mode].values()):
                            raise FileNotFoundError(f'A valid frame is missing its image, label, or mask: {mr/frame}')
                    valid.append(dict(frame=frame, paths=paths))
                sequences.append(dict(sequence=rel.as_posix(), target=target.name, valid=valid, skipped=skipped))
    if not sequences or not any(s['valid'] for s in sequences):
        raise ValueError('No valid paired target annotations found')
    return sequences


def split_sequences(sequences, fraction, seed):
    rng = random.Random(seed)
    for name in sorted({s['target'] for s in sequences}):
        group = [s for s in sequences if s['target'] == name and s['valid']]
        order = list(range(len(group)))
        rng.shuffle(order)
        count = min(len(group)-1, max(1, round(len(group)*fraction))) if fraction > 0 and len(group)>1 else 0
        held_out = set(order[:count])
        for i, sequence in enumerate(group):
            sequence['split'] = 'val' if i in held_out else 'train'


def kmeans(features, count, seed=0, iterations=100):
    """Use deterministic k-means++ initialization and reassign empty clusters; prototypes are cluster arithmetic means."""
    x = np.asarray(features, np.float32)
    if len(x) < count or len(np.unique(x, axis=0)) < count:
        raise ValueError(f'Fewer than {count} valid keyframes or distinct features; lower the MSE threshold or add data')
    rng = np.random.default_rng(seed)
    centers = [x[rng.integers(len(x))].copy()]
    d = np.sum((x-centers[0])**2, axis=1)
    for _ in range(1, count):
        next_index = rng.choice(len(x), p=d.astype(np.float64)/d.sum(dtype=np.float64))
        centers.append(x[next_index].copy())
        d = np.minimum(d, np.sum((x-centers[-1])**2, axis=1))
    centers = np.array(centers)
    previous = None
    for _ in range(iterations):
        distances = np.maximum((x*x).sum(1)[:,None]+(centers*centers).sum(1)[None]-2*x@centers.T, 0)
        labels = distances.argmin(1)
        for k in range(count):
            if not np.any(labels == k):
                residual = distances[np.arange(len(x)),labels]
                sizes = np.bincount(labels, minlength=count)
                residual[sizes[labels] <= 1] = -1
                labels[int(residual.argmax())] = k
        centers = np.stack([x[labels==k].mean(0) for k in range(count)])
        if previous is not None and np.array_equal(previous, labels):
            break
        previous = labels.copy()
    return centers, labels


def assign(features, centers):
    return ((features[:,None,:]-centers[None,:,:])**2).sum(-1).argmin(1)


def prepare(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.dataset_root = args.dataset_root.resolve()
    args.mask_root = (args.mask_root or args.dataset_root/'mask').resolve()
    args.label_root = (args.label_root or args.dataset_root/'label').resolve()
    args.output_root = args.output_root.resolve()
    args.vae = args.vae.resolve()
    if args.image_size < 32 or args.image_size % 16 or args.feature_grid < 1:
        raise ValueError('image_size must be at least 32 and divisible by 16; feature_grid must be positive')
    if args.mse_threshold < 0 or not math.isfinite(args.mse_threshold) or not 0<=args.val_fraction<1 or args.batch_size<1:
        raise ValueError('Invalid MSE threshold, validation fraction, or batch size')
    if any(args.output_root == p or p in args.output_root.parents or args.output_root in p.parents
           for p in (args.dataset_root, args.mask_root, args.label_root)):
        raise ValueError('Prepared output must be separate from source data, labels, and masks, with no nested input/output paths')
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError('Prepared output is not empty; choose a new directory to avoid mixing cached features')
    sequences = discover(args)
    split_sequences(sequences, args.val_fraction, args.seed)
    targets = sorted({s['target'] for s in sequences}, key=natural_key)
    if any(not any(s['target']==name and s['valid'] for s in sequences) for name in targets):
        raise ValueError('Selected targets lack valid frames; cannot construct N prototypes')
    args.output_root.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    manifest = dict(schema=SCHEMA, completed=False, config=config, targets=targets,
                    descriptor='frozen VAE posterior mean -> adaptive average pool -> flatten -> L2 normalize',
                    prototype_count=len(targets), samples=[], sequences=[], warnings=[])
    manifest_path = args.output_root/'manifest.json'
    write_json(manifest_path, manifest)
    anchors, identities = {}, []
    for number, seq in enumerate(sequences, 1):
        manifest['sequences'].append({k:v for k,v in seq.items() if k != 'valid'})
        for frame in seq['valid']:
            entry = dict(pair_id=len(manifest['samples']), sequence=seq['sequence'], target=seq['target'],
                         class_index=targets.index(seq['target']), frame=frame['frame'], split=seq['split'], modalities={})
            for mode in MODES:
                paths = frame['paths'][mode]
                identities.extend(file_identity(path) for path in paths.values())
                crop, mask, box, dataset_class = foreground_crop(**{k+'_path':v for k,v in paths.items()},
                    size=args.image_size, mode=mode, resize_mode=args.resize_mode)
                key = (mode, seq['target'], seq['split'], seq['sequence'] if args.keyframe_reset=='sequence' else '')
                mse = None if key not in anchors else frame_mse(anchors[key], crop, args.mse_scale)
                selected = mse is None or mse > args.mse_threshold
                if selected:
                    anchors[key] = crop.copy()
                rel = Path(mode)/Path(seq['sequence'])/frame['frame']
                crop_path, mask_path = args.output_root/'crops'/rel, args.output_root/'crop_masks'/rel
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(crop_path),crop) or not cv2.imwrite(str(mask_path),mask):
                    raise IOError(crop_path)
                entry['modalities'][mode] = dict(crop=crop_path.relative_to(args.output_root).as_posix(),
                    mask=mask_path.relative_to(args.output_root).as_posix(), source_image=str(paths['image']),
                    source_mask=str(paths['mask']), source_label=str(paths['label']), bbox_xyxy=box,
                    dataset_class_id=dataset_class, keyframe=selected, mse_to_last_keyframe=mse)
            manifest['samples'].append(entry)
        print(f'crop [{number}/{len(sequences)}] {seq["sequence"]}: {len(seq["valid"])} valid',flush=True)
    vae = load_vae(args.vae, args.device)
    latent_cache = {}
    for mode in MODES:
        means, stds, features = [], [], []
        for start in range(0,len(manifest['samples']),args.batch_size):
            group = manifest['samples'][start:start+args.batch_size]
            images = []
            for row in group:
                image = cv2.imread(str(args.output_root/row['modalities'][mode]['crop']),cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                images.append(torch.from_numpy(image.copy()).permute(2,0,1).float()/127.5-1)
            with torch.inference_mode():
                dist = vae.encode(torch.stack(images).to(args.device)).latent_dist
                mean, std = dist.mean*LATENT_SCALE, dist.std*LATENT_SCALE
                descriptor = F.normalize(F.adaptive_avg_pool2d(mean.float(),args.feature_grid).flatten(1),dim=1)
            means.append(mean.cpu().half()); stds.append(std.cpu().half()); features.append(descriptor.cpu())
            if start % (args.batch_size*10)==0:
                print(f'encode {mode}: {min(start+args.batch_size,len(manifest["samples"]))}/{len(manifest["samples"])}',flush=True)
        latent_cache[mode] = dict(mean=torch.cat(means),std=torch.cat(stds),features=torch.cat(features))
    del vae
    train = np.array([r['split']=='train' for r in manifest['samples']])
    banks, assignments, key_indices, counts = {}, {}, {}, {}
    for mode in MODES:
        keys = np.array([i for i,r in enumerate(manifest['samples']) if r['split']=='train' and r['modalities'][mode]['keyframe']])
        x = latent_cache[mode]['features'].numpy()
        centers, cluster_ids = kmeans(x[keys],len(targets),args.seed)
        banks[mode] = centers
        assignments[mode] = assign(x,centers)
        key_indices[mode] = keys.tolist()
        counts[mode] = len(keys)
        for i,k in zip(keys,cluster_ids):
            manifest['samples'][int(i)]['modalities'][mode]['keyframe_cluster'] = int(k)
        if len(keys) <= len(targets)*2:
            manifest['warnings'].append(f'{mode}: Only {len(keys)} training keyframes; threshold {args.mse_threshold} may be too large. Consider a lower threshold or per-sequence reset')
    # Align arbitrary independent cluster indices using paired training-frame co-occurrence, excluding validation data.
    overlap = np.zeros((len(targets),len(targets)),np.float64)
    np.add.at(overlap,(assignments['visible'][train],assignments['infrared'][train]),1)
    normalized = overlap/np.maximum(np.sqrt(overlap.sum(1)[:,None]*overlap.sum(0)[None,:]),1)
    row_indices, permutation = linear_sum_assignment(-normalized)
    assert np.array_equal(row_indices,np.arange(len(targets)))
    banks['infrared'] = banks['infrared'][permutation]
    inverse = np.argsort(permutation)
    for i, row in enumerate(manifest['samples']):
        row['modalities']['visible']['cluster'] = int(assignments['visible'][i])
        row['modalities']['infrared']['cluster'] = int(inverse[assignments['infrared'][i]])
        if 'keyframe_cluster' in row['modalities']['infrared']:
            row['modalities']['infrared']['keyframe_cluster'] = int(inverse[row['modalities']['infrared']['keyframe_cluster']])
    tensor_bank = torch.stack([torch.from_numpy(banks[m]) for m in MODES])
    for mode in MODES:
        torch_save(args.output_root/f'latents_{mode}.pt',latent_cache[mode])
    source_fingerprint = digest(identities)
    vae_identity = [file_identity(p) for p in sorted(Path(args.vae).glob('*')) if p.suffix in ('.json','.bin','.safetensors')]
    manifest.update(completed=True, source_fingerprint=source_fingerprint, vae_identity=vae_identity,
                    keyframe_counts=counts, valid_frame_pairs=len(manifest['samples']),
                    train_pairs=int(train.sum()), val_pairs=int((~train).sum()),
                    infrared_cluster_permutation=permutation.tolist(), cluster_cooccurrence=overlap.tolist(),
                    prototypes_file='prototypes.pt')
    manifest['cache_id'] = digest(dict(config=config,source=source_fingerprint,vae=vae_identity,prototypes=tensor_bank.tolist()))
    torch_save(args.output_root/'prototypes.pt',dict(C=tensor_bank,targets=targets,cache_id=manifest['cache_id'],
                 mode_order=MODES,keyframe_indices=key_indices,infrared_permutation=permutation.tolist()))
    write_json(manifest_path,manifest)
    groups = {mode:{name:[dict(pair_id=r['pair_id'],crop=r['modalities'][mode]['crop'],split=r['split'])
                      for r in manifest['samples'] if r['target']==name and r['modalities'][mode]['keyframe']]
                   for name in targets} for mode in MODES}
    write_json(args.output_root/'keyframes.json',groups)
    print(json.dumps({k:manifest[k] for k in ('valid_frame_pairs','train_pairs','val_pairs','keyframe_counts','prototype_count','warnings')},ensure_ascii=False),flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,default=DEFAULT_DATA)
    p.add_argument('--label-root',type=Path)
    p.add_argument('--mask-root',type=Path)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--vae',type=Path,default=PROJECT.parent/'weights/vae')
    p.add_argument('--image-size',type=int,default=256)
    p.add_argument('--feature-grid',type=int,default=8)
    p.add_argument('--mse-threshold',type=float,default=.5)
    p.add_argument('--mse-scale',choices=['unit','byte'],default='unit')
    p.add_argument('--keyframe-reset',choices=['target','sequence'],default='target')
    p.add_argument('--resize-mode',choices=['stretch','letterbox'],default='stretch')
    p.add_argument('--val-fraction',type=float,default=.1)
    p.add_argument('--angles',type=float,nargs='+')
    p.add_argument('--distances',type=float,nargs='+')
    p.add_argument('--sequence-names',nargs='+')
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda:0')
    prepare(p.parse_args())


if __name__=='__main__':
    main()
