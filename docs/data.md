# Dataset Format

## Dataset roots

| Purpose | Default directory |
| --- | --- |
| Generator source sequences | `datasets/generator_train/dataset_400_2/` |
| Generator prepared cache | `datasets/generator_train/prepared/` |
| CARLA optimization sequences | `datasets/carla/dataset_400_2/` |
| Sequence split lists | `datasets/splits/` |

Set `generator_dataset` and `carla_dataset` in `configs/paths.json`. They may reference the same dataset, so duplicate copies are unnecessary. Empty directories contain `.gitkeep` placeholders only.

## Paired sequence format

```text
dataset_400_2/
├── vi_camera_view_30/
│   └── distance_09m/
│       └── 10_pedestrian_0002/
│           ├── 014971.png
│           └── ...
├── ir_camera_view_30/
│   └── distance_09m/10_pedestrian_0002/...
├── vi_camera_view_45/...
├── ir_camera_view_45/...
├── ...
├── label/
│   ├── vi_camera_view_30/distance_09m/10_pedestrian_0002/
│   │   ├── annotations.csv
│   │   ├── 014971.txt
│   │   └── ...
│   └── ir_camera_view_30/...
├── mask/
│   ├── vi_camera_view_30/distance_09m/10_pedestrian_0002/
│   │   ├── 014971_mask.png
│   │   └── ...
│   └── ir_camera_view_30/...
└── metadata/
    └── ...
```

The visible (`vi`) and infrared (`ir`) sequences must have matching relative paths, frame names, and image dimensions. Images are individual frames rather than video containers. Directory names encode camera viewpoint and distance. Supported experiment settings include angles of 30, 45, 60, 75, 90, 105, 120, and 135 degrees and distances of 9, 12, 15, 18, and 21 meters. `09m` means 9 meters.

## Labels and masks

- A valid target frame has one YOLO box: `class_id center_x center_y width height`, normalized by image dimensions.
- `annotations.csv` records frame identity and annotation status. Its fields are defined by `build_fixed_camera_annotations.py`.
- Masks use the original image resolution, with target pixels equal to 255 and background pixels equal to 0.
- When the target has left the frame, save an all-zero mask and omit the per-frame target label. Tracking labels include only valid targets and preserve an explicit filename index.
- Acquisition metadata describes projected target boxes, target pixel counts, and frustum intersection. It associates detections and distinguishes confirmed target absence from missed detections.

The annotation builder infers visible frames and mirrors annotations to paired grayscale frames. This assumes that the modalities are spatially aligned and share the same target geometry; separate sensors may require calibration and independent annotations.

## Prepared generator cache

```text
prepared/
├── manifest.json
├── keyframes.json
├── prototypes.pt
├── latents_visible.pt
├── latents_infrared.pt
├── crops/visible/...
├── crops/infrared/...
└── crop_masks/...
```

The manifest records preprocessing settings, sample identity, splits, bounding boxes, and keyframe assignments. Only a cache marked as completed can be used for training. Regenerate the cache after changing preprocessing settings, source data, or VAE identity.

## Sequence splits

Place `train.json`, `val.json`, `development.json`, and `test.json` under `datasets/splits/`. Each file is a list of relative visible-sequence paths:

```json
[
  "vi_camera_view_30/distance_09m/10_pedestrian_0002"
]
```

This example illustrates the schema, not a recommended experimental split. Split complete sequences rather than frames. `development.json` contains the train/validation union; only sequences listed in `train.json` may update patches. Keep test sequences excluded from updates. An end-to-end held-out evaluation must also exclude test sequences from generator training and prototype construction.
