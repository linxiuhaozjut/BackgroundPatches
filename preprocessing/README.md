# Annotation and Patch Placement

`build_fixed_camera_annotations.py` generates YOLO boxes, valid-target tracking labels, and binary masks for fixed-camera paired sequences. It associates segmentation instances using acquisition metadata, retries missed targets, and writes zero masks when target absence is confirmed.

`apply_paired_patches.py` provides two interfaces:

- `apply_paired_patches`: moving-camera placement with shrinking and viewpoint-dependent motion.
- `apply_static_paired_patches`: fixed-camera placement with constant position, size, and perspective throughout a sequence.

Both interfaces accept `dataset_root`, `patch_dir`, `mask_root`, and `output_root`. The patch directory contains `visible.png` and `infrared.png`. Both modality masks jointly protect the target. See [dataset format](../docs/data.md) and [placement geometry](../docs/optimization.md).
