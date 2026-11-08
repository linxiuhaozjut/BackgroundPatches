# Checkpoint Layout

| Path | Expected artifact |
| --- | --- |
| `vae/` | Matching VAE `config.json` and model weights |
| `generator/dit_backbone.pt` | Compatible DiT-S/2 backbone initialization |
| `tracker/vipt.pth.tar` | ViPT checkpoint used for box queries |
| `tracker/ostrack_initialization.pth.tar` | Optional tracker-training initialization |
| `segmentation/yolov8n-seg.pt` | Segmentation checkpoint for annotation construction |

Trained conditional generator checkpoints are written under `outputs/generator/base/` and `outputs/generator/prototype/`. Sampling requires the same VAE specification used during training. ViPT architecture configuration must match its checkpoint. Weight files are not distributed with this repository.
