# Configuration Reference

All paths in `configs/paths.json` are resolved from the repository root. Use a fresh output directory for training or generation, or select the supported resume option where applicable. Remove placeholder files from directories that must be empty before executing a stage.

## Preprocessing and generator

| Setting | Value |
| --- | --- |
| Foreground size | 256x256 |
| MSE scale / threshold | [0,1] / strictly greater than 0.5 |
| Keyframe reset | Per target identity |
| VAE descriptor grid | 8x8 |
| Descriptor dimension | 256 |
| Prototype count | Number of target identities |
| Backbone | DiT-S/2 |
| Base stage | 1,000 steps, 8 pairs/batch, learning rate 1e-4 |
| Prototype stage | 2,000 additional steps, 8 pairs/batch, learning rate 5e-5 |
| EMA decay | 0.99 |
| Paired/shared/private weights | 0.05 / 0.01 / 0.01 |
| Prototype weight / temperature | 0.5 / 0.1 |
| Prototype interval / branch batch | Every 4 steps / 1 pair |
| Prototype warmup / timestep range | 100 steps / 100 through 600 |
| Sampling | EMA, DDIM, 100 steps, CFG 3, seed 42 |

The base profile disables the prototype objective. The prototype stage warm-starts from the base conditional EMA checkpoint, resetting optimizer state and stage step count. These are configurable profiles, not universal convergence criteria.

## Black-box optimizer

| Setting | Value |
| --- | --- |
| Iterations | 12 |
| Sequences per step | 10 |
| Random directions | 4 |
| Initial perturbation scale | 0.06 |
| Initial learning rate | 0.02 |
| IoU / guidance / TV weights | 0.6 / 0.3 / 0.1 |
| TV proximal strength / steps | 0.06 / 50 |
| Random seed | 42 |
| Worker devices | 0 through 7; adjust to available resources |
| Visualization interval | 0: final visualization only |

The pixel-wise profile disables TV and has no proximal mapping. Command-line defaults may differ from these experiment profiles; use the JSON files to inspect the exact command preview.

## Command preview

```bash
python scripts/pipeline.py prepare
python scripts/pipeline.py train-base
python scripts/pipeline.py train-prototype
python scripts/pipeline.py generate
python scripts/pipeline.py optimize
```

The preview prints stage commands only. Direct execution of each entry point requires its dependencies and input artifacts. For ViPT, ensure `tracker/` is on the Python import path and select the YAML matching the checkpoint. The optional clean-data tracker training script is `optimization/train_dataset4002_full.py`.
