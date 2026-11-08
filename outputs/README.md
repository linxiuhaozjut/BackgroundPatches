# Generated Outputs

- `generator/base/` and `generator/prototype/`: generator checkpoints and training metrics.
- `patches/initial/`: sampled `visible.png`, `infrared.png`, `generation.json`, and `conditions.pt`.
- `optimization/coherent/` and `optimization/pixel/`: patch optimization runs.
- `tracker/`: optional clean-tracker training artifacts and tracking outputs.

Optimization runs contain `initial_patch/`, `current_patch/`, `final_patch/`, `latest/`, and numerical history. `latest/` stores the most recent visualization set. Generated data and checkpoints are excluded from version control.
