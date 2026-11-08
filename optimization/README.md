# Black-Box Patch Optimization

- `optimize_dataset4002_coherent.py`: full-resolution stochastic updates with TV loss and proximal candidate regularization.
- `optimize_dataset4002_blackbox.py`: pixel-wise variant without proximal regularization.
- `train_dataset4002_full.py`: optional full-depth ViPT training on clean paired sequences.
- `export_blackbox_preview.py`: preview rendering from saved runs.
- `summarize_simple_attack.py`: sequence-level metric aggregation.
- `verify_blackbox_patch.py`, `verify_coherent_patch.py`: existing placement/output and regularization checks.

`Experiment` manages paired sequences and patch composition. `BoxOracle` adapts the ViPT interface to bounding-box-only queries. See [optimization documentation](../docs/optimization.md).
