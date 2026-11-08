# ViPT Tracker Integration

`lib/` contains the model, tracker interface, configuration, and supporting utilities. `experiments/vipt/deep_rgbt.yaml` defines the paired tracking configuration.

The optimization module uses `BoxOracle` to initialize the tracker and retrieve bounding boxes without differentiating through the tracker. Optional clean-data training is implemented in `../optimization/train_dataset4002_full.py`.

Provide a checkpoint under `../weights/tracker/` and a matching model configuration. Environment settings are derived from the repository structure through the framework configuration modules.
