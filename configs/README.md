# Configuration Files

- `paths.json`: dataset, cache, checkpoint, split, and output paths relative to the repository root.
- `prepare.json`: foreground resolution, keyframe selection, VAE descriptor settings, and data split.
- `generator_base.json`: base conditional generator training.
- `generator_prototype.json`: prototype-constrained warm-start training.
- `sampling.json`: paired diffusion sampling.
- `optimization_coherent.json`: color-coherent black-box optimization.
- `optimization_pixel.json`: pixel-wise optimization without TV regularization.

`scripts/pipeline.py` reads these files to produce stage commands. See [configuration reference](../docs/configuration.md).
