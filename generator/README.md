# Conditional Patch Generator

| Module | Responsibility |
| --- | --- |
| `paired_dit_data.py` | Foreground extraction, keyframes, VAE descriptors, prototype clustering, and latent caching |
| `paired_dit_model.py` | Modality-specific projection, prototype fusion, q construction, and conditional DiT |
| `paired_prototype_loss.py` | Smooth maximum cosine distance to the complete prototype set |
| `train_paired_dit.py` | Training, EMA, validation, checkpointing, and resume |
| `sample_paired_dit.py` | Paired sampling and the `generate_paired_patches` API |
| `evaluate_prototype_samples.py` | Prototype-distance measurements for exported samples |
| `models.py`, `diffusion/` | DiT backbone and diffusion implementation |

See [generator documentation](../docs/generator.md) for equations and [configuration](../docs/configuration.md) for parameter profiles.
