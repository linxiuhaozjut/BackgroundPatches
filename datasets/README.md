# Dataset Directories

- `generator_train/dataset_400_2/`: paired source images and annotations for generator training.
- `generator_train/prepared/`: generated foreground images, prototypes, and latent cache.
- `carla/dataset_400_2/`: paired CARLA sequences used for patch optimization and evaluation.
- `splits/`: train, validation, development, and test sequence lists.

Configure dataset roots in `../configs/paths.json`; both source roots may reference the same dataset. `.gitkeep` files retain empty directory structure and contain no data. See [data documentation](../docs/data.md).
