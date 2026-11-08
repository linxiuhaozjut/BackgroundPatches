# Patch Placement and Black-Box Optimization

## Patch placement

`apply_static_paired_patches(dataset_root, patch_dir, mask_root, output_root)` handles fixed-camera sequences. `patch_dir` contains `visible.png` and `infrared.png`. Both modalities use the same geometry and the union of their target masks.

At the nearest distance, the initial patch side equals half the shorter image dimension. Side length scales inversely with distance. The patch plane is rotated according to the camera viewpoint and projected with a pinhole model. At 90 degrees it is frontal; other viewpoints produce trapezoidal perspective. Fixed-camera mode retains the initial position, size, and perspective throughout the sequence. The optimizer uses a 9-meter reference distance for the supplied distance convention.

White mask pixels protect the target. An empty mask is allowed when acquisition metadata confirms target absence. A conservative projected-box guard can protect a missed target without changing segmentation labels. Moving-camera placement is also available through `apply_paired_patches`, with shrinking and viewpoint-dependent motion.

## Tracker interface

`BoxOracle` initializes ViPT from the first valid ground-truth box and returns subsequent predicted boxes. The optimizer receives boxes only; it does not differentiate through the tracker or consume internal tracker features. `Experiment.images` composites frames in memory using the same placement geometry as the sequence exporter.

## Losses

```text
L_iou = IoU(predicted_box, ground_truth_box)
L_guide = 0.5 * [((pred_cx - patch_cx) / image_width)^2
               + ((pred_cy - patch_cy) / image_height)^2]
TV(P) = 0.5 * [mean(abs(vertical differences))
             + mean(abs(horizontal differences))]
L_pixel = [TV(P_visible) + TV(P_infrared)] / 2
L_total = 0.6 L_iou + 0.3 L_guide + 0.1 L_pixel
```

The optimizer minimizes this objective. Box losses are averaged over valid non-initialization frames within each sequence, then over sequences. Frames without valid ground truth are excluded from box scoring. ViPT produces one joint box from the paired modalities; TV is computed separately and averaged.

## Finite-difference update

Each direction contains independent Rademacher (+1/-1) entries for every RGB pixel/channel and every infrared pixel. For D directions:

```text
sigma_t = sigma_0 / (t + 1)^0.101
g_m = mean_d[(L_plus_d - L_minus_d) / (2 sigma_t) * delta_dm]
norm = sqrt(mean(g_visible^2) + mean(g_infrared^2))
rate_t = rate_0 / (t + 1)^0.3
proposal_m = clip(P_m - rate_t * g_m / max(norm, 1e-12), 0, 1)
```

The update uses a normalized estimate, not its sign. Positive/negative perturbations and updated pixels are clipped to [0,1]. The normalization is the square root of the sum of the two modality-wise mean squares, not one mean over all concatenated channels.

## Color-coherent and pixel-wise variants

`optimize_dataset4002_coherent.py` applies a TV proximal mapping and PNG-equivalent quantization before querying each candidate. It considers positive/negative probes, the gradient proposal, a smoothing-only proposal, and the unchanged patch. The lowest-loss candidate on the current sequence batch is retained. The proximal strength is separate from the TV loss weight.

`optimize_dataset4002_blackbox.py` retains full-resolution pixel-wise updates without the proximal mapping. `configs/optimization_pixel.json` sets TV weight to zero.

With four directions, the coherent implementation performs twelve batch evaluations per iteration: current patch (1), signed probes (8), smoothing (1), gradient proposal (1), and accepted-candidate reporting (1). Twelve iterations with ten sequences therefore use 1,440 sequence evaluations for the update loop. This excludes placement checks, clean/initial/final evaluations, and test evaluation. Frame-query counts depend on sequence length and cannot be inferred from sequence-query counts alone.

## Outputs

Each run writes `initial_patch/`, `current_patch/`, `final_patch/`, numerical history, and `latest/` visualizations. The final paired patch is stored as `final_patch/visible.png` and `final_patch/infrared.png`. Visualization uses green ground-truth boxes and red predicted boxes. Only the latest visualization set is retained instead of storing videos for every iteration.

Use complete-sequence train/validation/test splits. A lower minibatch loss does not guarantee improved dataset-wide or held-out attack performance; evaluate final patches with the intended tracker and split protocol.
