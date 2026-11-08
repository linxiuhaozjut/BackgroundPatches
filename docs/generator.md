# Paired Conditional Generator

## Foreground extraction and keyframes

`paired_dit_data.py` checks that each YOLO box agrees with its binary mask, tightly crops the target, removes background pixels, and resizes the crop to a common resolution. The resized mask is reapplied so background pixels remain zero.

Keyframes are selected independently for each modality and target identity. The first valid frame is selected. Each subsequent frame is compared with the most recently selected keyframe; it is selected only when MSE is strictly greater than the threshold. The default scale is [0,1], the threshold is 0.5, and the comparison state continues across sequences belonging to the same target. Per-sequence reset is configurable.

A high threshold can retain only one keyframe per target. Inspect `keyframes.json` before training; the code reports sparse selections rather than silently changing the threshold. By default, all valid foreground frames provide diffusion supervision, while selected keyframes define the prototypes.

## Prototype construction

A frozen VAE encodes each foreground crop. Posterior means are adaptively pooled to an 8x8 grid, flattened, and L2-normalized. Four latent channels yield a 256-dimensional descriptor. Infrared frames are converted to grayscale and replicated across three channels for the shared VAE.

K-means clusters each modality independently with K=N, where N is the number of target-directory identities. Each cluster arithmetic mean defines a prototype C. Cluster indices are aligned across modalities using training-pair co-occurrence and Hungarian matching. Validation frames do not contribute to prototypes or alignment.

## Conditioning

For modality m and aligned prototype index i:

```text
h_mi = LayerNorm_m(W_m C_mi)
c_i  = MLP(concat(h_visible_i, h_infrared_i))
q_mi = concat(c_i, C_mi)
```

The complete q set is projected and pooled with learnable attention. Its pooled representation, timestep embedding, and modality embedding condition the DiT blocks. C remains fixed; projection, normalization, fusion, attention, and DiT parameters are trained jointly. Sampling does not require a target-class identifier.

## Training objectives

The base objective combines diffusion supervision with paired noise consistency, shared prototype alignment, and modality-specific prototype reconstruction:

```text
L_base = L_diffusion + 0.05 L_pair + 0.01 L_shared + 0.01 L_private
```

The optional prototype objective decodes the predicted clean latent, re-encodes the image through the frozen VAE, and compares its normalized descriptor with every prototype in the corresponding modality:

```text
d_i = 1 - cosine(feature, C_i)
L_proto = mean[tau * (logsumexp(d_i / tau) - log(N))]
```

Larger distances receive more weight. VAE weights are frozen, but gradients pass through its image decoding and encoding operations to the generator. This branch uses a single-step clean-image estimate, not backpropagation through the full sampling chain.

The provided profiles define a base training stage followed by a prototype-constrained warm start. Warm start loads conditional EMA weights and resets optimizer state and step count; resume restores a compatible training state. See [configuration](configuration.md) for values.

## Sampling

`sample_paired_dit.py::generate_paired_patches` uses a trained conditional checkpoint and VAE to generate paired patches. EMA and DDIM are the defaults. Both modalities share initial noise, while conditions and modality embeddings differ.

Outputs include `visible.png` (RGB), `infrared.png` (grayscale), `generation.json`, and `conditions.pt`. Multiple samples are stored in `pair_000`, `pair_001`, and subsequent directories. Select a complete pair and pass its directory to the optimizer. Batch size and pair index are part of the sampling protocol; a seed alone does not identify a sample from a differently sized batch.

The diffusion model learns from foreground images with q as a condition. The generated pair becomes an initial patch for a separate tracker-based optimization stage.
