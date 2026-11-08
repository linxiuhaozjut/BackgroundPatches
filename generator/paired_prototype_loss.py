"""Frozen-image-encoder distances with a smooth worst-prototype objective.

The target bank stays fixed. Gradients flow through VAE decoding/encoding into
the predicted clean latent and DiT; generated-image features are never detached.
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F
from paired_dit_data import LATENT_SCALE, load_vae


def prototype_distances(features, prototypes):
    """Cosine distances [2,B,N], preserving all prototypes, not just their mean."""
    return (1 - torch.einsum('mbd,mnd->mbn', F.normalize(features, dim=-1),
                             F.normalize(prototypes, dim=-1))).clamp(0, 2)


def smooth_worst(distances, temperature=.1):
    """tau*log(mean(exp(d/tau))); gradients are softmax(d/tau).

    The log(N) correction preserves the distance scale. For fixed N/tau this
    has the same gradients as logsumexp, increasingly emphasizing large gaps.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('prototype temperature must be finite and > 0')
    return temperature * (torch.logsumexp(distances / temperature, dim=-1)
                          - math.log(distances.shape[-1]))


class FrozenImagePrototypeLoss(nn.Module):
    def __init__(self, vae_path, prototypes, device, temperature=.1):
        super().__init__()
        self.vae = load_vae(vae_path, device)
        self.register_buffer('prototypes', prototypes.detach().float().clone().to(device))
        self.grid = math.isqrt(prototypes.shape[-1] // 4)
        if 4 * self.grid ** 2 != prototypes.shape[-1]:
            raise ValueError('prototype dimension must equal 4*feature_grid**2')
        self.temperature = temperature
        smooth_worst(torch.zeros(1, 1, prototypes.shape[1]), temperature)

    def image_features(self, images):
        """Images [2,B,3,H,W] in [-1,1]; IR is differentiably converted to gray."""
        vi, ir = images.unbind(0)
        weights = ir.new_tensor([.299, .587, .114]).view(1, 3, 1, 1)
        gray = (ir * weights).sum(1, keepdim=True).expand(-1, 3, -1, -1)
        # Exactly the preparation descriptor: frozen VAE posterior mean + pool + L2.
        mean = self.vae.encode(torch.cat([vi, gray])).latent_dist.mean * LATENT_SCALE
        features = F.normalize(F.adaptive_avg_pool2d(mean.float(), self.grid).flatten(1), dim=-1)
        return features.reshape(2, len(vi), -1)

    def forward(self, clean_latents):
        if clean_latents.ndim != 5 or clean_latents.shape[0] != 2:
            raise ValueError('clean_latents must have shape [2,B,4,H,W]')
        batch = clean_latents.shape[1]
        decoded = self.vae.decode(clean_latents.flatten(0, 1) / LATENT_SCALE).sample
        # Match the sampler's [-1,1] image clamp before exporting PNGs.
        images = decoded.clamp(-1, 1).reshape(2, batch, *decoded.shape[1:])
        return self.from_images(images)

    def from_images(self, images):
        features = self.image_features(images)
        distances = prototype_distances(features, self.prototypes)
        loss = smooth_worst(distances, self.temperature).mean()
        return loss, dict(features=features, distances=distances,
                          worst=distances.max(-1).values.mean(),
                          mean=distances.mean(),
                          farthest_weights=torch.softmax(distances / self.temperature, dim=-1))


def predicted_clean_pair(model, diffusion, latents, timestep):
    """One-step x0 estimate at a bounded noise level, with q always enabled.

    This is a training surrogate, not a full generated diffusion trajectory.
    Final generated PNGs must be evaluated separately.
    """
    batch = latents.shape[1]
    z = latents.flatten(0, 1)
    t = timestep.repeat(2)
    noise = torch.randn_like(latents[0]).repeat(2, 1, 1, 1)
    noisy = diffusion.q_sample(z, t, noise=noise)
    ids = torch.arange(2, device=z.device).repeat_interleave(batch)
    eps = model(noisy, t, ids, torch.zeros(2 * batch, dtype=torch.bool, device=z.device))[:, :4]
    return diffusion._predict_xstart_from_eps(noisy, t, eps).reshape_as(latents)
