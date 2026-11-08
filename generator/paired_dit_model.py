"""C -> h=LayerNorm(WC) -> fused c -> q=[c,C] -> dual-modality conditional DiT."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models import DiT_models, DiT


class PrototypeConditioner(nn.Module):
    def __init__(self, prototypes, hidden_size):
        super().__init__()
        if prototypes.ndim != 3 or prototypes.shape[0] != 2 or not torch.isfinite(prototypes).all():
            raise ValueError('C must be a finite [2,N,D] tensor')
        self.register_buffer('C',prototypes.float().clone())
        dim = prototypes.shape[-1]
        self.W = nn.ModuleList([nn.Linear(dim,hidden_size,bias=False) for _ in range(2)])
        self.LN = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(2)])
        self.fusion = nn.Sequential(nn.Linear(2*hidden_size,hidden_size),nn.GELU(),nn.Linear(hidden_size,hidden_size))
        self.q_projection = nn.Linear(hidden_size+dim,hidden_size)
        self.pool_query = nn.Parameter(torch.randn(2,hidden_size)*.02)
        self.reconstruct = nn.ModuleList([nn.Linear(hidden_size,dim) for _ in range(2)])

    def forward(self):
        h = torch.stack([self.LN[i](self.W[i](self.C[i])) for i in range(2)])
        c = self.fusion(torch.cat([h[0],h[1]],dim=-1))
        q = torch.stack([torch.cat([c,self.C[i]],dim=-1) for i in range(2)])
        tokens = self.q_projection(q)
        attention = torch.softmax((tokens*self.pool_query[:,None]).sum(-1)/tokens.shape[-1]**.5,dim=-1)
        context = (attention[:,:,None]*tokens).sum(1)
        return context, dict(h=h,c=c,q=q,pool_attention=attention)

    def auxiliary_losses(self, state):
        h = state['h']
        # Contrast paired prototypes to learn shared structure; reconstruct each C to retain modality-specific information.
        if h.shape[1] > 1:
            similarities = F.normalize(h[0],dim=-1)@F.normalize(h[1],dim=-1).T/.1
            labels = torch.arange(h.shape[1],device=h.device)
            shared = (F.cross_entropy(similarities,labels)+F.cross_entropy(similarities.T,labels))/2
        else:
            shared = 1-F.cosine_similarity(h[0],h[1],dim=-1).mean()
        private = sum(F.mse_loss(self.reconstruct[i](h[i]),self.C[i]) for i in range(2))/2
        return shared, private


class PairedConditionalDiT(nn.Module):
    def __init__(self, prototypes, image_size=256, architecture='DiT-S/2', condition_dropout=.1):
        super().__init__()
        if image_size%16 or not 0<=condition_dropout<1:
            raise ValueError('Image size must be divisible by 16 and condition dropout must be in [0,1)')
        if architecture=='DiT-Tiny/2':
            self.backbone = DiT(input_size=image_size//8,patch_size=2,hidden_size=96,depth=2,num_heads=4,num_classes=2)
        else:
            self.backbone = DiT_models[architecture](input_size=image_size//8,num_classes=2)
        hidden_size = self.backbone.pos_embed.shape[-1]
        # Replace the unused class-embedding lookup with the vector conditioner.
        self.backbone.y_embedder = nn.Identity()
        self.conditioner = PrototypeConditioner(prototypes,hidden_size)
        self.modality_embedding = nn.Embedding(2,hidden_size)
        self.null_condition = nn.Parameter(torch.zeros(2,hidden_size))
        self.condition_dropout = condition_dropout
        self.last_condition_state = None

    def forward(self,x,t,mode_ids,drop_mask=None):
        if mode_ids.shape!=(len(x),) or mode_ids.dtype!=torch.long:
            raise ValueError('mode_ids must be a batch-length long tensor: visible=0, infrared=1')
        context, state = self.conditioner()
        self.last_condition_state = state
        condition = context[mode_ids]
        if drop_mask is None:
            drop_mask = torch.rand(len(x),device=x.device)<self.condition_dropout if self.training else torch.zeros(len(x),dtype=torch.bool,device=x.device)
        condition = torch.where(drop_mask[:,None],self.null_condition[mode_ids],condition)
        b = self.backbone
        tokens = b.x_embedder(x)+b.pos_embed
        condition = b.t_embedder(t)+condition+self.modality_embedding(mode_ids)
        for block in b.blocks:
            tokens = block(tokens,condition)
        return b.unpatchify(b.final_layer(tokens,condition))

    def forward_with_cfg(self,x,t,mode_ids,cfg_scale):
        # Apply CFG to all four latent noise channels; use the conditional branch for variance.
        drops = torch.cat([torch.zeros(len(x),device=x.device,dtype=torch.bool),
                           torch.ones(len(x),device=x.device,dtype=torch.bool)])
        conditional, unconditional = self.forward(torch.cat([x,x]),torch.cat([t,t]),
                                                  torch.cat([mode_ids,mode_ids]),drops).chunk(2)
        channels = self.backbone.in_channels
        eps = unconditional[:,:channels]+cfg_scale*(conditional[:,:channels]-unconditional[:,:channels])
        return torch.cat([eps,conditional[:,channels:]],dim=1)


def load_backbone_weights(model,path):
    """Load a compatible backbone explicitly and reject insufficient parameter-shape coverage."""
    checkpoint = torch.load(path,map_location='cpu',weights_only=False)
    source = checkpoint.get('ema',checkpoint.get('model',checkpoint))
    source = {k.removeprefix('module.').removeprefix('backbone.'):v for k,v in source.items()}
    target = model.backbone.state_dict()
    accepted = {k:v for k,v in source.items() if k in target and v.shape==target[k].shape}
    total = sum(v.numel() for k,v in target.items() if k!='pos_embed')
    loaded = sum(v.numel() for k,v in accepted.items() if k!='pos_embed')
    if loaded/max(total,1)<.9:
        raise ValueError(f'Pretrained weights do not match the architecture; only {loaded/max(total,1):.1%} of backbone parameters match')
    missing,unexpected = model.backbone.load_state_dict(accepted,strict=False)
    return dict(path=str(path),loaded_parameter_fraction=loaded/total,missing=missing,unexpected=unexpected)
