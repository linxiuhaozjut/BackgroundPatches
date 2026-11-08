#!/usr/bin/env python3
"""Generate visible.png and infrared.png from a vector-conditioned checkpoint."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from diffusion import create_diffusion
from paired_dit_data import load_vae,write_json,torch_save,LATENT_SCALE
from paired_dit_model import PairedConditionalDiT


@torch.inference_mode()
def generate_paired_patches(checkpoint_path,output_root,*,vae_path=None,num_pairs=1,
                           sampling_steps=50,cfg_scale=3.,seed=42,device='cuda:0',sampler='ddim',use_ema=True):
    """Public generation interface; read the supplied checkpoint and VAE without downloading weights."""
    checkpoint_path=Path(checkpoint_path).resolve();output_root=Path(output_root).resolve()
    if num_pairs<1 or not 2<=sampling_steps<=1000 or not math.isfinite(cfg_scale) or cfg_scale<0:
        raise ValueError('Invalid pair count, sampling steps, or CFG scale')
    if sampler not in ('ddim','ddpm'):
        raise ValueError(sampler)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError('Generation output directory is not empty; choose a new directory')
    state=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    if state.get('format')!='paired-conditional-dit-v1':
        raise ValueError('Expected a vector-conditioned checkpoint from train_paired_dit.py; class-conditioned weights are not supported here')
    config=state['config'];weights=state['ema' if use_ema else 'model']
    C=weights['conditioner.C']
    model=PairedConditionalDiT(C,config['image_size'],config['model'],config['condition_dropout']).to(device)
    model.load_state_dict(weights,strict=True);model.eval()
    cache_path=Path(config['cache_root'])
    if vae_path is None and config.get('vae'):
        vae_path=config['vae']
    if vae_path is None:
        if not (cache_path/'manifest.json').is_file():
            raise FileNotFoundError('Cache directory is unavailable; specify the training VAE using vae_path/--vae')
        manifest=json.loads((cache_path/'manifest.json').read_text())
        if manifest['cache_id']!=config['cache_id']:
            raise ValueError('Cache and checkpoint identities do not match')
        vae_path=manifest['config']['vae']
    vae=load_vae(vae_path,device)
    diffusion=create_diffusion(str(sampling_steps))
    torch.manual_seed(seed)
    latent_size=config['image_size']//8
    # Share initial random noise between paired modalities; distinct conditions and modality IDs do not enforce identical pixels.
    initial=torch.randn(num_pairs,4,latent_size,latent_size,device=device)
    z=initial.repeat(2,1,1,1)
    ids=torch.arange(2,device=device).repeat_interleave(num_pairs)
    kwargs=dict(mode_ids=ids,cfg_scale=cfg_scale)
    loop=diffusion.ddim_sample_loop if sampler=='ddim' else diffusion.p_sample_loop
    latents=loop(model.forward_with_cfg,z.shape,noise=z,clip_denoised=False,model_kwargs=kwargs,
                 progress=True,device=device)
    decoded=vae.decode(latents/LATENT_SCALE).sample
    if not torch.isfinite(decoded).all():
        raise FloatingPointError('Generated output contains non-finite values')
    arrays=((decoded.clamp(-1,1)+1)*127.5).round().byte().permute(0,2,3,1).cpu().numpy()
    output_root.mkdir(parents=True,exist_ok=True)
    records=[]
    for i in range(num_pairs):
        directory=output_root if num_pairs==1 else output_root/f'pair_{i:03d}'
        directory.mkdir(parents=True,exist_ok=True)
        vi=cv2.cvtColor(arrays[i],cv2.COLOR_RGB2BGR)
        ir=cv2.cvtColor(arrays[i+num_pairs],cv2.COLOR_RGB2GRAY)
        for name,array in [('visible.png',vi),('infrared.png',ir)]:
            if not cv2.imwrite(str(directory/name),array):
                raise IOError(directory/name)
        records.append(dict(visible=str(directory/'visible.png'),infrared=str(directory/'infrared.png')))
    context,conditions=model.conditioner()
    torch_save(output_root/'conditions.pt',{k:v.detach().cpu() for k,v in conditions.items()}|
               dict(C=model.conditioner.C.cpu(),context=context.cpu()))
    metadata=dict(completed=True,checkpoint=str(checkpoint_path),checkpoint_step=state['step'],
        cache_id=config['cache_id'],vae=str(Path(vae_path).resolve()),image_size=config['image_size'],
        seed=seed,sampling_steps=sampling_steps,cfg_scale=cfg_scale,sampler=sampler,use_ema=use_ema,
        paired_initial_noise=True,outputs=records,
        note='Conditional foreground samples; detector/tracker attack objectives have not been optimized or evaluated')
    write_json(output_root/'generation.json',metadata)
    print(json.dumps(metadata,ensure_ascii=False),flush=True)
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--vae',type=Path)
    p.add_argument('--num-pairs',type=int,default=1)
    p.add_argument('--sampling-steps',type=int,default=50)
    p.add_argument('--cfg-scale',type=float,default=3.)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--sampler',choices=['ddim','ddpm'],default='ddim')
    p.add_argument('--raw-model',action='store_true',help='Use raw model weights instead of EMA')
    a=p.parse_args()
    generate_paired_patches(a.ckpt,a.output_root,vae_path=a.vae,num_pairs=a.num_pairs,
        sampling_steps=a.sampling_steps,cfg_scale=a.cfg_scale,seed=a.seed,device=a.device,
        sampler=a.sampler,use_ema=not a.raw_model)


if __name__=='__main__':
    main()
