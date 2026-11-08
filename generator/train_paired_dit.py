#!/usr/bin/env python3
"""Train a dual-modality vector-conditioned DiT with paired foreground images, backbone initialization, EMA, and resumable checkpoints."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F

from diffusion import create_diffusion
from models import DiT_models
from paired_dit_data import write_json,torch_save,MODES
from paired_dit_model import PairedConditionalDiT,load_backbone_weights
from paired_prototype_loss import FrozenImagePrototypeLoss,predicted_clean_pair


class LatentPairs(Dataset):
    def __init__(self,root,split='train',training_frames='all'):
        self.root=Path(root).resolve()
        self.manifest=json.loads((self.root/'manifest.json').read_text())
        if not self.manifest.get('completed'):
            raise ValueError('Data preparation is incomplete')
        self.indices=[i for i,r in enumerate(self.manifest['samples']) if r['split']==split and
                      (training_frames=='all' or any(r['modalities'][m]['keyframe'] for m in MODES))]
        self.cache=[torch.load(self.root/f'latents_{m}.pt',map_location='cpu',weights_only=True) for m in MODES]
        expected=len(self.manifest['samples'])
        size=self.manifest['config']['image_size']//8
        for c in self.cache:
            if c['mean'].shape!=(expected,4,size,size) or c['std'].shape!=c['mean'].shape:
                raise ValueError('Latent cache dimensions do not match the manifest')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self,index):
        i=self.indices[index]
        return (torch.stack([c['mean'][i] for c in self.cache]),
                torch.stack([c['std'][i] for c in self.cache]))


@torch.no_grad()
def update_ema(ema,model,decay):
    parameters=dict(model.named_parameters())
    for name,p in ema.named_parameters():
        p.lerp_(parameters[name],1-decay)
    buffers=dict(model.named_buffers())
    for name,b in ema.named_buffers():
        b.copy_(buffers[name])


def sample_batch(mean,std,device):
    # Batch layout: visible samples first, followed by the paired infrared samples.
    mean=mean.transpose(0,1).flatten(0,1).to(device).float()
    std=std.transpose(0,1).flatten(0,1).to(device).float()
    return mean+std*torch.randn_like(std)


@torch.no_grad()
def evaluate(model,diffusion,dataset,args,prototype_objective=None):
    if not len(dataset):
        return None
    model.eval()
    devices=[torch.device(args.device).index or 0] if str(args.device).startswith('cuda') else []
    # Use fixed validation randomness without changing training noise, dropout, or checkpoint RNG state.
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(args.seed+1701)
        generator=torch.Generator().manual_seed(args.seed+1701)
        loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=0,generator=generator)
        total,count=0.,0
        proto=[];worst=[]
        for number,(mean,std) in enumerate(loader):
            if number>=args.eval_batches:
                break
            z=sample_batch(mean,std,args.device)
            n=len(mean)
            ids=torch.arange(2,device=args.device).repeat_interleave(n)
            t=torch.randint(diffusion.num_timesteps,(n,),device=args.device).repeat(2)
            noise=torch.randn_like(z[:n]).repeat(2,1,1,1)
            values=diffusion.training_losses(model,z,t,noise=noise,model_kwargs=dict(mode_ids=ids))
            total+=float(values['loss'].sum());count+=len(z)
            if prototype_objective is not None:
                b=min(n,args.prototype_batch_size)
                paired=z.reshape(2,n,*z.shape[1:])[:,:b]
                ts=torch.full((b,),args.prototype_t_max,device=z.device,dtype=torch.long)
                estimate=predicted_clean_pair(model,diffusion,paired,ts)
                p,stats=prototype_objective(estimate)
                proto.append(float(p));worst.append(float(stats['worst']))
    result=dict(val_diffusion_loss=total/count)
    if proto:
        result.update(val_prototype_loss=float(np.mean(proto)),val_worst_distance=float(np.mean(worst)))
    result['val_selection_score']=result['val_diffusion_loss']+args.prototype_weight*result.get('val_prototype_loss',0.)
    return result


def train(args):
    if args.max_steps<1 or args.batch_size<1 or args.lr<=0 or not 0<=args.ema_decay<1:
        raise ValueError('Invalid training steps, batch size, learning rate, or EMA settings')
    if min(args.pair_weight,args.shared_weight,args.private_weight,args.prototype_weight)<0 or args.eval_every<1 or args.save_every<1 or args.eval_batches<1 or args.log_every<1:
        raise ValueError('Loss weights must be nonnegative; logging, evaluation, and save intervals must be positive')
    if args.prototype_every<1 or args.prototype_batch_size<1 or args.prototype_warmup<0 or not 1<=args.prototype_t_min<=args.prototype_t_max<=999:
        raise ValueError('Invalid prototype-loss interval, batch size, warmup, or timestep range')
    if sum(bool(x) for x in (args.init_ckpt,args.resume,args.warm_start))>1:
        raise ValueError('--init-ckpt, --resume, and --warm-start are mutually exclusive')
    args.cache_root=args.cache_root.resolve();args.output_root=args.output_root.resolve()
    if args.cache_root==args.output_root or args.cache_root in args.output_root.parents or args.output_root in args.cache_root.parents:
        raise ValueError('Training output must be separate from the data cache')
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise FileExistsError('Training directory is not empty; use --resume or choose another directory')
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    if str(args.device).startswith('cuda'):
        torch.cuda.set_device(torch.device(args.device))
    training=LatentPairs(args.cache_root,'train',args.training_frames)
    validation=LatentPairs(args.cache_root,'val',args.training_frames)
    if not len(training):
        raise ValueError('The training set is empty')
    bank=torch.load(args.cache_root/'prototypes.pt',map_location='cpu',weights_only=True)
    manifest=training.manifest
    if bank['cache_id']!=manifest['cache_id']:
        raise ValueError('Prototype and data-cache identities do not match')
    image_size=manifest['config']['image_size']
    model=PairedConditionalDiT(bank['C'],image_size,args.model,args.condition_dropout).to(args.device)
    initial_weights=None
    if args.init_ckpt:
        if args.resume:
            raise ValueError('--init-ckpt and --resume cannot be used together')
        initial_weights=load_backbone_weights(model,args.init_ckpt)
    if args.warm_start:
        source=torch.load(args.warm_start,map_location='cpu',weights_only=False)
        if source['config']['cache_id']!=bank['cache_id'] or source['config']['model']!=args.model:
            raise ValueError('Warm-start architecture or prototype cache mismatch')
        if not torch.equal(source['ema']['conditioner.C'],bank['C']):
            raise ValueError('The fixed warm-start prototypes have changed')
        model.load_state_dict(source['ema'],strict=True)
        initial_weights=dict(path=str(args.warm_start),kind='full conditional EMA warm start; optimizer/step reset')
    ema=deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=0.)
    diffusion=create_diffusion('')
    prototype_objective=FrozenImagePrototypeLoss(manifest['config']['vae'],bank['C'],args.device,
                         args.prototype_temperature) if args.prototype_weight>0 else None
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(image_size=image_size,cache_id=bank['cache_id'],initial_weights=initial_weights,
                  vae=manifest['config']['vae'],vae_identity=manifest['vae_identity'],
                  targets=bank['targets'],latent_scale=.18215,
                  training_objective='paired foreground denoising + frozen-image-encoder smooth worst-prototype distance; no adversarial objective',
                  train_frame_pairs=len(training),val_frame_pairs=len(validation))
    step,epoch,batch_cursor,best=0,0,0,float('inf')
    best_score=float('inf')
    if args.resume:
        state=torch.load(args.resume,map_location='cpu',weights_only=False)
        previous=state['config']
        for name in ('cache_id','model','image_size','condition_dropout','batch_size','seed','training_frames',
                     'lr','pair_weight','shared_weight','private_weight','ema_decay',
                     'prototype_weight','prototype_temperature','prototype_every','prototype_batch_size',
                     'prototype_t_min','prototype_t_max','prototype_warmup'):
            if name not in previous:
                if name=='prototype_weight' and config[name]!=0:
                    raise ValueError('This checkpoint has no prototype constraint; use --warm-start to change the objective')
                if name.startswith('prototype_'):
                    continue
            if config[name]!=previous[name]:
                raise ValueError(f'Resume parameter mismatch: {name}; expected {previous[name]}')
        model.load_state_dict(state['model'],strict=True);ema.load_state_dict(state['ema'],strict=True)
        optimizer.load_state_dict(state['optimizer'])
        step,epoch,batch_cursor=state['step'],state['epoch'],state['batch_cursor']
        best=state['best_val_loss']
        best_score=state.get('best_val_score',best)
        if args.max_steps<=step:
            raise ValueError('Resume max_steps must exceed the completed step count')
        torch.set_rng_state(state['rng']['torch'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state['rng']['cuda'])
        np.random.set_state(state['rng']['numpy']);random.setstate(state['rng']['python'])
    args.output_root.mkdir(parents=True,exist_ok=True)
    config['initial_weights']=initial_weights if not args.resume else previous.get('initial_weights')
    write_json(args.output_root/'train_config.json',config)
    log=(args.output_root/'metrics.jsonl').open('a')
    started=time.time()

    def save(name):
        torch_save(args.output_root/name,dict(format='paired-conditional-dit-v1',model=model.state_dict(),
            ema=ema.state_dict(),optimizer=optimizer.state_dict(),config=config,step=step,epoch=epoch,
            batch_cursor=batch_cursor,best_val_loss=best,best_val_score=best_score,
            rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                     numpy=np.random.get_state(),python=random.getstate())))

    try:
        while step<args.max_steps:
            generator=torch.Generator().manual_seed(args.seed+epoch)
            loader=DataLoader(training,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,
                              drop_last=False,generator=generator,pin_memory=str(args.device).startswith('cuda'))
            model.train()
            for batch,(mean,std) in enumerate(loader):
                if batch<batch_cursor:
                    continue
                z=sample_batch(mean,std,args.device)
                n=len(mean)
                ids=torch.arange(2,device=args.device).repeat_interleave(n)
                t=torch.randint(diffusion.num_timesteps,(n,),device=args.device).repeat(2)
                noise=torch.randn_like(z[:n]).repeat(2,1,1,1)
                drops=(torch.rand(n,device=args.device)<args.condition_dropout).repeat(2)
                captured={}
                def forward(x,ts,**kw):
                    value=model(x,ts,**kw)
                    captured['epsilon']=value[:,:4]
                    return value
                losses=diffusion.training_losses(forward,z,t,noise=noise,model_kwargs=dict(mode_ids=ids,drop_mask=drops))
                loss_dit=losses['loss'].mean()
                eps_vi,eps_ir=captured['epsilon'].chunk(2)
                loss_pair=F.mse_loss(eps_vi,eps_ir)
                loss_shared,loss_private=model.conditioner.auxiliary_losses(model.last_condition_state)
                loss=loss_dit+args.pair_weight*loss_pair+args.shared_weight*loss_shared+args.private_weight*loss_private
                proto_record={}
                if prototype_objective is not None and step%args.prototype_every==0:
                    b=min(n,args.prototype_batch_size)
                    paired=z.reshape(2,n,*z.shape[1:])[:,:b]
                    ts=torch.randint(args.prototype_t_min,args.prototype_t_max+1,(b,),device=z.device)
                    estimate=predicted_clean_pair(model,diffusion,paired,ts)
                    proto_loss,proto_stats=prototype_objective(estimate)
                    ramp=min(1.,(step+1)/max(args.prototype_warmup,1))
                    # Multiply by interval so prototype_weight is an average per-step weight.
                    loss=loss+args.prototype_weight*args.prototype_every*ramp*proto_loss
                    proto_record=dict(prototype_loss=float(proto_loss.detach()),
                        prototype_worst_distance=float(proto_stats['worst'].detach()),
                        prototype_mean_distance=float(proto_stats['mean'].detach()),prototype_ramp=ramp)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite training loss: step={step}')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();update_ema(ema,model,args.ema_decay)
                step+=1;batch_cursor=batch+1
                if step%args.log_every==0 or step==1 or proto_record:
                    record=dict(step=step,epoch=epoch,loss=float(loss.detach()),diffusion=float(loss_dit.detach()),
                        pair_consistency=float(loss_pair.detach()),shared=float(loss_shared.detach()),
                        private=float(loss_private.detach()),grad_norm=float(grad_norm),elapsed_s=time.time()-started)
                    record.update(proto_record)
                    log.write(json.dumps(record)+'\n');log.flush()
                    if step%args.log_every==0 or step==1:
                        print(json.dumps(record),flush=True)
                if step%args.eval_every==0 or step==args.max_steps:
                    value=evaluate(ema,diffusion,validation,args,prototype_objective)
                    model.train()
                    if value is not None:
                        log.write(json.dumps(dict(step=step,**value))+'\n');log.flush()
                        print(json.dumps(dict(step=step,**value)),flush=True)
                        best=min(best,value['val_diffusion_loss'])
                        if value['val_selection_score']<best_score:
                            best_score=value['val_selection_score'];save('best.pt')
                if step%args.save_every==0 or step==args.max_steps:
                    save('last.pt')
                    if step%args.save_every==0:
                        save(f'step_{step:07d}.pt')
                if step>=args.max_steps:
                    break
            if batch_cursor==len(loader):
                epoch+=1;batch_cursor=0
        write_json(args.output_root/'training_result.json',dict(completed=True,step=step,cache_id=bank['cache_id'],
                    best_val_loss=best if math.isfinite(best) else None,
                    best_val_score=best_score if math.isfinite(best_score) else None,elapsed_s=time.time()-started))
    finally:
        log.close()
    return model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--model',choices=list(DiT_models)+['DiT-Tiny/2'],default='DiT-S/2')
    p.add_argument('--init-ckpt',type=Path)
    p.add_argument('--resume',type=Path)
    p.add_argument('--warm-start',type=Path,help='Load the full conditional EMA and reset optimizer and steps when changing the objective')
    p.add_argument('--max-steps',type=int,default=20000)
    p.add_argument('--batch-size',type=int,default=8,help='Number of frame pairs; each step processes twice as many images')
    p.add_argument('--training-frames',choices=['all','keyframes'],default='all')
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--condition-dropout',type=float,default=.1)
    p.add_argument('--pair-weight',type=float,default=.05)
    p.add_argument('--shared-weight',type=float,default=.01)
    p.add_argument('--private-weight',type=float,default=.01)
    p.add_argument('--prototype-weight',type=float,default=.5,help='Smooth maximum prototype-distance weight; 0 disables it')
    p.add_argument('--prototype-temperature',type=float,default=.1)
    p.add_argument('--prototype-every',type=int,default=4,help='Interval for differentiable VAE decoding/encoding; scale the loss by this interval')
    p.add_argument('--prototype-batch-size',type=int,default=1,help='Number of pairs in the additional image-feature branch')
    p.add_argument('--prototype-t-min',type=int,default=100)
    p.add_argument('--prototype-t-max',type=int,default=600)
    p.add_argument('--prototype-warmup',type=int,default=100)
    p.add_argument('--ema-decay',type=float,default=.9999)
    p.add_argument('--save-every',type=int,default=1000)
    p.add_argument('--eval-every',type=int,default=500)
    p.add_argument('--eval-batches',type=int,default=8)
    p.add_argument('--log-every',type=int,default=20)
    p.add_argument('--num-workers',type=int,default=0)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda:0')
    train(p.parse_args())


if __name__=='__main__':
    main()
