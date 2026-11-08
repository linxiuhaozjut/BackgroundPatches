#!/usr/bin/env python3
"""Train the unmodified full-depth/deep-prompt ViPT on clean dataset_400_2."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import yaml

from optimize_dataset4002_blackbox import DATA, PLACER, load_placer, box_from_label, write_json

PROJECT=Path(__file__).resolve().parent


def plain(value):
    if isinstance(value,dict): return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [plain(v) for v in value]
    return value


def prepare(root,out,seed):
    placer=load_placer(PLACER)
    pairs,_=placer.discover_pairs(root,root/'mask',None,None,None)
    records=[]
    for p in pairs:
        vi=p['vi'];ir=p['ir']
        image=cv2.imread(str(root/vi/p['files'][0].name))
        h,w=image.shape[:2]
        rows=[]
        for row in csv.DictReader((root/'label'/vi/'annotations.csv').open()):
            if row['status']!='ok': continue
            name=row['frame']
            box=box_from_label(root/'label'/vi/(Path(name).stem+'.txt'),w,h)
            if min(box[2:])<=0: raise ValueError('Invalid GT')
            rows.append(dict(frame=name,box=box))
        if len(rows)>=2:
            records.append(dict(sequence=vi.as_posix(),infrared=ir.as_posix(),target=vi.name,
                angle=p['angle'],distance=p['distance'],frames=rows))
    rng=random.Random(seed);splits={k:[] for k in ('train','val','test')}
    for target in sorted({r['target'] for r in records}):
        group=sorted([r['sequence'] for r in records if r['target']==target]);rng.shuffle(group)
        n=max(1,round(.1*len(group)))
        splits['val'].extend(group[:n]);splits['test'].extend(group[n:2*n]);splits['train'].extend(group[2*n:])
    sets=[set(v) for v in splits.values()]
    assert all(not(sets[i]&sets[j]) for i in range(3) for j in range(i))
    assert set.union(*sets)=={r['sequence'] for r in records}
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'manifest.json',records)
    for k,v in splits.items(): write_json(out/(k+'.json'),sorted(v))
    return records,splits


class CleanPairs(Dataset):
    def __init__(self,root,records,names,size,seed,training):
        self.root=root;self.records=[r for r in records if r['sequence'] in set(names)]
        self.size=size;self.seed=seed;self.training=training

    def __len__(self): return self.size

    def __getitem__(self,index):
        from lib.train.data.processing_utils import jittered_center_crop
        rng=np.random if self.training else np.random.RandomState(self.seed+index)
        rec=self.records[index%len(self.records)]
        count=len(rec['frames']);search=int(rng.randint(1,count))
        template=0 if rng.rand()<.5 else int(rng.randint(max(0,search-16),search))
        tensors={}
        flip=self.training and rng.rand()<.5
        for mode,frame_id,factor,size in (('template',template,2.,128),('search',search,4.,256)):
            row=rec['frames'][frame_id]
            vi=cv2.imread(str(self.root/rec['sequence']/row['frame']))
            ir=cv2.imread(str(self.root/rec['infrared']/row['frame']))
            if vi is None or ir is None: raise FileNotFoundError(row['frame'])
            image=np.concatenate([cv2.cvtColor(vi,cv2.COLOR_BGR2RGB),cv2.cvtColor(ir,cv2.COLOR_BGR2RGB)],axis=2)
            box=torch.tensor(row['box'],dtype=torch.float32)
            if flip:
                image=image[:,::-1].copy();box[0]=image.shape[1]-box[0]-box[2]
            extract=box.clone()
            if mode=='search':
                wh=box[2:]*torch.tensor(np.exp(rng.randn(2)*.1),dtype=torch.float32)
                center=box[:2]+box[2:]/2+wh.prod().sqrt()*torch.tensor(rng.rand(2)-.5,dtype=torch.float32)
                extract=torch.cat([center-wh/2,wh])
            crops,boxes,_,_=jittered_center_crop([image],[extract],[box],factor,size)
            tensor=torch.from_numpy(crops[0].transpose(2,0,1).copy()).float()/255
            mean=torch.tensor([.485,.456,.406]*2)[:,None,None]
            std=torch.tensor([.229,.224,.225]*2)[:,None,None]
            tensors[mode+'_images']=(tensor-mean)/std
            tensors[mode+'_anno']=boxes[0]
        return tensors


def worker_seed(worker_id):
    seed=torch.initial_seed()%2**32;np.random.seed(seed);random.seed(seed);cv2.setNumThreads(0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,default=DATA)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--steps',type=int,default=800)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',type=int,default=0)
    p.add_argument('--initialization',choices=('scratch','ostrack'),default='scratch')
    p.add_argument('--warm-start',type=Path,help='Strict full-model weights; optimizer resets, steps count this stage only')
    p.add_argument('--reset-head',action='store_true',help='Reinitialize the unchanged CENTER head before clean-data training')
    p.add_argument('--yaml-name',default='full_clean_dataset4002')
    p.add_argument('--save-steps',type=int,nargs='+',default=[100,200,400,800])
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();out=a.output_root.resolve()
    if a.steps<1 or a.batch_size<1 or not 0<a.lr<1: raise ValueError('Invalid training settings')
    if not a.yaml_name.replace('_','').isalnum(): raise ValueError('Invalid YAML name')
    if out.exists() and any(out.iterdir()): raise FileExistsError('Use new output directory')
    cfg_file=PROJECT.parent/'tracker/experiments/vipt'/(a.yaml_name+'.yaml')
    if cfg_file.exists(): raise FileExistsError('Use a new YAML name')
    torch.manual_seed(a.seed);random.seed(a.seed);np.random.seed(a.seed)
    torch.set_num_threads(4);cv2.setNumThreads(0)
    records,splits=prepare(a.dataset_root,out/'splits',a.seed)
    print(json.dumps({k:len(v) for k,v in splits.items()}),flush=True)
    if a.prepare_only: return
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    torch.cuda.set_device(a.device);device=torch.device('cuda',a.device)
    from lib.config.vipt.config import cfg,update_config_from_file
    from lib.models.vipt import build_viptrack
    from lib.train.actors import ViPTActor
    from lib.utils.box_ops import giou_loss
    from lib.utils.focal_loss import FocalLoss
    from types import SimpleNamespace
    update_config_from_file(str(PROJECT.parent/'tracker/experiments/vipt/deep_rgbt.yaml'))
    cfg.MODEL.PRETRAIN_FILE=str(PROJECT.parent/'weights/tracker/ostrack_initialization.pth.tar') if a.initialization=='ostrack' else ''
    cfg.TRAIN.DROP_PATH_RATE=0.;cfg.TRAIN.LR=a.lr;cfg.TRAIN.BATCH_SIZE=a.batch_size
    cfg.TRAIN.CE_START_EPOCH=0;cfg.TRAIN.CE_WARM_EPOCH=0
    # Preserve the full original network: 12 blocks, 768 channels, 12 deep prompts,
    # CE stages [3,6,9] with keep ratio .7 and the original CENTER head.
    yaml_name=a.yaml_name
    cfg_file.write_text(yaml.safe_dump(plain(cfg),sort_keys=False))
    (out/'model_config.yaml').write_text(cfg_file.read_text())
    net=build_viptrack(cfg,training=True).to(device)
    if a.initialization=='ostrack':
        net.backbone.patch_embed_prompt.load_state_dict(net.backbone.patch_embed.state_dict())
    if a.warm_start:
        if a.initialization!='scratch': raise ValueError('Use scratch initialization with explicit warm-start')
        restored=torch.load(a.warm_start,map_location='cpu',weights_only=False)
        net.load_state_dict(restored['net'],strict=True)
    if a.reset_head:
        if a.warm_start: raise ValueError('Do not combine reset-head with strict warm-start')
        for module in net.box_head.modules():
            if hasattr(module,'reset_parameters'): module.reset_parameters()
    assert len(net.backbone.blocks)==12 and len(net.backbone.prompt_blocks)==12
    names=[]
    for name,param in net.named_parameters():
        param.requires_grad=True
        if param.requires_grad: names.append(name)
    optimizer=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=a.lr,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda')
    settings=SimpleNamespace(batchsize=a.batch_size,num_template=1)
    actor=ViPTActor(net,{'giou':giou_loss,'l1':torch.nn.functional.l1_loss,'focal':FocalLoss()},
        {'giou':2.,'l1':5.,'focal':1.},settings,cfg)
    generator=torch.Generator().manual_seed(a.seed)
    train=DataLoader(CleanPairs(a.dataset_root,records,splits['train'],a.steps*a.batch_size,a.seed,True),
        batch_size=a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=True,
        worker_init_fn=worker_seed,generator=generator,persistent_workers=a.workers>0)
    val=DataLoader(CleanPairs(a.dataset_root,records,splits['val'],320,10000,False),
        batch_size=a.batch_size,num_workers=4,pin_memory=True)
    config=vars(a).copy();config={k:str(v) if isinstance(v,Path) else v for k,v in config.items()}
    config.update(model=f'Full ViPT: 12 transformer blocks, vipt_deep, original CE and CENTER head; {a.initialization} initialization',
        trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),
        total_parameters=sum(p.numel() for p in net.parameters()),trainable_names=names,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),yaml=yaml_name,
        clean_only=True,adversarial_training=False,checkpoint_selection='best.pth.tar uses clean validation crop IoU; named step checkpoints allow explicitly reported development screening',
        original_architecture_preserved=True,backbone_depth=12,prompt_type='vipt_deep',
        ce_locations=list(cfg.MODEL.BACKBONE.CE_LOC),ce_keep_ratio=list(cfg.MODEL.BACKBONE.CE_KEEP_RATIO),
        warm_start_sha256=hashlib.sha256(a.warm_start.read_bytes()).hexdigest() if a.warm_start else None,
        initialization_description='strict clean full-model warm start; optimizer reset' if a.warm_start else a.initialization,
        split_counts={k:len(v) for k,v in splits.items()},
        valid_frame_pairs={k:sum(len(r['frames']) for r in records if r['sequence'] in set(v)) for k,v in splits.items()},
        train_loss='2*GIoU + 5*L1 + focal',amp=True,
        augmentation='horizontal flip p=.5; search center jitter 1.0, scale jitter .1; no adversarial data')
    write_json(out/'train_config.json',config)
    def data_to_device(data):
        data={k:v.unsqueeze(0).to(device,non_blocking=True) for k,v in data.items()};data['epoch']=1
        return data
    def evaluate():
        net.eval();values=[]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            for data in val:
                _,status=actor(data_to_device(data));values.append(status)
        return {k:float(np.mean([s[k] for s in values])) for k in values[0]}
    def save(name,step,metrics):
        state={k:v.detach().cpu() for k,v in net.state_dict().items()}
        torch.save(dict(net=state,step=step,train_config=config,validation=metrics),out/name)
    initial=evaluate();history=[dict(step=0,validation=initial)]
    write_json(out/'history.json',history)
    best=-float('inf');start=time.time();running=[]
    for step,data in enumerate(train,1):
        net.train();actor.fix_bns();net.backbone.pos_drop.eval()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16): loss,status=actor(data_to_device(data))
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite clean training loss')
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],.1)
        scaler.step(optimizer);scaler.update();running.append(status)
        if step%20==0: print(json.dumps(dict(step=step,loss=status['Loss/total'],iou=status['IoU'],elapsed_s=time.time()-start)),flush=True)
        if step%100==0 or step==a.steps:
            metrics=evaluate();row=dict(step=step,validation=metrics,train={k:float(np.mean([r[k] for r in running])) for k in status})
            history.append(row);running=[];write_json(out/'history.json',history)
            save('last.pth.tar',step,metrics)
            if step in a.save_steps: save(f'step_{step:04d}.pth.tar',step,metrics)
            if metrics['IoU']>best: best=metrics['IoU'];save('best.pth.tar',step,metrics)
            print(json.dumps(dict(validation_step=step,validation=metrics,best_clean_validation_iou=best)),flush=True)
    write_json(out/'result.json',dict(completed=True,steps=a.steps,best_clean_validation_iou=best,
        initial_validation=initial,last_validation=metrics,elapsed_s=time.time()-start))


if __name__=='__main__': main()
