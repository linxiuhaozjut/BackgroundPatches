#!/usr/bin/env python3
"""Universal paired-patch SPSA: tracker API outputs only, no tracker gradients."""
from __future__ import annotations
import argparse
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from collections import OrderedDict
import os
from pathlib import Path
import random
import shutil
import time

import cv2
import numpy as np

PROJECT=Path(__file__).resolve().parent
DATA=PROJECT.parent/'datasets/carla/dataset_400_2'
PLACER=PROJECT.parent/'preprocessing/apply_paired_patches.py'
# The bundled tracker framework lives in a separate directory.
import sys
sys.path.insert(0, str(PROJECT.parent/'tracker'))


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def select_sequences(pairs,path):
    """Explicit sequence split; never split frames from the same sequence."""
    names=json.loads(Path(path).read_text())
    if not isinstance(names,list) or not names or len(names)!=len(set(names)):
        raise ValueError('Sequence list must be a nonempty list of unique relative paths')
    table={p['vi'].as_posix():p for p in pairs}
    missing=set(names)-set(table)
    if missing: raise ValueError(f'Unknown sequences: {sorted(missing)}')
    return [table[name] for name in names]


def load_placer(path):
    import sys
    spec=importlib.util.spec_from_file_location('paired_patch_placement',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def iou(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    top=np.maximum(a[:2],b[:2]);bottom=np.minimum(a[:2]+a[2:],b[:2]+b[2:])
    intersection=float(np.maximum(bottom-top,0).prod())
    return intersection/max(float(a[2:].prod()+b[2:].prod()-intersection),1e-9)


def tv(patches):
    return float(np.mean([.5*(np.abs(np.diff(p,axis=0)).mean()+np.abs(np.diff(p,axis=1)).mean()) for p in patches]))


def weighted_loss(overlap,guide,pixel,weights):
    return float(np.dot([overlap,guide,pixel],weights))


def pixel_directions(patches,rng):
    """Independent +/-1 for EVERY source pixel/channel; no resized grid."""
    return [rng.choice([-1.,1.],size=p.shape).astype(np.float32) for p in patches]


def spsa_gradient(directions,plus_loss,minus_loss,sigma):
    return [(plus_loss-minus_loss)/(2*sigma)*d for d in directions]


def loss_parts(pred,truth,center,width,height):
    overlap=iou(pred,truth)
    pc=np.asarray(pred[:2])+np.asarray(pred[2:])/2
    guide=float(np.square((pc-np.asarray(center))/np.array([width,height])).sum()/2)
    return overlap,guide


def load_patch(directory):
    vi=cv2.imread(str(directory/'visible.png'),cv2.IMREAD_COLOR)
    ir=cv2.imread(str(directory/'infrared.png'),cv2.IMREAD_GRAYSCALE)
    if vi is None or ir is None or vi.shape[:2]!=ir.shape or vi.shape[0]!=vi.shape[1]:
        raise ValueError('Need same-size square visible.png and infrared.png')
    return [vi.astype(np.float32)/255,ir.astype(np.float32)/255]


def save_patch(directory,patches):
    directory.mkdir(parents=True,exist_ok=True)
    for name,p in zip(('visible','infrared'),patches):
        assert cv2.imwrite(str(directory/(name+'.png')),np.rint(np.clip(p,0,1)*255).astype(np.uint8))


def box_from_label(path,w,h):
    values=[float(x) for x in path.read_text().split()]
    if len(values)!=5 or not np.isfinite(values).all(): raise ValueError(path)
    _,cx,cy,bw,bh=values
    return [(cx-bw/2)*w,(cy-bh/2)*h,bw*w,bh*h]


class BoxOracle:
    """Only this adapter knows ViPT. The optimizer consumes boxes, not tensors."""
    def __init__(self,device,yaml_name,checkpoint=None):
        import torch
        from lib.test.parameter.vipt import parameters
        from lib.test.tracker.vipt import ViPTTrack
        torch.cuda.set_device(device)
        # Trusted checkpoints may contain metadata rejected by the PyTorch 2.6 weights-only default.
        os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
        with contextlib.redirect_stdout(io.StringIO()):
            params=parameters(yaml_name)
            if checkpoint: params.checkpoint=str(checkpoint)
            params.debug=0
            self.tracker=ViPTTrack(params)
        self.checkpoint=params.checkpoint;self.queries=0;self.frames=0
        self.tracker.network.requires_grad_(False)

    def start(self,vi,ir,box):
        image=np.concatenate([cv2.cvtColor(vi,cv2.COLOR_BGR2RGB),cv2.cvtColor(ir,cv2.COLOR_BGR2RGB)],axis=2)
        self.tracker.initialize(image,{'init_bbox':list(box)})
        self.queries+=1;self.frames+=1

    def track(self,vi,ir):
        import torch
        image=np.concatenate([cv2.cvtColor(vi,cv2.COLOR_BGR2RGB),cv2.cvtColor(ir,cv2.COLOR_BGR2RGB)],axis=2)
        with torch.inference_mode(): result=self.tracker.track(image)
        box=np.asarray(result['target_bbox'],float)
        if box.shape!=(4,) or not np.isfinite(box).all() or np.any(box[2:]<=0):
            raise ValueError('Tracker returned invalid box')
        self.frames+=1
        return box.tolist()


class Experiment:
    def __init__(self,args,load_tracker=True):
        self.args=args;self.placer=load_placer(args.placement_script)
        self.root=args.output_root.resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        self.dataset=args.dataset_root.resolve();self.masks=args.mask_root.resolve()
        self.pairs,self.nearest=self.placer.discover_pairs(self.dataset,self.masks,args.angles,args.distances,args.sequence_names)
        if getattr(args,'sequence_list',None): self.pairs=select_sequences(self.pairs,args.sequence_list)
        self.nearest=9. # Physical nearest distance is dataset parameter, not the minimum of a filtered subset.
        for pair in self.pairs:
            pair['meta']=self.placer.fixed_frame_metadata(self.dataset,pair)
            pair['annotations']={r['frame']:r for r in csv.DictReader((self.dataset/'label'/pair['vi']/'annotations.csv').open())}
        self.oracle=None if args.workers>1 or not load_tracker else BoxOracle(args.device,args.yaml,args.checkpoint)
        self.pool=None;self.worker=False
        if args.workers>1 and load_tracker:
            context=mp.get_context('spawn')
            devices=context.Queue()
            for device in args.devices[:args.workers]: devices.put(device)
            self.pool=ProcessPoolExecutor(args.workers,mp_context=context,
                initializer=worker_init,initargs=(args,devices))
        self.initial=load_patch(args.patch_dir)
        self.size=self.initial[0].shape[0]
        self.cache={};self.cache_key=None

    def reset_dir(self,path):
        path=Path(path)
        if path.parent.resolve()!=self.root or path.name not in ('scratch','latest') or path.is_symlink():
            raise ValueError('Refusing to remove an unrelated directory')
        if path.exists(): shutil.rmtree(path)
        path.mkdir()

    def prepared(self,pair):
        first=cv2.imread(str(self.dataset/pair['vi']/pair['files'][0].name))
        h,w=first.shape[:2]
        geom=self.placer.patch_geometry(w,h,pair['angle'],pair['distance'],self.nearest,0,len(pair['files']))
        rows=[]
        for path in pair['files']:
            images=[self.placer.read_image(self.dataset/pair[m]/path.name) for m in ('vi','ir')]
            masks=[self.placer.read_image(self.masks/pair[m]/(path.stem+'_mask.png')) for m in ('vi','ir')]
            protected=(masks[0]>0)|(masks[1]>0)
            meta=pair['meta'].get(path.name)
            absent=bool(meta and meta['semantic_target_pixels']==0 and meta['target_intersects_frustum'] is False)
            skip=any(not mask.any() for mask in masks) and not absent
            if skip:
                guard=self.placer.projected_bbox_guard(meta,h,w)
                if guard is not None: protected|=guard;skip=False
            annotation=pair['annotations'][path.name]
            truth=None
            if annotation['status']=='ok':
                truth=box_from_label(self.dataset/'label'/pair['vi']/(path.stem+'.txt'),w,h)
            rows.append(dict(name=path.name,images=images,protected=protected.astype(np.uint8)*255,
                             skip=skip,truth=truth,status=annotation['status']))
        valid=next((i for i,r in enumerate(rows) if r['truth'] is not None),None)
        if valid is None: raise ValueError(f'No initialization label: {pair["vi"]}')
        return rows,geom,valid

    def batch_data(self,pairs):
        key=tuple(p['vi'].as_posix() for p in pairs)
        if self.cache_key!=key:
            self.cache={p['vi'].as_posix():self.prepared(p) for p in pairs}
            self.cache_key=key
        return self.cache

    def images(self,row,patches,geometry):
        result=[]
        for image,p in zip(row['images'],patches):
            if row['skip']: result.append(image.copy());continue
            color=np.rint(p*255).astype(np.uint8)
            if color.ndim==2: color=cv2.cvtColor(color,cv2.COLOR_GRAY2BGR)
            out,_=self.placer.composite_patch(image,color,np.ones(p.shape[:2],np.float32),row['protected'],geometry)
            assert np.array_equal(out[row['protected']>0],image[row['protected']>0])
            result.append(out)
        return [cv2.cvtColor(x,cv2.COLOR_GRAY2BGR) if x.ndim==2 else x for x in result]

    def disk_placed(self,pair,patches):
        """Use the existing public four-path static placement function unchanged."""
        scratch=self.root/'scratch';self.reset_dir(scratch)
        source=scratch/'source';source.mkdir()
        for mode in ('vi','ir'):
            dest=source/pair[mode];dest.parent.mkdir(parents=True,exist_ok=True)
            dest.symlink_to(self.dataset/pair[mode],target_is_directory=True)
        (source/'metadata').symlink_to(self.dataset/'metadata',target_is_directory=True)
        patchdir=scratch/'patch';save_patch(patchdir,patches)
        with contextlib.redirect_stdout(io.StringIO()):
            self.placer.apply_static_paired_patches(source,patchdir,self.masks,scratch/'placed',
                nearest_distance_m=9.,save_previews=False)
        return scratch/'placed'

    def verify_placement(self,patches):
        pair=self.pairs[0]
        rows,geometry,_=self.prepared(pair)
        placed=self.disk_placed(pair,patches)
        for row in rows:
            reference=self.images(row,patches,geometry)
            for mode,image in zip(('vi','ir'),reference):
                saved=cv2.imread(str(placed/pair[mode]/(Path(row['name']).stem+'.png')))
                if saved is None or not np.array_equal(saved,image):
                    raise AssertionError('Public static placement differs from streamed rendering')
        report=dict(passed=True,sequence=pair['vi'].as_posix(),paired_frames=len(rows),
            check='public four-path function and streamed images exactly equal; target pixels protected')
        write_json(self.root/'placement_verification.json',report)
        self.reset_dir(self.root/'scratch');(self.root/'scratch').rmdir()
        return report

    def evaluate(self,patches,pairs,visualize=False,clean=False,disk=False):
        if self.pool:
            if visualize: self.reset_dir(self.root/'latest')
            jobs=[self.pool.submit(worker_evaluate,patches,pair,visualize,clean) for pair in pairs]
            reports=[]
            for index,job in enumerate(jobs):
                reports.append(job.result())
                if disk and ((index+1)%40==0 or index==len(jobs)-1):
                    print(f'evaluation {index+1}/{len(jobs)} clean={clean} visual={visualize}',flush=True)
            result=dict(loss=float(np.mean([r['loss'] for r in reports])),
                iou=float(np.mean([r['iou'] for r in reports])),guide=float(np.mean([r['guide'] for r in reports])),
                tv=0. if clean else tv(patches),scored_frames=sum(r['scored_frames'] for r in reports),
                ignored_frames=sum(r['ignored_frames'] for r in reports),
                sequences=[s for r in reports for s in r['sequences']],
                aggregation='mean of per-sequence means; GT-absent/missed/init frames excluded',
                tracker_sequence_queries=sum(r['tracker_sequence_queries'] for r in reports),
                tracker_frames=sum(r['tracker_frames'] for r in reports))
            result.update(loss_weights=[self.args.iou_weight,self.args.guide_weight,self.args.tv_weight],
                          tv_enabled=self.args.tv_weight>0)
            if visualize: write_json(self.root/'latest'/'evaluation.json',result)
            return result
        overlaps=[];guides=[];reports=[];scored=0;ignored=0
        start_queries=self.oracle.queries;start_frames=self.oracle.frames
        if visualize and not self.worker: self.reset_dir(self.root/'latest')
        cache=None if disk else self.batch_data(pairs)
        for number,pair in enumerate(pairs):
            rows,geometry,start=self.prepared(pair) if disk else cache[pair['vi'].as_posix()]
            last_scored=max(i for i,row in enumerate(rows) if row['truth'] is not None)
            # Validate the in-memory renderer against the public four-path function
            # on the first full sequence. Stream the rest to avoid temporary datasets.
            placed=self.disk_placed(pair,patches) if disk and number==0 and not clean and not self.worker else None
            writers=[];records=[];seq_iou=[];seq_guide=[]
            dest=self.root/'latest'/pair['vi']
            if visualize:
                dest.mkdir(parents=True,exist_ok=True)
                h,w=rows[0]['images'][0].shape[:2]
                for mode in ('visible','infrared'):
                    writer=cv2.VideoWriter(str(dest/(mode+'.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),self.args.fps,(w,h))
                    if not writer.isOpened(): raise RuntimeError('Video writer unavailable')
                    writers.append(writer)
            try:
                for index,row in enumerate(rows):
                    # Later frames cannot affect earlier tracker boxes. Numerical
                    # queries may stop after the last usable GT; final videos still
                    # track the complete sequence, including the absent-target tail.
                    if not visualize and index>last_scored:
                        ignored+=len(rows)-index
                        break
                    if clean:
                        images=[cv2.cvtColor(im,cv2.COLOR_GRAY2BGR) if im.ndim==2 else im.copy() for im in row['images']]
                    elif placed:
                        images=[cv2.imread(str(placed/pair[mode]/(Path(row['name']).stem+'.png'))) for mode in ('vi','ir')]
                        reference=self.images(row,patches,geometry)
                        assert all(np.array_equal(x,y) for x,y in zip(images,reference)),'memory/disk placement mismatch'
                    else: images=self.images(row,patches,geometry)
                    pred=None
                    if index==start:
                        self.oracle.start(*images,row['truth']);pred=row['truth']
                    elif index>start: pred=self.oracle.track(*images)
                    active=index>start and row['truth'] is not None
                    if active:
                        a,b=loss_parts(pred,row['truth'],geometry.center_xy,*images[0].shape[1::-1])
                        seq_iou.append(a);seq_guide.append(b);scored+=1
                    else: ignored+=1
                    records.append(dict(frame=row['name'],status=row['status'],gt_xywh=row['truth'],
                                        pred_xywh=pred,scored=active,initialization=index==start))
                    if visualize:
                        for mode,image,writer in zip(('visible','infrared'),images,writers):
                            image=image.copy()
                            for box,color,label in ((row['truth'],(0,255,0),'GT'),(pred,(0,0,255),'TRACK')):
                                if box is not None:
                                    x,y,w,h=[int(round(v)) for v in box]
                                    cv2.rectangle(image,(x,y),(x+w,y+h),color,2)
                                    cv2.putText(image,label,(max(0,x),max(15,y-4)),0,.5,color,1)
                            cv2.putText(image,row['name']+' | '+row['status'],(8,22),0,.5,(255,255,255),1)
                            writer.write(image)
                            if index==len(rows)-1: cv2.imwrite(str(dest/(mode+'_last.png')),image)
                            if self.args.save_frames:
                                frames=dest/mode;frames.mkdir(exist_ok=True)
                                cv2.imwrite(str(frames/(Path(row['name']).stem+'.jpg')),image)
            finally:
                for writer in writers: writer.release()
            if not seq_iou: raise ValueError('No scored post-initialization frames')
            overlaps.append(float(np.mean(seq_iou)));guides.append(float(np.mean(seq_guide)))
            reports.append(dict(sequence=pair['vi'].as_posix(),iou=overlaps[-1],guide=guides[-1],scored_frames=len(seq_iou)))
            if visualize: write_json(dest/'boxes.json',records)
            if disk and not self.worker and (number%20==0 or number==len(pairs)-1):
                print(f'evaluation {number+1}/{len(pairs)} clean={clean} visual={visualize}',flush=True)
        pixel=0. if clean else tv(patches)
        weights=[self.args.iou_weight,self.args.guide_weight,self.args.tv_weight]
        report=dict(loss=weighted_loss(float(np.mean(overlaps)),float(np.mean(guides)),pixel,weights),
            iou=float(np.mean(overlaps)),guide=float(np.mean(guides)),tv=pixel,
            scored_frames=scored,ignored_frames=ignored,sequences=reports,
            aggregation='mean of per-sequence means; GT-absent/missed/init frames excluded',
            tracker_sequence_queries=self.oracle.queries-start_queries,tracker_frames=self.oracle.frames-start_frames,
            loss_weights=weights,tv_enabled=self.args.tv_weight>0)
        if visualize and not self.worker: write_json(self.root/'latest'/'evaluation.json',report)
        return report


def worker_init(args,devices):
    global WORKER
    import copy
    import torch
    args=copy.copy(args);args.device=devices.get();args.workers=1
    torch.set_num_threads(2);cv2.setNumThreads(1)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    WORKER=Experiment(args);WORKER.worker=True;WORKER.lru=OrderedDict()


def worker_evaluate(patches,pair,visualize,clean):
    key=pair['vi'].as_posix()
    if key not in WORKER.lru:
        WORKER.lru[key]=WORKER.prepared(pair)
        if len(WORKER.lru)>16: WORKER.lru.popitem(last=False)
    WORKER.lru.move_to_end(key)
    WORKER.cache_key=(key,);WORKER.cache={key:WORKER.lru[key]}
    return WORKER.evaluate(patches,[pair],visualize=visualize,clean=clean)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,default=DATA)
    p.add_argument('--mask-root',type=Path)
    p.add_argument('--patch-dir',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--placement-script',type=Path,default=PLACER)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--yaml',default='deep_rgbt')
    p.add_argument('--device',type=int,default=0)
    p.add_argument('--iterations',type=int,default=40)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--devices',type=int,nargs='+',default=list(range(8)))
    p.add_argument('--sequences-per-step',type=int,default=10,help='stratified target mini-batch; 0=all')
    p.add_argument('--directions',type=int,default=4)
    p.add_argument('--sigma',type=float,default=.02)
    p.add_argument('--learning-rate',type=float,default=.005)
    p.add_argument('--iou-weight',type=float,default=.6)
    p.add_argument('--guide-weight',type=float,default=.3)
    p.add_argument('--tv-weight',type=float,default=0.,help='0 disables TV in optimization; raw TV is logged only')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--fps',type=float,default=10)
    p.add_argument('--angles',type=float,nargs='+')
    p.add_argument('--distances',type=float,nargs='+')
    p.add_argument('--sequence-names',nargs='+')
    p.add_argument('--sequence-list',type=Path,help='JSON list of relative visible sequence paths to evaluate')
    p.add_argument('--optimization-list',type=Path,help='JSON subset allowed for patch optimization; other sequences only evaluated')
    p.add_argument('--evaluate-only',action='store_true',help='Evaluate supplied patch with no patch updates')
    p.add_argument('--evaluation-patch-dir',type=Path,help='With --evaluate-only, compare this final patch against --patch-dir initialization')
    p.add_argument('--save-frames',action='store_true')
    p.add_argument('--no-visualization',action='store_true',help='Numerical evaluation only; final delivery normally keeps videos')
    p.add_argument('--visualize-every',type=int,default=0,help='0=only final full-set videos')
    p.add_argument('--skip-clean-baseline',action='store_true')
    p.add_argument('--verify-placement-only',action='store_true',help='compare public/static streamed rendering without tracker queries')
    a=p.parse_args()
    if a.evaluation_patch_dir and not a.evaluate_only:
        raise ValueError('--evaluation-patch-dir requires --evaluate-only')
    a.mask_root=a.mask_root or a.dataset_root/'mask'
    for source in (a.dataset_root,a.mask_root,a.patch_dir):
        s=source.resolve();o=a.output_root.resolve()
        if s==o or s in o.parents or o in s.parents: raise ValueError('Output must be separate from inputs')
    if a.output_root.exists() and any(a.output_root.iterdir()): raise FileExistsError('Use a new empty run directory')
    if min(a.iterations,a.directions)<1 or not 0<a.sigma<1 or not 0<a.learning_rate<1:
        raise ValueError('Invalid optimization settings')
    weights=[a.iou_weight,a.guide_weight,a.tv_weight]
    if not np.isfinite(weights).all() or min(weights)<0 or sum(weights)<=0:
        raise ValueError('Loss weights must be finite, nonnegative and not all zero')
    if a.workers<1 or a.workers>len(a.devices) or len(set(a.devices))!=len(a.devices):
        raise ValueError('Workers need distinct devices')
    if a.sequences_per_step<0: raise ValueError('Invalid batch size')
    import torch
    torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed);torch.set_num_threads(2)
    exp=Experiment(a,load_tracker=not a.verify_placement_only);rng=np.random.default_rng(a.seed)
    if a.verify_placement_only:
        print(json.dumps(exp.verify_placement(exp.initial)))
        return
    patches=[x.copy() for x in exp.initial]
    optimization_pairs=select_sequences(exp.pairs,a.optimization_list) if a.optimization_list else exp.pairs
    save_patch(exp.root/'initial_patch',patches)
    configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    configuration.update(loss_weights=weights,tracker_checkpoint=str(a.checkpoint or PROJECT/'output/checkpoints/train/vipt/deep_rgbt/ViPTrack_ep0060.pth.tar'),
        placement_sha256=hashlib.sha256(a.placement_script.read_bytes()).hexdigest(),
        optimizer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        optimization='joint visible RGB and single-channel IR SPSA; independent full-resolution pixel/channel directions; no grid resize',
        perturbation_shape=[list(p.shape) for p in patches],
        optimized_pixel_channels=sum(p.size for p in patches),tv_enabled=a.tv_weight>0,
        sequence_count=len(exp.pairs),nearest_distance_m=9.,blackbox_outputs='target_bbox only; no backward on tracker')
    configuration['optimization_pool_sequences']=[p['vi'].as_posix() for p in optimization_pairs]
    if (a.patch_dir/'generation.json').exists():
        configuration['initial_generation']=json.loads((a.patch_dir/'generation.json').read_text())
        shutil.copyfile(a.patch_dir/'generation.json',exp.root/'initial_patch'/'generation.json')
        training=Path(configuration['initial_generation']['checkpoint']).parent/'train_config.json'
        if training.exists(): shutil.copyfile(training,exp.root/'initial_patch'/'training_parameters.json')
    source_verification={}
    for name in ('visible.png','infrared.png'):
        source=a.patch_dir/name;stored=exp.root/'initial_patch'/name
        original=cv2.imread(str(source),cv2.IMREAD_UNCHANGED)
        saved=cv2.imread(str(stored),cv2.IMREAD_UNCHANGED)
        equal=np.array_equal(original,saved)
        if not equal: raise AssertionError('Initialization differs from supplied DiT PNG')
        source_verification[name]=dict(source=str(source.resolve()),pixels_equal=equal,
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            stored_sha256=hashlib.sha256(stored.read_bytes()).hexdigest())
    write_json(exp.root/'initial_source_verification.json',source_verification)
    write_json(exp.root/'config.json',configuration)
    start_time=time.time()
    exp.verify_placement(patches)
    clean=None
    if not a.skip_clean_baseline: clean=exp.evaluate(patches,exp.pairs,clean=True,disk=True)
    baseline=exp.evaluate(patches,exp.pairs,disk=True)
    write_json(exp.root/'baseline.json',dict(clean=clean,initial_patch=baseline))
    targets=sorted({x['vi'].name for x in optimization_pairs});groups={t:[x for x in optimization_pairs if x['vi'].name==t] for t in targets}
    queues={t:[] for t in targets};seen=set()
    history=[]
    for iteration in range(0 if a.evaluate_only else a.iterations):
        if a.sequences_per_step==0 or a.sequences_per_step>=len(optimization_pairs): batch=optimization_pairs
        else:
            batch=[]
            for j in range(a.sequences_per_step):
                group=groups[targets[(iteration*a.sequences_per_step+j)%len(targets)]]
                target=targets[(iteration*a.sequences_per_step+j)%len(targets)]
                if not queues[target]: queues[target]=list(rng.permutation(len(group)))
                batch.append(group[queues[target].pop()])
        seen.update(x['vi'].as_posix() for x in batch)
        current=exp.evaluate(patches,batch);grad=[np.zeros_like(x) for x in patches];candidates=[(current,patches,'unchanged')]
        sigma=a.sigma/(iteration+1)**.101
        for direction in range(a.directions):
            perturb=pixel_directions(patches,rng)
            plus=[np.clip(x+sigma*d,0,1) for x,d in zip(patches,perturb)]
            minus=[np.clip(x-sigma*d,0,1) for x,d in zip(patches,perturb)]
            fp=exp.evaluate(plus,batch);fm=exp.evaluate(minus,batch)
            estimate=spsa_gradient(perturb,fp['loss'],fm['loss'],sigma)
            for g,estimate_m in zip(grad,estimate): g+=estimate_m/a.directions
            candidates.extend([(fp,plus,'plus'),(fm,minus,'minus')])
        norm=float(np.sqrt(sum(float(np.mean(g*g)) for g in grad)))
        rate=a.learning_rate/(iteration+1)**.3
        proposal=[np.clip(x-rate*g/max(norm,1e-12),0,1) for x,g in zip(patches,grad)]
        fp=exp.evaluate(proposal,batch);candidates.append((fp,proposal,'gradient'))
        best,patches,selected=min(candidates,key=lambda x:x[0]['loss'])
        # Store exactly the update that will be used next, always quantized as the PNG renderer.
        patches=[np.rint(x*255).astype(np.float32)/255 for x in patches]
        save_patch(exp.root/'current_patch',patches)
        latest=exp.evaluate(patches,batch,visualize=a.visualize_every>0 and (iteration+1)%a.visualize_every==0)
        record=dict(iteration=iteration+1,before=current,after=latest,selected=selected,gradient_rms=norm,sigma=sigma,rate=rate)
        history.append(record);write_json(exp.root/'history.json',history)
        print(json.dumps(dict(iteration=iteration+1,loss=latest['loss'],iou=latest['iou'],guide=latest['guide'],tv=latest['tv'],selected=selected)),flush=True)
        # No historical images/videos are retained. Only numerical logs survive.
    if a.evaluation_patch_dir:
        patches=load_patch(a.evaluation_patch_dir)
        if any(p.shape!=q.shape for p,q in zip(patches,exp.initial)):
            raise ValueError('Evaluation patches must match initial patch shapes')
    save_patch(exp.root/'final_patch',patches)
    exp.cache.clear();exp.cache_key=None
    final=exp.evaluate(patches,exp.pairs,visualize=not a.no_visualization,disk=True)
    result=dict(completed=True,iterations=len(history),sequence_count=len(exp.pairs),clean=clean,
        initial_patch=baseline,final_patch=final,elapsed_s=time.time()-start_time,
        initial_patch_dir=str(exp.root/'initial_patch'),final_patch_dir=str(exp.root/'final_patch'),
        latest_visualization=str(exp.root/'latest'),all_sequences_evaluated=True,
        optimization_unique_sequences=len(seen),optimization_sequences=sorted(seen),
        note='mini-batch optimization; full-set improvement measured separately, not guaranteed')
    write_json(exp.root/'result.json',result)
    if exp.pool: exp.pool.shutdown(wait=True)
    exp.reset_dir(exp.root/'scratch');(exp.root/'scratch').rmdir()
    print(json.dumps({k:result[k] for k in ('completed','iterations','sequence_count','elapsed_s','final_patch_dir')}),flush=True)


if __name__=='__main__': main()
