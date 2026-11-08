"""Evaluate actual exported PNGs against every frozen prototype; no reranking."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from paired_dit_data import write_json
from paired_prototype_loss import FrozenImagePrototypeLoss


def evaluate(root, objective):
    root=Path(root)
    info=json.loads((root/'generation.json').read_text())
    results=[]
    for index,record in enumerate(info['outputs']):
        images=[]
        for mode in ('visible','infrared'):
            bgr=cv2.imread(record[mode],cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(record[mode])
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            images.append(torch.from_numpy(rgb.copy()).permute(2,0,1).float()/127.5-1)
        with torch.inference_mode():
            loss,stats=objective.from_images(torch.stack(images)[:,None].to(objective.prototypes.device))
        per_mode={}
        for m,mode in enumerate(('visible','infrared')):
            distances=stats['distances'][m,0].cpu().tolist()
            image=cv2.imread(record[mode],cv2.IMREAD_COLOR)
            per_mode[mode]=dict(distances=distances,worst_distance=max(distances),
                mean_distance=float(np.mean(distances)),farthest_cluster=int(np.argmax(distances)),
                mean_saturation=float(cv2.cvtColor(image,cv2.COLOR_BGR2HSV)[:,:,1].mean()))
        results.append(dict(pair_index=index,smooth_worst_loss=float(loss),modalities=per_mode))
    summary={}
    for mode in ('visible','infrared'):
        summary[mode]={key:float(np.mean([r['modalities'][mode][key] for r in results]))
                       for key in ('worst_distance','mean_distance','mean_saturation')}
    report=dict(checkpoint=info['checkpoint'],checkpoint_step=info['checkpoint_step'],
        seed=info['seed'],sampling_steps=info['sampling_steps'],cfg_scale=info['cfg_scale'],
        num_pairs=len(results),summary=summary,pairs=results,
        metric='cosine distance to each fixed normalized C in frozen VAE image descriptor space; lower is closer',
        interpretation='proximity in this encoder only; not proof of semantic universality or attack effectiveness')
    write_json(root/'prototype_distances.json',report)
    return report,info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-root',type=Path,required=True)
    p.add_argument('--generated',type=Path,required=True)
    p.add_argument('--baseline',type=Path)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--temperature',type=float,default=.1)
    a=p.parse_args()
    manifest=json.loads((a.cache_root/'manifest.json').read_text())
    bank=torch.load(a.cache_root/'prototypes.pt',map_location='cpu',weights_only=True)
    objective=FrozenImagePrototypeLoss(manifest['config']['vae'],bank['C'],a.device,a.temperature)
    result,info=evaluate(a.generated,objective)
    assert info['cache_id']==bank['cache_id']
    if a.baseline:
        baseline,baseinfo=evaluate(a.baseline,objective)
        for key in ('cache_id','seed','sampling_steps','cfg_scale','sampler','image_size','paired_initial_noise','use_ema'):
            assert info[key]==baseinfo[key],f'baseline sampling mismatch: {key}'
        assert len(info['outputs'])==len(baseinfo['outputs'])
        comparison=dict(before=baseline['summary'],after=result['summary'],
            delta={m:{k:result['summary'][m][k]-baseline['summary'][m][k]
                      for k in ('worst_distance','mean_distance','mean_saturation')}
                   for m in ('visible','infrared')},
            num_pairs=len(result['pairs']),seed=info['seed'],negative_distance_delta_is_improvement=True)
        write_json(a.generated/'prototype_comparison.json',comparison)
        print(json.dumps(comparison),flush=True)
    else:
        print(json.dumps(result['summary']),flush=True)


if __name__=='__main__':
    main()
