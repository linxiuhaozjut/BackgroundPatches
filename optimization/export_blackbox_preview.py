#!/usr/bin/env python3
"""Export representative videos/frames and all-set measured metrics."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_frame(path,index):
    capture=cv2.VideoCapture(str(path));capture.set(cv2.CAP_PROP_POS_FRAMES,index)
    ok,image=capture.read();capture.release()
    if not ok: raise RuntimeError(str(path))
    return image


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    run=a.run;out=run/'preview';out.mkdir(exist_ok=True)
    result=json.loads((run/'result.json').read_text())
    config=json.loads((run/'config.json').read_text())
    generation_path=run/'initial_patch'/'generation.json'
    generation=json.loads(generation_path.read_text()) if generation_path.exists() else {}
    metrics={name:{k:result[name][k] for k in ('loss','iou','guide','tv','scored_frames','ignored_frames')}
             for name in ('clean','initial_patch','final_patch') if result[name]}
    metrics.update(optimization_unique_sequences=result.get('optimization_unique_sequences'),
                   sequence_count=result['sequence_count'],elapsed_s=result['elapsed_s'])
    (out/'metrics_summary.json').write_text(json.dumps(metrics,indent=2)+'\n')
    by_sequence={stage:{s['sequence']:s for s in result[stage]['sequences']}
                 for stage in ('clean','initial_patch','final_patch') if result[stage]}
    grouping={}
    for dimension,part in (('angle',0),('distance',1),('target',2)):
        keys=sorted({sequence.split('/')[part] for sequence in by_sequence['final_patch']})
        grouping[dimension]={}
        for key in keys:
            grouping[dimension][key]={stage:float(np.mean([value['iou'] for sequence,value in table.items()
                if sequence.split('/')[part]==key])) for stage,table in by_sequence.items()}
    grouping['final_better_than_initial_sequences']=sum(
        by_sequence['final_patch'][s]['iou']<by_sequence['initial_patch'][s]['iou']
        for s in by_sequence['final_patch'])
    (out/'group_metrics.json').write_text(json.dumps(grouping,indent=2)+'\n')
    grid=np.full((2*290,2*276,3),245,np.uint8)
    changes={}
    for row,stage in enumerate(('initial_patch','final_patch')):
        for col,mode in enumerate(('visible','infrared')):
            image=cv2.imread(str(run/stage/(mode+'.png')))
            image=cv2.resize(image,(256,256))
            y=row*290;x=col*276
            label=(f'DiT step {generation.get("checkpoint_step","?")}' if row==0 else 'ViPT optimized')+' '+mode
            cv2.putText(grid,label,(x+8,y+19),0,.42,(0,0,0),1)
            grid[y+26:y+282,x+10:x+266]=image
    cv2.imwrite(str(out/'patch_comparison.png'),grid)
    detail=np.full((3*290,2*276,3),245,np.uint8)
    for col,mode in enumerate(('visible','infrared')):
        initial=cv2.imread(str(run/'initial_patch'/(mode+'.png')),cv2.IMREAD_UNCHANGED)
        final=cv2.imread(str(run/'final_patch'/(mode+'.png')),cv2.IMREAD_UNCHANGED)
        delta=final.astype(np.int16)-initial.astype(np.int16)
        changes[mode]=dict(changed_pixel_channels=int(np.count_nonzero(delta)),
            total_pixel_channels=int(delta.size),changed_fraction=float(np.count_nonzero(delta)/delta.size),
            mean_abs_change_8bit=float(np.abs(delta).mean()),max_abs_change_8bit=int(np.abs(delta).max()))
        # Fixed source crop [96:160,96:160] shows individual pixels, not grid cells.
        crops=[initial[96:160,96:160],final[96:160,96:160],
               np.clip(np.abs(delta[96:160,96:160])*8,0,255).astype(np.uint8)]
        for row,(crop,label) in enumerate(zip(crops,('DiT crop','ViPT crop','abs change x8'))):
            if crop.ndim==2: crop=cv2.cvtColor(crop,cv2.COLOR_GRAY2BGR)
            crop=cv2.resize(crop,(256,256),interpolation=cv2.INTER_NEAREST)
            y=row*290;x=col*276
            cv2.putText(detail,label+' '+mode,(x+8,y+19),0,.45,(0,0,0),1)
            detail[y+26:y+282,x+10:x+266]=crop
    cv2.imwrite(str(out/'pixel_detail.png'),detail)
    changes.update(loss_weights=config.get('loss_weights'),
                   optimized_pixel_channels=config.get('optimized_pixel_channels'),
                   dit_checkpoint=generation.get('checkpoint'),dit_step=generation.get('checkpoint_step'))
    (out/'pixel_changes.json').write_text(json.dumps(changes,indent=2)+'\n')
    frames=[]
    for mode in ('visible','infrared'):
        grid=np.zeros((3*330,3*400,3),np.uint8)
        for row,angle in enumerate((30,90,105)):
            for col,distance in enumerate((9,15,21)):
                relative=Path(f'vi_camera_view_{angle}/distance_{distance:02d}m/01_jeep')
                source=run/'latest'/relative
                if not (source/'boxes.json').exists():
                    cv2.putText(grid,f'{angle} deg | {distance:02d}m | held out',(col*400+8,row*330+20),0,.48,(255,255,255),1)
                    continue
                records=json.loads((source/'boxes.json').read_text())
                valid=[i for i,r in enumerate(records) if r['scored']]
                index=valid[len(valid)//2]
                image=read_frame(source/(mode+'.mp4'),index)
                cv2.imwrite(str(out/f'{mode}_{angle}_{distance:02d}m.png'),image)
                if mode=='visible':
                    frames.append(dict(sequence=str(relative),frame=records[index]['frame'],
                        selection='middle valid post-initialization frame; fixed angles/distances, target 01_jeep'))
                tile=cv2.resize(image,(400,300),interpolation=cv2.INTER_AREA)
                y=row*330;x=col*400
                grid[y+30:y+330,x:x+400]=tile
                title=f'{angle} deg | {distance:02d}m | frame {records[index]["frame"]}'
                cv2.putText(grid,title,(x+8,y+20),0,.48,(255,255,255),1)
        cv2.imwrite(str(out/(mode+'_examples.jpg')),grid,[cv2.IMWRITE_JPEG_QUALITY,93])
    (out/'preview_selection.json').write_text(json.dumps(frames,indent=2)+'\n')
    print(json.dumps(metrics))


if __name__=='__main__': main()
