#!/usr/bin/env python3
"""Report attack effect with explicit sequence-disjoint test results."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import cv2
from optimize_dataset4002_blackbox import write_json


def summarize(rows):
    if not rows: return dict(sequences=0)
    result=dict(sequences=len(rows))
    for key in ('clean_iou','initial_iou','final_iou','drop_from_clean','extra_drop_from_initial'):
        result[key]=float(np.mean([r[key] for r in rows]))
    for key in ('any_drop_from_clean','drop_at_least_010','drop_at_least_020','final_below_050','extra_drop_at_least_002'):
        count=sum(r[key] for r in rows);result[key]=dict(count=count,fraction=count/len(rows))
    adequate=[r for r in rows if r['clean_iou']>=.5]
    result['clean_iou_at_least_050']=dict(sequences=len(adequate),
        drop_at_least_010_count=sum(r['drop_at_least_010'] for r in adequate),
        drop_at_least_010_fraction=sum(r['drop_at_least_010'] for r in adequate)/len(adequate) if adequate else None)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--development-run',type=Path,required=True)
    p.add_argument('--test-run',type=Path,required=True);p.add_argument('--splits',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    splits={k:set(json.loads((a.splits/(k+'.json')).read_text())) for k in ('train','val','test')}
    assert not(splits['train']&splits['val'] or splits['train']&splits['test'] or splits['val']&splits['test'])
    dev=json.loads((a.development_run/'result.json').read_text());test=json.loads((a.test_run/'result.json').read_text())
    assert set(dev['optimization_sequences'])<=splits['train']
    assert not set(dev['optimization_sequences'])&splits['test']
    assert test['iterations']==0
    assert {r['sequence'] for r in test['final_patch']['sequences']}==splits['test']
    assert {r['sequence'] for r in dev['final_patch']['sequences']}==splits['train']|splits['val']
    configurations=[json.loads((run/'config.json').read_text()) for run in (a.development_run,a.test_run)]
    assert configurations[0]['tracker_checkpoint']==configurations[1]['tracker_checkpoint']
    assert configurations[0]['yaml']==configurations[1]['yaml']
    for mode in ('visible','infrared'):
        for stage in ('initial_patch','final_patch'):
            left=cv2.imread(str(a.development_run/stage/(mode+'.png')),cv2.IMREAD_UNCHANGED)
            right=cv2.imread(str(a.test_run/stage/(mode+'.png')),cv2.IMREAD_UNCHANGED)
            assert left is not None and np.array_equal(left,right), 'Test must use exactly the development patches'
    rows=[]
    for report in (dev,test):
        tables={stage:{r['sequence']:r['iou'] for r in report[stage]['sequences']} for stage in ('clean','initial_patch','final_patch')}
        for name in sorted(tables['clean']):
            split=next(k for k,v in splits.items() if name in v)
            clean,initial,final=[tables[k][name] for k in ('clean','initial_patch','final_patch')]
            parts=name.split('/')
            rows.append(dict(sequence=name,split=split,angle=parts[0],distance=parts[1],target=parts[2],
                clean_iou=clean,initial_iou=initial,final_iou=final,drop_from_clean=clean-final,
                extra_drop_from_initial=initial-final,any_drop_from_clean=final<clean,
                drop_at_least_010=clean-final>=.1,drop_at_least_020=clean-final>=.2,
                final_below_050=final<.5,extra_drop_at_least_002=initial-final>=.02))
    assert len({r['sequence'] for r in rows})==len(rows)==sum(len(v) for v in splits.values())
    output=dict(definition='sequence mean IoU drop >= 0.10 vs clean; init/absent/missed GT frames excluded',
        split_integrity='passed; no test sequences used for tracker training, clean validation, or patch updates',
        evaluation_scope='These splits describe tracker training and patch updates. End-to-end holdout status also depends on the generator and prototype training splits, which this report does not verify.',
        checkpoint=configurations[0]['tracker_checkpoint'],yaml=configurations[0]['yaml'],
        all=summarize(rows),splits={k:summarize([r for r in rows if r['split']==k]) for k in splits})
    for dimension in ('angle','distance','target'):
        output[dimension]={v:summarize([r for r in rows if r[dimension]==v]) for v in sorted({r[dimension] for r in rows})}
    a.output.mkdir(parents=True,exist_ok=True);write_json(a.output/'attack_summary.json',output)
    with (a.output/'per_sequence.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # Fixed examples, selected by condition rather than attack success.
    selection=[]
    for mode in ('visible','infrared'):
        grid=np.zeros((990,1200,3),np.uint8)
        for row,angle in enumerate((30,90,105)):
            for col,distance in enumerate((9,15,21)):
                relative=Path(f'vi_camera_view_{angle}/distance_{distance:02d}m/01_jeep')
                candidates=[run/'latest'/relative for run in (a.development_run,a.test_run)]
                sources=[s for s in candidates if (s/'boxes.json').exists()]
                assert len(sources)==1
                source=sources[0];records=json.loads((source/'boxes.json').read_text())
                valid=[i for i,r in enumerate(records) if r['scored']];index=valid[len(valid)//2]
                capture=cv2.VideoCapture(str(source/(mode+'.mp4')))
                capture.set(cv2.CAP_PROP_POS_FRAMES,index);ok,image=capture.read();capture.release()
                assert ok
                cv2.imwrite(str(a.output/f'{mode}_{angle}_{distance:02d}m.png'),image)
                tile=cv2.resize(image,(400,300));y=row*330;x=col*400
                grid[y+30:y+330,x:x+400]=tile
                split=next(k for k,v in splits.items() if relative.as_posix() in v)
                cv2.putText(grid,f'{angle} deg | {distance:02d}m | {split}',(x+8,y+20),0,.48,(255,255,255),1)
                if mode=='visible': selection.append(dict(sequence=relative.as_posix(),split=split,frame=records[index]['frame']))
        cv2.imwrite(str(a.output/(mode+'_examples.jpg')),grid,[cv2.IMWRITE_JPEG_QUALITY,93])
    write_json(a.output/'preview_selection.json',selection)
    # Also show a held-out sequence without choosing by attack strength.
    test_name=sorted(splits['test'])[0];source=a.test_run/'latest'/test_name
    records=json.loads((source/'boxes.json').read_text())
    valid=[i for i,r in enumerate(records) if r['scored']];index=valid[len(valid)//2]
    for mode in ('visible','infrared'):
        capture=cv2.VideoCapture(str(source/(mode+'.mp4')))
        capture.set(cv2.CAP_PROP_POS_FRAMES,index);ok,image=capture.read();capture.release();assert ok
        cv2.imwrite(str(a.output/('test_'+mode+'_example.png')),image)
    write_json(a.output/'test_example.json',dict(sequence=test_name,source=str(source),frame=records[index]['frame'],
        selection='lexicographically first test sequence; middle valid post-initialization frame'))
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
