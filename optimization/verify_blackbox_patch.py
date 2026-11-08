#!/usr/bin/env python3
"""Check loss definitions and saved full-sequence videos against label validity."""
import argparse
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
from optimize_dataset4002_blackbox import (iou,loss_parts,tv,load_patch,save_patch,
                                          weighted_loss,pixel_directions,spsa_gradient)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);a=p.parse_args()
    assert iou([0,0,10,10],[0,0,10,10])==1
    assert iou([0,0,10,10],[10,0,10,10])==0
    assert abs(iou([0,0,10,10],[5,0,10,10])-1/3)<1e-12
    overlap,guide=loss_parts([45,45,10,10],[45,45,10,10],(50,50),100,100)
    assert overlap==1 and guide==0
    assert abs(loss_parts([55,45,10,10],[45,45,10,10],(50,50),100,100)[1]-.005)<1e-12
    patches=[np.full((16,16,3),.5,np.float32),np.full((16,16),.5,np.float32)]
    assert tv(patches)==0
    checker=(np.indices((16,16)).sum(0)%2).astype(np.float32)
    assert tv([np.repeat(checker[:,:,None],3,axis=2),checker])==1
    # Pixel-level directions must not tie neighbors together into a grid cell.
    rng=np.random.default_rng(42)
    directions=pixel_directions([np.zeros((256,256,3)),np.zeros((256,256))],rng)
    assert directions[0].shape==(256,256,3) and directions[1].shape==(256,256)
    for direction in directions:
        assert set(np.unique(direction))=={-1.,1.}
        assert .45<float(np.mean(direction[1:]!=direction[:-1]))<.55
        assert .45<float(np.mean(direction[:,1:]!=direction[:,:-1]))<.55
    # A complete orthogonal +/-1 basis recovers distinct adjacent-pixel gradients.
    coefficients=np.array([[-3.,1.],[2.,-1.]],np.float32)
    basis=np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]],np.float32)
    estimate=np.zeros_like(coefficients);sigma=.02;origin=np.full((2,2),.5,np.float32)
    for direction in basis.reshape(4,2,2):
        fp=float(np.sum(coefficients*(origin+sigma*direction)))
        fm=float(np.sum(coefficients*(origin-sigma*direction)))
        estimate+=spsa_gradient([direction],fp,fm,sigma)[0]/4
    assert np.allclose(estimate,coefficients,atol=1e-5)
    assert weighted_loss(.5,.1,0.,[.6,.3,0.])==weighted_loss(.5,.1,100.,[.6,.3,0.])
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);save_patch(root,patches);loaded=load_patch(root)
        assert loaded[0].shape==(16,16,3) and loaded[1].shape==(16,16)
        assert np.allclose(loaded[1],128/255)
    result=dict(loss_checks='passed',png_modality_checks='passed',
                pixel_directions='independent at neighboring pixels/channels',
                adjacent_pixel_gradient_recovery='passed',disabled_tv_invariance='passed')
    if a.run:
        report=json.loads((a.run/'result.json').read_text())
        assert report['completed']
        for stage in ('initial_patch','final_patch'):
            vi=cv2.imread(str(a.run/stage/'visible.png'),cv2.IMREAD_UNCHANGED)
            ir=cv2.imread(str(a.run/stage/'infrared.png'),cv2.IMREAD_UNCHANGED)
            assert vi.ndim==3 and vi.shape[2]==3 and ir.ndim==2
            assert vi.shape[:2]==ir.shape and ir.shape[0]==ir.shape[1]
        assert report['initial_patch']['scored_frames']==report['final_patch']['scored_frames']
        if report['clean']: assert report['clean']['scored_frames']==report['final_patch']['scored_frames']
        n=report['sequence_count'];files=list((a.run/'latest').rglob('boxes.json'))
        assert len(files)==n
        videos=0;frames=0;scored=0
        for path in files:
            records=json.loads(path.read_text());scored+=sum(r['scored'] for r in records)
            assert all(not r['scored'] for r in records if r['gt_xywh'] is None or r['initialization'])
            for mode in ('visible','infrared'):
                capture=cv2.VideoCapture(str(path.parent/(mode+'.mp4')))
                assert capture.isOpened()
                assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT))==len(records)
                ok,first=capture.read();assert ok
                capture.set(cv2.CAP_PROP_POS_FRAMES,len(records)-1);ok,last=capture.read();assert ok
                capture.release();videos+=1;frames+=len(records)
                assert (path.parent/(mode+'_last.png')).exists()
        assert scored==report['final_patch']['scored_frames']
        loss=report['final_patch']
        config=json.loads((a.run/'config.json').read_text())
        weights=config.get('loss_weights',[.6,.3,.1])
        assert abs(loss['loss']-weighted_loss(loss['iou'],loss['guide'],loss['tv'],weights))<1e-10
        if 'perturbation_shape' in config:
            assert config['perturbation_shape']==[list(vi.shape),list(ir.shape)]
            assert config['optimized_pixel_channels']==vi.size+ir.size
            assert 'grid_size' not in config
            source=json.loads((a.run/'initial_source_verification.json').read_text())
            assert all(value['pixels_equal'] for value in source.values())
        result.update(sequences=n,videos=videos,video_frames=frames,scored_frames=scored,
            optimization_unique_sequences=report.get('optimization_unique_sequences'))
        (a.run/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__': main()
