"""Numerical checks for the color continuity loss and TV proximal mapping."""
import json
from types import SimpleNamespace
import numpy as np
from optimize_dataset4002_coherent import tv, tv_prox, coherent_candidate

def objective(u,f,weight):
    return .5*float(np.square(u-f).sum())+weight*(float(np.abs(np.diff(u,axis=0)).sum())+float(np.abs(np.diff(u,axis=1)).sum()))

def main():
    constant=np.full((32,32,3),.4,np.float32)
    assert np.array_equal(tv_prox(constant),constant)
    assert np.array_equal(tv_prox(constant,0),constant)
    # Closed-form two-pixel ROF solution; also exercises Neumann boundaries.
    tiny=np.array([[0.,1.],[0.,1.]],np.float32)
    assert np.allclose(tv_prox(tiny,.12,100),[[.12,.88],[.12,.88]],atol=1e-6)
    rng=np.random.default_rng(42)
    clean=np.broadcast_to(np.linspace(.2,.8,64,dtype=np.float32)[None,:,None],(64,64,3)).copy()
    noisy=np.clip(clean+rng.normal(0,.06,clean.shape),0,1).astype(np.float32)
    smooth=tv_prox(noisy,.12,100)
    assert smooth.shape==noisy.shape and 0<=smooth.min()<=smooth.max()<=1
    assert tv([smooth])<.25*tv([noisy])
    assert np.square(smooth-clean).mean()<np.square(noisy-clean).mean()
    assert objective(smooth,noisy,.12)<objective(noisy,noisy,.12)
    assert abs(float(smooth.mean()-noisy.mean()))<1e-6
    edge=np.zeros((64,64),np.float32);edge[:,32:]=1
    preserved=tv_prox(edge,.12,100)
    assert preserved[:,33:].mean()-preserved[:,:31].mean()>.95
    gray=noisy[:,:,0]
    assert np.allclose(tv_prox(noisy,.12,50)[:,:,0],tv_prox(gray,.12,50))
    args=SimpleNamespace(coherence_strength=.12,coherence_steps=50)
    quantized=coherent_candidate([noisy,gray],args)
    assert all(np.max(np.abs(p*255-np.rint(p*255)))<1e-4 for p in quantized)
    assert tv([clean])<tv([noisy])
    print(json.dumps(dict(passed=True,noisy_tv=tv([noisy]),smoothed_tv=tv([smooth]),
                         edge_contrast=float(preserved[:,33:].mean()-preserved[:,:31].mean()),
                         checks=['constant invariance','noisy vs smooth loss','proximal objective decrease',
                         'mean preservation','edge retention','RGB/gray consistency','PNG quantization'])))

if __name__=='__main__': main()
