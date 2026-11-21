"""Overlap-blended 20×20 latent tiles: bounded memory, approximate full decoding."""
import numpy as np
from .ncnn_runtime import NCNNModel


def decode(z, model_dir, threads=3, packing=False):
    if z.shape!=(1,4,32,32):
        raise ValueError("Tiled decoder expects one 32×32 latent")
    model=NCNNModel(model_dir/"vae_tile20",threads=threads,packing=packing)
    canvas=np.zeros((1,3,256,256),dtype=np.float32)
    normalization=np.zeros((1,1,256,256),dtype=np.float32)
    overlap=64
    for y in [0,12]:
        for x in [0,12]:
            tile=model(np.ascontiguousarray(z[:,:,y:y+20,x:x+20]))
            wy=np.ones(160,dtype=np.float32);wx=wy.copy()
            ramp=np.linspace(0,1,overlap,dtype=np.float32)
            if y>0:wy[:overlap]=ramp
            else:wy[-overlap:]=ramp[::-1]
            if x>0:wx[:overlap]=ramp
            else:wx[-overlap:]=ramp[::-1]
            weight=(wy[:,None]*wx[None,:])[None,None]
            target=(slice(None),slice(None),slice(y*8,y*8+160),slice(x*8,x*8+160))
            canvas[target]+=tile*weight
            normalization[:,:,y*8:y*8+160,x*8:x*8+160]+=weight
    if (normalization<=0).any():raise ValueError("Tile blending left an uncovered pixel")
    return canvas/normalization
