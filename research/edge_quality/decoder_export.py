"""Exact one-frame specialization of the pinned Qwen-Image VAE decoder.

For the first frame each causal Conv3d sees zeros at earlier times, so only
its final temporal kernel slice contributes. Temporal upsamplers skip their
time-conv for that frame. Spatial operations retain the original weights.
This is architecture specialization, not distillation or a decoder substitute.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from diffusers import AutoencoderKLQwenImage

from research.supra.alternative_assets import ROOT as ASSETS, REPOS
from research.supra.ncnn_runtime import NCNNModel
from .common import OUTPUT, save_json, sha256


def conv2d(source):
    if isinstance(source,nn.Identity):
        return nn.Identity()
    kernel=source.kernel_size[-2:]
    if hasattr(source,"_padding"):
        padding=(source._padding[2],source._padding[0])
    else:
        padding=source.padding[-2:]
    target=nn.Conv2d(source.in_channels,source.out_channels,kernel,
                     stride=source.stride[-2:],padding=padding,bias=source.bias is not None)
    target.weight=nn.Parameter(source.weight.detach()[:,:,-1].clone() if source.weight.ndim==5 else source.weight.detach().clone(),requires_grad=False)
    if source.bias is not None:
        target.bias=nn.Parameter(source.bias.detach().clone(),requires_grad=False)
    return target


class Norm2D(nn.Module):
    def __init__(self,source):
        super().__init__()
        self.scale=source.scale
        self.gain=nn.Parameter(source.gamma.detach().reshape(1,-1,1,1).clone(),requires_grad=False)

    def forward(self,x):
        # F.normalize clamps the L2 norm at 1e-12; it does not add RMS epsilon.
        # Keep this recognizable to PNNX so NCNN emits one Normalize layer,
        # avoiding multiple full-resolution temporary feature maps.
        return F.normalize(x,p=2,dim=1,eps=1e-12)*self.scale*self.gain


class Residual2D(nn.Module):
    def __init__(self,source):
        super().__init__()
        self.shortcut=conv2d(source.conv_shortcut)
        self.norm1=Norm2D(source.norm1)
        self.norm2=Norm2D(source.norm2)
        self.conv1=conv2d(source.conv1)
        self.conv2=conv2d(source.conv2)

    def forward(self,x):
        return self.shortcut(x)+self.conv2(F.silu(self.norm2(self.conv1(F.silu(self.norm1(x))))))


class Attention2D(nn.Module):
    def __init__(self,source):
        super().__init__()
        self.norm=Norm2D(source.norm)
        self.to_qkv=source.to_qkv
        self.proj=source.proj

    def forward(self,x):
        b,c,h,w=x.shape
        q,k,v=self.to_qkv(self.norm(x)).flatten(2).transpose(1,2).chunk(3,dim=-1)
        keys=k.transpose(-1,-2)
        pieces=[]
        # Keep global attention, but bound the query-score workspace. PNNX
        # unrolls these fixed-shape chunks into separately released GEMMs.
        for start in range(0,h*w,128):
            scores=torch.softmax(torch.matmul(q[:,start:start+128],keys)*(c**-.5),dim=-1)
            pieces.append(torch.matmul(scores,v))
        out=torch.cat(pieces,dim=1).transpose(1,2).reshape(b,c,h,w)
        return x+self.proj(out)


class Decoder2D(nn.Module):
    def __init__(self,vae):
        super().__init__()
        source=vae.decoder
        self.post_quant=conv2d(vae.post_quant_conv)
        self.conv_in=conv2d(source.conv_in)
        mid=[]
        mid.append(Residual2D(source.mid_block.resnets[0]))
        for attn,resnet in zip(source.mid_block.attentions,source.mid_block.resnets[1:]):
            mid.extend([Attention2D(attn),Residual2D(resnet)])
        self.mid=nn.Sequential(*mid)
        up=[]
        for block in source.up_blocks:
            stage=[Residual2D(r) for r in block.resnets]
            if block.upsamplers is not None:
                resample=block.upsamplers[0].resample
                # Integer nearest and nearest-exact doubling select identical pixels.
                stage.extend([nn.Upsample(scale_factor=2,mode="nearest"),resample[1]])
            up.append(nn.Sequential(*stage))
        self.up=nn.Sequential(*up)
        self.norm_out=Norm2D(source.norm_out)
        self.conv_out=conv2d(source.conv_out)

    def forward(self,z):
        x=self.mid(self.conv_in(self.post_quant(z)))
        return self.conv_out(F.silu(self.norm_out(self.up(x)))).clamp(-1,1)


class ExportPart(nn.Module):
    """Give sequential children a valid prefix in PNNX's generated Python."""
    def __init__(self,part):
        super().__init__()
        self.part=part

    def forward(self,x):
        return self.part(x)


@torch.inference_mode()
def main():
    import pnnx
    parser=argparse.ArgumentParser()
    parser.add_argument("--latent-size",type=int,default=32)
    parser.add_argument("--skip-export",action="store_true")
    parser.add_argument("--validate-only",action="store_true")
    parser.add_argument("--parts",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(7)
    directory=OUTPUT/"decoder"
    directory.mkdir(parents=True,exist_ok=True)
    prefix=directory/f"decoder_{args.latent_size}"
    if args.validate_only:
        fixture=np.load(directory/f"fixture-{args.latent_size}.npz")
        runtime=NCNNModel(prefix,threads=4,packing=True)
        started=time.perf_counter()
        actual=runtime(fixture["latent"])
        error=actual-fixture["expected"]
        record=dict(seconds=time.perf_counter()-started,
            relative_rmse=float(np.sqrt(np.mean(error**2))/np.sqrt(np.mean(fixture["expected"]**2))),
            max_abs=float(np.max(np.abs(error))),weight_bytes=Path(str(prefix)+".ncnn.bin").stat().st_size)
        save_json(directory/f"ncnn-validation-{args.latent_size}.json",record)
        print(json.dumps(record),flush=True)
        if record["relative_rmse"]>.01:
            raise RuntimeError("NCNN decoder differs too far from FP32 reference")
        return
    vae=AutoencoderKLQwenImage.from_pretrained(ASSETS/"inko-vae",local_files_only=True,torch_dtype=torch.float32).eval().requires_grad_(False)
    vae.encoder=nn.Identity()
    vae.quant_conv=nn.Identity()
    model=Decoder2D(vae).eval().requires_grad_(False)
    validations=[]
    for size in (8,16,args.latent_size):
        z=torch.randn(1,16,size,size)*2.38
        expected=vae.decode(z.unsqueeze(2)).sample.squeeze(2)
        rewritten=model(z)
        error=(expected-rewritten).float()
        rel=(error.square().mean().sqrt()/expected.square().mean().sqrt().clamp_min(1e-8)).item()
        validations.append(dict(latent_size=size,relative_rmse=rel,max_abs=error.abs().max().item()))
        print(json.dumps(validations[-1]),flush=True)
        if rel>2e-5:
            raise RuntimeError("One-frame decoder rewrite is not numerically equivalent")
    save_json(directory/"rewrite-validation.json",dict(source_revision=REPOS["inko-vae"][1],
        tests=validations,original_decoder_parameters=sum(p.numel() for p in vae.decoder.parameters()),
        specialized_parameters=sum(p.numel() for p in model.parameters()),
        scope="First frame only; no temporal history; full spatial image."))
    np.savez(directory/f"fixture-{args.latent_size}.npz",latent=z.numpy(),expected=rewritten.numpy())
    if args.skip_export:
        return
    if args.parts:
        components=[("stem",nn.Sequential(model.post_quant,model.conv_in,model.mid))]
        for i,stage in enumerate(model.up):
            components.extend((f"up{i}_res{j}",stage[j]) for j in range(3))
            if len(stage)>3:
                # Avoid copying the upsampled high-channel tensor between parts.
                components.append((f"up{i}_resize",nn.Sequential(*list(stage.children())[3:])))
        components.append(("head",nn.Sequential(model.norm_out,nn.SiLU(),model.conv_out,nn.Hardtanh(-1,1))))
        current=z
        parts=[]
        for name,module in components:
            part_prefix=directory/f"part_{args.latent_size}_{name}"
            shape=list(current.shape)
            pnnx.export(ExportPart(module),str(part_prefix)+".pt",(current,),check_trace=False,fp16=True)
            current=module(current)
            kind="resize" if name.endswith("_resize") else "residual" if "_res" in name else name
            parts.append(dict(name=name,kind=kind,prefix=part_prefix.name,input_shape=shape,output_shape=list(current.shape)))
        save_json(directory/f"parts-{args.latent_size}.json",parts)
        print("exported split decoder",flush=True)
        return
    pnnx.export(model,str(prefix)+".pt",(z,),check_trace=False,fp16=True)
    runtime=NCNNModel(prefix,threads=4,packing=True)
    started=time.perf_counter()
    actual=runtime(z.numpy())
    error=actual-rewritten.numpy()
    record=dict(seconds=time.perf_counter()-started,
                relative_rmse=float(np.sqrt(np.mean(error**2))/np.sqrt(np.mean(rewritten.numpy()**2))),
                max_abs=float(np.max(np.abs(error))),weight_bytes=Path(str(prefix)+".ncnn.bin").stat().st_size)
    save_json(directory/f"ncnn-validation-{args.latent_size}.json",record)
    print(json.dumps(record),flush=True)
    if record["relative_rmse"]>.01:
        raise RuntimeError("NCNN decoder differs too far from FP32 reference")


if __name__=="__main__":
    main()
