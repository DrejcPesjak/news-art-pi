"""Groupwise activation-aware GPTQ recovery for Inko's packed weights.

Uses only the frozen development captions and teacher sampling trajectories.
No gradients, image training corpus or evaluation captions are used. Error
feedback is confined to groups of 64 inputs, unlike full-Hessian GPTQ.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from safetensors.torch import load_file,save_file

from research.supra.alternative_assets import ROOT as ASSETS
from .common import OUTPUT,prompts,save_json,sha256
from .inko_quant import imports


@torch.inference_mode()
def calibrate(args):
    _,Inko,sample_ode,_,_=imports()
    model=Inko.from_single_file(str(args.checkpoint/"model.safetensors")).to(device="cuda",dtype=torch.bfloat16).eval().requires_grad_(False)
    hessians={}
    counts={}
    enabled=[False]
    def hook(name):
        def record(module,inputs,output):
            if not enabled[0]:
                return
            x=inputs[0].detach().float().reshape(-1,module.in_features)
            if module.in_features%64:
                raise ValueError("Calibration expects group-aligned input dimensions")
            # Subsample spatial tokens deterministically, retaining all text/time cases.
            if len(x)>256:
                x=x[::max(1,len(x)//256)]
            grouped=x.reshape(-1,module.in_features//64,64).permute(1,2,0)
            with torch.autocast("cuda",enabled=False):
                h=torch.bmm(grouped,grouped.transpose(1,2))
            if name not in hessians:
                hessians[name]=torch.zeros_like(h)
                counts[name]=0
            hessians[name].add_(h)
            counts[name]+=len(x)
        return record
    handles=[]
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.Linear) and name.startswith("blocks."):
            handles.append(module.register_forward_hook(hook(name)))
    started=time.perf_counter()
    for row in prompts("development"):
        context=load_file(str(OUTPUT/"conditioning"/f'{row["id"]}.safetensors'),device="cuda")
        empty=load_file(str(OUTPUT/"conditioning/empty.safetensors"),device="cuda")
        for seed in (0,1):
            noise=torch.randn(1,16,args.size//8,args.size//8,generator=torch.Generator().manual_seed(seed)).cuda()
            step=[0]
            def velocity(z,t):
                selected=set(np.linspace(0,args.steps-1,5,dtype=int).tolist())
                enabled[0]=step[0] in selected
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    conditional=model(z.to(torch.bfloat16),t,context["text"],context["pooled"]).float()
                    unconditional=model(z.to(torch.bfloat16),t,empty["text"],empty["pooled"]).float() if args.guidance!=1 else None
                step[0]+=1
                return unconditional+args.guidance*(conditional-unconditional) if unconditional is not None else conditional
            sample_ode(velocity,noise,steps=args.steps)
        print("calibrated",row["id"],round(time.perf_counter()-started,2),flush=True)
    for handle in handles:
        handle.remove()
    tensors={name:(h/counts[name]).cpu().contiguous() for name,h in hessians.items()}
    save_file(tensors,str(OUTPUT/f"{args.calibration_name}.safetensors"))
    save_json(OUTPUT/f"{args.calibration_name}.json",dict(split="development",prompts=prompts("development"),
        seeds=[0,1],steps=args.steps,selected_steps=np.linspace(0,args.steps-1,5,dtype=int).tolist(),size=args.size,guidance=args.guidance,
        seconds=time.perf_counter()-started,counts=counts,sha256=sha256(OUTPUT/f"{args.calibration_name}.safetensors")))


@torch.inference_mode()
def recover(args):
    from safetensors import safe_open
    source_directory=OUTPUT/"weights"/args.base_variant
    directory=OUTPUT/"weights"/args.output_variant
    directory.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((source_directory/"manifest.json").read_text())
    manifest["variant"]=args.output_variant
    manifest["calibration_sha256"]=sha256(OUTPUT/f"{args.calibration_name}.safetensors")
    manifest["method"]="Groupwise Hessian error feedback; 64 input columns; damping 1% mean diagonal"
    hessians=load_file(str(OUTPUT/f"{args.calibration_name}.safetensors"),device="cuda")
    metrics=[]
    with safe_open(str(args.checkpoint/"model.safetensors"),framework="pt",device="cpu") as source:
        for name,meta in manifest["tensors"].items():
            if meta["bits"]!=4:
                for key in ("q","scales","file"):
                    if key in meta:
                        shutil.copyfile(source_directory/meta[key],directory/meta[key])
                continue
            original=source.get_tensor(name).float().cuda()
            rows,columns=original.shape
            groups=columns//64
            h=hessians[name.removesuffix(".weight")].float().clone()
            # Normalize Hessians before damping to avoid conditioning-scale issues.
            magnitude=h.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(1e-8)
            h=h/magnitude[:,None,None]
            h.diagonal(dim1=-2,dim2=-1).add_(.01)
            factor=torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)),upper=True)
            weight=original.reshape(rows,groups,64).permute(1,0,2).contiguous()
            # Use the tested MSE scale choices as fixed scales for error feedback.
            scales=torch.from_numpy(np.load(source_directory/meta["scales"])).cuda().T.contiguous()
            quantized=torch.empty_like(weight,dtype=torch.int8)
            for index in range(64):
                current=weight[:,:,index].clone()
                q=(current*scales).round().clamp(-7,7)
                quantized[:,:,index]=q.to(torch.int8)
                error=(current-q/scales)/factor[:,index,index,None]
                weight[:,:,index:]-=error[:,:,None]*factor[:,None,index,index:]
            signed=quantized.permute(1,0,2).reshape(rows,columns).cpu().numpy()
            packed=(signed[:,0::2].astype(np.uint8)&15)|((signed[:,1::2].astype(np.uint8)&15)<<4)
            np.save(directory/meta["q"],packed)
            np.save(directory/meta["scales"],scales.T.cpu().numpy())
            reconstructed=(quantized.float()/scales[:,:,None]).permute(1,0,2).reshape(rows,columns)
            relative=((reconstructed-original).square().mean().sqrt()/original.square().mean().sqrt()).item()
            meta["relative_rmse"]=relative
            metrics.append(dict(name=name,relative_rmse=relative))
            print(name,relative,flush=True)
    referenced={meta[key] for meta in manifest["tensors"].values() for key in ("q","scales","file") if key in meta}
    manifest["files"]={name:dict(bytes=(directory/name).stat().st_size,sha256=sha256(directory/name)) for name in sorted(referenced)}
    manifest["total_bytes"]=sum(v["bytes"] for v in manifest["files"].values())
    save_json(directory/"manifest.json",manifest)
    save_json(directory/"recovery.json",metrics)


def main():
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["calibrate","recover"])
    parser.add_argument("--checkpoint",type=Path,default=ASSETS/"inko")
    parser.add_argument("--calibration-name",default="calibration")
    parser.add_argument("--base-variant",default="q4-mixed")
    parser.add_argument("--output-variant",default="q4-gptq")
    parser.add_argument("--size",type=int,default=256)
    parser.add_argument("--steps",type=int,default=20)
    parser.add_argument("--guidance",type=float,default=3)
    args=parser.parse_args()
    {"calibrate":calibrate,"recover":recover}[args.command](args)


if __name__=="__main__":
    main()
