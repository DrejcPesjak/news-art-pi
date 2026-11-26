"""Train a small Qwen-latent decoder, initialized from existing TAESD weights.

TAESD's SD latent space is not compatible with Inko. This experiment changes
the input to sixteen channels and trains all decoder parameters against the
original Qwen VAE. Inputs are normalized Inko latents, not raw VAE latents.
It is a lossy, newly trained substitution; the exact decoder stays available.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file,load_file

from research.supra.assets import MODEL_DIR,REVISIONS
from research.supra.alternative_assets import ROOT as ASSETS,REPOS
from .common import OUTPUT,prompts,save_json,sheet,sha256
from .distill import DIRECTORY as DISTILL

DIRECTORY=OUTPUT/"tiny-qwen"


@torch.no_grad()
def dataset(args):
    from diffusers import AutoencoderKLQwenImage
    from .inko_quant import imports
    _,_,_,decode_latents,vae_stats_from_config=imports()
    model=AutoencoderKLQwenImage.from_pretrained(ASSETS/"inko-vae",local_files_only=True,
                                               torch_dtype=torch.bfloat16).cuda().eval()
    model.encoder=nn.Identity()
    model.quant_conv=nn.Identity()
    torch.nn.utils.convert_conv3d_weight_memory_format(model,torch.channels_last_3d)
    mean,std=vae_stats_from_config(model.config,"cuda")
    rows=json.loads((DISTILL/"prompts.json").read_text())
    validation={row["id"] for row in rows if row.get("split")=="development"}
    others=[row for row in rows if row["id"] not in validation]
    validation.update(row["id"] for row in others[::14][:24])
    destination=DIRECTORY/"data"
    destination.mkdir(exist_ok=True)
    records=[]
    for index,row in enumerate(rows):
        path=destination/f'{row["id"]}.npz'
        if not path.exists():
            with np.load(DISTILL/"trajectories"/f'{row["id"]}.npz') as source:
                latent=source["final"].copy()
            z=torch.from_numpy(latent).cuda().float()
            image=decode_latents(z*std+mean,model)[0]
            np.savez(path,z=latent,image=np.asarray(image))
        records.append(dict(row,file=path.name,decoder_split="validation" if row["id"] in validation else "training"))
        if index%40==0:
            print("decoder targets",index+1,"/",len(rows),flush=True)
    save_json(DIRECTORY/"dataset.json",dict(records=records,
        source_vae_revision=REPOS["inko-vae"][1],source_trajectories_sha256=sha256(DISTILL/"trajectories.json"),
        input="normalized Inko latent, without VAE mean/std; teacher uses z*std+mean",
        training=sum(r["decoder_split"]=="training" for r in records),
        validation=sum(r["decoder_split"]=="validation" for r in records)))


def create():
    from diffusers import AutoencoderTiny
    base=AutoencoderTiny.from_pretrained(MODEL_DIR/"tiny",local_files_only=True,torch_dtype=torch.float32)
    model=AutoencoderTiny(latent_channels=16)
    state=base.decoder.state_dict()
    # Preserve the variance of the pretrained input projection as an initial
    # guess. It is not a latent-space equivalence; training must learn alignment.
    state["layers.0.weight"]=state["layers.0.weight"].repeat(1,4,1,1)/2
    model.decoder.load_state_dict(state,strict=True)
    model.encoder=nn.Identity()
    return model.decoder


def ssim(a,b):
    a=(a+1)/2
    b=(b+1)/2
    ma=F.avg_pool2d(a,7,stride=1,padding=3)
    mb=F.avg_pool2d(b,7,stride=1,padding=3)
    va=(F.avg_pool2d(a*a,7,stride=1,padding=3)-ma*ma).clamp_min(0)
    vb=(F.avg_pool2d(b*b,7,stride=1,padding=3)-mb*mb).clamp_min(0)
    cov=F.avg_pool2d(a*b,7,stride=1,padding=3)-ma*mb
    return (((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))).mean()


def pixels(value):
    return ((value[0].float().clamp(-1,1)+1)*127.5).round().byte().permute(1,2,0).cpu().numpy()


@torch.no_grad()
def evaluate(model,step,validation):
    destination=DIRECTORY/"evaluation"
    destination.mkdir(exist_ok=True)
    metrics=[]
    records=[]
    for row in validation:
        z=torch.from_numpy(row["z"].copy()).cuda().float()
        target=torch.from_numpy(row["image"].copy()).cuda().permute(2,0,1).unsqueeze(0).float()/127.5-1
        with torch.autocast("cuda",dtype=torch.bfloat16):
            predicted=model(z).float().clamp(-1,1)
        mse=((predicted-target)/2).square().mean().item()
        metrics.append(dict(id=row["id"],psnr=-10*math.log10(max(mse,1e-12)),ssim=ssim(predicted,target).item()))
        if row["id"] in {x["id"] for x in prompts("development")[:4]}:
            for variant,value in (("original-Qwen-VAE",target),(f"tiny-{step}",predicted)):
                path=destination/f'{row["id"]}-{variant}.png'
                if not path.exists():
                    Image.fromarray(pixels(value)).save(path)
                records.append(dict(row,id=row["id"],prompt=row["prompt"],seed=0,variant=variant,image=str(path)))
    sheet(records,destination/f"checkpoint-{step}.jpg",seed=0,size=256)
    report=dict(step=step,mean_psnr=sum(r["psnr"] for r in metrics)/len(metrics),
                mean_ssim=sum(r["ssim"] for r in metrics)/len(metrics),examples=metrics)
    save_json(destination/f"checkpoint-{step}.json",report)
    print("evaluation",json.dumps({k:v for k,v in report.items() if k!="examples"}),flush=True)
    return report


def train(args):
    torch.manual_seed(117)
    random.seed(117)
    model=create().cuda().train()
    rows=json.loads((DIRECTORY/"dataset.json").read_text())["records"]
    train=[]
    validation=[]
    for row in rows:
        with np.load(DIRECTORY/"data"/row["file"]) as source:
            value=dict(row,z=source["z"].copy(),image=source["image"].copy())
        (train if row["decoder_split"]=="training" else validation).append(value)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,betas=(.9,.99),weight_decay=1e-5)
    ema={name:parameter.detach().clone() for name,parameter in model.named_parameters()}
    start=time.perf_counter()
    records=[]
    save_json(DIRECTORY/"training-config.json",dict(parameters=sum(p.numel() for p in model.parameters()),
        steps=args.steps,batch=args.batch,lr=args.lr,training=len(train),validation=len(validation),
        base_revision=REVISIONS.get("tiny"),input_channels=16,normalized_input=True,
        loss="pixel L1 + .2*(1-SSIM) + .1*finite-difference L1",seed=117))
    for step in range(args.steps):
        selected=random.choices(train,k=args.batch)
        z=torch.from_numpy(np.concatenate([row["z"] for row in selected],axis=0)).cuda().float()
        target=torch.from_numpy(np.stack([row["image"] for row in selected])).cuda().permute(0,3,1,2).float()/127.5-1
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            prediction=model(z).float()
        pixel_loss=(prediction-target).abs().mean()
        structure=1-ssim(prediction,target)
        gradient=((prediction[:,:,1:]-prediction[:,:,:-1])-(target[:,:,1:]-target[:,:,:-1])).abs().mean()
        gradient=gradient+((prediction[:,:,:,1:]-prediction[:,:,:,:-1])-(target[:,:,:,1:]-target[:,:,:,:-1])).abs().mean()
        loss=pixel_loss+.2*structure+.1*gradient
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite tiny decoder loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        with torch.no_grad():
            for name,parameter in model.named_parameters():
                ema[name].lerp_(parameter,.01)
        record=dict(step=step+1,loss=loss.item(),pixel_l1=pixel_loss.item(),ssim_loss=structure.item(),
                    seconds=time.perf_counter()-start,peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20)
        records.append(record)
        if (step+1)%20==0 or step==0:
            print(json.dumps(record),flush=True)
            save_json(DIRECTORY/"progress.json",record)
        if (step+1)%args.checkpoint_every==0 or step+1==args.steps:
            live={name:parameter.detach().clone() for name,parameter in model.named_parameters()}
            with torch.no_grad():
                for name,parameter in model.named_parameters():
                    parameter.copy_(ema[name])
            checkpoint=DIRECTORY/f"decoder-{step+1}.safetensors"
            save_file({name:value.cpu().contiguous() for name,value in model.state_dict().items()},str(checkpoint),
                      metadata={"input":"normalized Inko latent; 16 channels","source":"TAESD initialization, Qwen VAE distillation"})
            evaluate(model,step+1,validation)
            with torch.no_grad():
                for name,parameter in model.named_parameters():
                    parameter.copy_(live[name])
            save_json(DIRECTORY/"training-log.json",records)
            gc.collect()
            torch.cuda.empty_cache()


class ExportCore(nn.Module):
    def __init__(self,decoder):
        super().__init__()
        self.decoder=decoder

    def forward(self,z):
        return self.decoder(z).clamp(-1,1)


@torch.no_grad()
def export(args):
    import pnnx
    model=create().eval()
    model.load_state_dict(load_file(str(args.checkpoint)),strict=True)
    target=OUTPUT/"decoder"
    prefix=target/"tiny_qwen"
    z=torch.randn(1,16,32,32,generator=torch.Generator().manual_seed(18))
    expected=model(z).clamp(-1,1).numpy()
    np.savez(target/"tiny-qwen-fixture.npz",latent=z.numpy(),expected=expected)
    pnnx.export(ExportCore(model),str(prefix)+".pt",(z,),check_trace=False,fp16=True)
    save_json(target/"tiny-qwen.json",dict(checkpoint=str(args.checkpoint),sha256=sha256(args.checkpoint),
        parameters=sum(p.numel() for p in model.parameters()),normalized_input=True,
        prefix=prefix.name,lossy=True,source_vae_revision=REPOS["inko-vae"][1],
        initialization_repo=REVISIONS["tiny"][0],initialization_revision=REVISIONS["tiny"][1]))
    print("exported tiny Qwen decoder",flush=True)


def main():
    torch.set_num_threads(4)
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["dataset","train","export"])
    parser.add_argument("--steps",type=int,default=6000)
    parser.add_argument("--batch",type=int,default=2)
    parser.add_argument("--lr",type=float,default=0.0002)
    parser.add_argument("--checkpoint-every",type=int,default=1000)
    parser.add_argument("--checkpoint",type=Path)
    args=parser.parse_args()
    if args.command=="export" and not args.checkpoint:
        parser.error("export requires a decoder checkpoint")
    {"dataset":dataset,"train":train,"export":export}[args.command](args)


if __name__=="__main__":
    main()
