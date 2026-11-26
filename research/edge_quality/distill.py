"""Experimental guided-trajectory regression into a merged Inko LoRA.

The target is the teacher's average guided velocity over each coarse interval.
This is a simple supervised trajectory experiment, not LCM/DMD or a claim of
general distillation quality. Training and checkpoints remain laptop-only.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file,save_file

from research.supra.alternative_assets import ROOT as ASSETS,REPOS,INKO_CODE_REVISION
from .common import OUTPUT,prompts,save_json,sha256,sheet
from .inko_quant import imports

DIRECTORY=OUTPUT/"distill"


def dataset(args):
    import requests
    from huggingface_hub import HfApi
    dataset_name="kaupane/inko-rollout-25k"
    info=HfApi().dataset_info(dataset_name)
    def fetch(offset):
        response=requests.get("https://datasets-server.huggingface.co/rows",
            params=dict(dataset=dataset_name,config="default",split="train",offset=offset,length=100),timeout=45)
        response.raise_for_status()
        return response.json()["rows"]
    selected=[]
    seen=set()
    offsets=list(range(0,16000,1000))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for batch in pool.map(fetch,offsets):
            for item in batch:
                row=item["row"]
                caption=row.get("prompt","").strip()
                if not caption or sum(ord(c)<128 for c in caption)/len(caption)<.95:
                    continue
                # Short captions keep the training context close to news scenes.
                # Preserve a complete first sentence when one occurs in the cap.
                caption=caption[:400]
                if len(caption)>=400:
                    end=caption.rfind(". ")
                    caption=caption[:end+1] if end>80 else caption.rsplit(" ",1)[0]+"."
                if any(word in caption.lower() for word in ("typography","calligraphy","lettering","caption","inscription","poster","logo","title","written","text reads")):
                    continue
                caption=caption.rstrip(". ")+". oil painting"
                if caption not in seen:
                    selected.append(dict(id=f"public-{row['row_id']}",prompt=caption,source=dataset_name,row_id=row["row_id"]))
                    seen.add(caption)
                if len(selected)>=args.public_count:
                    break
            if len(selected)>=args.public_count:
                break
    if HfApi().dataset_info(dataset_name).sha!=info.sha:
        raise RuntimeError("Source dataset changed during capture; retry with stable provenance")
    subjects=[("a cargo ship","sailing across the sea"),("a rescue boat","moving between flooded houses"),
        ("a doctor and a patient","in a hospital room"),("a group of women football players","celebrating a victory"),
        ("a rocket","launching from a coastal spaceport"),("a farmer and a tractor","harvesting wheat"),
        ("a swimmer","in the ocean"),("a bride and a groom","standing in a garden"),
        ("a singer with a microphone","performing on a stage"),("a train","crossing a bridge"),
        ("a firefighter","fighting a forest fire"),("a spacecraft with solar panels","orbiting a planet"),
        ("solar panels","covering rolling green hills"),("penguins","standing on an iceberg"),
        ("a jellyfish","floating underwater"),("a basketball player","jumping toward a hoop"),
        ("a construction worker","inspecting a damaged road"),("a scientist","working in a laboratory"),
        ("a red bus","driving along a city street"),("an airplane","flying above mountains"),
        ("a crowd of people","walking across a town square"),("a satellite dish","standing beside a research station"),
        ("a fishing boat","returning to a harbor"),("an astronaut","walking on the moon")]
    settings=["in soft morning light","under a cloudy sky","in warm afternoon light","with strong light and shadow","with bright rich colors","with muted earthy colors"]
    for i,(subject,action) in enumerate(subjects):
        for j,setting in enumerate(settings):
            selected.append(dict(id=f"synthetic-{i:02d}-{j}",prompt=f"An oil painting of {subject} {action}, {setting}.",source="handwritten scene combinations"))
    selected.extend(dict(row,source="frozen development prompts") for row in prompts("development"))
    save_json(DIRECTORY/"prompts.json",selected)
    save_json(DIRECTORY/"dataset.json",dict(source_dataset=dataset_name,source_revision=info.sha,
        offsets=offsets,public_prompts=sum(r["source"]==dataset_name for r in selected),
        total_prompts=len(selected),images_downloaded=0,caption_processing="English ASCII fraction >.95; capped at 400 characters; selected typography terms excluded; painting style appended",
        prompts_sha256=sha256(DIRECTORY/"prompts.json")))
    print(json.dumps(dict(total=len(selected),public=sum(r["source"]==dataset_name for r in selected))),flush=True)


@torch.inference_mode()
def encode(args):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    imports()
    from src.utils.encode_text import encode_text
    model=AutoModelForCausalLM.from_pretrained(ASSETS/"inko-text",local_files_only=True,
        dtype=torch.bfloat16,attn_implementation="sdpa").cuda().eval()
    tokenizer=AutoTokenizer.from_pretrained(ASSETS/"inko-text",local_files_only=True)
    tokenizer.padding_side="right"
    target=DIRECTORY/"conditioning"
    target.mkdir(exist_ok=True)
    rows=[dict(id="empty",prompt="")]+json.loads((DIRECTORY/"prompts.json").read_text())
    for index,row in enumerate(rows):
        path=target/f'{row["id"]}.safetensors'
        if path.exists():
            from safetensors import safe_open
            with safe_open(str(path),framework="pt",device="cpu") as cached:
                if cached.metadata().get("prompt")!=row["prompt"]:
                    raise RuntimeError(f"Cached conditioning prompt changed: {row['id']}")
            continue
        text,mask,pooled=encode_text([row["prompt"]],model,tokenizer,pooling=True,
                                   exit_layer=20,exit_mode="stop_at_layer",fast_slice=True)
        save_file({"text":text.cpu().contiguous(),"pooled":pooled.cpu().contiguous()},str(path),metadata={"prompt":row["prompt"]})
        if index%25==0:
            print("encoded",index,"/",len(rows),flush=True)


@torch.inference_mode()
def trajectories(args):
    _,Inko,_,_,_=imports()
    model=Inko.from_single_file(str(ASSETS/"inko/model.safetensors")).cuda().to(torch.bfloat16).eval().requires_grad_(False)
    target=DIRECTORY/"trajectories"
    target.mkdir(exist_ok=True)
    expected_config=dict(size=args.size,teacher_steps=args.teacher_steps,student_steps=args.student_steps,
        guidance=args.guidance,prompts_sha256=sha256(DIRECTORY/"prompts.json"),source_revision=REPOS["inko"][1])
    config_path=target/"run-config.json"
    if config_path.exists() and json.loads(config_path.read_text())!=expected_config:
        raise RuntimeError("Trajectory settings changed; use a fresh experiment directory")
    save_json(config_path,expected_config)
    rows=json.loads((DIRECTORY/"prompts.json").read_text())
    empty=load_file(str(DIRECTORY/"conditioning/empty.safetensors"),device="cuda")
    segments=args.teacher_steps//args.student_steps
    if segments*args.student_steps!=args.teacher_steps:
        raise ValueError("Teacher steps must be divisible by student steps")
    shift=max(1.,args.size/256)
    uniform=torch.linspace(0,1,args.teacher_steps+1,dtype=torch.float32)
    schedule=uniform/(shift-(shift-1)*uniform)
    started=time.perf_counter()
    records=[]
    for index,row in enumerate(rows):
        path=target/f'{row["id"]}.npz'
        if path.exists():
            records.append(dict(id=row["id"],file=path.name))
            continue
        context=load_file(str(DIRECTORY/"conditioning"/f'{row["id"]}.safetensors'),device="cuda")
        seed=1000+index
        z=torch.randn(1,16,args.size//8,args.size//8,generator=torch.Generator().manual_seed(seed)).cuda()
        starts=[]
        velocities=[]
        times=[]
        for step in range(args.teacher_steps):
            if step%segments==0:
                start=z.clone()
                t0=schedule[step].item()
            t=schedule[step].expand(1).cuda()
            with torch.autocast("cuda",dtype=torch.bfloat16):
                conditional=model(z.to(torch.bfloat16),t,context["text"],context["pooled"]).float()
                unconditional=model(z.to(torch.bfloat16),t,empty["text"],empty["pooled"]).float()
            velocity=unconditional+args.guidance*(conditional-unconditional)
            z=z+velocity*(schedule[step+1]-schedule[step]).item()
            if (step+1)%segments==0:
                starts.append(start[0].cpu().numpy().astype(np.float16))
                velocities.append(((z-start)/(schedule[step+1].item()-t0))[0].cpu().numpy().astype(np.float16))
                times.append(t0)
        np.savez(path,z=np.stack(starts),velocity=np.stack(velocities),t=np.array(times,np.float32),
                 final=z.cpu().numpy().astype(np.float16))
        records.append(dict(id=row["id"],file=path.name,seed=seed))
        save_json(DIRECTORY/"trajectory-progress.json",dict(done=index+1,total=len(rows),seconds=time.perf_counter()-started))
        if index%10==0:
            print("teacher",index+1,"/",len(rows),round(time.perf_counter()-started,1),"seconds",flush=True)
    save_json(DIRECTORY/"trajectories.json",dict(size=args.size,teacher_steps=args.teacher_steps,
        student_steps=args.student_steps,guidance=args.guidance,source_revision=REPOS["inko"][1],
        code_revision=INKO_CODE_REVISION,records=records,seconds=time.perf_counter()-started))


class LoRALinear(nn.Module):
    def __init__(self,base,rank):
        super().__init__()
        self.base=base.requires_grad_(False)
        self.down=nn.Linear(base.in_features,rank,bias=False,device=base.weight.device,dtype=torch.float32)
        self.up=nn.Linear(rank,base.out_features,bias=False,device=base.weight.device,dtype=torch.float32)
        nn.init.kaiming_uniform_(self.down.weight,a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.scale=1.

    def forward(self,x):
        return self.base(x)+self.up(self.down(x))*self.scale


def attach(model,rank):
    for name,child in list(model.named_children()):
        if isinstance(child,nn.Linear):
            setattr(model,name,LoRALinear(child,rank))
        else:
            attach(child,rank)


def adapter_state(model):
    return {name:parameter.detach().cpu().contiguous() for name,parameter in model.named_parameters() if parameter.requires_grad}


def merge(model,target,metadata,base_checkpoint=ASSETS/"inko"):
    from safetensors import safe_open
    target.mkdir(parents=True,exist_ok=True)
    adapters={name:module for name,module in model.named_modules() if isinstance(module,LoRALinear)}
    tensors={}
    if adapters:
        with safe_open(str(base_checkpoint/"model.safetensors"),framework="pt",device="cpu") as source:
            for name in source.keys():
                value=source.get_tensor(name)
                module=adapters.get(name.removesuffix(".weight")) if name.endswith(".weight") else None
                if module is not None:
                    value=(value.float()+module.up.weight.detach().float().cpu()@module.down.weight.detach().float().cpu()*module.scale).to(value.dtype)
                tensors[name]=value.contiguous()
    else:
        tensors={name:value.detach().cpu().contiguous() for name,value in model.state_dict().items()}
    save_file(tensors,str(target/"model.safetensors"),metadata={"experiment":"guided trajectory LoRA","base_revision":REPOS["inko"][1]})
    save_json(target/"transformer_config.json",model.get_config())
    config=json.loads((base_checkpoint/"pipeline_config.json").read_text())
    trajectory_config=metadata.get("trajectory_config",{})
    if trajectory_config:
        config.update(height=trajectory_config["size"],width=trajectory_config["size"],
                      num_inference_steps=trajectory_config["student_steps"],guidance_scale=1.)
    save_json(target/"pipeline_config.json",config)
    save_json(target/"training.json",dict(metadata,checkpoint_sha256=sha256(target/"model.safetensors")))


def clear_rope_caches(model):
    for block in model.blocks:
        block.attn.rope._cache.clear()


@torch.no_grad()
def evaluate(model,step,rows,context,trajectory_config):
    from diffusers import AutoencoderKLQwenImage
    _,_,sample_ode,decode_latents,vae_stats_from_config=imports()
    vae=AutoencoderKLQwenImage.from_pretrained(ASSETS/"inko-vae",local_files_only=True,
                                             torch_dtype=torch.bfloat16).cuda().eval()
    vae.encoder=nn.Identity()
    vae.quant_conv=nn.Identity()
    torch.nn.utils.convert_conv3d_weight_memory_format(vae,torch.channels_last_3d)
    mean,std=vae_stats_from_config(vae.config,"cuda")
    destination=DIRECTORY/"evaluation"
    destination.mkdir(exist_ok=True)
    size=trajectory_config["size"]
    steps=trajectory_config["student_steps"]
    records=[]
    errors=[]
    lookup={row["id"]:index for index,row in enumerate(rows)}
    for row in prompts("development")[:4]:
        index=lookup[row["id"]]
        with np.load(DIRECTORY/"trajectories"/f'{row["id"]}.npz') as data:
            teacher=torch.from_numpy(data["final"].copy()).cuda().float()
        seed=1000+index
        noise=torch.randn(1,16,size//8,size//8,generator=torch.Generator().manual_seed(seed)).cuda()
        conditioning=context[row["id"]]
        def velocity(z,t):
            with torch.autocast("cuda",dtype=torch.bfloat16):
                return model(z.to(torch.bfloat16),t,conditioning["text"],conditioning["pooled"]).float()
        student=sample_ode(velocity,noise,steps=steps)
        errors.append(float((student-teacher).square().mean().sqrt()/teacher.square().mean().sqrt()))
        for variant,latent in ((f"teacher-{trajectory_config['teacher_steps']}-guided",teacher),(f"student-{step}-{steps}-single",student)):
            path=destination/f'{row["id"]}-{variant}.png'
            if not path.exists():
                decode_latents(latent*std+mean,vae)[0].save(path)
            records.append(dict(row,variant=variant,seed=seed,image=str(path)))
    # Each row has its own teacher seed; use seed zero only as a sheet grouping label.
    grouped=[dict(record,seed=0) for record in records]
    sheet(grouped,destination/f"checkpoint-{step}.jpg",seed=0,size=256)
    report=dict(step=step,mean_latent_relative_rmse=sum(errors)/len(errors),per_prompt=errors)
    save_json(destination/f"checkpoint-{step}.json",report)
    print("evaluation",json.dumps(report),flush=True)
    del vae
    clear_rope_caches(model)
    gc.collect()
    torch.cuda.empty_cache()


def train(args):
    _,Inko,_,_,_=imports()
    torch.manual_seed(777)
    random.seed(777)
    base_checkpoint=args.base_checkpoint or ASSETS/"inko"
    dtype=torch.float32 if args.full_model else torch.bfloat16
    model=Inko.from_single_file(str(base_checkpoint/"model.safetensors")).cuda().to(dtype).eval().requires_grad_(args.full_model)
    if not args.full_model:
        attach(model,args.rank)
    if args.resume:
        resume_file=DIRECTORY/f"merged-{args.resume}/model.safetensors" if args.full_model else DIRECTORY/f"adapter-{args.resume}.safetensors"
        restored=load_file(str(resume_file),device="cuda")
        with torch.no_grad():
            for name,parameter in model.named_parameters():
                if parameter.requires_grad:
                    parameter.copy_(restored[name])
    model.activation_checkpointing=True
    # Eval mode preserves inference positional policies; autograd remains enabled.
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=args.lr,betas=(.9,.999),weight_decay=0,fused=args.full_model)
    rows=json.loads((DIRECTORY/"prompts.json").read_text())
    samples=[]
    for row in rows:
        path=DIRECTORY/"trajectories"/f'{row["id"]}.npz'
        if not path.exists():
            continue
        with np.load(path) as data:
            samples.append(dict(id=row["id"],z=data["z"],velocity=data["velocity"],t=data["t"]))
    if len(samples)<32:
        raise RuntimeError("Need at least 32 caption trajectories before training")
    context={sample["id"]:load_file(str(DIRECTORY/"conditioning"/f'{sample["id"]}.safetensors'),device="cuda") for sample in samples}
    start=time.perf_counter()
    records=json.loads((DIRECTORY/"training-log.json").read_text()) if args.resume else []
    records=[record for record in records if record["step"]<=args.resume] if args.resume else []
    elapsed_offset=records[-1]["seconds"] if records else 0
    state_path=DIRECTORY/f"trainer-state-{args.resume}.pt"
    optimizer_restored=False
    if args.resume and state_path.exists():
        state=torch.load(state_path,map_location="cpu",weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        optimizer_restored=True
    elif args.resume:
        # The original first run saved adapters only. Reproduce its sampling
        # stream, while explicitly recording the fresh optimizer on this resume.
        for _ in range(args.resume*args.batch):
            sample=random.choice(samples)
            random.randrange(len(sample["t"]))
    trajectory_config=json.loads((DIRECTORY/"trajectories.json").read_text())
    config=dict(rank=args.rank,lr=args.lr,steps=args.steps,
        batch=args.batch,trajectories=len(samples),trainable_parameters=sum(p.numel() for p in parameters),
        objective="MSE to teacher average CFG velocity over each coarse interval",seed=777,
        resume_step=args.resume,optimizer_restored=optimizer_restored,
        full_model=args.full_model,base_checkpoint=str(base_checkpoint))
    prior_config=json.loads((DIRECTORY/"training-config.json").read_text()) if (DIRECTORY/"training-config.json").exists() else None
    history=json.loads((DIRECTORY/"training-history.json").read_text()) if (DIRECTORY/"training-history.json").exists() else ([prior_config] if prior_config else [])
    history.append(config)
    save_json(DIRECTORY/"training-history.json",history)
    save_json(DIRECTORY/"training-config.json",config)
    if not args.resume:
        evaluate(model,0,rows,context,trajectory_config)
    for step in range(args.resume,args.steps):
        if args.warmup:
            warm=min(1.,(step+1)/args.warmup)
            progress=max(0.,(step-args.warmup)/max(1,args.steps-args.warmup))
            rate=args.lr*warm*(.1+.9*.5*(1+math.cos(math.pi*progress)))
            for group in optimizer.param_groups:
                group["lr"]=rate
        optimizer.zero_grad(set_to_none=True)
        losses=[]
        for micro in range(args.batch):
            sample=random.choice(samples)
            index=random.randrange(len(sample["t"]))
            z=torch.from_numpy(sample["z"][index:index+1].copy()).cuda().to(torch.bfloat16)
            target=torch.from_numpy(sample["velocity"][index:index+1].copy()).cuda().float()
            t=torch.tensor([sample["t"][index]],device="cuda",dtype=torch.float32)
            conditioning=context[sample["id"]]
            with torch.autocast("cuda",dtype=torch.bfloat16):
                prediction=model(z,t,conditioning["text"],conditioning["pooled"]).float()
                loss=(prediction-target).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite distillation loss")
            (loss/args.batch).backward()
            losses.append(loss.item())
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
        optimizer.step()
        record=dict(step=step+1,loss=sum(losses)/len(losses),gradient_norm=float(norm),
                    seconds=elapsed_offset+time.perf_counter()-start,peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20)
        records.append(record)
        if (step+1)%10==0 or step==0:
            print(json.dumps(record),flush=True)
            save_json(DIRECTORY/"training-progress.json",record)
        if (step+1)%args.checkpoint_every==0 or step+1==args.steps:
            checkpoint=DIRECTORY/f"adapter-{step+1}.safetensors"
            if not args.full_model:
                save_file(adapter_state(model),str(checkpoint))
            save_json(DIRECTORY/"training-log.json",records)
            merge(model,DIRECTORY/f"merged-{step+1}",dict(record,rank=args.rank,
                  trajectory_config=json.loads((DIRECTORY/"trajectories.json").read_text()) if (DIRECTORY/"trajectories.json").exists() else {}),base_checkpoint)
            evaluate(model,step+1,rows,context,trajectory_config)
            torch.save(dict(next_step=step+1,optimizer=optimizer.state_dict(),python_rng=random.getstate(),
                            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),
                       DIRECTORY/f"trainer-state-{step+1}.pt")
        if time.perf_counter()-start>args.max_seconds:
            if (step+1)%args.checkpoint_every:
                if not args.full_model:
                    save_file(adapter_state(model),str(DIRECTORY/f"adapter-{step+1}.safetensors"))
                save_json(DIRECTORY/"training-log.json",records)
                merge(model,DIRECTORY/f"merged-{step+1}",dict(record,rank=args.rank,trajectory_config=trajectory_config),base_checkpoint)
            break
    print("training finished",flush=True)


def main():
    global DIRECTORY
    torch.set_num_threads(4)
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["dataset","encode","trajectories","train"])
    parser.add_argument("--public-count",type=int,default=192)
    parser.add_argument("--size",type=int,default=384)
    parser.add_argument("--teacher-steps",type=int,default=24)
    parser.add_argument("--student-steps",type=int,default=8)
    parser.add_argument("--guidance",type=float,default=4.5)
    parser.add_argument("--rank",type=int,default=32)
    parser.add_argument("--steps",type=int,default=3000)
    parser.add_argument("--batch",type=int,default=2)
    parser.add_argument("--lr",type=float,default=1e-4)
    parser.add_argument("--checkpoint-every",type=int,default=500)
    parser.add_argument("--max-seconds",type=int,default=7200)
    parser.add_argument("--resume",type=int,default=0)
    parser.add_argument("--experiment-dir",type=Path,default=DIRECTORY)
    parser.add_argument("--base-checkpoint",type=Path)
    parser.add_argument("--full-model",action="store_true")
    parser.add_argument("--warmup",type=int,default=0)
    args=parser.parse_args()
    DIRECTORY=args.experiment_dir
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    {"dataset":dataset,"encode":encode,"trajectories":trajectories,"train":train}[args.command](args)


if __name__=="__main__":
    main()
