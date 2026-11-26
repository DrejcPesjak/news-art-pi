"""Initialize a physically smaller Inko through measured weight surgery.

Retain evenly spaced transformer blocks, preserve paired SwiGLU channels using
activation-weighted output energy, and rescale retained residual gates. This
initializer needs quality evaluation/recovery; pruning is not equivalence.
"""
import argparse
import json
from pathlib import Path
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import load_file,save_file

from .common import OUTPUT,save_json,sha256


def main():
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",type=Path,default=OUTPUT/"distill/merged-500")
    parser.add_argument("--calibration",type=Path,default=OUTPUT/"calibration-distilled-500.safetensors")
    parser.add_argument("--output",type=Path,default=OUTPUT/"shrink/base")
    parser.add_argument("--single-blocks",type=int,default=12)
    parser.add_argument("--ff-width",type=int,default=2048)
    parser.add_argument("--gate-scale",type=float,default=1.5)
    parser.add_argument("--source-blocks",type=int,nargs="+")
    args=parser.parse_args()
    if (args.output/"model.safetensors").exists():
        parser.error("Use a fresh output directory")
    config=json.loads((args.checkpoint/"transformer_config.json").read_text())
    if config["double_stream_depth"]!=1 or (not args.source_blocks and config["single_stream_depth"]%args.single_blocks):
        parser.error("This initializer expects one double block and an evenly divisible single-block count")
    old_ff=round(config["hidden_size"]*config["mlp_ratio"])
    if not 0<args.ff_width<=old_ff or args.ff_width%64:
        parser.error("FF width must be group-aligned and no larger than the source")
    stride=config["single_stream_depth"]//args.single_blocks
    selected=args.source_blocks or [0]+list(range(stride,config["single_stream_depth"]+1,stride))
    if selected!=sorted(set(selected)) or selected[0]!=0 or selected[-1]>config["single_stream_depth"]:
        parser.error("Source blocks must be sorted/unique, retain block zero, and exist in the source")
    mapping={old:new for new,old in enumerate(selected)}
    hessians=load_file(str(args.calibration))
    tensors={}
    channels={}
    energies={}
    with safe_open(str(args.checkpoint/"model.safetensors"),framework="pt",device="cpu") as source:
        for name in source.keys():
            if name.startswith("blocks."):
                old=int(name.split(".")[1])
                if old not in mapping:
                    continue
                new=mapping[old]
                target_name=name.replace(f"blocks.{old}.",f"blocks.{new}.",1)
            else:
                old=None
                target_name=name
            value=source.get_tensor(name)
            if ".mlp" in name and (".up_proj." in name or ".down_proj." in name):
                prefix=name.rsplit(".",2)[0]
                if prefix not in channels:
                    down=source.get_tensor(prefix+".down_proj.weight").float()
                    h=hessians[prefix+".down_proj"]
                    activation_energy=h.diagonal(dim1=-2,dim2=-1).contiguous().reshape(-1)
                    score=down.square().sum(0)*activation_energy
                    keep=score.topk(args.ff_width).indices.sort().values
                    channels[prefix]=keep
                    energies[prefix]=float(score[keep].sum()/score.sum().clamp_min(1e-8))
                keep=channels[prefix]
                if ".up_proj." in name:
                    value=value[torch.cat([keep,keep+old_ff])]
                elif name.endswith(".weight"):
                    value=value[:,keep]
            if old is not None and old!=0 and ".modulation.1." in name:
                value=value.clone()
                value[2*config["hidden_size"]:]*=args.gate_scale
            tensors[target_name]=value.contiguous()
    config["single_stream_depth"]=len(selected)-1
    config["mlp_ratio"]=args.ff_width/config["hidden_size"]
    args.output.mkdir(parents=True,exist_ok=True)
    save_file(tensors,str(args.output/"model.safetensors"),metadata={"method":"block retention, activation-energy paired FF pruning, residual gate rescaling"})
    save_json(args.output/"transformer_config.json",config)
    shutil.copyfile(args.checkpoint/"pipeline_config.json",args.output/"pipeline_config.json")
    parameters=sum(value.numel() for name,value in tensors.items() if ".rope." not in name and name!="t_embedder.factor")
    report=dict(source_checkpoint=str(args.checkpoint),source_sha256=sha256(args.checkpoint/"model.safetensors"),
                calibration=str(args.calibration),calibration_sha256=sha256(args.calibration),
                source_blocks=selected,ff_width=args.ff_width,gate_scale=args.gate_scale,
                parameters=parameters,retained_ff_energy=energies,
                checkpoint_sha256=sha256(args.output/"model.safetensors"))
    save_json(args.output/"surgery.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="retained_ff_energy"}),flush=True)


if __name__=="__main__":
    main()
