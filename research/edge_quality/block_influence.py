"""Measure block changes on real development teacher trajectory states."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from .common import OUTPUT,prompts,save_json,sha256
from .inko_quant import imports


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",type=Path,default=OUTPUT/"distill/merged-500")
    args=parser.parse_args()
    torch.set_num_threads(4)
    _,Inko,_,_,_=imports()
    model=Inko.from_single_file(str(args.checkpoint/"model.safetensors")).cuda().to(torch.bfloat16).eval().requires_grad_(False)
    values={i:[] for i in range(len(model.blocks))}
    def hook(index):
        def record(module,inputs,outputs):
            before=inputs[0].float()
            after=outputs[0].float()
            change=(after-before).norm()/before.norm().clamp_min(1e-8)
            values[index].append(change.item())
        return record
    for index,block in enumerate(model.blocks):
        block.register_forward_hook(hook(index))
    for row in prompts("development"):
        context=load_file(str(OUTPUT/"distill/conditioning"/f'{row["id"]}.safetensors'),device="cuda")
        with np.load(OUTPUT/"distill/trajectories"/f'{row["id"]}.npz') as data:
            for index in (0,2,4,6,7):
                z=torch.from_numpy(data["z"][index:index+1].copy()).cuda().to(torch.bfloat16)
                t=torch.tensor([data["t"][index]],device="cuda",dtype=torch.float32)
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    model(z,t,context["text"],context["pooled"])
        print("measured",row["id"],flush=True)
    records=[dict(block=index,relative_change=sum(change)/len(change)) for index,change in values.items()]
    singles=records[1:]
    selected=sorted(r["block"] for r in sorted(singles,key=lambda r:r["relative_change"],reverse=True)[:12])
    report=dict(checkpoint_sha256=sha256(args.checkpoint/"model.safetensors"),
        split="development",source="Actual guided teacher trajectory states, not random late-time noise",
        records=records,selected=[0]+selected)
    save_json(OUTPUT/"shrink/trajectory-block-influence.json",report)
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    main()
