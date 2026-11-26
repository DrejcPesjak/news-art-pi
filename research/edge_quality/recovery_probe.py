"""Compare smaller initializations on teacher-trajectory velocity targets."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from .common import OUTPUT,prompts,save_json
from .inko_quant import imports


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("checkpoints",type=Path,nargs="+")
    args=parser.parse_args()
    torch.set_num_threads(4)
    _,Inko,_,_,_=imports()
    cases=[]
    for row in prompts("development"):
        context=load_file(str(OUTPUT/"distill/conditioning"/f'{row["id"]}.safetensors'),device="cuda")
        with np.load(OUTPUT/"distill/trajectories"/f'{row["id"]}.npz') as source:
            for index in (0,2,4,6,7):
                cases.append(dict(id=row["id"],context=context,
                    z=torch.from_numpy(source["z"][index:index+1].copy()).cuda().to(torch.bfloat16),
                    t=torch.tensor([source["t"][index]],device="cuda",dtype=torch.float32),
                    target=torch.from_numpy(source["velocity"][index:index+1].copy()).cuda().float()))
    reports=[]
    for path in args.checkpoints:
        model=Inko.from_single_file(str(path/"model.safetensors")).cuda().to(torch.bfloat16).eval().requires_grad_(False)
        losses=[]
        for row in cases:
            with torch.autocast("cuda",dtype=torch.bfloat16):
                value=model(row["z"],row["t"],row["context"]["text"],row["context"]["pooled"]).float()
            losses.append((value-row["target"]).square().mean().item())
        report=dict(checkpoint=str(path),mean_mse=sum(losses)/len(losses),max_mse=max(losses))
        reports.append(report)
        print(json.dumps(report),flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    save_json(OUTPUT/"shrink/recovery-probe.json",reports)


if __name__=="__main__":
    main()
