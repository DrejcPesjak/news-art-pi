"""Separate-process PyTorch fixtures and NumPy/NCNN runtime validation."""
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np

from .common import OUTPUT,save_json


def fixture(args):
    import torch
    from safetensors.torch import load_file
    from .inko_quant import imports,apply_weights
    from research.supra.alternative_assets import ROOT as ASSETS
    _,Inko,_,_,_=imports()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    model=Inko.from_single_file(str(ASSETS/"inko/model.safetensors")).float().to(args.device).eval().requires_grad_(False)
    apply_weights(model,OUTPUT/"weights"/args.variant)
    conditioning=load_file(str(OUTPUT/"conditioning/ship.safetensors"))
    z=torch.randn(1,16,args.size//8,args.size//8,generator=torch.Generator().manual_seed(123)).to(args.device)
    text=conditioning["text"].float().to(args.device)
    pooled=conditioning["pooled"].float().to(args.device)
    records={"z":z.cpu().numpy(),"text":text.cpu().numpy(),"pooled":pooled.cpu().numpy(),"t":np.array([args.time],np.float32)}
    def hook(index):
        def capture(module,inputs,outputs):
            records[f"block_{index}_image"]=outputs[0].detach().cpu().numpy()
            records[f"block_{index}_text"]=outputs[1].detach().cpu().numpy()
        return capture
    for i,block in enumerate(model.blocks):
        block.register_forward_hook(hook(i))
    with torch.inference_mode():
        result=model(z,torch.tensor([args.time],device=args.device),text,pooled)
    records["expected"]=result.cpu().numpy()
    np.savez(OUTPUT/f"fixture-{args.variant}-{args.size}-{args.time}.npz",**records)
    print("saved independent FP32 reference fixture",flush=True)


def validate(args):
    from .array_runtime import InkoArray
    fixture=np.load(OUTPUT/f"fixture-{args.variant}-{args.size}-{args.time}.npz")
    runtime=InkoArray(OUTPUT/"weights"/args.variant,backend=args.backend)
    blocks=[]
    def trace(index,x,text):
        expected=fixture[f"block_{index}_image"][0]
        error=x-expected
        rel=float(np.sqrt(np.mean(error**2))/max(np.sqrt(np.mean(expected**2)),1e-8))
        blocks.append(dict(block=index,relative_rmse=rel,max_abs=float(np.max(np.abs(error)))))
        print(json.dumps(blocks[-1]),flush=True)
    started=time.perf_counter()
    actual=runtime.forward(fixture["z"],float(fixture["t"][0]),fixture["text"],fixture["pooled"],trace=trace)
    error=actual-fixture["expected"]
    report=dict(variant=args.variant,backend=args.backend,size=args.size,time=args.time,
                seconds=time.perf_counter()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                relative_rmse=float(np.sqrt(np.mean(error**2))/np.sqrt(np.mean(fixture["expected"]**2))),
                max_abs=float(np.max(np.abs(error))),blocks=blocks,timing=runtime.timing)
    save_json(OUTPUT/f"array-validation-{args.variant}-{args.size}-{args.time}-{args.backend}.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("blocks","timing")}),flush=True)
    if report["relative_rmse"]>1e-3:
        raise RuntimeError("Array runtime failed full-model numerical parity")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["fixture","validate"])
    parser.add_argument("--variant",default="q4")
    parser.add_argument("--size",type=int,default=256)
    parser.add_argument("--time",type=float,default=.3)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--backend",choices=["ncnn","numpy"],default="ncnn")
    args=parser.parse_args()
    {"fixture":fixture,"validate":validate}[args.command](args)


if __name__=="__main__":
    main()
