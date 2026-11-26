"""Pack only Qwen3's first twenty layers and validate feature extraction."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np

from research.supra.alternative_assets import ROOT as ASSETS,REPOS
from .common import OUTPUT,prompts,save_json,sha256
from .quant import quantize,dequantize


def pack(args):
    from safetensors import safe_open
    import torch
    torch.set_num_threads(4)
    directory=OUTPUT/"text"/f"q{args.bits}"
    directory.mkdir(parents=True,exist_ok=True)
    manifest=dict(format=1,source_revision=REPOS["inko-text"][1],exit_layer=20,tensors={},
                  config=json.loads((ASSETS/"inko-text/config.json").read_text()))
    source_path=ASSETS/"inko-text/model.safetensors"
    with safe_open(str(source_path),framework="pt",device="cpu") as source:
        for name in source.keys():
            if name=="model.embed_tokens.weight":
                np.save(directory/"embedding.npy",source.get_tensor(name).to(torch.float16).numpy())
                continue
            if not name.startswith("model.layers.") or int(name.split(".")[2])>=20:
                continue
            weight=source.get_tensor(name).float().numpy()
            stem=name.replace(".","_")
            if weight.ndim==2:
                packed,scales=quantize(weight,args.bits,64,mse=True)
                np.save(directory/(stem+".q.npy"),packed)
                np.save(directory/(stem+".s.npy"),scales)
                meta=dict(shape=list(weight.shape),bits=args.bits,group=64,q=stem+".q.npy",scales=stem+".s.npy")
            else:
                np.save(directory/(stem+".npy"),weight)
                meta=dict(shape=list(weight.shape),bits=32,file=stem+".npy")
            manifest["tensors"][name]=meta
    shutil.copyfile(ASSETS/"inko-text/tokenizer.json",directory/"tokenizer.json")
    manifest["files"]={p.name:dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in directory.iterdir() if p.name!="manifest.json"}
    manifest["total_bytes"]=sum(meta["bytes"] for meta in manifest["files"].values())
    save_json(directory/"manifest.json",manifest)
    print(json.dumps(dict(storage_mib=manifest["total_bytes"]/2**20,
        embedding_mib=(directory/"embedding.npy").stat().st_size/2**20)),flush=True)


def fixture(args):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from .inko_quant import imports
    imports()
    from src.utils.encode_text import encode_text
    model=AutoModelForCausalLM.from_pretrained(ASSETS/"inko-text",local_files_only=True,dtype=torch.float32,
                                             attn_implementation="eager").to("cuda").eval().requires_grad_(False)
    tokenizer=AutoTokenizer.from_pretrained(ASSETS/"inko-text",local_files_only=True)
    directory=OUTPUT/"text"/f"q{args.bits}"
    manifest=json.loads((directory/"manifest.json").read_text())
    for name,parameter in model.named_parameters():
        meta=manifest["tensors"].get(name)
        if meta is None:
            continue
        if "q" in meta:
            weight=dequantize(np.load(directory/meta["q"]),np.load(directory/meta["scales"]),meta["shape"],meta["bits"],meta["group"])
        else:
            weight=np.load(directory/meta["file"])
        parameter.data.copy_(torch.from_numpy(weight).to("cuda"))
    cases=[dict(id="empty",prompt="")]+prompts(args.split)
    for row in cases:
        with torch.inference_mode():
            text,mask,pooled=encode_text([row["prompt"]],model,tokenizer,pooling=True,
                                       exit_layer=20,exit_mode="stop_at_layer",fast_slice=True)
        np.savez(directory/f'fixture-{row["id"]}.npz',text=text.cpu().numpy(),pooled=pooled.cpu().numpy())
    print("saved Qwen feature references",flush=True)


def validate(args):
    from .array_runtime import QwenArray
    from .qwen_tokenizer import QwenTokenizer
    import resource
    directory=OUTPUT/"text"/f"q{args.bits}"
    cases=[dict(id="empty",prompt="")]+prompts(args.split)
    tokenizer=QwenTokenizer(directory/"tokenizer.json")
    ids=[tokenizer.prompt(row["prompt"]) for row in cases]
    # The tokenizer is not resident during neural feature extraction on the Pi.
    tokenizer.bpe.cache_clear()
    del tokenizer
    gc.collect()
    runtime=QwenArray(directory,backend=args.backend)
    records=[]
    for row,tokens in zip(cases,ids):
        reference=np.load(directory/f'fixture-{row["id"]}.npz')
        started=time.perf_counter()
        text,pooled=runtime.encode(tokens)
        error=text-reference["text"]
        record=dict(id=row["id"],seconds=time.perf_counter()-started,
                    relative_rmse=float(np.linalg.norm(error)/np.linalg.norm(reference["text"])),
                    max_abs=float(np.max(np.abs(error))))
        records.append(record)
        print(json.dumps(record),flush=True)
        if record["relative_rmse"]>1e-3:
            raise RuntimeError("Portable Qwen features failed FP32 reference parity")
    save_json(directory/f"validation-{args.backend}.json",dict(records=records,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["pack","fixture","validate"])
    parser.add_argument("--bits",type=int,choices=[4,8],default=8)
    parser.add_argument("--split",choices=["development","evaluation"],default="development")
    parser.add_argument("--backend",choices=["ncnn","numpy"],default="ncnn")
    args=parser.parse_args()
    {"pack":pack,"fixture":fixture,"validate":validate}[args.command](args)


if __name__=="__main__":
    main()
