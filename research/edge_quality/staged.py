"""Offline staged Inko prototype: NumPy/NCNN/regex/Pillow only at runtime.

The parent runs text, sampling and decoding in fresh sequential processes.
This is a research runner, not a production worker integration.
"""
import argparse
import gc
import json
import os
import platform
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

from .common import OUTPUT,save_json,sha256


def limit_memory(mib):
    if mib:
        ceiling=int(mib*2**20)
        resource.setrlimit(resource.RLIMIT_DATA,(ceiling,ceiling))


def encode(args):
    from .qwen_tokenizer import QwenTokenizer
    from .array_runtime import QwenArray
    directory=args.models/"text"/f"q{args.text_bits}"
    tokenizer=QwenTokenizer(directory/"tokenizer.json")
    texts=(args.prompt,"") if args.guidance!=1 else (args.prompt,)
    full_ids=[tokenizer.prompt(prompt,2086) for prompt in texts]
    ids=[tokens[:args.max_text_tokens] for tokens in full_ids]
    tokenizer.bpe.cache_clear()
    del tokenizer
    gc.collect()
    model=QwenArray(directory,threads=args.threads,backend=args.backend)
    positive,positive_pooled=model.encode(ids[0])
    values=dict(text=positive,pooled=positive_pooled)
    if args.guidance!=1:
        negative,negative_pooled=model.encode(ids[1])
        values.update(negative=negative,negative_pooled=negative_pooled)
    np.savez(args.work/"conditioning.npz",**values)
    return dict(prompt_tokens=len(ids[0]),negative_tokens=len(ids[1]) if len(ids)>1 else 0,
                negative_encoded=args.guidance!=1,
                max_text_tokens=args.max_text_tokens,prompt_truncated=len(ids[0])<len(full_ids[0]),
                text_bits=args.text_bits,manifest_sha256=sha256(directory/"manifest.json"))


def sample(args):
    from .array_runtime import InkoArray
    conditioning=np.load(args.work/"conditioning.npz")
    model=InkoArray(args.models/"weights"/args.variant,threads=args.threads,backend=args.backend)
    if args.noise:
        z=np.load(args.noise).astype(np.float32)
    else:
        # NumPy's PCG64 sequence differs from PyTorch; save noise for parity checks.
        z=np.random.default_rng(args.seed).standard_normal((1,16,args.size//8,args.size//8),dtype=np.float32)
    np.save(args.work/"noise.npy",z)
    shift=max(1.,args.size/256)
    uniform=np.linspace(0,1,args.steps+1,dtype=np.float32)
    schedule=uniform/(np.float32(shift)-np.float32(shift-1)*uniform)
    started=time.perf_counter()
    for index in range(args.steps):
        t=float(schedule[index])
        v=model.forward(z,t,conditioning["text"],conditioning["pooled"])
        if args.guidance!=1:
            negative=model.forward(z,t,conditioning["negative"],conditioning["negative_pooled"])
            v=negative+np.float32(args.guidance)*(v-negative)
        z=z+v*np.float32(schedule[index+1]-schedule[index])
        if not np.isfinite(z).all():
            raise ValueError("Non-finite sampling state")
        save_json(args.work/"progress.json",dict(stage="sample",step=index+1,total=args.steps,
                                               seconds=time.perf_counter()-started))
        print(f"step {index+1}/{args.steps}: {time.perf_counter()-started:.2f}s",flush=True)
    np.save(args.work/"latent.npy",z)
    return dict(steps=args.steps,guidance=args.guidance,size=args.size,variant=args.variant,
                seed=args.seed,noise_source=str(args.noise) if args.noise else "NumPy PCG64",
                time_shift=shift,manifest_sha256=sha256(args.models/"weights"/args.variant/"manifest.json"))


def decode(args):
    from .decoder_runtime import DecoderPart
    from PIL import Image
    if args.decoder=="tiny-qwen":
        from research.supra.ncnn_runtime import NCNNModel
        metadata=json.loads((args.models/"decoder/tiny-qwen.json").read_text())
        if not metadata.get("normalized_input"):
            raise ValueError("Tiny Qwen decoder requires an explicit normalized-input contract")
        model=NCNNModel(args.models/"decoder"/metadata["prefix"],threads=args.threads,packing=True)
        decoded=model(np.load(args.work/"latent.npy"))
        pixels=np.rint((decoded[0].clip(-1,1)+1)*127.5).transpose(1,2,0).astype(np.uint8)
        Image.fromarray(pixels).save(args.work/"image.png")
        return dict(image=str(args.work/"image.png"),decoder="Tiny Qwen, locally distilled lossy decoder",
                    decoder_metadata_sha256=sha256(args.models/"decoder/tiny-qwen.json"))
    config=json.loads((args.models/"decoder/vae-config.json").read_text())
    mean=np.array(config["latents_mean"],dtype=np.float32).reshape(1,16,1,1)
    std=np.array(config["latents_std"],dtype=np.float32).reshape(1,16,1,1)
    latent=np.load(args.work/"latent.npy")*std+mean
    parts_file=args.models/"decoder"/f"parts-{args.size//8}.json"
    if parts_file.exists():
        decoded=latent
        for part in json.loads(parts_file.read_text()):
            print("decoder",part["name"],list(decoded.shape),flush=True)
            model=DecoderPart(args.models/"decoder"/part["prefix"],threads=args.threads)
            name=part["name"]
            kind=part.get("kind") or ("resize" if name.endswith("_resize") else "residual" if "_res" in name else name)
            decoded=model.tiled(decoded,kind,part["output_shape"][1],rows=args.decoder_tile_rows)
            del model
            gc.collect()
            # NCNN graphs use different channel/resolution workspaces. Return
            # unused glibc allocations between parts instead of retaining them.
            import ctypes
            libc=ctypes.CDLL(None)
            if hasattr(libc,"malloc_trim"):
                libc.malloc_trim(0)
    else:
        model=DecoderPart(args.models/"decoder"/f"decoder_{args.size//8}",threads=args.threads)
        decoded=model(latent)
    pixels=np.rint((decoded[0].clip(-1,1)+1)*127.5).transpose(1,2,0).astype(np.uint8)
    Image.fromarray(pixels).save(args.work/"image.png")
    return dict(image=str(args.work/"image.png"),decoder="Original Qwen VAE, exact single-frame 2D specialization",
                decoder_tile_rows=args.decoder_tile_rows,parts_manifest_sha256=sha256(parts_file) if parts_file.exists() else None)


def run(args):
    report=dict(prompt=args.prompt,size=args.size,steps=args.steps,guidance=args.guidance,
                seed=args.seed,variant=args.variant,text_bits=args.text_bits,decoder=args.decoder,stages={})
    start=time.perf_counter()
    commands=["encode","sample","decode"]
    if args.cached_conditioning:
        import shutil
        shutil.copyfile(args.cached_conditioning,args.work/"conditioning.npz")
        commands.remove("encode")
        report["cached_conditioning"]=str(args.cached_conditioning)
    for command in commands:
        cmd=[sys.executable,"-m","research.edge_quality.staged",command,
             "--models",str(args.models),"--work",str(args.work),"--prompt",args.prompt,
             "--size",str(args.size),"--steps",str(args.steps),"--guidance",str(args.guidance),
             "--seed",str(args.seed),"--variant",args.variant,"--text-bits",str(args.text_bits),
             "--threads",str(args.threads),"--limit-mib",str(args.limit_mib),
             "--backend",args.backend,"--max-text-tokens",str(args.max_text_tokens)]
        cmd.extend(["--decoder-tile-rows",str(args.decoder_tile_rows)])
        cmd.extend(["--decoder",args.decoder])
        if args.noise:
            cmd.extend(["--noise",str(args.noise)])
        env=dict(os.environ,OPENBLAS_NUM_THREADS="1" if args.backend=="ncnn" else str(args.threads),
                 OMP_NUM_THREADS=str(args.threads),MALLOC_ARENA_MAX="1")
        with (args.work/f"{command}.log").open("w") as log:
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            try:
                remaining=args.timeout-(time.perf_counter()-start)
                code=process.wait(timeout=max(.01,remaining))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                    process.wait()
                raise RuntimeError(f"Inko research pipeline exceeded {args.timeout}s")
        if code:
            raise RuntimeError(f"{command} failed with status {code}; see {args.work/command}.log")
        report["stages"][command]=json.loads((args.work/f"{command}.json").read_text())
        save_json(args.work/"result.json",report)
    report["total_seconds"]=time.perf_counter()-start
    save_json(args.work/"result.json",report)
    print(json.dumps(report,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["run","encode","sample","decode"])
    parser.add_argument("--models",type=Path,default=OUTPUT)
    parser.add_argument("--work",type=Path,default=OUTPUT/"native-run")
    parser.add_argument("--prompt",default="A cargo ship burning at sea, smoke rising into the sky. oil painting")
    parser.add_argument("--variant")
    parser.add_argument("--text-bits",type=int,choices=[4,8])
    parser.add_argument("--size",type=int)
    parser.add_argument("--steps",type=int)
    parser.add_argument("--guidance",type=float)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--limit-mib",type=int,default=480)
    parser.add_argument("--max-text-tokens",type=int,default=256)
    parser.add_argument("--decoder-tile-rows",type=int,default=32)
    parser.add_argument("--decoder",choices=["exact","tiny-qwen"])
    parser.add_argument("--timeout",type=int,default=7200)
    parser.add_argument("--backend",choices=["ncnn","numpy"],default="ncnn")
    parser.add_argument("--noise",type=Path)
    parser.add_argument("--cached-conditioning",type=Path)
    args=parser.parse_args()
    recipe_path=args.models/"recipe.json"
    recipe=json.loads(recipe_path.read_text()) if recipe_path.exists() else {}
    for key,default in dict(variant="q4-gptq",text_bits=8,size=256,steps=12,guidance=3,decoder="exact").items():
        if getattr(args,key) is None:
            setattr(args,key,recipe.get(key,default))
    # All large products use NCNN's thread pool. NumPy's separate BLAS pool
    # otherwise reserves ~140 MiB of unused virtual data on this laptop and
    # consumes the allocation limit despite a much smaller measured RSS.
    os.environ["OPENBLAS_NUM_THREADS"]=str(args.threads) if args.backend=="numpy" and args.command in ("encode","sample") else "1"
    global np
    import numpy as np
    if args.size%16 or args.size<16 or args.size>1184 or args.steps<1 or args.threads<1:
        parser.error("size must be a multiple of 16 in [16,1184]; steps/threads must be positive")
    if not np.isfinite(args.guidance) or args.guidance<0:
        parser.error("guidance must be finite and nonnegative")
    if args.limit_mib<0 or args.timeout<=0 or args.decoder_tile_rows<1 or args.seed<0:
        parser.error("require a nonnegative allocation limit/seed and positive timeout/tile rows")
    if args.max_text_tokens<39 or args.max_text_tokens>2086:
        parser.error("max-text-tokens must be in [39,2086]")
    args.work.mkdir(parents=True,exist_ok=True)
    if args.command=="run":
        run(args)
        return
    limit_memory(args.limit_mib)
    started=time.perf_counter()
    record={"encode":encode,"sample":sample,"decode":decode}[args.command](args)
    record.update(seconds=time.perf_counter()-started,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  allocation_limit_mib=args.limit_mib,
                  machine=platform.machine(),threads=args.threads,backend="ncnn" if args.command=="decode" else args.backend,
                  numpy_version=np.__version__,
                  ncnn_version=getattr(sys.modules.get("ncnn"),"__version__",None),
                  forbidden_imports_loaded=[name for name in ("torch","transformers","diffusers") if name in sys.modules])
    if record["forbidden_imports_loaded"]:
        raise RuntimeError("Heavy ML dependency imported in the portable runtime")
    save_json(args.work/f"{args.command}.json",record)
    print(json.dumps(record),flush=True)


if __name__=="__main__":
    main()
