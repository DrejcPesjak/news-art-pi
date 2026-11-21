"""Standalone CPU pipeline: release each component's RAM by exiting its process."""
import argparse
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["generate","encode","sample","decode"])
    parser.add_argument("--model-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--prompt",default="a sea jellyfish floating in the pitch-black ocean depths")
    parser.add_argument("--steps",type=int,default=25)
    parser.add_argument("--guidance",type=float,default=3)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--threads",type=int,default=3)
    parser.add_argument("--initial-latent",type=Path)
    args=parser.parse_args()
    if args.steps<1 or not 1<=args.threads<=16:
        raise ValueError("Require positive steps and 1–16 threads")
    args.output.mkdir(parents=True,exist_ok=True)
    if args.stage=="generate":
        # Keep the supervisor lightweight: numerical libraries are imported only
        # in children, and no two weight sets coexist in RAM.
        started=time.perf_counter()
        for stage in ["encode","sample","decode"]:
            command=[sys.executable,"-m","research.supra.staged",stage,
                     "--model-dir",str(args.model_dir),"--output",str(args.output),
                     "--prompt",args.prompt,"--steps",str(args.steps),
                     "--guidance",str(args.guidance),"--seed",str(args.seed),
                     "--threads",str(args.threads)]
            if args.initial_latent:
                command.extend(["--initial-latent",str(args.initial_latent)])
            subprocess.run(command,check=True)
        summary={"total_seconds":time.perf_counter()-started,"steps":args.steps,
                 "prompt":args.prompt,"machine":platform.machine(),
                 "stages":{stage:json.loads((args.output/(stage+".json")).read_text())
                           for stage in ["encode","sample","decode"]}}
        (args.output/"result.json").write_text(json.dumps(summary,indent=2)+"\n")
        print(json.dumps(summary,indent=2),flush=True)
        return

    import numpy as np
    from .ncnn_runtime import NCNNModel
    started=time.perf_counter()
    if args.stage=="encode":
        import sentencepiece as spm
        tokenizer=spm.SentencePieceProcessor(model_file=str(args.model_dir/"spiece.model"))
        ids=tokenizer.encode(args.prompt,out_type=int)[:127]+[1]
        mask=np.zeros((1,128),dtype=np.float32);mask[0,:len(ids)]=1
        ids=ids+[0]*(128-len(ids))
        embedding=np.load(args.model_dir/"embedding.npy",mmap_mode="r")
        embedded=embedding[np.asarray(ids)][None].astype(np.float32)
        del embedding
        model=NCNNModel(args.model_dir/"text",threads=args.threads)
        loaded=time.perf_counter()
        context=model(embedded,mask)
        np.savez(args.output/"context.npz",context=context,mask=mask)
    elif args.stage=="sample":
        conditional=np.load(args.output/"context.npz")
        unconditional=np.load(args.model_dir/"unconditional.npz")
        if args.initial_latent:
            z=np.load(args.initial_latent).astype(np.float32)
        else:
            z=np.random.default_rng(args.seed).standard_normal((1,4,32,32),dtype=np.float32)
        if z.shape!=(1,4,32,32):raise ValueError("Initial latent must have shape (1,4,32,32)")
        model=NCNNModel(args.model_dir/"dit",threads=args.threads)
        loaded=time.perf_counter()
        dt=np.float32(1/args.steps)
        for index in range(args.steps):
            t=np.array([index*float(dt)],dtype=np.float32)
            conditioned=model(z,t,conditional["context"],conditional["mask"],scalar_inputs=(1,))
            if args.guidance>1:
                unconditioned=model(z,t,unconditional["context"],unconditional["mask"],scalar_inputs=(1,))
                velocity=unconditioned+np.float32(args.guidance)*(conditioned-unconditioned)
            else:
                velocity=conditioned
            z=z+dt*velocity
            progress={"step":index+1,"steps":args.steps,"elapsed_seconds":time.perf_counter()-loaded}
            (args.output/"progress.json").write_text(json.dumps(progress)+"\n")
            print(json.dumps(progress),flush=True)
        np.save(args.output/"latent.npy",z)
    else:
        from PIL import Image
        if (args.model_dir/"vae_tile20.ncnn.param").exists():
            from .tiled_decode import decode
            loaded=time.perf_counter()
            decoded=decode(np.load(args.output/"latent.npy"),args.model_dir,threads=args.threads)
        else:
            model=NCNNModel(args.model_dir/"vae",threads=args.threads)
            loaded=time.perf_counter()
            decoded=model(np.load(args.output/"latent.npy"))
        pixels=np.rint((np.clip(decoded[0],-1,1)+1)*127.5).transpose(1,2,0).astype(np.uint8)
        Image.fromarray(pixels).save(args.output/"image.png")
    finished=time.perf_counter()
    metrics={"stage":args.stage,"load_seconds":loaded-started,
             "compute_seconds":finished-loaded,"total_seconds":finished-started,
             "peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
             "threads":args.threads,"machine":platform.machine()}
    (args.output/(args.stage+".json")).write_text(json.dumps(metrics,indent=2)+"\n")
    print(json.dumps(metrics),flush=True)


if __name__=="__main__":main()
