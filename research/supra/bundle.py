"""Create or verify a checksummed runtime bundle (no PyTorch on Pi)."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def verify(directory):
    manifest=json.loads((directory/"manifest.json").read_text())
    for name,metadata in manifest.items():
        if Path(name).name!=name:raise ValueError("Manifest entries must be filenames")
        path=directory/name
        digest=hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda:stream.read(1024*1024),b""):
                digest.update(block)
        if path.stat().st_size!=metadata["bytes"] or digest.hexdigest()!=metadata["sha256"]:
            raise ValueError(f"Artifact integrity check failed: {name}")
    print(f"Verified {len(manifest)} runtime artifacts",flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["create","verify"])
    parser.add_argument("directory",type=Path)
    parser.add_argument("--source",type=Path,default=Path("artifacts/supra/ncnn"))
    args=parser.parse_args()
    if args.command=="verify":
        verify(args.directory)
        return
    import numpy as np
    import torch
    from .assets import MODEL_DIR
    for filename in ["validation.json","text-validation.json","vae_tile20-validation.json"]:
        if not json.loads((args.source/filename).read_text())["passed"]:
            raise ValueError(f"Refusing to bundle failed export: {filename}")
    args.directory.mkdir(parents=True,exist_ok=True)
    names=[f"{component}.ncnn.{suffix}" for component in ["dit","text","vae_tile20"] for suffix in ["param","bin"]]
    names.extend(["embedding.npy"])
    names.append("vae.ncnn.param")
    for name in names:
        shutil.copyfile(args.source/name,args.directory/name)
    # Full-frame and tile graphs use byte-identical weights. Store them once.
    full_weights=args.directory/"vae.ncnn.bin"
    if full_weights.exists() or full_weights.is_symlink():full_weights.unlink()
    full_weights.symlink_to("vae_tile20.ncnn.bin")
    names.append("vae.ncnn.bin")
    shutil.copyfile(MODEL_DIR/"text/spiece.model",args.directory/"spiece.model")
    checkpoint=torch.load(MODEL_DIR/"supra/model_final_ema.pt",map_location="cpu",weights_only=True,mmap=True)
    config=checkpoint["config"]
    np.savez(args.directory/"unconditional.npz",context=config["uncond_text"].float().numpy()[None],
             mask=config["uncond_mask"].float().numpy()[None])
    names.extend(["spiece.model","unconditional.npz"])
    manifest={}
    for name in names:
        path=args.directory/name
        digest=hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda:stream.read(1024*1024),b""):
                digest.update(block)
        manifest[name]={"bytes":path.stat().st_size,"sha256":digest.hexdigest()}
    (args.directory/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    verify(args.directory)


if __name__=="__main__":main()
