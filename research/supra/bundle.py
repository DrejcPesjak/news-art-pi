"""Create or verify a checksummed runtime bundle (no PyTorch on Pi)."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(directory):
    manifest=json.loads((directory/"manifest.json").read_text())
    for name,metadata in manifest.items():
        if Path(name).name!=name:raise ValueError("Manifest entries must be filenames")
        path=directory/name
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Bundle symlinks must stay within the bundle")
        if path.stat().st_size!=metadata["bytes"] or digest_file(path)!=metadata["sha256"]:
            raise ValueError(f"Artifact integrity check failed: {name}")
    print(f"Verified {len(manifest)} runtime artifacts",flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["create","verify"])
    parser.add_argument("directory",type=Path)
    parser.add_argument("--source",type=Path,default=Path("artifacts/supra/ncnn"))
    parser.add_argument("--decoder",choices=["tiny","full","tiled","all"],default="tiny")
    args=parser.parse_args()
    if args.command=="verify":
        verify(args.directory)
        return
    import numpy as np
    import torch
    from .assets import MODEL_DIR, REVISIONS
    components = ["dit", "text"] + {"tiny": ["tiny"], "full": ["vae"],
                                      "tiled": ["vae_tile20"], "all": ["tiny", "vae_tile20"]}[args.decoder]
    checks = ["validation.json", "text-validation.json"]
    checks.extend(component + "-validation.json" for component in components[2:])
    if args.decoder == "all":
        checks.append("vae-validation.json")
    for filename in checks:
        validation=json.loads((args.source/filename).read_text())
        if not validation["passed"]:
            raise ValueError(f"Refusing to bundle failed export: {filename}")
        if filename=="validation.json":
            contexts={row["context"] for row in validation.get("results",[])}
            if not {"jellyfish","solar","rocket","basketball","unconditional"}<=contexts:
                raise ValueError("Run validate_ncnn with real conditional contexts before bundling")
    if args.directory.exists() and any(args.directory.iterdir()):
        raise ValueError("Create a fresh bundle directory; existing bundles are not overwritten")
    args.directory.mkdir(parents=True,exist_ok=True)
    names=[f"{component}.ncnn.{suffix}" for component in components for suffix in ["param","bin"]]
    names.extend(["embedding.npy"])
    if args.decoder == "all":
        names.append("vae.ncnn.param")
    for name in names:
        shutil.copyfile(args.source/name,args.directory/name)
    if args.decoder == "all":
        full_weights = args.directory / "vae.ncnn.bin"
        if digest_file(args.source / "vae.ncnn.bin") == digest_file(args.source / "vae_tile20.ncnn.bin"):
            full_weights.symlink_to("vae_tile20.ncnn.bin")
        else:
            shutil.copyfile(args.source / "vae.ncnn.bin", full_weights)
        names.append("vae.ncnn.bin")
    shutil.copyfile(MODEL_DIR/"text/spiece.model",args.directory/"spiece.model")
    checkpoint=torch.load(MODEL_DIR/"supra/model_final_ema.pt",map_location="cpu",weights_only=True,mmap=True)
    config=checkpoint["config"]
    np.savez(args.directory/"unconditional.npz",context=config["uncond_text"].float().numpy()[None],
             mask=config["uncond_mask"].float().numpy()[None])
    names.extend(["spiece.model","unconditional.npz"])
    used_revisions = ["supra", "text", "tiny" if args.decoder == "tiny" else "vae"]
    if args.decoder == "all":
        used_revisions.append("tiny")
    pipeline = {"generator": "Supra2-IMG", "decoder": args.decoder, "resolution": [256, 256],
                "runtime": "NCNN FP32 arithmetic with FP16 weight storage",
                "components": {key: REVISIONS[key] for key in used_revisions}}
    (args.directory / "pipeline.json").write_text(json.dumps(pipeline, indent=2) + "\n")
    names.append("pipeline.json")
    manifest={}
    for name in names:
        path=args.directory/name
        manifest[name]={"bytes":path.stat().st_size,"sha256":digest_file(path)}
    (args.directory/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    verify(args.directory)


if __name__=="__main__":main()
