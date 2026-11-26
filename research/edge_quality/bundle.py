"""Build/verify an offline research Inko bundle without modifying production."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import tarfile

from .common import ROOT,OUTPUT,save_json,sha256


RUNTIME_FILES=["__init__.py","array_runtime.py","quant.py","qwen_tokenizer.py",
               "staged.py","decoder_runtime.py","common.py","bundle.py"]


def verify(directory):
    manifest=json.loads((directory/"bundle.json").read_text())
    for relative,metadata in manifest["files"].items():
        path=directory/relative
        if not path.is_file() or path.stat().st_size!=metadata["bytes"] or sha256(path)!=metadata["sha256"]:
            raise ValueError(f"Bundle file failed verification: {relative}")
    return manifest


def build(args):
    destination=args.output
    if (destination/"bundle.json").exists():
        raise ValueError("Use a fresh output directory; do not overwrite a verified bundle")
    destination.mkdir(parents=True,exist_ok=True)
    def copy(source,relative):
        target=destination/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    for filename in RUNTIME_FILES:
        copy(Path(__file__).parent/filename,Path("research/edge_quality")/filename)
    copy(ROOT/"research/supra/ncnn_runtime.py",Path("research/supra/ncnn_runtime.py"))
    copy(ROOT/"research/supra/__init__.py",Path("research/supra/__init__.py"))
    for filename in ("research/__init__.py",):
        if (ROOT/filename).exists():
            copy(ROOT/filename,Path(filename))
    component_manifests={}
    for relative in (f"weights/{args.variant}",f"text/q{args.text_bits}"):
        source=args.models/relative
        manifest=json.loads((source/"manifest.json").read_text())
        component_manifests[relative]=manifest
        for name,metadata in manifest["files"].items():
            path=source/name
            if sha256(path)!=metadata["sha256"]:
                raise ValueError(f"Source component checksum mismatch: {path}")
            copy(path,Path("models")/relative/name)
        copy(source/"manifest.json",Path("models")/relative/"manifest.json")
    for size in args.sizes:
        latent=size//8
        parts_path=args.models/"decoder"/f"parts-{latent}.json"
        parts=json.loads(parts_path.read_text())
        copy(parts_path,Path("models/decoder")/parts_path.name)
        for part in parts:
            for extension in (".ncnn.param",".ncnn.bin"):
                filename=part["prefix"]+extension
                copy(args.models/"decoder"/filename,Path("models/decoder")/filename)
    copy(args.models/"decoder/vae-config.json",Path("models/decoder/vae-config.json"))
    if args.decoder=="tiny-qwen":
        metadata=json.loads((args.models/"decoder/tiny-qwen.json").read_text())
        for extension in (".ncnn.param",".ncnn.bin"):
            name=metadata["prefix"]+extension
            copy(args.models/"decoder"/name,Path("models/decoder")/name)
        copy(args.models/"decoder/tiny-qwen.json",Path("models/decoder/tiny-qwen.json"))
        copy(ROOT/"models/supra/tiny/README.md",Path("licenses/taesd/README.md"))
        copy(ROOT/"models/supra/tiny/revision.txt",Path("licenses/taesd/revision.txt"))
    save_json(destination/"models/recipe.json",dict(variant=args.variant,text_bits=args.text_bits,
        size=args.sizes[0],steps=args.steps,guidance=args.guidance,decoder=args.decoder))
    for name in ("inko","inko-text","inko-vae"):
        source=ROOT/"models/alternatives"/name
        for filename in ("README.md","LICENSE"):
            if (source/filename).exists():
                copy(source/filename,Path("licenses")/name/filename)
    copy(Path(__file__).parent/"README.md",Path("RESEARCH.md"))
    (destination/"README.txt").write_text(
        "Offline Inko research candidate. Not validated on the Raspberry Pi.\n"
        "Runtime dependencies: Python >=3.11, NumPy, NCNN, Pillow, regex. No Torch.\n"
        "From this directory, use:\n"
        f"python3 -m research.edge_quality.staged run --models models --variant {args.variant} "
        f"--text-bits {args.text_bits} --size {args.sizes[0]} --steps {args.steps} --guidance {args.guidance:g} --decoder {args.decoder} "
        "--work output --prompt 'A ship at sea. oil painting'\n"
        "Each stage has a 480 MiB allocation limit; this is not an RSS or swap cap.\n"
        "NCNN uses four threads; NumPy BLAS uses one to avoid redundant reserves.\n"
        "Verify with: python3 -m research.edge_quality.bundle verify --output .\n")
    files={str(path.relative_to(destination)):dict(bytes=path.stat().st_size,sha256=sha256(path))
           for path in sorted(destination.rglob("*")) if path.is_file()}
    manifest=dict(format="news-art-inko-research-v1",default_variant=args.variant,
        text_bits=args.text_bits,sizes=args.sizes,default_steps=args.steps,default_guidance=args.guidance,
        decoder=args.decoder,production_ready=False,
        note="Laptop-validated research bundle; ARMv7 runtime and Pi performance require physical verification",
        runtime_dependencies={name:importlib.metadata.version(name) for name in ("numpy","ncnn","Pillow","regex")},
        component_sources={key:{name:value[name] for name in ("source_revision","code_revision","checkpoint_sha256","calibration_sha256") if name in value}
                           for key,value in component_manifests.items()},
        files=files,total_bytes=sum(value["bytes"] for value in files.values()))
    save_json(destination/"bundle.json",manifest)
    verify(destination)
    if args.archive:
        with tarfile.open(str(destination)+".tar.gz","w:gz",compresslevel=1) as archive:
            archive.add(destination,arcname=destination.name)
    print(json.dumps(dict(directory=str(destination),files=len(files),mib=manifest["total_bytes"]/2**20,
                          verified=True)),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["build","verify"])
    parser.add_argument("--models",type=Path,default=OUTPUT)
    parser.add_argument("--output",type=Path,default=OUTPUT/"candidate-bundle")
    parser.add_argument("--variant",default="q4-gptq")
    parser.add_argument("--text-bits",type=int,default=8)
    parser.add_argument("--sizes",type=int,nargs="+",default=[256,384])
    parser.add_argument("--steps",type=int,default=20)
    parser.add_argument("--guidance",type=float,default=4.5)
    parser.add_argument("--decoder",choices=["exact","tiny-qwen"],default="exact")
    parser.add_argument("--archive",action="store_true")
    args=parser.parse_args()
    if args.command=="verify":
        manifest=verify(args.output)
        print(json.dumps(dict(verified=True,files=len(manifest["files"]),mib=manifest["total_bytes"]/2**20)))
    else:
        build(args)


if __name__=="__main__":
    main()
