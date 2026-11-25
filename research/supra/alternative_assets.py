"""Pinned, verified assets for the laptop-only alternative-model comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import requests

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "90")
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub import get_token

ROOT = Path("models/alternatives")
INKO_CODE_REVISION = "9f8f79cd7491bf96b13d43f290cc9a281afd5787"
INKO_CODE_FILES = ["src/__init__.py", "src/pipeline.py", "src/models/__init__.py",
                   "src/models/inko.py", "src/models/dit_blocks.py", "src/models/varlen_attention.py",
                   "src/flow/__init__.py", "src/flow/paths.py", "src/flow/solvers.py",
                   "src/utils/__init__.py", "src/utils/device.py", "src/utils/encode_text.py",
                   "src/utils/prompt_contract.py", "src/utils/vae_codec.py", "pyproject.toml"]
REPOS = {
    "surjo": ("SurjoLabs/Surjo-Image-Preview", "c545be850b4e5d9b45b706826fa6fd45a2651251"),
    "surjo-text": ("LiquidAI/LFM2.5-Encoder-230M", "0b649ad0c684378b03d4d8304f7577a662ab89bc"),
    "surjo-vae": ("SurjoLabs/FLUX.2-small-decoder-vae", "65d3176be61832f8496ca33f0a82fa1d3de89abd"),
    "inko": ("kaupane/Inko", "2f690976f5ce9aadad7c76f786d3c503f86795e4"),
    "inko-text": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
    "inko-vae": ("REPA-E/e2e-qwenimage-vae", "9cb4c88de60bc7b1865c7f93abc01745992b8784"),
    "flux-lora": ("alimama-creative/FLUX.1-Turbo-Alpha", "7b29bf041b413949089d78b826a5c9726e3c8e50"),
    "flux-base": ("black-forest-labs/FLUX.1-dev", "45d95c497444bbaebe766e1528792fc9158c0c3a"),
    "flux-quant": ("diffusers/FLUX.1-dev-bnb-4bit", "27114c0933fb58aea21f707e113c25e8c6558487"),
    "flux-text": ("diffusers/t5-nf4", "3a051953358916d471d2b9da3758ec34a74b172c"),
}


def wanted(key, name):
    if key == "flux-quant":
        return name.startswith("transformer/") or name == "README.md"
    if key == "flux-base":
        return name.startswith(("text_encoder/", "tokenizer/", "tokenizer_2/", "vae/", "scheduler/")) or name == "model_index.json"
    return (name.endswith((".safetensors", ".json", ".py", ".model", ".txt", ".jinja"))
            and not name.startswith(("assets/", "samples/", "demo/", "workflows/"))) or name.upper().startswith(("README", "LICENSE"))


def inko_code():
    directory = ROOT/"inko-reviewed"
    directory.mkdir(parents=True, exist_ok=True)
    previous_path = directory/"code-manifest.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    manifest = {"source":"https://github.com/Mtrya/inko", "revision":INKO_CODE_REVISION,"files":{}}
    for name in INKO_CODE_FILES:
        path = directory/name
        valid = (path.exists() and previous.get("revision") == INKO_CODE_REVISION
                 and previous.get("files",{}).get(name) == hashlib.sha256(path.read_bytes()).hexdigest())
        if not valid:
            response = requests.get(f"https://raw.githubusercontent.com/Mtrya/inko/{INKO_CODE_REVISION}/{name}",timeout=30)
            response.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
        manifest["files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory/"REVISION").write_text(INKO_CODE_REVISION+"\n")
    (directory/"code-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


def ranged_file(repo, revision, file, directory):
    """Bounded parallel HTTP ranges; validate every range and final LFS digest."""
    target = directory/file.rfilename
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == file.size:
        return target
    temporary = target.with_name(target.name+".ranges")
    progress = target.with_name(target.name+".ranges.json")
    block = 16*2**20
    done = set(json.loads(progress.read_text())) if progress.exists() else set()
    with temporary.open("ab"):
        pass
    fd = os.open(temporary, os.O_RDWR)
    os.ftruncate(fd, file.size)
    # Reuse complete prefix blocks from the interrupted sequential Hub download.
    if not done and file.lfs:
        cached = directory/".cache/huggingface/download"/Path(file.rfilename).parent
        matches = list(cached.glob("*"+file.lfs.sha256+"*.incomplete"))
        if matches:
            source = max(matches,key=lambda p:p.stat().st_size)
            with source.open("rb") as stream:
                for i in range(source.stat().st_size//block):
                    os.pwrite(fd, stream.read(block), i*block)
                    done.add(i)
    lock = threading.Lock()
    total = (file.size+block-1)//block
    token = get_token()
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{file.rfilename}"
    def transfer(i):
        if i in done:
            return
        start, end = i*block, min(file.size,(i+1)*block)-1
        headers = {"Range":f"bytes={start}-{end}"}
        if token:
            headers["Authorization"] = "Bearer "+token
        for attempt in range(3):
            try:
                with requests.get(url,headers=headers,stream=True,timeout=(15,45)) as response:
                    if response.status_code == 429:
                        time.sleep(min(60,int(response.headers.get("Retry-After","10"))))
                        response.raise_for_status()
                    response.raise_for_status()
                    expected = f"bytes {start}-{end}/{file.size}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("Unexpected HTTP range response")
                    offset = start
                    for data in response.iter_content(2**20):
                        if offset+len(data)>end+1:
                            raise ValueError("Range response exceeded expected size")
                        os.pwrite(fd,data,offset)
                        offset += len(data)
                    if offset != end+1:
                        raise ValueError("Incomplete HTTP range")
                with lock:
                    done.add(i)
                    progress.write_text(json.dumps(sorted(done)))
                    if len(done)%8 == 0 or len(done)==total:
                        print(f"transfer {repo}/{file.rfilename}: {len(done)}/{total} blocks",flush=True)
                return
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2*(attempt+1))
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(transfer,range(total)))
    finally:
        os.close(fd)
    temporary.replace(target)
    progress.unlink(missing_ok=True)
    return target


def download(key):
    repo, revision = REPOS[key]
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    directory = ROOT/key
    manifest = {"repo": repo, "revision": revision, "files": {}}
    for file in info.siblings:
        if not wanted(key, file.rfilename):
            continue
        if file.lfs and file.size > 32*2**20:
            path = ranged_file(repo,revision,file,directory)
        else:
            path = Path(hf_hub_download(repo, file.rfilename, revision=revision, local_dir=directory))
        digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
        expected = file.lfs.sha256 if file.lfs else None
        if expected and digest != expected:
            raise ValueError(f"Checksum mismatch: {repo}/{file.rfilename}")
        manifest["files"][file.rfilename] = {"bytes": path.stat().st_size, "sha256": digest, "upstream_lfs_sha256": expected}
        print(f"verified {key}/{file.rfilename}", flush=True)
    (directory/"download-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    if key == "inko":
        inko_code()
    return key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", choices=list(REPOS))
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download, args.models))


if __name__ == "__main__":
    main()
