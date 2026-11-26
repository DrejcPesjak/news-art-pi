"""Pinned Inko reference, grouped-weight quantization and quality comparisons."""
import argparse
import gc
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

from research.supra.alternative_assets import ROOT as ASSETS, REPOS, INKO_CODE_REVISION
from .common import OUTPUT, prompts, save_json, sheet, sha256
from .quant import quantize, dequantize


def imports():
    sys.path.insert(0, str((ASSETS / "inko-reviewed").resolve()))
    from src.pipeline import InkoPipeline
    from src.models.inko import Inko
    from src.flow.solvers import sample_ode
    from src.utils.vae_codec import decode_latents, vae_stats_from_config
    return InkoPipeline, Inko, sample_ode, decode_latents, vae_stats_from_config


@torch.inference_mode()
def encode(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    imports()
    from src.utils.encode_text import encode_text
    directory = OUTPUT / "conditioning"
    directory.mkdir(parents=True, exist_ok=True)
    text = AutoModelForCausalLM.from_pretrained(ASSETS/"inko-text", local_files_only=True,
            dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(ASSETS/"inko-text", local_files_only=True)
    tokenizer.padding_side = "right"
    for row in [dict(id="empty", prompt="")] + prompts(args.split):
        tensors = encode_text([row["prompt"]], text, tokenizer, pooling=True,
                              exit_layer=20, exit_mode="stop_at_layer", fast_slice=True)
        context, mask, pooled = tensors
        save_file({"text":context.cpu().contiguous(), "mask":mask.cpu().contiguous(),
                   "pooled":pooled.cpu().contiguous()}, str(directory / f'{row["id"]}.safetensors'),
                  metadata={"prompt":row["prompt"], "text_revision":REPOS["inko-text"][1]})
        save_json(directory / f'{row["id"]}.json', dict(row, text_shape=list(context.shape),
                  mask=mask.cpu().tolist(), source_revision=REPOS["inko-text"][1]))
    print("encoded", len(prompts(args.split)), "prompts and empty conditioning", flush=True)


def pack(args):
    """Export one tensor at a time, avoiding full-model resident copies."""
    directory = args.weight_output or OUTPUT / "weights" / args.variant
    directory.mkdir(parents=True, exist_ok=True)
    record = {"format":1, "variant":args.variant, "group":64, "tensors":{},
              "source_revision":REPOS["inko"][1], "code_revision":INKO_CODE_REVISION,
              "config":json.loads((args.checkpoint/"transformer_config.json").read_text()),
              "checkpoint_sha256":sha256(args.checkpoint/"model.safetensors")}
    with safe_open(str(args.checkpoint/"model.safetensors"), framework="pt", device="cpu") as source:
        for name in source.keys():
            if ".rope." in name:
                # Geometry is reconstructed analytically; never load unused tables.
                continue
            weight = source.get_tensor(name).float().numpy()
            # Keep small input/output/conditioning projections and norms intact.
            is_linear = weight.ndim == 2 and name.startswith("blocks.") and name.endswith("weight")
            bits = 8 if args.variant == "q8" or (args.variant != "q4" and ".modulation" in name) else 4
            stem = name.replace(".", "_")
            if is_linear:
                packed, scales = quantize(weight, bits, 64, mse=True)
                np.save(directory / (stem+".q.npy"), packed)
                np.save(directory / (stem+".s.npy"), scales)
                restored = dequantize(packed, scales, weight.shape, bits)
                metadata = dict(shape=list(weight.shape), bits=bits, group=64,
                                q=stem+".q.npy", scales=stem+".s.npy",
                                relative_rmse=float(np.linalg.norm(restored-weight)/max(np.linalg.norm(weight),1e-12)))
            else:
                np.save(directory / (stem+".npy"), weight.astype(np.float32))
                metadata = dict(shape=list(weight.shape), bits=32, file=stem+".npy")
            record["tensors"][name] = metadata
    referenced={meta[key] for meta in record["tensors"].values() for key in ("q","scales","file") if key in meta}
    record["files"] = {name:dict(bytes=(directory/name).stat().st_size,sha256=sha256(directory/name)) for name in sorted(referenced)}
    record["total_bytes"] = sum(v["bytes"] for v in record["files"].values())
    save_json(directory / "manifest.json", record)
    print(json.dumps({"variant":args.variant,"storage_mib":record["total_bytes"]/2**20}),flush=True)


def apply_weights(model, directory):
    manifest = json.loads((directory/"manifest.json").read_text())
    for name, parameter in model.named_parameters():
        meta = manifest["tensors"][name]
        if "q" in meta:
            weight = dequantize(np.load(directory/meta["q"]), np.load(directory/meta["scales"]),
                                meta["shape"],meta["bits"],meta["group"])
        else:
            weight = np.load(directory/meta["file"])
        parameter.data.copy_(torch.from_numpy(weight).to(device=parameter.device,dtype=parameter.dtype))


@torch.inference_mode()
def generate(args):
    _, Inko, sample_ode, decode_latents, vae_stats_from_config = imports()
    from diffusers import AutoencoderKLQwenImage
    directory = OUTPUT / "inko"
    directory.mkdir(parents=True, exist_ok=True)
    transformer = Inko.from_single_file(str(args.checkpoint/"model.safetensors")).to(device="cuda",dtype=torch.bfloat16).eval()
    if args.variant != "bf16":
        apply_weights(transformer, args.weight_output or OUTPUT/"weights"/args.variant)
    if args.tiny_checkpoint:
        from .tiny_decoder import create
        vae=create()
        vae.load_state_dict(load_file(str(args.tiny_checkpoint)),strict=True)
        vae=vae.to(device="cuda",dtype=torch.bfloat16).eval()
        mean=std=None
    else:
        vae = AutoencoderKLQwenImage.from_pretrained(ASSETS/"inko-vae",local_files_only=True,
                                                   torch_dtype=torch.bfloat16).to("cuda").eval()
        vae.encoder = torch.nn.Identity()
        vae.quant_conv = torch.nn.Identity()
        torch.nn.utils.convert_conv3d_weight_memory_format(vae, torch.channels_last_3d)
        mean, std = vae_stats_from_config(vae.config, "cuda")
    records_path = directory / "records.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else []
    def context(identifier):
        tensors = load_file(str(OUTPUT/"conditioning"/f"{identifier}.safetensors"),device="cuda")
        return tensors["text"], tensors["pooled"], None
    unconditional = context("empty")
    variant = f"{args.label or args.variant}-{args.size}px-{args.steps}s-cfg{args.guidance:g}"
    if args.tiny_checkpoint:
        variant+="-tiny"
    for row in prompts(args.split):
        conditional = context(row["id"])
        for seed in args.seeds:
            path = directory / f'{row["id"]}-{seed}-{variant}.png'
            if any(r["image"] == str(path) for r in records):
                continue
            noise = torch.randn(1,16,args.size//8,args.size//8,
                                generator=torch.Generator(device="cpu").manual_seed(seed)).cuda()
            def velocity(x,t):
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    v = transformer(x.to(torch.bfloat16),t,*conditional).float()
                    if args.guidance == 1:
                        return v
                    empty = transformer(x.to(torch.bfloat16),t,*unconditional).float()
                    return empty + args.guidance*(v-empty)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            latent = sample_ode(velocity,noise,steps=args.steps)
            save_file({"latent":latent.cpu()},str(path.with_suffix(".safetensors")))
            if args.tiny_checkpoint:
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    decoded=vae(latent.to(torch.bfloat16))
                pixels=((decoded[0].float().clamp(-1,1)+1)*127.5).round().byte().permute(1,2,0).cpu().numpy()
                Image.fromarray(pixels).save(path)
            else:
                images = decode_latents(latent*std+mean,vae)
                images[0].save(path)
            torch.cuda.synchronize()
            record = dict(row,seed=seed,variant=variant,image=str(path),steps=args.steps,
                          guidance=args.guidance,size=args.size,seconds=time.perf_counter()-started,
                          peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                          source_revision=REPOS["inko"][1],code_revision=INKO_CODE_REVISION,
                          checkpoint=str(args.checkpoint),decoder="tiny-qwen" if args.tiny_checkpoint else "original-Qwen-VAE",
                          tiny_checkpoint=str(args.tiny_checkpoint) if args.tiny_checkpoint else None)
            records.append(record)
            save_json(records_path,records)
            print(json.dumps({k:record[k] for k in ("id","seed","variant","seconds")}),flush=True)
    selected = [r for r in records if r["split"] == args.split]
    for seed in args.seeds:
        sheet(selected,directory/f"{args.split}-seed-{seed}.jpg",seed=seed)


def main():
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["encode","pack","generate"])
    parser.add_argument("--variant",default="bf16")
    parser.add_argument("--checkpoint",type=Path,default=ASSETS/"inko")
    parser.add_argument("--weight-output",type=Path)
    parser.add_argument("--label")
    parser.add_argument("--tiny-checkpoint",type=Path)
    parser.add_argument("--split",choices=["development","evaluation"],default="development")
    parser.add_argument("--seeds",nargs="+",type=int,default=[0,1])
    parser.add_argument("--size",type=int,default=512)
    parser.add_argument("--steps",type=int,default=20)
    parser.add_argument("--guidance",type=float,default=4.5)
    args=parser.parse_args()
    if args.command=="pack" and args.variant=="bf16":
        parser.error("pack requires a quantized variant")
    if args.command=="pack" and args.variant=="q4-gptq":
        parser.error("use research.edge_quality.gptq recover for GPTQ weights")
    {"encode":encode,"pack":pack,"generate":generate}[args.command](args)


if __name__=="__main__":
    main()
