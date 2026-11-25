"""Laptop-only comparison of requested generators on frozen news prompts.

Run each model in a separate process. Grids use recommended model recipes,
not equal compute, resolution or identical latent noise across architectures.
Surjo sampling follows its pinned upstream model card; Inko uses the reviewed
upstream pipeline. FLUX uses Diffusers plus the requested Alimama adapter.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import sys
import time

import numpy as np
from PIL import Image
import torch

from .alternative_assets import ROOT, REPOS

OUTPUT = Path("artifacts/model-comparison")


def key(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()[:20]


def prompts():
    rows = json.loads((OUTPUT/"prompts.json").read_text())
    return list({key(row["prompt"]): row["prompt"] for row in rows}.items())


def clear():
    gc.collect()
    torch.cuda.empty_cache()


class Surjo:
    settings = {"steps": 50, "guidance": 3.0, "width": 256, "height": 256,
                "loop_iters": 2, "precision": "bf16", "quantized": False}

    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        from diffusers import AutoencoderKLFlux2
        path = ROOT/"surjo/modeling_surjo_image_preview.py"
        spec = importlib.util.spec_from_file_location("reviewed_surjo", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.SurjoImagePreviewDiT.from_pretrained(ROOT/"surjo", torch_dtype=torch.bfloat16,
                                                                local_files_only=True).to("cuda").eval()
        self.tokenizer = AutoTokenizer.from_pretrained(ROOT/"surjo-text", local_files_only=True)
        self.text = AutoModelForMaskedLM.from_pretrained(ROOT/"surjo-text", trust_remote_code=True,
                                                        local_files_only=True, dtype=torch.bfloat16).lfm2.to("cuda").eval()
        self.vae = AutoencoderKLFlux2.from_pretrained(ROOT/"surjo-vae/vae", torch_dtype=torch.bfloat16,
                                                     local_files_only=True).to("cuda").eval()

    @torch.inference_mode()
    def __call__(self, prompt):
        ids = self.tokenizer(["", prompt], padding="max_length", truncation=True,
                             max_length=128, return_tensors="pt").to("cuda")
        context = self.text(**ids).last_hidden_state.to(torch.bfloat16)
        mask = ids["attention_mask"].to(torch.bfloat16)
        gen = torch.Generator(device="cuda").manual_seed(0)
        z = torch.randn(1, 128, 16, 16, device="cuda", generator=gen, dtype=torch.bfloat16)
        for i in range(50):
            t = torch.full((1,), i/50, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                velocity = self.model(torch.cat([z,z]), torch.cat([t,t]), context,
                                      ctx_mask=mask, deepsup=False, loop_iters=2)
            empty, conditional = velocity.float().chunk(2)
            z = (z.float()+(empty+3*(conditional-empty))/50).to(torch.bfloat16)
        if not torch.isfinite(z).all():
            raise ValueError("Surjo returned non-finite latent")
        mean = self.vae.bn.running_mean.float().view(1,-1,1,1)
        std = torch.sqrt(self.vae.bn.running_var.float().view(1,-1,1,1)+float(self.vae.config.batch_norm_eps))
        latent = (z.float()*std+mean).to(torch.bfloat16)
        latent = latent.view(1,32,2,2,16,16).permute(0,1,4,2,5,3).reshape(1,32,32,32)
        decoded = self.vae.decode(latent).sample
        if not torch.isfinite(decoded).all():
            raise ValueError("Surjo decoder returned non-finite pixels")
        pixels = ((decoded[0].clamp(-1,1)+1)/2).permute(1,2,0).float().cpu().numpy()
        return Image.fromarray((pixels*255).astype(np.uint8))


class Inko:
    settings = {"steps": 20, "guidance": 4.5, "width": 896, "height": 896,
                "precision": "bf16", "quantized": False}

    def __init__(self):
        sys.path.insert(0, str((ROOT/"inko-reviewed").resolve()))
        from src.pipeline import InkoPipeline
        self.pipe = InkoPipeline.from_pretrained(ROOT/"inko", torch_dtype=torch.bfloat16,
                                                text_encoder_path=ROOT/"inko-text",
                                                vae_path=ROOT/"inko-vae", local_files_only=True).to("cuda")

    def __call__(self, prompt):
        return self.pipe(prompt, width=896, height=896, num_inference_steps=20,
                         guidance_scale=4.5, generator=torch.Generator(device="cpu").manual_seed(0)).images[0]


def flux_encode():
    from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler
    from transformers import T5EncoderModel, CLIPTextModel, CLIPTokenizer, T5TokenizerFast
    from safetensors.torch import save_file
    directory = OUTPUT/"flux-embeddings"
    directory.mkdir(parents=True, exist_ok=True)
    text = T5EncoderModel.from_pretrained(ROOT/"flux-text", local_files_only=True,
                                          device_map="cuda", dtype=torch.bfloat16)
    clip = CLIPTextModel.from_pretrained(ROOT/"flux-base/text_encoder", local_files_only=True,
                                         dtype=torch.bfloat16).to("cuda")
    pipe = FluxPipeline(scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(ROOT/"flux-base/scheduler"),
                        vae=None, transformer=None, text_encoder=clip, text_encoder_2=text,
                        tokenizer=CLIPTokenizer.from_pretrained(ROOT/"flux-base/tokenizer", local_files_only=True),
                        tokenizer_2=T5TokenizerFast.from_pretrained(ROOT/"flux-base/tokenizer_2", local_files_only=True))
    for index,(identifier,prompt) in enumerate(prompts()):
        path = directory/(identifier+".safetensors")
        metrics_path = directory/(identifier+".json")
        if path.exists() and metrics_path.exists():
            continue
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            embedding, pooled, _ = pipe.encode_prompt(prompt, prompt_2=prompt, device=torch.device("cuda"), max_sequence_length=512)
        if not torch.isfinite(embedding).all() or not torch.isfinite(pooled).all():
            raise ValueError("FLUX text encoder returned non-finite embeddings")
        torch.cuda.synchronize()
        metrics = {"seconds":time.perf_counter()-started,
                   "peak_cuda_allocated_mib":torch.cuda.max_memory_allocated()/2**20}
        save_file({"prompt_embeds":embedding.cpu().contiguous(), "pooled_prompt_embeds":pooled.cpu().contiguous()}, str(path),
                  metadata={"prompt":prompt,"text_model_revision":REPOS["flux-text"][1]})
        metrics_path.write_text(json.dumps(metrics,indent=2)+"\n")
        print(f"encoded FLUX prompt {index+1}/{len(prompts())}",flush=True)


class Flux:
    settings = {"steps": 8, "guidance": 3.5, "width": 512, "height": 512,
                "precision": "NF4 transformer + bf16 compute; NF4 T5", "quantized": True,
                "adapter_scale": 1.0, "base": "FLUX.1-dev", "decoder": "original FLUX VAE",
                "adapter_precision": "bf16",
                "note": "512px for laptop memory, below upstream 1024px recipe"}

    def __init__(self):
        from diffusers import FluxTransformer2DModel, FluxPipeline, FlowMatchEulerDiscreteScheduler
        self.transformer = FluxTransformer2DModel.from_pretrained(ROOT/"flux-quant/transformer", local_files_only=True,
                                                                 torch_dtype=torch.bfloat16, device_map="cuda")
        self.pipe = FluxPipeline(scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(ROOT/"flux-base/scheduler"),
                                 vae=None, transformer=self.transformer, text_encoder=None,
                                 text_encoder_2=None, tokenizer=None, tokenizer_2=None)
        self.pipe.load_lora_weights(ROOT/"flux-lora", weight_name="diffusion_pytorch_model.safetensors",
                                    local_files_only=True, low_cpu_mem_usage=True)
        # Quantized base layers have uint8 storage; do not cast the base weights.
        # Keep the adapter at the author's BF16 precision instead of FP32.
        for name,parameter in self.transformer.named_parameters():
            if "lora_" in name:
                parameter.data = parameter.data.to(dtype=torch.bfloat16)
        adapters = list(self.transformer.peft_config)
        if not adapters or not any("lora_B" in name and torch.count_nonzero(parameter).item()
                                   for name,parameter in self.transformer.named_parameters()):
            raise ValueError("Turbo adapter was not loaded with nonzero weights")
        self.pipe.set_adapters(adapters, adapter_weights=[1.0]*len(adapters))
        from safetensors import safe_open
        with safe_open(str(ROOT/"flux-lora/diffusion_pytorch_model.safetensors"), framework="pt", device="cpu") as checkpoint:
            expected = sum(math.prod(checkpoint.get_slice(n).get_shape()) for n in checkpoint.keys()
                           if len(checkpoint.get_slice(n).get_shape()) == 2)
        parameters = sum(p.numel() for n,p in self.transformer.named_parameters() if "lora_" in n)
        if parameters != expected:
            raise ValueError(f"Adapter parameter mismatch: loaded {parameters}, checkpoint {expected}")
        self.settings = dict(type(self).settings, active_adapters=adapters,
                             adapter_parameters=parameters)
        self.pipe.set_progress_bar_config(disable=True)

    @torch.inference_mode()
    def __call__(self, prompt):
        from safetensors.torch import load_file, save_file
        path = OUTPUT/"flux-embeddings"/(key(prompt)+".safetensors")
        embeddings = {k:v.to("cuda") for k,v in load_file(str(path)).items()}
        latent = self.pipe(**embeddings, height=512, width=512, num_inference_steps=8,
                           guidance_scale=3.5, output_type="latent",
                           generator=torch.Generator(device="cpu").manual_seed(0)).images
        directory = OUTPUT/"flux-latents"
        directory.mkdir(parents=True, exist_ok=True)
        save_file({"latent":latent.cpu().contiguous()}, str(directory/(key(prompt)+".safetensors")))
        return None


def flux_decode():
    from diffusers import AutoencoderKL, FluxPipeline
    from safetensors.torch import load_file
    vae = AutoencoderKL.from_pretrained(ROOT/"flux-base/vae", local_files_only=True,
                                        torch_dtype=torch.bfloat16).to("cuda").eval()
    for identifier,prompt in prompts():
        record_path = OUTPUT/"flux"/(identifier+".json")
        if not record_path.exists():
            continue
        path = OUTPUT/"flux"/(identifier+".png")
        if path.exists():
            continue
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        z = load_file(str(OUTPUT/"flux-latents"/(identifier+".safetensors")))["latent"].to("cuda")
        with torch.inference_mode():
            z = FluxPipeline._unpack_latents(z,512,512,8)
            decoded = vae.decode(z/vae.config.scaling_factor+vae.config.shift_factor).sample
        if not torch.isfinite(decoded).all():
            raise ValueError("FLUX decoder returned non-finite output")
        pixels = ((decoded[0].float().clamp(-1,1)+1)/2).permute(1,2,0).cpu().numpy()
        Image.fromarray(np.rint(pixels*255).astype(np.uint8)).save(path)
        record = json.loads(record_path.read_text())
        record["decode_seconds"] = time.perf_counter()-started
        record["decode_peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated()/2**20
        encoding = json.loads((OUTPUT/"flux-embeddings"/(identifier+".json")).read_text())
        record["encode_seconds"] = encoding["seconds"]
        record["encode_peak_cuda_allocated_mib"] = encoding["peak_cuda_allocated_mib"]
        record["total_inference_seconds"] = record["seconds"]+record["encode_seconds"]+record["decode_seconds"]
        record["status"] = "complete"
        record_path.write_text(json.dumps(record,indent=2)+"\n")
        print("decoded FLUX",identifier,flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model",choices=["surjo","inko","flux","flux-encode","flux-decode"])
    parser.add_argument("--limit",type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.model == "flux-encode":
        flux_encode();return
    if args.model == "flux-decode":
        flux_decode();return
    started = time.perf_counter()
    pipeline = {"surjo":Surjo,"inko":Inko,"flux":Flux}[args.model]()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter()-started
    directory = OUTPUT/args.model
    directory.mkdir(parents=True,exist_ok=True)
    selected = prompts()[:args.limit] if args.limit else prompts()
    asset_keys = {"surjo":["surjo","surjo-text","surjo-vae"],
                  "inko":["inko","inko-text","inko-vae"],
                  "flux":["flux-base","flux-lora","flux-quant","flux-text"]}[args.model]
    assets = {name:{"repo":REPOS[name][0],"revision":REPOS[name][1]} for name in asset_keys}
    if args.model == "inko":
        assets["reviewed_code_revision"] = (ROOT/"inko-reviewed/REVISION").read_text().strip()
    runner_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for index,(identifier,prompt) in enumerate(selected):
        record_path = directory/(identifier+".json")
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if (record.get("prompt") == prompt and record.get("settings") == pipeline.settings
                    and record.get("assets") == assets and record.get("runner_sha256") == runner_digest):
                if (directory/(identifier+".png")).exists() or (args.model == "flux" and (OUTPUT/"flux-latents"/(identifier+".safetensors")).exists()):
                    continue
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        image = pipeline(prompt)
        torch.cuda.synchronize()
        seconds = time.perf_counter()-started
        if image is not None:
            image.save(directory/(identifier+".png"))
        record = {"model":args.model,"prompt":prompt,"seed":0,"settings":pipeline.settings,
                  "assets":assets,"runner_sha256":runner_digest,"torch_version":torch.__version__,
                  "gpu":torch.cuda.get_device_name(),
                  "seconds":seconds,"load_seconds":load_seconds,
                  "peak_cuda_allocated_mib":torch.cuda.max_memory_allocated()/2**20,
                  "peak_process_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  "status":"complete" if image is not None else "latent-ready"}
        record["timing_scope"] = ("sampling only; encoding/decoding recorded separately"
                                   if args.model == "flux" else "encoding, sampling and decoding; excludes model loading")
        record_path.write_text(json.dumps(record,indent=2)+"\n")
        print(f'{args.model} {index+1}/{len(selected)} {identifier}: {seconds:.2f}s, CUDA {record["peak_cuda_allocated_mib"]:.0f} MiB',flush=True)


if __name__ == "__main__":
    main()
