from contextlib import nullcontext
import gc
import logging
import resource
import time

import numpy as np
from PIL import Image
import torch
from transformers import AutoTokenizer, T5EncoderModel
from diffusers import AutoencoderKL

from .assets import MODEL_DIR
from .model import SupraDiT, MAX_CTX_LEN, VAE_SCALE

log = logging.getLogger(__name__)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Pipeline:
    def __init__(self, precision="baseline", device="cuda", threads=4):
        started = time.perf_counter()
        self.device = torch.device(device)
        self.precision = precision
        torch.set_num_threads(threads)
        self.dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        if precision == "bf16" and self.device.type != "cuda":
            raise ValueError("bf16 mode is a CUDA experiment, not an ARM assumption")
        # The checkpoint contains tensors/config only. No unrestricted pickle load.
        checkpoint = torch.load(MODEL_DIR / "supra/model_final_ema.pt", map_location="cpu",
                                weights_only=True, mmap=True)
        self.config = checkpoint.get("config", {})
        weights = checkpoint.get("ema", checkpoint.get("model", checkpoint))
        with torch.device("meta"):
            self.model = SupraDiT()
        self.model.load_state_dict(weights, strict=True, assign=True)
        self.model = self.model.to(device=self.device, dtype=self.dtype).eval().requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "text", local_files_only=True)
        self.text = T5EncoderModel.from_pretrained(MODEL_DIR / "text", local_files_only=True,
                                                  dtype=self.dtype).to(self.device).eval().requires_grad_(False)
        self.vae = AutoencoderKL.from_pretrained(MODEL_DIR / "vae", local_files_only=True,
                                                 torch_dtype=self.dtype).to(self.device).eval().requires_grad_(False)
        self.context_length = int(self.config.get("ctx_len", MAX_CTX_LEN))
        # Only decoding is used in text-to-image inference. Drop the unused encoder
        # in the optimized mode; baseline retains the upstream full VAE.
        if precision != "baseline":
            self.vae.encoder = None
            self.vae.quant_conv = None
        del weights, checkpoint
        gc.collect()
        synchronize(self.device)
        self.load_seconds = time.perf_counter() - started
        self.components = {name: {
            "parameters": sum(p.numel() for p in module.parameters()),
            "weight_mib": sum(p.numel()*p.element_size() for p in module.parameters()) / 2**20,
        } for name, module in (("dit", self.model), ("text_encoder", self.text), ("vae", self.vae))}
        log.info("Loaded %s in %.2fs: %s", precision, self.load_seconds, self.components)

    def autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else nullcontext()

    @torch.inference_mode()
    def encode(self, prompt, guidance=3.0):
        tokens = self.tokenizer(prompt, padding="max_length", truncation=True,
                                max_length=self.context_length, return_tensors="pt").to(self.device)
        with self.autocast():
            context = self.text(**tokens).last_hidden_state.float()
        mask = tokens["attention_mask"].float()
        if guidance > 1:
            if "uncond_text" in self.config:
                unconditional = self.config["uncond_text"].to(self.device).float().unsqueeze(0)
                unconditional_mask = self.config["uncond_mask"].to(self.device).float().unsqueeze(0)
            else:
                empty = self.tokenizer("", padding="max_length", truncation=True,
                                       max_length=self.context_length, return_tensors="pt").to(self.device)
                with self.autocast():
                    unconditional = self.text(**empty).last_hidden_state.float()
                unconditional_mask = empty["attention_mask"].float()
            context = torch.cat([context, unconditional])
            mask = torch.cat([mask, unconditional_mask])
        return context, mask

    @torch.inference_mode()
    def sample(self, context, mask, steps, seed, guidance):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        z = torch.randn(1, 4, 32, 32, device=self.device, generator=generator)
        dt = 1.0 / steps
        # Baseline follows upstream's per-step autocast. Optimized mode has native
        # bf16 weights and keeps a single autocast context around the sampler.
        outer = self.autocast() if self.precision != "baseline" else nullcontext()
        with outer:
            for index in range(steps):
                t = torch.full((1,), index*dt, device=self.device)
                with self.autocast() if self.precision == "baseline" else nullcontext():
                    if guidance > 1:
                        both = self.model(torch.cat([z, z]), torch.cat([t, t]), context, mask)
                        conditional, unconditional = both.float().chunk(2)
                        velocity = unconditional + guidance*(conditional-unconditional)
                    else:
                        velocity = self.model(z, t, context, mask).float()
                z = z + dt*velocity
        return z

    @torch.inference_mode()
    def generate(self, prompt, steps=50, seed=0, guidance=3.0, output=None):
        if steps < 1:
            raise ValueError("steps must be positive")
        synchronize(self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        context, mask = self.encode(prompt, guidance)
        synchronize(self.device)
        encoded = time.perf_counter()
        latent = self.sample(context, mask, steps, seed, guidance)
        synchronize(self.device)
        sampled = time.perf_counter()
        with self.autocast():
            decoded = self.vae.decode(latent / VAE_SCALE).sample
        if not torch.isfinite(decoded).all():
            raise ValueError("Decoder returned non-finite pixels")
        pixels = ((decoded[0].float().clamp(-1, 1)+1)/2).permute(1, 2, 0)
        pixels = (pixels.cpu().numpy()*255).round().astype(np.uint8)
        synchronize(self.device)
        ended = time.perf_counter()
        image = Image.fromarray(pixels)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            image.save(output)
        metrics = {
            "precision": self.precision, "steps": steps, "seed": seed, "guidance": guidance,
            "text_seconds": encoded-start, "sample_seconds": sampled-encoded,
            "decode_seconds": ended-sampled, "total_seconds": ended-start,
            "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(self.device)/2**20 if self.device.type == "cuda" else None,
            "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(self.device)/2**20 if self.device.type == "cuda" else None,
            "pixel_mean": float(pixels.mean()), "pixel_std": float(pixels.std()),
        }
        return image, metrics
