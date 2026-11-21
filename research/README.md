# Supra2 experiments

These experiments run separately from the Pi's production slideshow. The demo
backend and deployment branch do not depend on PyTorch or these research packages.
Weights and full-resolution experiment outputs are excluded from Git.

## Environment

This laptop reuses **`~/venvs/ml`**, a shared uv environment (Python 3.12.12,
PyTorch 2.12.1+cu130, torchvision 0.27.1+cu130). `.venv` is an ignored symlink to it.
Do not use `uv sync` here: that could remove packages other projects require.
The additional packages were installed with `uv pip install` while constraining
every existing version, so the existing stack was not upgraded or downgraded.
`requirements.txt` in this directory records the experiment dependencies.

```bash
~/venvs/ml/bin/python -m research.supra.assets
HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.supra.run generate \
  --prompt "a sea jellyfish floating in the pitch-black ocean depths" \
  --steps 50 --seed 0 --guidance 3 --output artifacts/supra/baseline
HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.supra.run benchmark
~/venvs/ml/bin/python -m unittest discover -s research/tests -v
```

Downloads are pinned by commit in `supra/assets.py`. The HTTP transfer path is
used because the Xet transfer stalled on this network. No paid APIs, remote image
generation, `trust_remote_code` or unrestricted pickle loading are used.

## Model and licenses

| Component | Pinned revision | Declared license |
| --- | --- | --- |
| [SupraLabs/Supra2-IMG](https://huggingface.co/SupraLabs/Supra2-IMG) | `b6186a6e8fa9a846c44b804f42639ce4228e3b94` | Apache-2.0 |
| [google/flan-t5-base](https://huggingface.co/google/flan-t5-base) | `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2` | Apache-2.0 |
| [stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse) | `31f26fdeee1355a5c34592e401dd41e45d25a493` | MIT |

Licenses above are the upstream model-card declarations. The Supra repository
does not include a separate LICENSE file at the pinned revision. The architecture
in `supra/model.py` is adapted from its reviewed `inference.py`, with attribution;
its network helpers and unrestricted checkpoint loading are omitted. The
checkpoint is loaded with `weights_only=True`. Checkpoint weights are not
redistributed by this repository.

Supra is one image-generation model, but its full inference pipeline includes a
text encoder and VAE. The DiT's parameter count alone is not a RAM estimate for
the complete application. The base output is 256 × 256. The upstream recommended
baseline is seed 0, guidance 3, and 50 Euler sampling steps.

## Measurement method

`supra/prompts.json` defines four fixed prompts. The benchmark compares the
50-step baseline with native bf16 weights and 50/25/10/5 steps. Seed and guidance
are fixed. Each precision gets a short CUDA warmup; model loading is timed
separately. CUDA synchronization brackets each stage.

Baseline follows upstream's fp32 resident weights with per-step bf16 CUDA
autocast. The optimized mode stores weights in bf16, keeps autocast around the
sampler, and removes the unused VAE encoder. No pruning, distillation or training
is claimed. These remain experiments requiring measured quality and speed.

Metrics separate text encoding, denoising and decoding. CUDA allocated memory
and reserved memory are reported separately. Process RSS is the process-lifetime
high-water mark; it must not be read as per-image or per-variant RAM. GPU figures
do not predict ARM CPU latency.

The comparison sheet supports manual inspection of subject recognition,
composition and obvious artifacts. Pixel RMSE only measures change from a
reference image; it is not a semantic or aesthetic quality score. This small
sample cannot establish broad image quality, FID, or prompt-adherence performance.

## ARM runtime investigation

The Pi runs ARMv7, Python 3.13, and has 920 MiB RAM shared with the desktop/display.
The current ONNX Runtime wheel set has no ARMv7 build. NCNN has a compatible
`cp313-manylinux_2_31_armv7l` wheel; its import has been tested on this Pi. It is
installed in `~/.local/share/news-art-pi/ncnn-runtime`, outside the system Python
and production application's import path.

NCNN conversion uses [PNNX](https://github.com/Tencent/ncnn/tree/master/tools/pnnx).
Original six-dimensional patch rearrangements exceed NCNN's tensor rank. The
export wrapper replaces patch projection with stride-two convolution and
unpatchifying with pixel shuffle. Explicit attention avoids the converter's
problematic masked SDPA path. Every export must pass a numerical check; the mere
presence of `.param` and `.bin` files does not establish correctness.

```bash
~/venvs/ml/bin/python -m research.supra.export dit \
  --output artifacts/supra/ncnn-fp32
```

`--fp16-storage` is an experimental disk-storage compression option. It does not
mean the Pi can compute in fp16 or that runtime RAM halves. FP32/FP16 export
validation outcomes are recorded alongside the artifacts. Do not deploy an
artifact whose validation says `passed: false`.

Pi feasibility requires all three components, sequential loading to control peak
RAM, numerical/visual validation, and actual on-device latency measurements. A
working DiT export alone would not make the Pi self-sufficient for generation.
