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

See [RESULTS.md](RESULTS.md) for measured timings, memory, sample comparisons,
and the distinction between conversion validation and image quality.

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

## Staged CPU pipeline

The text encoder is exported without its large embedding lookup table. The table
is memory-mapped separately; only prompt rows are gathered into RAM. SentencePiece
tokenizes locally. Each component then runs in a fresh process so its weight
memory is released before loading the next. NCNN computes in fp32; compact fp16
weight files do not imply fp16 arithmetic on ARMv7.

```bash
# Export the two remaining components after the verified DiT export above:
HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.supra.component_export text
HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.supra.component_export vae
HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.supra.component_export vae --tile-size 20
# Validate real conditional contexts (requires both fp32 and compact DiT exports):
HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.supra.validate_ncnn
# Bundle only runtime weights/graphs/tokenizer, with integrity checks:
~/venvs/ml/bin/python -m research.supra.bundle create artifacts/supra/runtime-bundle
~/venvs/ml/bin/python -m research.supra.bundle verify artifacts/supra/runtime-bundle
~/venvs/ml/bin/python -m research.supra.staged generate \
  --model-dir artifacts/supra/runtime-bundle --output artifacts/supra/cpu-image \
  --steps 25 --threads 4 --decoder tiled
```

`--decoder full` uses the original whole-frame decoder graph. `--decoder tiled`
uses four overlapping latent tiles with blended borders; it is a memory/quality
tradeoff, not numerically equivalent to whole-frame decoding. Full and tiled
graphs have byte-identical weights, so the runtime bundle stores those once.

The CPU pipeline's default random generator is NumPy PCG64. The same numeric seed
does not imply the same initial noise as CUDA PyTorch. Pass `--initial-latent` with
a saved `(1,4,32,32)` float32 `.npy` file for controlled cross-runtime comparisons.
No prompt/result API or remote generator is used.

On this Pi the experiment checkout and weights are under
`~/.local/share/news-art-pi/{research-checkout,supra-models}`. Python imports NCNN
only when `PYTHONPATH=$HOME/.local/share/news-art-pi/ncnn-runtime` is explicitly
set. The production slideshow has no such override and keeps using demo images.
The experiment also requires system packages `python3-numpy`, `python3-pil`, and
`python3-sentencepiece`. The target does not need uv, CUDA, PyTorch, Transformers,
Diffusers, or a Hugging Face/GitHub login.

The Pi kernel does not currently expose the memory cgroup controller. Systemd
accepts `MemoryMax`/`MemorySwapMax` but cannot enforce them on this kernel. Use
`--memory-mib 480` to bound each process's Linux data/anonymous-allocation segment
with `RLIMIT_DATA`. This is not an exact RSS limit and does not forbid OS swapping;
shared libraries and file-backed mappings are accounted differently. The time
limit and low scheduling priority below are independently enforced.

For example, from the checkout directory in an SSH session:

```bash
systemd-run --user --wait --pipe --collect --unit=news-art-model-trial \
  -p MemoryMax=520M -p MemorySwapMax=0 -p Nice=10 -p RuntimeMaxSec=1200 \
  -p WorkingDirectory="$PWD" \
  /usr/bin/env PYTHONPATH="$HOME/.local/share/news-art-pi/ncnn-runtime" \
  /usr/bin/python3 -m research.supra.staged generate \
  --model-dir "$HOME/.local/share/news-art-pi/supra-models" \
  --output "$HOME/.local/share/news-art-pi/supra-results/image" \
  --steps 10 --threads 4 --decoder tiled --memory-mib 480
```

Reports separate model loading, computation, and per-stage peak RSS. The process
data limit is inherited by child processes; the heavyweight stages do not overlap.
A failed experiment does not modify the production display's state or services.
