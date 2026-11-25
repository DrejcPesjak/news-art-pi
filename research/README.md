# Supra2 experiments

The host-side experiments use PyTorch, Transformers, and Diffusers. Production
uses the staged CPU module with NCNN, NumPy, SentencePiece, and Pillow; it does
not require the host-side ML stack.
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
| Optional distilled decoder: [madebyollin/taesd](https://huggingface.co/madebyollin/taesd) | `614f76814bbe30edbe2e627ace1c2234c81a2c0e` | MIT |

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

### Automatic headline-to-scene comparison

Compare the old production prompt, a raw headline with the new style, and the
historical rule-based converter (not production v3). Reuses the shared laptop environment and local
weights; no extra model, package, or API. Example using the recovered overnight
headline list (local artifact, not committed):

```bash
HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.supra.compare_scenes \
  --headlines artifacts/overnight-2026-11-22/headlines.json \
  --sequences 79 78 28 36 34 68 48 52 40 29 65 77 --seeds 0 1
```

Open `artifacts/scene-comparison/index.html` for grids and exact converted prompts.
The runner uses 25 steps, CFG 3, and TAESD for all columns. Initial noise is fixed
across each row. CUDA seeds differ from the Pi's NumPy noise, so these are prompt
comparisons, not exact Pi reproductions. The rules omit specific details and may
misclassify ambiguous topics. This small selected sample does not establish a
general reduction in text artifacts. These local experiments do not deploy code.

### Publisher context and traditional NLP experiment

The UDPipe/sentence-selection prototype below is laptop research only.
Production v3 uses the separate brief-caption converter described later;
it does not import this UDPipe/Trafilatura stack.
Optional dependencies are in `requirements-nlp.txt`. Install with the shared
environment's existing versions constrained. No pre-existing versions changed
in the measured run.

The UDPipe English EWT model comes from the URL linked by the
[upstream course](https://ufal.mff.cuni.cz/courses/npfl125):

```bash
mkdir -p models/nlp
curl -L --fail \
  https://lindat.mff.cuni.cz/repository/server/api/core/bitstreams/8d51e3d3-1f87-47fe-8975-c197ddf7907b/content \
  -o models/nlp/english-ewt.udpipe
sha256sum models/nlp/english-ewt.udpipe
```

Measured SHA-256: `784bd0fa85e3d831fd02a55290d0acfd05c953159dc38cc33d52e1b28add9957`
(16,309,608 bytes). Assets stay outside Git. UDPipe 1 UD models have
non-commercial/share-alike terms; consult the [model license](https://ufal.mff.cuni.cz/udpipe/1/models)
before redistribution or broader use.

Download NLTK resources into the private laptop cache:

```bash
~/venvs/ml/bin/python - <<'PY'
from pathlib import Path
import nltk
directory = Path.home()/'.cache/news-art-nltk'
directory.mkdir(mode=0o700, parents=True, exist_ok=True)
for name in ['punkt_tab', 'averaged_perceptron_tagger_eng',
             'maxent_ne_chunker_tab', 'words', 'wordnet']:
    nltk.download(name, download_dir=str(directory))
PY
```

Repeat from the ignored feed snapshot `artifacts/context-nlp/google-feed.xml`:

```bash
~/venvs/ml/bin/python -m research.supra.context_probe
~/venvs/ml/bin/python -m research.supra.nlp_scenes
HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.supra.compare_context
~/venvs/ml/bin/python -m unittest discover -s research/tests -v
```

Feed indices refer to this snapshot, not a future reordered feed. `context_probe`
uses an undocumented Google redirect RPC and Trafilatura to extract publisher
metadata/opening text. It caches pages/results and records HTTP failures; no
paywall bypass. Related RSS headlines are not treated as article abstracts.

`nlp_scenes` uses NLTK sentence segmentation, UDPipe syntax and WordNet lexical
classes to rank source propositions. NER annotations are diagnostics; they do
not supply knowledge of an entity's appearance. An early SVO-only rewrite lost
coordinated actions and misread irregular verbs. Final image prompts preserve
the selected source sentence; unsuitable parses fall back to the headline.
Headline-only NLP therefore often leaves its input unchanged.

Open `artifacts/context-nlp/images/index.html`: six grids compare headline/style,
topic templates, NLP headline selection, and NLP article-context selection over
12 articles and two seeds (96 images). Settings are 25 steps, CFG 3 and TAESD,
with fixed noise across each row. The gallery links prompts, source evidence,
NER annotations, failures and measurements. This combined NLP prototype peaked
around 507 MiB RSS on the laptop. ARMv7 installation/memory remain untested; do
not load it alongside Pi image inference without measurement.

### Alternative image generators: frozen news-prompt comparison

The laptop comparison uses the same 48 prompt slots from the context experiment,
seed 0. There are 32 distinct strings; duplicate prompt/seed outputs are reused
within a model. The existing Supra2/TAESD images provide a fourth row. The frozen
input is `artifacts/model-comparison/prompts.json`, including the baseline paths.
It deliberately preserves the previous context failures and NLP fallbacks.

Requested generators and completed test recipes:

| Generator | Image size | Steps / guidance | Components |
| --- | --- | --- | --- |
| Supra2 baseline | 256² | 25 / 3 | Existing previous experiment, TAESD |
| [FLUX Turbo Alpha](https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha) | 512² | 8 / 3.5 | Requested LoRA at scale 1 on NF4 FLUX.1-dev, NF4 T5, original FLUX VAE |
| [Inko](https://huggingface.co/kaupane/Inko) | 896² | 20 / 4.5 | BF16 upstream pipeline, Qwen3 early-exit encoder, original Qwen VAE |
| [Surjo Preview](https://huggingface.co/SurjoLabs/Surjo-Image-Preview) | 256² | 50 / 3 | BF16, loop iterations 2, LFM2.5 bidirectional encoder, original small FLUX.2 decoder |

FLUX Turbo Alpha is an adapter, not a standalone small generator. Its base
requires Hugging Face access (already available on this laptop). The comparison
uses the documented [Diffusers NF4 base](https://huggingface.co/diffusers/FLUX.1-dev-bnb-4bit)
and a pinned [NF4 T5 encoder](https://huggingface.co/diffusers/t5-nf4) for laptop
memory. The 512px FLUX setting is below the upstream 1024px example. This is
neither equal compute nor equal resolution; identical seeds do not give identical
noise across architectures. Laptop measurements do not establish Pi viability.

Revisions are pinned in `supra/alternative_assets.py`. Downloads validate the
final LFS SHA-256 and record manifests. Large files use resumable HTTP ranges;
completed chunks survive an interrupted transfer. Inko's inference-only source
subset is downloaded at a pinned GitHub revision, rather than installing its
training dependencies. Surjo's custom model/encoder files and Inko's inference
code were inspected before execution. Do not run an unreviewed revision.

Using the shared environment and optional `requirements-alternatives.txt`, from
the repo root (the frozen input must already exist):

```bash
~/venvs/ml/bin/python -m research.supra.alternative_assets \
  surjo surjo-text surjo-vae inko inko-text inko-vae \
  flux-lora flux-base flux-quant flux-text
HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.supra.alternative_compare surjo
HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.supra.alternative_compare inko
HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.supra.alternative_compare flux-encode
HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.supra.alternative_compare flux
HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.supra.alternative_compare flux-decode
~/venvs/ml/bin/python -m research.supra.alternative_gallery
```

The FLUX encoder, sampler and decoder run in separate processes to avoid
simultaneous GPU residency. The runner checks that the requested Turbo adapter
is active with nonzero weights. Timings separate these stages; native models'
timings include encoding/sampling/decoding but exclude model loading. JSON
sidecars record prompts, seed, settings, revisions, source digest, timings and
memory. Resume valid cached stages instead of restarting whole experiments.

Open `artifacts/model-comparison/index.html`: twelve 4×4 article grids have four
model rows and four prompt-format columns. Expand the per-model contact sheets
for additional 4×4 views of distinct prompts. Originals retain native resolution;
only plot tiles resize to 256px. `--partial` renders explicit pending cells;
without it the gallery refuses to claim completion if any model output is absent.
All weights and generated artifacts stay outside Git. No Pi deployment occurs.

To compare Inko's photo-style baseline with oil-painting prompts at 896px and
512px (four frozen headlines, seed 0, 20 steps, CFG 4.5):

```bash
HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.supra.inko_resolution
```

Open `artifacts/inko-resolution/index.html` for the three-row comparison, exact
prompts, native images, component weight counts and GPU allocation measurements.
The script keeps all components resident and measures allocated, not reserved,
CUDA memory. It does not quantize weights or change the production configuration.

### Brief oil-painting captions

The `brief_scenes` experiment actually prunes source noun phrases,
using NLTK POS/NER and WordNet. It retains a role/object plus at most one setting
or second subject, starts with `a`, and appends `. oil painting`. Identifying
proper names are removed in the generic variant; nationality adjectives survive.
Its name-stripped variant was subsequently deployed on the Pi as revision
`bebff43`; the experiment now imports the shared converter from
`news_art/brief_scenes.py`. It does not infer roles such as singer from a name.
Production runs it in a separate child process before NCNN; see the main README
for caption data bootstrap and `RESULTS.md` for the first completed Pi run.

Using the existing frozen publisher-context report and private NLTK resources:

```bash
~/venvs/ml/bin/python -m research.supra.brief_scenes
HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.supra.compare_brief
~/venvs/ml/bin/python -m unittest research.tests.test_brief_scenes -v
```

Open `artifacts/supra-brief/index.html`: six 4×4 grids, 12 articles, four variants,
two seeds, 96 rendered images. Columns compare full headlines, shortened captions
with names retained, shortened captions with names stripped, and explicitly
handwritten controls. Every column uses oil painting, 256px, 25 steps, CFG 3 and
TAESD. Two handwritten controls preserve Drew's drug-lord and ship examples
verbatim. The others are illustrative references, not automatic NLP outputs.
Exact source text, prompts, NER diagnostics and image settings are saved beside
the gallery. No downloads, environment changes or deployment are needed.

### Original precision and step comparisons

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
# Compact storage version used for the CPU runtime:
~/venvs/ml/bin/python -m research.supra.export dit --fp16-storage
```

`--fp16-storage` is an experimental disk-storage compression option. It does not
mean the Pi can compute in fp16 or that runtime RAM halves. FP32/FP16 export
validation outcomes are recorded alongside the artifacts. Do not deploy an
artifact whose validation says `passed: false`.

The Pi trials load the components sequentially, validate real outputs, and report
on-device timings. A working DiT export alone does not establish complete image
generation. The practical path uses the distilled decoder described below.

## Staged CPU pipeline

The text encoder is exported without its large embedding lookup table. The table
is memory-mapped separately; only prompt rows are gathered into RAM. SentencePiece
tokenizes locally. Each component then runs in a fresh process so its weight
memory is released before loading the next. NCNN computes in fp32; compact fp16
weight files do not imply fp16 arithmetic on ARMv7.

```bash
# Export the remaining runtime components after the verified DiT exports above:
HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.supra.component_export text
HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.supra.component_export tiny
# Validate real conditional contexts (requires both fp32 and compact DiT exports):
HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.supra.validate_ncnn
# Bundle only runtime weights/graphs/tokenizer, with integrity checks:
~/venvs/ml/bin/python -m research.supra.bundle create artifacts/supra/runtime-bundle --decoder tiny
~/venvs/ml/bin/python -m research.supra.bundle verify artifacts/supra/runtime-bundle
~/venvs/ml/bin/python -m research.supra.staged generate \
  --model-dir artifacts/supra/runtime-bundle --output artifacts/supra/cpu-image \
  --steps 25 --threads 4 --decoder tiny --packed-decoder
```

`--decoder full` uses the original whole-frame decoder graph. `--decoder tiled`
uses four overlapping latent tiles with blended borders; it is a memory/quality
tradeoff, not numerically equivalent to whole-frame decoding. Full and tiled
graphs have byte-identical weights, so the runtime bundle stores those once.
To create these reference exports, run `component_export vae` and
`component_export vae --tile-size 20`. Choose `bundle create ... --decoder full`,
`--decoder tiled`, or `--decoder all` to package them. The default minimal bundle
contains only the tiny decoder. Bundle creation requires a fresh output directory
and checks that the DiT passed all five real-context validations.

`--decoder tiny` substitutes the compatible TAESD distilled decoder for the
original SD VAE. Supra2's DiT and Flan-T5 text encoder remain unchanged: this is
one image generator with a different decoding component, not a second image
generator. The decoder drops from about 49.5M to 1.22M parameters. This is a lossy
component replacement, not an exact quantization or distillation trained here.
Compare [decoder samples](results/decoder-comparison.jpg) before choosing it.
The tiny decoder consumes the already-scaled latent directly; unlike the original
VAE, its input must **not** be divided by 0.18215.

```bash
HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.supra.component_export tiny
HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.supra.compare_decoders
```

`--packed-decoder` enables channel packing for decoding while keeping direct
convolution, Winograd disabled, and fp32 arithmetic. Its memory/runtime behavior
must be measured on the target, not inferred from x86 results.

The CPU pipeline's default random generator is NumPy PCG64. The same numeric seed
does not imply the same initial noise as CUDA PyTorch. Pass `--initial-latent` with
a saved `(1,4,32,32)` float32 `.npy` file for controlled cross-runtime comparisons.
No prompt/result API or remote generator is used.

On this Pi the experiment checkout and weights are under
`~/.local/share/news-art-pi/{research-checkout,supra-models}`. Python imports NCNN
only when `PYTHONPATH=$HOME/.local/share/news-art-pi/ncnn-runtime` is explicitly
set. Production instead uses `/opt/news-art-pi/runtime` and the verified model link
`/var/lib/news-art-pi/models/supra`, installed by `deployment/install-models.sh`.
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
  -p Nice=10 -p RuntimeMaxSec=1200 \
  -p WorkingDirectory="$PWD" \
  /usr/bin/env PYTHONPATH="$HOME/.local/share/news-art-pi/ncnn-runtime" \
  /usr/bin/python3 -m research.supra.staged generate \
  --model-dir "$HOME/.local/share/news-art-pi/supra-models" \
  --output "$HOME/.local/share/news-art-pi/supra-results/image" \
  --steps 25 --threads 4 --decoder tiny --packed-decoder --memory-mib 480
```

Reports separate model loading, computation, and per-stage peak RSS. The process
data limit is inherited by child processes; the heavyweight stages do not overlap.
A failed experiment does not modify the production display's state or services.
