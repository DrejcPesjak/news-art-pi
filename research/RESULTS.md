# Supra2 experiment results

## Brief scene captions, 22 November 2026

Deployment follow-up: at Drew's request, v3 name-stripped captions and oil painting
were deployed on revision `bebff43`. The production worker shares the same
converter (`news_art/brief_scenes.py`); all 12 frozen generic prompts matched after
moving code and reusing the NER classifier within each child process. NLTK/lxml
and private local NLTK data were installed on the Pi. Heavy NLP preparation runs
in a child that exits before NCNN starts. Publisher context uses metadata and
article/main paragraphs instead of laptop trafilatura extraction.

All 15 candidate-release tests and runtime checks passed on the Pi. First live
poster 135 completed in 629.96 s including publisher fetch/caption preparation,
with context available. Prompt: `a Sunrise and a box office. oil painting`.
HDMI inspection confirmed publication, but this first result still has prominent
pseudo-lettering and the caption retains a word from a movie title. V3 is not a
complete name-removal or fake-lettering fix. Evidence and exact metadata:
`artifacts/deployment/captions-v3/first-live-{state.json,image.png,hdmi.png}`.
Runtime smoke check peak child RSS was 183.4 MiB / 59.67 s while the old Supra
job was sampling; this is not isolated normal caption latency.

Local Supra2 + TAESD comparison: 12 frozen articles × four prompt variants ×
two seeds (0/1), 96 rendered images, 256px, 25 steps, CFG 3. All variants use
the same `oil painting` suffix, so shortening/name removal is compared without
changing the style. Some named/generic captions coincide and produce duplicate
images; these are 96 test slots, not 96 distinct prompts. Same initial noise
within each seed; laptop CUDA noise is not a Pi reproducibility guarantee.

Manual inspection of all six grids shows substantially less prominent fake
lettering for the brief generic captions, especially crime, soccer, climate
and phone rows. Anatomy remains poor and the ship's drone impact is not reliably
depicted. There is no blinded numerical quality rating or broad improvement claim.
The supplied FBI-list reference still produces an abstract poster for seed 0,
but a portrait for seed 1; short prompts are not a seed-independent fix.

The automatic extractive converter preserves useful roles in `a Mexican drug
lord`, `a British swimmer at sea`, and `a couple getting married`. It also loses
important distinctions: soccer becomes `a trophy and a crowd`; a space capsule
becomes `a cargo capsule at sea`; `a fossil fuel` produces
an unhelpful object. Handwritten controls illustrate the remaining gap and are
not silently substituted into automatic results. NER removes names but cannot
supply a missing common role or resolve model ambiguity by itself.

Three semantic regression tests pass: retaining a swimmer while stripping names,
using article context to recover a couple, and avoiding invented marriage for
an unrelated couple. These grids were generated on the laptop before the
deployment follow-up above. Evidence:
`artifacts/supra-brief/index.html`, `prompts.json`, `results.json`, six grids.

The original baseline below was measured on 21 November 2026. This is a small engineering evaluation, not a
published model-quality benchmark. Raw results and comparison images are in
[`results/`](results/).

## Laptop baseline and optimizations

Hardware: RTX 3070 Laptop GPU, 8 GiB VRAM. Reused `~/venvs/ml`; all pre-existing
package versions were preserved. All three downloaded model files were verified
against their pinned Hugging Face LFS SHA-256 digests.

First reference image: seed 0, CFG 3, 50 steps, 256 × 256. Generation after model
loading took 1.359 s; loading itself took 0.818 s. This first-run timing includes
CUDA initialization/warmup effects inside the generation call, unlike the warm
comparison below. Python interpreter/import startup is not included.

Four fixed prompts and one seed per prompt, with warmup per precision. Each table
row is the median end-to-end inference time over those four images; CUDA memory
is the maximum allocated peak observed in that row. Model loading and image file
writing are excluded from these warm timings.

| Resident precision | Steps | Warm median | Peak allocated CUDA memory |
| --- | --- | --- | --- |
| baseline | 50 | 0.557 s | 1365 MiB |
| bf16 | 50 | 0.531 s | 740 MiB |
| bf16 | 25 | 0.289 s | 740 MiB |
| bf16 | 10 | 0.146 s | 740 MiB |
| bf16 | 5 | 0.101 s | 740 MiB |

Native bf16 weights, removal of the unused VAE encoder, and fewer steps gave
about 1.9× faster generation at 25 steps and about 46% less peak allocated VRAM
than the 50-step baseline. This is a measured bundle of changes, not a claim that
casting weights alone doubles speed.

[Step/precision comparison](results/comparison.jpg) ·
[Guidance comparison](results/guidance-comparison.jpg)

Visual inspection: 25 steps preserved the major composition and much of the
50-step detail in these examples. Ten steps lost detail; five was visibly softer.
CFG 1 avoids the unconditional pass, but produced less coherent subjects in this
sample, so CFG 3 remains the reference. No general quality guarantee follows
from four prompts and a single seed.

The model also has baseline prompt-adherence problems: the solar-farm prompt
became a building, and the basketball prompt produced incorrect anatomy and
sports equipment. Those errors already occur at 50 steps. Pixel RMSE is not a
semantic-quality metric and is not used to claim those images are correct.

## NCNN CPU path

All three components were exported without requiring PyTorch on the target.
Actual prompt contexts and the stored unconditional context passed DiT checks.
Relative RMS error from compressed FP16 weight storage was approximately 0.05%
for the DiT, 0.095% for the encoder, and 0.045% for the full decoder in the tested
fixtures. This tests numerical conversion, not full-pipeline semantic quality.

The unit-variance random context used in an early stress fixture was far outside
the observed encoder output scale and amplified numerical differences. Final
validation uses real encoder outputs; the conditional and unconditional results
are recorded explicitly in `results/ncnn-validation.json`.

On x86, the default full decoder peaked at 1.64 GiB RSS. Disabling convolution
transforms reduced it to about 667 MiB, still too much for the Pi alongside its
desktop. Four overlapping 20 × 20 latent tiles reduced the laptop's staged decode
peak to about 486 MiB. Tiling is approximate: normalization and attention see
local context. A jellyfish sample remained recognizable without obvious seams;
that does not establish correctness for every image.

One complete staged/tiled CPU run on the laptop took 19.4 s for 25 guided steps.
Each component ran in a fresh process, keeping the peak under about 500 MiB per
stage. These single CPU trials are diagnostics, not a robust CPU benchmark.

## Distilled decoder optimization

The original SD decoder proved too expensive on the Pi. An unpacked end-to-end
trial was stopped after 12 min 54 s with decoding still running; a packed original
decoder trial later failed with NCNN error `-100` under the allocation limit.
Neither is reported as a completed image latency.

The practical alternative is [TAESD](https://huggingface.co/madebyollin/taesd), a
pretrained MIT-licensed distilled decoder compatible with the same SD latent
space. It replaces the original decoding component; Supra2's DiT and Flan-T5
stay unchanged. This is one image generator with a different decoder, not another
text-to-image model. No distillation training was performed in this project.

The decoder shrinks from about 49.5M to 1.22M parameters; its NCNN weight file is
2.34 MiB instead of 94.47 MiB. PNNX estimates 35.4G versus 624.1G FLOPs per frame.
Those operation counts explain a direction, not a latency prediction.

[Same-latent decoder comparison](results/decoder-comparison.jpg): the tiny decoder
preserved composition and subjects in the four tested prompts, with changed or
softer fine details. It does not fix the generator's prompt-adherence errors.
Its exported NCNN output differed by about 0.08% relative RMS from its own PyTorch
implementation on the test fixture. That is conversion error, not error relative
to the original SD VAE.

On the Pi, decoding the retained 10-step jellyfish latent took **21.55 s including
loading**, with **118 MiB peak RSS**. Comparing that image to a laptop PyTorch
TAESD decode of the exact same Pi-generated latent gave pixel RMSE **0.047 on the
0–255 scale**. This validates decoding, not the whole pipeline's prompt quality.

The distilled weights are pinned at `614f76814bbe30edbe2e627ace1c2234c81a2c0e`
and SHA-256 verified. TAESD takes already-scaled SD latents directly; dividing by
0.18215, as required for the original VAE, would be incorrect for this decoder.

## Raspberry Pi

Pi 2 B Rev 1.1, ARMv7, Raspbian 13, Python 3.13. NCNN 1.0.20261020 has a matching
wheel and works on the device. SentencePiece and NumPy are local dependencies.
Tests keep the existing news display running. This kernel lacks the memory cgroup
controller, so systemd memory/swap caps cannot be relied on here. Subsequent
pipeline trials use an explicit per-process `RLIMIT_DATA` allocation limit,
alongside systemd's time limit and low scheduling priority. This is not an exact
RSS cap and does not disable OS swapping. No laptop service is used by this
runtime; weights reside on the Pi.

### Completed end-to-end trial

**The Pi generated a 256 × 256 image in 570.41 seconds (9 min 30.4 s)** using
25 Euler steps, CFG 3, four CPU threads, NCNN fp32 arithmetic, compact fp16
weights, and the packed TAESD decoder. This was one uninterrupted automatic run,
including component loads, fresh-process startup, and saving the PNG. All image
computation and model assets were on the Pi. The laptop only launched the test
and collected logs/results.

| Stage | Load | Computation and output | Stage total | Peak process RSS |
| --- | --- | --- | --- | --- |
| Flan-T5 encoding | 9.81 s | 5.97 s | 15.77 s | 364.75 MiB |
| Supra2 sampling, 25 guided steps | 11.06 s | 516.18 s | 527.23 s | 447.61 MiB |
| Distilled decoding | 0.45 s | 21.31 s | 21.76 s | 117.99 MiB |

Process startup/supervision adds about 5.65 s beyond the stage totals. Peak RSS is
for each heavyweight process, not the whole system or a sum of the three peaks;
stages do not overlap. The desktop and demo display stayed running. The model
process allocation limit was 480 MiB. `get_throttled` remained `0x0`; the
post-run temperature was 47.6°C.

[Pi-generated image](results/pi-image-25.png) ·
[Raw timing report](results/pi-25-step.json) ·
[Runtime/image comparison](results/runtime-comparison.jpg)

The prompt was the upstream-style jellyfish test, with an explicit saved CUDA
seed-0 initial latent for cross-runtime comparisons. The runtime itself used no
GPU. Tested experiment code: `d4e692b2fa9dd2204c57fab15e7e01a9d2331d41`.

This meets the ten-minute target in one trial, with about 30 seconds of margin.
It is not a latency guarantee for every background workload. Fewer steps can
provide more scheduling margin, with a quality tradeoff. GPU timings above must
not be extrapolated to ARM. Four laptop prompts and one uninterrupted 25-step Pi trial are a
small engineering sample, not a broad quality or reliability evaluation.

### Deployment status at the end of the research trial

At the end of the 21 November research trial, `main` contained these experiments
and `deploy` still ran the demo slideshow. The 22 November integration connects
this runtime to live headlines; see the current [application README](../README.md).

The integration uses service-readable models/runtime paths, bounds inference
with an allocation limit and a process-group timeout, and measures refreshes from
job start time. It retains the previous poster until successful publication and
logs locally through the service journal.

## Local automatic scene-prompt evaluation, 22 November

The overnight feed exposed a frequent failure: large pseudo-lettering on nearly
empty backgrounds. Added offline topic rules in `news_art/scenes.py` to turn
recognized headlines into concrete generic scene captions, with an unchanged
headline fallback for unknown topics. The rules were developed using this
overnight headline set, so this is a development evaluation, not a held-out test.

Generated 72 laptop CUDA images: 12 selected overnight headlines, seeds 0 and 1,
and three prompt variants. All use 25 steps, CFG 3 and TAESD. Columns compare
the old production wrapper/style, the original headline plus a photographic
style, and the automatic scene plus the same photographic style. The second
and third columns isolate the scene conversion. The selected poster sequences
are 79, 78, 28, 36, 34, 68, 48, 52, 40, 29, 65 and 77.

Visual inspection found substantially less prominent pseudo-lettering with the
converted scenes, especially in the medical, finance, politics, technology and
swimming examples. Simply changing the wrapper/style often still produced text.
The scene examples retain visible model errors: distorted people, weak spacecraft
details, and flooding that can resemble ordinary coastal water. Generic scenes
omit names, exact locations and event details, and the fixed rules can misclassify
ambiguous headlines. No OCR score or general quality/success rate is claimed.

Local grids, exact prompts and all-headline rule assignments are in ignored
`artifacts/scene-comparison/`; reproduce with the command in `research/README.md`.
All 52 supplied overnight headlines matched a rule, which measures coverage of
this input set rather than classification correctness. All tested prompts fit
within 128 tokens. This evaluation changes no model weights or Pi deployment.

## Article-context and NLP evaluation, 22 November

Audited a new 38-item Google News RSS snapshot. Descriptions were related-title
lists, not usable article abstracts. Resolved 12 selected publisher URLs and
obtained metadata summaries/opening text from nine. WESH returned HTTP 451;
AP and NewsNation returned HTTP 403. Those failures remain explicit headline
fallback cases, not successful context examples. Extraction required filtering
browser-support boilerplate and repeated title fragments; subtitle metadata can
be incomplete or promotional even after this filtering.

The prototype combines NLTK sentence segmentation, UDPipe dependency parsing,
WordNet lexical classes for proposition ranking, and NER diagnostics. Direct
SVO reconstruction was rejected after it dropped coordinated actions and
misread "has swum" as possession. Final prompts preserve a selected source
sentence, with headline fallback for unsuitable parses. NER labels were inspected
but do not themselves provide a visual scene or the appearance of named people.

Generated **96 laptop CUDA images**: all 12 attempted articles, seeds 0 and 1,
four prompt variants (headline/style; existing topic template; NLP headline;
NLP article context). Every row fixes initial noise, 25 steps, CFG 3, and TAESD.
All prompts fit the context limit; the largest was 54 tokens. Eight research
checks passed, including negation, coordinated-action preservation, sentence
segmentation, boilerplate filtering and the existing export checks.

Visual inspection found mixed results. The Dragon capsule/context examples show
a more recognizable capsule, and wedding context introduces clearer wedding
clothing in some images. Abstract politics and crime descriptions still produce
pseudo-lettering or generic portraits; the phone context can be worse than its
headline. Mercury context still produces lettering. Headline-only NLP changes
little, and the soccer parse falls back even though article text is available.
Templates generally produce cleaner illustrative scenes but sacrifice specific
meaning; the ship-briefing example even matches the broad military template,
demonstrating the existing rules' brittleness. No automatic image-quality score
or general improvement rate is claimed.

On the laptop, the combined NLP process loaded in 3.59 s and took 12.68 s for
24 conversions, with **506.58 MiB process-lifetime peak RSS**. This includes NER
and WordNet overhead and is not a per-headline memory figure. It has not been
benchmarked on ARMv7 and is not ready to assume safe concurrent Pi execution.
Existing shared-environment package versions were verified unchanged.

Full local evidence: `artifacts/context-nlp/images/index.html`, `articles.json`,
`feed-audit.json`, `nlp-prompts.json`, `nlp-metrics.json`, and
`assets-environment.json` under `artifacts/context-nlp/`. The source HTML, NLP
assets and generated images are ignored. No Pi service, model or deployment was
changed. Context acquisition is feasible but is not, by itself, a reliable fix
for the model's pseudo-text failure; sentence selection is not equivalent to
converting news into a concise visual description.

## Alternative generators: completed comparison, 22 November

Tested the requested FLUX.1-Turbo-Alpha, Inko and Surjo-Image-Preview on all 48
seed-0 prompt slots from the context experiment. There are 32 distinct strings;
identical prompt/seed outputs are reused. Existing Supra2/TAESD images provide
the baseline. The final gallery contains **12 complete 4×4 article grids (192
cells)** plus two 4×4 contact sheets per model. Original resolutions and exact
prompts/settings are retained; no Pi deployment occurred.

| Generator | Resolution | Steps / guidance | Median inference | Max allocated CUDA |
| --- | --- | --- | --- | --- |
| Surjo-Image-Preview | 256² | 50 / 3 | 1.73 s | 894 MiB |
| Inko | 896² | 20 / 4.5 | 11.22 s | 5102 MiB |
| FLUX Turbo Alpha, NF4 base/T5 | 512² | 8 / 3.5 | 11.47 s | 7300 MiB |

These are different native/practical recipes, not equal compute or resolution.
The laptop GPU reached 86°C and reduced clocks during sustained generation;
timings are descriptive, not controlled performance rankings. Native timings
include text encoding, sampling and decoding, excluding model loading. FLUX
stages use separate processes: median encode 0.334 s, sample 10.843 s, decode
0.245 s. The median of per-image summed stage times is 11.472 s (it need not
equal the sum of stage medians). FLUX's 7300 MiB figure is its sampling peak;
encoding peaked at 4583 MiB and decoding at 786 MiB. These peaks do not overlap.
Process-lifetime CPU RSS peaks were 2544 MiB, 5001 MiB and 7395 MiB respectively;
they include loading and must not be treated as ARM runtime requirements.

FLUX Turbo Alpha is the requested adapter on FLUX.1-dev, not a standalone small
model. It ran at 512px with a Diffusers NF4 base and NF4 T5 to fit 8 GiB VRAM,
below the upstream 1024px example. Verified all **346,972,160 adapter parameters**
were loaded, nonzero adapter weights were active, and adapter scale was 1.
Adapter arithmetic/storage remained BF16. No CPU offloading was needed during
sampling; the encoder and original VAE ran separately. This is a practical
quantized variant, not a full-precision reference result.

Visual inspection of the distinct-prompt contact sheets found FLUX strongest
overall in this sample, especially the soccer celebration, burning ship and
capsule examples. Inko also substantially reduced dominant pseudo-lettering and
produced more coherent scenes than Supra2, but confused some spacecraft with
helmet-like objects and did not reliably preserve named identities or gender.
Surjo improved many scenes but repeated generic portraits on abstract news and
still produced large fake lettering for soccer and medical headlines. Small
labels, anatomy and subject/scene errors remain even in the stronger models.
This is one seed on a small development prompt set, not a general quality score.

Verified every comparison prompt matches the frozen previous experiment, all
192 cell paths resolve, all 96 new distinct-model PNGs have the declared native
dimensions, baseline copies are byte-identical, and every FLUX record is complete.
Existing shared-environment package versions were preserved; only PEFT was added
for this experiment. Weights are pinned and LFS SHA-256 checked. Sampling code,
source revisions and reproduction commands are in `research/README.md`.

Evidence: ignored `artifacts/model-comparison/index.html`, `comparison.json`,
`summary.json`, per-model PNG/JSON files and per-component download manifests.
No alternative generator is integrated with the production worker or validated
on the Pi. Laptop image quality and GPU timings do not establish Pi feasibility.

### Inko painting style at 512px

Four frozen headline-only prompts (Mercury, soccer, wedding and ship) were tested
with `An expressive oil painting, visible brushstrokes, rich colors.` appended.
Same seed 0, BF16, 20 steps and CFG 4.5; all components remain on the GPU. The
comparison also reuses the previous photo-style 896px outputs. Eight new images
were generated; no production settings or weights changed.

| Oil-painting resolution | Median inference | Peak allocated CUDA |
| --- | --- | --- |
| 896² | 10.42 s | 5102 MiB |
| 512² | 3.41 s | 3312 MiB |

The smaller output reduced allocated GPU peak by about 35% and was about 3.1×
faster in this four-case sample. PyTorch's reserved allocator cache is not this
metric. Resizing changes latent noise dimensions and the upstream resolution
time shift, so the same seed need not preserve composition. The painterly style
remains visible at 512px, with subject errors still present (including sports
and spacecraft confusion). No broad quality equivalence is claimed.

Measured BF16 parameter storage: generator 532,496,992 parameters / 1015.66 MiB;
text encoder 596,049,920 / 1136.88 MiB; VAE 126,892,531 / 242.03 MiB. These sum to
2394.56 MiB (2.34 GiB) before activations/workspace. Lowering image resolution
does not shrink those weights. Thus 512px alone does not establish a fit within
the Pi 2's 920 MiB RAM; quantization, staging and an ARM runtime would still need
investigation. The comparison/measurements live in `artifacts/inko-resolution/`.

## Bounded Inko candidate, 26 November 2026

Inko now has a standalone NumPy/NCNN runtime with sequential text, sampling and
decoding processes. Groupwise packed 4-bit transformer weights (8-bit modulation),
8-bit Qwen text weights and a mapped embedding table avoid full model residency.
Multiplication remains FP32; packed storage is not an integer speedup.

A rank-32 adapter trained for 3000 updates on 344 locally generated teacher
trajectories reduces sampling from 24 guided teacher steps to eight CFG-1
student steps. A separately trained sixteen-channel Tiny Qwen decoder completed
6000 updates: 25.05 dB mean PSNR / 0.7996 SSIM on 32 validation images.

The final standalone 384px laptop CPU pipeline completed in 45.92 s. Peak child
RSS was 127.3 MiB (text), 212.3 MiB (sample), 248.1 MiB (decode); each had RLIMIT_DATA
480 MiB and imported no Torch/Transformers/Diffusers. Portable transformer
velocity relative error versus an independent FP32 fixture was 1.97e-6.
The memory limit is an allocation limit, not a strict RSS or swap cap.

Manual inspection of eight held-out captions with two seeds finds much more
recognizable scenes than the Supra comparison, while the fast candidate loses
texture/color versus original guided Inko. Anatomy, extra people, subject
confusion and pseudo-lettering remain. The original quantized 20-step CFG-4.5
model and exact decoder are retained as a quality fallback. This is a small
engineering comparison, not a broad quality benchmark.

Supra typography-subspace weight edits did not show a useful improvement.
A 244M-parameter block/MLP-pruned Inko remained visually unacceptable after
6000 recovery updates and is rejected. The successful candidate keeps the
dense 532M architecture and saves residency and forward passes instead.

Methods, limitations, training interruption/resume and commands are documented
in [edge_quality/README.md](edge_quality/README.md). Machine-readable evidence
is [results/edge-quality.json](results/edge-quality.json); the local visual
report is `artifacts/edge-quality/index.html`. The Pi was disconnected: no ARMv7
latency, system-memory fit, production integration or HDMI success is claimed.

Evaluation holds out exact caption strings, not subject categories: synthetic
training captions cover these subject families. This is not an out-of-domain
evaluation.
