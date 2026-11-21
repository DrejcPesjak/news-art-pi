# Supra2 experiment results

Measured on 21 November 2026. This is a small engineering evaluation, not a
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

### Deployment status and next integration

The public repository's `main` contains the experiments, numerical validation,
portable bundle builder, commands, and results. The active `deploy` branch still
runs the original RSS/demo-image application; the three supplied pictures keep
cycling. These experiments do **not** switch the production generator backend.

The next application change is to connect this measured runtime to headline
prompts and configuration, place runtime dependencies/models outside the service's
protected home directory, and schedule refreshes from job start time rather than
sleeping ten minutes after generation ends. Keep generation bounded and retain
the last successful poster. Unattended service logging should use the local
journal rather than an SSH-attached pipe.
