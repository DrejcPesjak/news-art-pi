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

## Raspberry Pi

Pi 2 B Rev 1.1, ARMv7, Raspbian 13, Python 3.13. NCNN 1.0.20261020 has a matching
wheel and works on the device. SentencePiece and NumPy are local dependencies.
Tests keep the existing news display running. This kernel lacks the memory cgroup
controller, so systemd memory/swap caps cannot be relied on here. Subsequent
pipeline trials use an explicit per-process `RLIMIT_DATA` allocation limit,
alongside systemd's time limit and low scheduling priority. This is not an exact
RSS cap and does not disable OS swapping. No laptop service is used by this
runtime; weights reside on the Pi.

Measured component results and the complete image trial are recorded below once
finished. GPU timings above must not be extrapolated to ARM.
