# Inko on a small ARMv7 device: laptop experiments

Started 25 November 2026. Production is unchanged. The Pi is disconnected;
all new timings and memory figures are laptop measurements until explicitly
identified otherwise. Use the existing `~/venvs/ml` environment, without `uv sync`.

## Methods and boundaries

* Freeze `prompts.json` before experiments: eight development and eight
  evaluation scenes. Seeds 0 and 1. Evaluation captions are excluded from
  calibration and distillation training.
* Test one specific gradient-free Supra2 weight-edit hypothesis: project away
  typography-related T5 conditioning directions from `ctx_proj.weight`.
  Four edits, original control, and a byte-identical restored-image control.
  This is semantic-subspace ablation, not proof of a localized text circuit.
  It did not produce convincing quality improvement on development scenes.
* Quantize Inko's transformer matrices in groups of 64 input columns, preserving
  other small projections/norms. Compare all-4-bit, 4-bit plus 8-bit modulation,
  and groupwise Hessian error feedback using development teacher activations.
  The last method is a bounded groupwise adaptation of GPTQ, not full-Hessian
  GPTQ. Its raw weight error can rise while activation reconstruction improves.
* Keep packed weights mapped and dequantize only the current matrix. Discard
  the process's mapped pages afterwards; Linux may retain reclaimable file
  cache. Execute optimized FP32 NCNN GEMM, avoiding the scalar int4 fallback.
  This saves memory; it does not promise integer-arithmetic speedups.
* Run Qwen3-0.6B through layer 20 only, before final normalization. Gather just
  the requested embedding rows from an FP16 table. The released tokenizer's
  NFC, byte-level BPE and special tokens are reproduced with Python + `regex`.
* Specialize the original REPA-E Qwen VAE for its first image frame. A causal
  temporal convolution then needs only its last temporal kernel slice. The
  first-frame temporal resampling convolution is skipped by the upstream VAE.
  All spatial weights remain unchanged: 25,291,955 parameters and about
  48.3 MiB NCNN FP16 storage. This exact decoder remains the quality fallback.
* Export decoder parts, execute full-frame attention, then use exact row tiles
  for local residual/resizing/head operations. Halos cover their full spatial
  receptive fields; channel normalization is local to each pixel. This differs
  from approximate tiling of an entire decoder containing global attention.
* Fresh sequential text/sample/decode processes enforce a 480 MiB RLIMIT_DATA.
  That is an allocation limit, not a strict RSS cap or an OS swap policy.
* Exploratory guidance/step distillation trains a rank-32 LoRA to regress the
  teacher's mean guided velocity over coarse trajectory intervals: 24 guided
  steps to 8 conditional steps at 384px. It uses 192 captured public captions,
  144 handwritten scene combinations and eight development captions. No image
  dataset is downloaded. Success must be judged on rolled-out images, not just
  a falling teacher-forced regression loss.

## Sources and pinned components

Component revisions remain those in `research/supra/alternative_assets.py`.
All original checkpoints stay intact. Derived checkpoints and generated evidence
live in ignored `artifacts/edge-quality/`; runtime reference sources are in
ignored `models/edge-quality/`.

* [Inko source](https://github.com/Mtrya/inko), code revision
  `9f8f79cd7491bf96b13d43f290cc9a281afd5787`.
* [Weight-space programming experiments](https://rikiyat.github.io/blog/weight-space-programming/)
  and [WeightCode](https://arxiv.org/abs/2611.25476) motivated the targeted edit;
  their reported results do not establish general image-generator repair.
* [GPTQ](https://arxiv.org/abs/2210.17323) motivates Hessian error compensation.
* [Inko rollout captions](https://huggingface.co/datasets/kaupane/inko-rollout-25k),
  captured dataset revision `9d9a9b5c0e1c435a793087428ae75c7eb4e75e10`.
  Exact selected IDs, processed captions and checksums are retained in the
  ignored distillation directory; these captions are not redistributed here.

## Reproduction

Run from the repository root; GPU commands require GPU access outside a sandbox
that hides NVIDIA devices. No packages or environment versions are changed.

```bash
HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.edge_quality.supra_edit
HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.edge_quality.inko_quant encode
~/venvs/ml/bin/python -m research.edge_quality.inko_quant pack --variant q4
~/venvs/ml/bin/python -m research.edge_quality.inko_quant pack --variant q4-mixed
HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.edge_quality.gptq calibrate
HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.edge_quality.gptq recover

HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.edge_quality.inko_quant generate --variant bf16
HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.edge_quality.inko_quant generate --variant q4-gptq

~/venvs/ml/bin/python -m research.edge_quality.qwen_export pack --bits 8
HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.edge_quality.qwen_export fixture --bits 8
OPENBLAS_NUM_THREADS=4 ~/venvs/ml/bin/python -m research.edge_quality.qwen_export validate --bits 8

HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.edge_quality.decoder_export --latent-size 32 --parts
HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.edge_quality.decoder_export --latent-size 48 --parts
cp models/alternatives/inko-vae/config.json artifacts/edge-quality/decoder/vae-config.json

OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 ~/venvs/ml/bin/python -m research.edge_quality.staged run \
  --variant q4-gptq --size 384 --steps 12 --guidance 3 \
  --work artifacts/edge-quality/example

~/venvs/ml/bin/python -m research.edge_quality.distill dataset
HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.edge_quality.distill encode
HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.edge_quality.distill trajectories
HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.edge_quality.distill train

~/venvs/ml/bin/python -m unittest discover -s research/tests -v
```

`staged` has no PyTorch/Transformers/Diffusers imports, and records whether any
were loaded in each child. Its parent enforces a process-group timeout. Runtime
dependencies are NumPy, NCNN, Pillow and `regex`; a fresh Pi may need the latter
installed through apt. This runner is not integrated into the production worker.
NCNN uses four threads, while the runner configures NumPy's otherwise-unused
BLAS pool to one: its extra laptop threads reserve substantial virtual data
which counts against RLIMIT_DATA even when RSS is low.

## Validation evidence

Independent FP32 fixtures retain every transformer block output and final
velocity; the portable implementation is compared after the fixture process
exits. Test actual text features and conditional/unconditional trajectories,
not only unrepresentative random text embeddings. Decoder rewrite checks cover
multiple spatial sizes. Packed formats, odd group tails, signed nibbles, zeros,
attention chunking, RoPE norms, tokenizer parity and dynamic GEMM have unit tests.

The first exploratory decoder exports failed the allocation limit despite
moderate RSS; those failures are retained in local logs. Replacing expanded
normalization, avoiding redundant batch copies, preserving NCNN input ownership,
splitting graph parts and bounding local convolution workspaces are necessary
parts of the runtime. Do not report an unlimited-memory recovery run as a
successful bounded run.

No laptop check proves ARMv7 package readiness, Pi latency, total system memory,
power health or HDMI behavior. Those require the physical Pi when it reconnects.

## Completed experiment, 26 November 2026

The practical result is a bounded local Inko runtime, with two recipes:

| Recipe | Transformer | Sampling | Decoder | Purpose |
| --- | --- | --- | --- | --- |
| Fast candidate | 532M dense, rank-32 LoRA merged, mixed 4/8-bit | 8 steps, CFG 1 | locally trained Tiny Qwen | First Pi trial |
| Quality fallback | original dense, activation-corrected mixed 4/8-bit | 20 steps, CFG 4.5 | exact one-frame Qwen | Preserve guided quality |

Both use a quantized Qwen text encoder in a separate process. Original BF16
weights total 2394.56 MiB before activations; the runtime never loads them
jointly. Packed transformer weights occupy about 361.7 MiB, and text assets
about 626.6 MiB including the mapped embedding table. Disk bytes are not peak
RAM. Four-bit matrices are expanded one at a time for FP32 multiplication.

The selected dense adapter completed 3000 updates on 344 teacher trajectories.
Development rollout latent relative RMSE fell from 0.6934 to 0.4775, a diagnostic
rather than an image quality score. An inference-mode RoPE cache interrupted the
first run after checkpoint 1500. The cache was fixed, a regression test added,
and training resumed from that checkpoint with a fresh optimizer and LR 5e-5
(initial LR 1e-4). `training-history.json` records this discontinuity. This is
coarse-trajectory regression on teacher states, not full distribution matching.

A separate 1,229,443-parameter decoder completed 6000 updates against the Qwen
VAE's images. It starts from TAESD, expands the input to sixteen channels, and
fine-tunes every decoder weight; unmodified TAESD cannot decode Inko latents.
Its 32-image validation set gives mean PSNR 25.05 dB and local-window SSIM
0.7996. The 312 training and 32 validation examples are disjoint for this
specific decoder training; the transformer saw the development examples.
Inputs are **normalized Inko latents**, with neither VAE mean/std nor SD scaling
applied. NCNN FP16 storage is about 2.35 MiB. It softens fine texture and can
change colors. The original exact decoder is also included in the bundle.

A physically smaller model retained 13/25 blocks and reduced MLP width from
3072 to 2048 (243,830,688 parameters). Activation statistics and block influence
selected the retained weights. Even after 6000 full-model recovery updates,
images remained abstract or garbled; development rollout error was 0.8355.
This model is rejected. High retained activation energy did not imply quality.
The targeted direct Supra weight edits also failed to establish a useful gain.

Sixteen held-out-caption outputs (eight prompts, two seeds) compare Supra, original
BF16 Inko, quantized guided Inko, and the final fast candidate. Manual inspection
finds recognizable subjects and compositions in many candidate images, with
less texture/color than guided Inko. Bad anatomy, pseudo-text and subject
mistakes remain. A spacecraft prompt can still produce a robot, and one wedding
seed produces an extra bride. Neither PSNR nor these few examples establish
broad generation quality. Check both seeds in the local report.

The final standalone 384px candidate CPU run took **45.92 s** on this laptop:
text peak 127.3 MiB, sample 212.3 MiB, decoder 248.1 MiB. All three fresh stages completed
under a 480 MiB allocation limit without Torch/Transformers/Diffusers. Numeric
portable-vs-FP32 velocity relative RMSE was 1.97e-6 at 384px. Earlier exact-decoder
validation at 512px used 316.9 MiB; Tiny decoding there used 401.4 MiB and 0.82 s.
A smaller decoder file can therefore still have a larger activation peak.
These figures do not predict ARMv7 speed or whole-system memory pressure.

## Final candidate commands

These assume the ignored experiment outputs already exist. GPU generation uses
cached Qwen conditioning; CPU staging creates its own conditioning from text.

```bash
HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.edge_quality.inko_quant pack \
  --variant q4-distilled-3000 --checkpoint artifacts/edge-quality/distill/merged-3000
HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.edge_quality.gptq calibrate \
  --checkpoint artifacts/edge-quality/distill/merged-3000 \
  --calibration-name calibration-distilled-3000 --size 384 --steps 8 --guidance 1
HF_HUB_OFFLINE=1 ~/venvs/ml/bin/python -m research.edge_quality.gptq recover \
  --checkpoint artifacts/edge-quality/distill/merged-3000 \
  --calibration-name calibration-distilled-3000 --base-variant q4-distilled-3000 \
  --output-variant q4-distilled-3000-gptq

~/venvs/ml/bin/python -m research.edge_quality.tiny_decoder dataset
~/venvs/ml/bin/python -m research.edge_quality.tiny_decoder train --steps 6000
~/venvs/ml/bin/python -m research.edge_quality.tiny_decoder export \
  --checkpoint artifacts/edge-quality/tiny-qwen/decoder-6000.safetensors

~/venvs/ml/bin/python -m research.edge_quality.staged run \
  --variant q4-distilled-3000-gptq --size 384 --steps 8 --guidance 1 \
  --decoder tiny-qwen --work artifacts/edge-quality/example-fast \
  --prompt 'A lighthouse on a rocky coast. oil painting'

~/venvs/ml/bin/python -m research.edge_quality.bundle build \
  --output artifacts/edge-quality/candidate-bundle --variant q4-distilled-3000-gptq \
  --sizes 384 512 256 --steps 8 --guidance 1 --decoder tiny-qwen --archive
~/venvs/ml/bin/python -m research.edge_quality.bundle verify \
  --output artifacts/edge-quality/candidate-bundle
~/venvs/ml/bin/python -m research.edge_quality.gallery \
  --student-variant q4-distilled-3000-gptq-384px-8s-cfg1-tiny
```

Open `artifacts/edge-quality/index.html` or the two
`final-evaluation-seed-{0,1}.jpg` contact sheets. Committed measurements are in
`research/results/edge-quality.json`; images, intermediate checkpoints and
bundles stay ignored. The archive includes standalone runtime code, component
checksums, model attribution, exact decoder parts and a default recipe.

The next hardware gate is an isolated trial on the actual Pi: verify ARMv7
NCNN/NumPy/regex, run the bundle with production still displaying its last
poster, check available RAM/swap/power and completed image/latency. Integrate
and deploy through Git only after that result is acceptable. No production
configuration, worker code or deployment branch was changed by this experiment.

Evaluation holds out exact caption strings, not subject categories: synthetic
training captions cover these subject families. This is not an out-of-domain
evaluation.

The final 512px run completed in 77.84 s, with 277.2 MiB
sampling RSS and 397.0 MiB decoder RSS, all stages limited to 480 MiB. Its first
attempt failed during pass four with NCNN allocation error -100. Bounding
transposed-weight GEMM packing to 2048 output rows per call fixed that observed
workspace failure; numerical parity remained unchanged. The failed logs are
retained. The default remains 384px for more memory headroom.
