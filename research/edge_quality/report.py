"""Build three portable HTML research reports with small, committed previews.

Reads retained local evidence; never loads or changes a generation model.
No network, training, inference, or external web assets are used.
"""
import argparse
import html
import json
import os
from pathlib import Path
import shutil
import tempfile

from PIL import Image

from .common import ROOT, OUTPUT, save_json, sha256

DESTINATION = ROOT / "research/reports/edge-quality"
VARIANTS = [
    ("Supra2", "baseline"),
    ("Original Inko · BF16", "bf16-384px-20s-cfg4.5"),
    ("Inko · quantized", "q4-gptq-384px-20s-cfg4.5"),
    ("Student · exact decoder", "q4-distilled-3000-gptq-384px-8s-cfg1"),
    ("Student · tiny decoder", "q4-distilled-3000-gptq-384px-8s-cfg1-tiny"),
]
STYLE = """
:root{--paper:#f5f2eb;--ink:#202b2b;--muted:#57625f;--line:#d7d9cd;--green:#236253;--gold:#a36b16;--red:#9a3e36}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:17px/1.65 system-ui,sans-serif}
nav{border-bottom:1px solid var(--line);padding:18px max(24px,calc((100vw - 1160px)/2));display:flex;gap:24px;flex-wrap:wrap;background:#fcfaf6}
a{color:var(--green);text-underline-offset:4px}nav a{font-size:14px}nav a[aria-current]{font-weight:750}
main{max-width:1160px;padding:50px 24px 70px;margin:auto}.eyebrow{text-transform:uppercase;letter-spacing:.14em;font-size:12px;font-weight:750;color:var(--green)}
h1{font:clamp(38px,6vw,68px)/1.08 Georgia,serif;max-width:920px;letter-spacing:-.035em;margin:16px 0 25px}
h2{font:32px/1.2 Georgia,serif;margin-top:52px}h3{font-size:20px}.lead{max-width:850px;font-size:21px;line-height:1.55}
p,li{max-width:920px}.muted,figcaption,small{color:var(--muted);font-size:14px}.note{border-left:4px solid var(--gold);padding:16px 22px;background:#ede7d8;margin:26px 0}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:32px 0}.card{padding:22px;border:1px solid var(--line);background:#fcfaf6;border-radius:10px}
.card strong{display:block;font:38px/1.1 Georgia,serif;margin-bottom:10px}.card span{color:var(--muted);font-size:14px}
.scroll{overflow-x:auto;margin:22px 0}table{width:100%;border-collapse:collapse;text-align:left;font-size:14px}th,td{padding:12px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted)}
.gallery{min-width:1000px}.gallery th:first-child{width:180px}.gallery td{padding:7px}.gallery img{width:160px;height:160px;object-fit:contain;border-radius:4px;display:block;background:#e7e4db}
figure{margin:28px 0}figure>img{max-width:100%;height:auto}figcaption{margin-top:9px}details{border-top:1px solid var(--line);padding:15px 0}summary{cursor:pointer;font-weight:650}
code,pre{font-family:ui-monospace,monospace;font-size:14px}pre{overflow:auto;background:#e9e7df;padding:20px;border-radius:8px}
.pill{display:inline-block;padding:2px 8px;border-radius:20px;font-size:12px;border:1px solid var(--line);white-space:nowrap}.ok{color:var(--green)}.bad{color:var(--red)}
.flow{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;counter-reset:flow}.flow>div{border-top:3px solid var(--green);padding:18px;background:#fcfaf6}.flow strong{display:block;margin-bottom:10px}
footer{margin-top:50px;padding-top:20px;border-top:1px solid var(--line);font-size:14px;color:var(--muted)}.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}
@media(max-width:700px){.cards,.flow,.pair{grid-template-columns:1fr}main{padding-top:28px}nav{gap:14px}.lead{font-size:19px}}
@media print{body{background:white}details{display:block}.scroll{overflow:visible}.gallery{min-width:0}.gallery img{width:100%;height:auto}nav{display:none}h2{break-after:avoid}.cards,figure,tr{break-inside:avoid}}
"""


def read(path):
    return json.loads(Path(path).read_text())


def esc(value):
    return html.escape(str(value), quote=True)


class Reports:
    def __init__(self, destination):
        self.destination = destination
        (destination / "assets").mkdir(parents=True, exist_ok=True)
        self.assets = {}

    def preview(self, path, width=240):
        path = Path(path)
        key = sha256(path)[:20] + f"-{width}"
        relative = f"assets/{key}.webp"
        target = self.destination / relative
        if not target.exists():
            with Image.open(path) as original:
                im = original.convert("RGB")
                im.thumbnail((width, width * 4))
                im.save(target, "WEBP", quality=72, method=6)
        self.assets[relative] = dict(
            source=str(path.resolve().relative_to(ROOT)),
            source_sha256=sha256(path), preview_sha256=sha256(target),
            bytes=target.stat().st_size,
        )
        return relative

    def page(self, filename, title, subtitle, content):
        links = [("index.html", "Experiment report"), ("weight-editing.html", "Weight editing"),
                 ("inko-evolution.html", "From Inko to the candidate")]
        nav = "".join(f'<a href="{p}"{" aria-current=page" if p == filename else ""}>{name}</a>' for p, name in links)
        page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)} · News Art Pi research</title><style>{STYLE}</style></head>
<body><nav aria-label="Reports">{nav}</nav><main>
<div class="eyebrow">News Art Pi / research record / 25–26 November 2026</div>
<h1>{esc(title)}</h1><p class="lead">{subtitle}</p>
{content}
<footer>Evidence and code: <a href="https://github.com/DrejcPesjak/news-art-pi/tree/research/inko-edge-quality">research/inko-edge-quality</a>.
<a href="data/edge-quality.json">Measurements</a> ·
<a href="data/edge-quality-cfg.json">CFG experiment</a> ·
<a href="methods.txt">Methods and commands</a>.
These reports contain compressed image previews, not model weights or training data. No external fonts, scripts or image servers are required.</footer>
</main></body></html>"""
        (self.destination / filename).write_text(page)

    def gallery(self, rows, variants, seeds=(0, 1)):
        output = []
        for seed in seeds:
            ids = list(dict.fromkeys(r["id"] for r in rows if r["seed"] == seed))
            lookup = {(r["id"], r["variant"]): r for r in rows if r["seed"] == seed}
            output.append(f'<details{" open" if seed == 0 else ""}><summary>Seed {seed} — {len(ids)} captions</summary><div class="scroll"><table class="gallery"><thead><tr><th>Caption</th>')
            output.extend(f"<th>{esc(label)}</th>" for label, _ in variants)
            output.append("</tr></thead><tbody>")
            for identifier in ids:
                prompt = next(r["prompt"] for r in rows if r["id"] == identifier)
                output.append(f"<tr><th scope=row>{esc(prompt)}</th>")
                for _, variant in variants:
                    record = lookup.get((identifier, variant))
                    if record is None:
                        raise ValueError(f"Missing comparison: {seed}/{identifier}/{variant}")
                    asset = self.preview(record["image"])
                    output.append(f'<td><a href="{asset}"><img src="{asset}" alt="{esc(identifier + ": " + variant)}" loading="lazy" width="160" height="160"></a></td>')
                output.append("</tr>")
            output.append("</tbody></table></div></details>")
        return "".join(output)

    def figure(self, path, caption, width=1050):
        asset = self.preview(path, width=width)
        return f'<figure><a href="{asset}"><img src="{asset}" alt="{esc(caption)}" loading="lazy"></a><figcaption>{caption}</figcaption></figure>'


def charts(destination, measurements):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "news-art-report-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "svg.hashsalt": "news-art-research",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.facecolor": "#f5f2eb", "figure.facecolor": "#f5f2eb",
                         "axes.labelcolor": "#202b2b", "text.color": "#202b2b"})
    green, red, gold = "#236253", "#9a3e36", "#a36b16"
    assets = destination / "assets"

    fig, ax = plt.subplots(figsize=(8.3, 3.4), layout="constrained")
    dense = measurements["dense_checkpoint_metrics"]
    small = read(OUTPUT / "small-checkpoint-summary.json")
    ax.plot([r["step"] for r in dense], [r["mean_latent_relative_rmse"] for r in dense],
            "o-", color=green, label="532M dense student")
    ax.plot([r["step"] for r in small], [r["mean_latent_relative_rmse"] for r in small],
            "s--", color=red, label="244M pruned student — rejected")
    ax.set(xlabel="Training updates", ylabel="Relative latent RMSE ↓", ylim=(0, 1.08),
           title="Rollout error on four development captions")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=.2)
    fig.savefig(assets / "training.svg", metadata={"Date": "2026-11-26"})
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), layout="constrained")
    tiny = read(OUTPUT / "tiny-qwen-checkpoint-summary.json")
    for ax, key, label in zip(axes, ["mean_psnr", "mean_ssim"], ["Mean PSNR (dB) ↑", "Mean local-window SSIM ↑"]):
        ax.plot([r["step"] for r in tiny], [r[key] for r in tiny], "o-", color=green)
        ax.set(xlabel="Decoder training updates", ylabel=label)
        ax.grid(axis="y", alpha=.2)
    fig.savefig(assets / "decoder-training.svg", metadata={"Date": "2026-11-26"})
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
    cfg = read(ROOT / "research/results/edge-quality-cfg.json")
    baseline, guided = cfg["cpu_cfg1"], cfg["cpu_cfg2"]
    runs = [baseline, guided, measurements["bounded_cpu_runs"]["candidate-native-512-bounded"]]
    labels = ["384px / CFG 1", "384px / CFG 2", "512px / CFG 1"]
    axes[0].barh(labels, [r["total_seconds"] for r in runs], color=[green, gold, "#5c7a89"])
    axes[0].set(xlabel="Complete CPU pipeline (seconds)", title="Separate laptop runs")
    for stage, color in zip(["encode", "sample", "decode"], [gold, green, "#5c7a89"]):
        index = ["encode", "sample", "decode"].index(stage)
        axes[1].bar([i + (index-1)*.23 for i in range(3)],
                    [r["stages"][stage]["peak_rss_mib"] for r in runs],
                    width=.23, label=stage, color=color)
    axes[1].set(xticks=range(3), xticklabels=["384 / 1", "384 / 2", "512 / 1"],
                ylabel="Peak child RSS (MiB)", xlabel="Resolution / CFG",
                title="Stages execute sequentially")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(assets / "cpu.svg", metadata={"Date": "2026-11-26"})
    plt.close(fig)
    # Matplotlib leaves spaces at SVG path line endings; keep generated text
    # clean for repository whitespace checks without changing the geometry.
    for path in assets.glob("*.svg"):
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def build(destination):
    report = Reports(destination)
    (destination / "data").mkdir(exist_ok=True)
    (destination / "source").mkdir(exist_ok=True)
    for name in ("edge-quality.json", "edge-quality-cfg.json", "edge-quality-cleanup.json"):
        shutil.copyfile(ROOT / "research/results" / name, destination / "data" / name)
    shutil.copyfile(Path(__file__).parent / "README.md", destination / "methods.txt")
    shutil.copyfile(Path(__file__).parent / "supra_edit.py", destination / "source/supra_edit.py")
    measurements = read(ROOT / "research/results/edge-quality.json")
    cfg = read(ROOT / "research/results/edge-quality-cfg.json")
    cleanup = read(ROOT / "research/results/edge-quality-cleanup.json")
    records = read(OUTPUT / "inko/records.json")
    supra = read(OUTPUT / "supra-evaluation/records.json")
    selected = [r for r in records if r["split"] == "evaluation" and r["variant"] in {v for _, v in VARIANTS}]
    selected += [r for r in supra if r["variant"] == "baseline"]
    assert len(selected) == 80
    charts(destination, measurements)

    overview = """
<div class="cards"><div class="card"><strong>45.9 s</strong><span>Complete 384px laptop CPU run, CFG 1</span></div><div class="card"><strong>248 MiB</strong><span>Largest child-process RSS in that run</span></div><div class="card"><strong>8 passes</strong><span>One conditional evaluation per sampling step</span></div></div>
<div class="note"><strong>Working laptop candidate; Pi trial pending.</strong> The Pi 2 B was disconnected. All new CPU measurements are x86_64 laptop results. Neither ARMv7 speed, total system memory, power nor HDMI behavior has been verified. Production code, settings and deployment were unchanged.</div>
<h2>The problem and the outcome</h2>
<p>Supra2 fit the existing Pi pipeline but often produced weak scenes and pseudo-lettering. Original Inko produced more recognizable pictures, but its BF16 generator, text encoder and VAE together contained about 2394.6 MiB of parameters before activations. The target Pi has about 920 MiB RAM.</p>
<p>The successful route keeps Inko’s dense 532M generator and reduces weight residency, model evaluations and decoder cost. It combines packed mixed 4/8-bit weights, a separately staged text encoder, eight-pass trajectory distillation and a newly trained tiny decoder. It is a research candidate rather than a demonstrated Pi deployment.</p>
<h2>What we tried</h2>
<div class="scroll"><table><thead><tr><th>Experiment</th><th>What happened</th><th>Decision</th></tr></thead><tbody>
<tr><td>Direct Supra weight edits</td><td>Removed selected typography-related conditioning directions at two ranks and strengths. No convincing scene-quality or lettering benefit.</td><td><span class="pill bad">Rejected hypothesis</span></td></tr>
<tr><td>Inko quantization</td><td>Compared all-4-bit, 8-bit modulation and groupwise activation-aware error compensation. Calibration used actual development trajectories.</td><td><span class="pill ok">Keep mixed 4/8-bit</span></td></tr>
<tr><td>Portable CPU runtime</td><td>NumPy/NCNN math, mapped weights, one dequantized matrix at a time and fresh sequential processes. No Torch, Transformers or Diffusers loaded.</td><td><span class="pill ok">Working on laptop</span></td></tr>
<tr><td>Exact original decoder</td><td>Specialized causal one-frame Qwen decoding to 2D, with exact local halos and full global attention. About 25.3M parameters / 48.3 MiB storage.</td><td><span class="pill ok">Quality fallback</span></td></tr>
<tr><td>Dense trajectory distillation</td><td>344 teacher trajectories; 3000 rank-32 adapter updates. Teacher: 24 steps / CFG 4.5. Student: 8 steps / CFG 1.</td><td><span class="pill ok">Selected checkpoint</span></td></tr>
<tr><td>Physical pruning and recovery</td><td>13 of 25 blocks, MLP width 2048 instead of 3072, 244M parameters. Images remained unacceptable after 6000 recovery updates.</td><td><span class="pill bad">Rejected model</span></td></tr>
<tr><td>Train a tiny Qwen decoder</td><td>16-channel TAESD-derived decoder, all weights fine-tuned for 6000 updates. 1.229M parameters; 25.05 dB PSNR / 0.7996 SSIM on 32 validation images.</td><td><span class="pill ok">Selected, visibly lossy</span></td></tr>
<tr><td>Additional CFG</td><td>Same student tested at 1, 2, 3 and 4.5. Extra guidance strengthens color/contrast, with mixed preferences and persistent coherence faults.</td><td><span class="pill">Default unchanged</span></td></tr>
</tbody></table></div>
<h2>Images: the useful gain and the remaining failures</h2>
<p>All eight exact evaluation captions and both seeds are shown below. Their subject families appear in synthetic training captions, so this is not an out-of-domain benchmark. Supra uses 256px / 25 steps / CFG 3; original and quantized guided Inko use 384px / 20 steps / CFG 4.5; both students use 384px / 8 steps / CFG 1. Seeds do not imply identical noise across architectures.</p>
<p>The fast candidate often keeps recognizable compositions. It loses some texture and color relative to guided Inko. The exact-vs-tiny student columns isolate the decoder substitution: softness is not solely a CFG issue. Wrong subjects, bad anatomy, duplicated people and pseudo-text remain; the spacecraft/robot confusion is visible even in the original model.</p>
"""
    overview += report.gallery(selected, VARIANTS)
    overview += """
<h2>Training diagnostics</h2>
<figure><img src="assets/training.svg" alt="Development latent error for dense and pruned models"><figcaption>Four development captions, using their teacher trajectories and seeds. Latent error is a regression diagnostic, not a perceptual-quality or prompt-coherence score. More training did not rescue the pruned model.</figcaption></figure>
<figure><img src="assets/decoder-training.svg" alt="Tiny decoder validation PSNR and SSIM over training"><figcaption>The tiny decoder used 312 training examples and 32 separate validation examples. Those splits apply to decoder training; the transformer saw the development captions.</figcaption></figure>
"""
    overview += report.figure(OUTPUT / "small/evaluation/checkpoint-6000.jpg",
                              "Rejected 244M model after 6000 recovery updates, alongside its teacher targets. Development images, not evaluation captions.", width=900)
    overview += """
<h2>Runtime, memory and numerical checks</h2>
<figure><img src="assets/cpu.svg" alt="Laptop CPU duration and per-stage memory"><figcaption>Separate four-thread laptop runs, with thermal/contention variation. The 480 MiB RLIMIT_DATA is an allocation limit, not an RSS or swap cap; it is deliberately not drawn as an RSS threshold. Child peaks are not added because stages do not overlap.</figcaption></figure>
<p>The standalone 384px CFG-1 bundle completed in 45.92 s: text 127.3 MiB, sampling 212.3 MiB and decoding 248.1 MiB peak RSS. CFG 2 took 94.52 s with a similar largest child peak. The final 512px run completed in about 78 s, with 277.2 MiB sampling and 397.0 MiB decoding RSS.</p>
<p>Portable transformer output matched an independently generated FP32 fixture to about 1.97×10⁻⁶ relative RMSE at 384px. Exact-decoder bounded validation covered 256, 384 and 512px; its FP16 storage introduces small numerical differences. The application’s 15 tests and research suite’s 27 tests passed at the model handoff. Numeric parity proves implementation agreement, not aesthetic quality or Pi performance.</p>
<h2>Failures that changed the implementation</h2>
<ul><li>Initial decoder exports exceeded the allocation limit. Fused normalization, split graphs, complete local halos and fewer copies bounded their workspaces.</li>
<li>NCNN residual operations required an owned input tensor; borrowing NumPy storage caused a crash in that decoder path.</li>
<li>Unused NumPy BLAS threads reserved enough virtual data to interfere with RLIMIT_DATA. NCNN keeps four threads; the unused BLAS pool is limited to one.</li>
<li>An inference-mode RoPE cache interrupted dense training after checkpoint 1500. The cache was fixed and regression-tested. Training resumed with a fresh optimizer at LR 5×10⁻⁵, after initial LR 10⁻⁴; this discontinuity is recorded.</li>
<li>The first final-checkpoint 512px trial failed on sampling pass four. Bounding NCNN transposed-weight packing to 2048 output rows fixed the observed allocation failure without changing numerical agreement.</li></ul>
<h2>CFG feedback: no universal winner</h2>
<p>Drew’s informal visual preference was approximately <strong>60% for CFG 2 and 40% for CFG 1</strong>. This was not a formal scored evaluation and no denominator was supplied. CFG 1.5 was suggested as a compromise; it has not been tested. No sampling settings, model weights or bundle defaults were changed in response.</p>
<p>Guidance and steps are independent controls. At eight steps our sequential runtime uses eight transformer evaluations for CFG 1 and sixteen for any tested CFG above 1. Going from CFG 2 to 4.5 changes the mixture strength, not the evaluation count. The student already tries to imitate a CFG-4.5 teacher, and its empty-prompt output was not explicitly trained in the distillation objective.</p>
"""
    cfg_variants = [(f"CFG {g} / 8 steps", f"q4-distilled-3000-gptq-384px-8s-cfg{g}-tiny") for g in ["1", "2", "3", "4.5"]]
    overview += report.gallery([r for r in records if r["variant"] in {v for _, v in cfg_variants} and r["split"] == "evaluation"], cfg_variants)
    overview += f"""
<h2>Cleanup and retained evidence</h2>
<p>The cleanup recovered <strong>{cleanup['free_space_increase_bytes']/2**30:.2f} GiB</strong> by removing superseded bundles, regenerable export scratch, intermediate merged checkpoints and older optimizer states. Dense checkpoint merges remain reconstructable from the retained adapters and original base model. All 117 loose experiment logs were compressed into one checksum-verified local archive.</p>
<p>Original downloads, production/deployment evidence, current bundles and archives, final dense and rejected-model checkpoints, latest resume states, captions, trajectories, images, metrics and the Supra edit tensor were kept. Model weights, raw logs and large artifacts remain excluded from Git. The complete local deletion inventory is <code>artifacts/edge-quality/cleanup-2026-11-26.json</code>; the compact committed summary is <a href="data/edge-quality-cleanup.json">here</a>.</p>
<h2>What remains before production</h2>
<p>Run an isolated complete-image trial on the actual ARMv7 Pi, with its real NCNN/NumPy/regex packages, system memory pressure, storage speed and power supply. Measure completed latency and inspect the output before integrating the backend into the worker. Verify services and HDMI after an authorized deployment through the production branch. More coherence work and a broader independent evaluation remain useful regardless of hardware fit.</p>
"""
    report.page("index.html", "A better generator within a small memory budget",
                "What we tried, what failed, and what the compressed Inko candidate actually demonstrates.", overview)

    edit_rows = read(OUTPUT / "supra-edit/records.json")
    protocol = read(OUTPUT / "supra-edit/protocol.json")
    assert len(edit_rows) == 80
    restored = sha256(OUTPUT / "supra-edit/restored-control.png")
    original = sha256(OUTPUT / "supra-edit/ship-0-baseline.png")
    assert restored == original
    edit_variants = [(v, v) for v in protocol.get("variants", dict.fromkeys(r["variant"] for r in edit_rows))]
    weight = """
<div class="note"><strong>Negative result for one specific hypothesis.</strong> This experiment did not establish a useful repair for Supra2. It does not establish that direct weight programming in general cannot work.</div>
<h2>The hypothesis</h2>
<p>Supra2 often produces unwanted pseudo-lettering. We tested whether conditioning directions associated with typography contributed to that behavior, and whether suppressing those directions directly in the text-conditioning projection could reduce it without retraining the generator.</p>
<p>The target was only <code>ctx_proj.weight</code>. The image generator, T5 encoder, decoder, prompts, seeds and sampler were otherwise held fixed within this experiment. No gradient updates or runtime hooks were used for the edit; the modified projection itself carried the change.</p>
<h2>Constructing and applying the edit</h2>
<ol><li>Encode twelve paired phrases with the existing Flan-T5 encoder: for example “printed letters” versus “plain colors”, and “typography and lettering” versus “texture and brushstrokes”.</li>
<li>Average valid token embeddings for each phrase, excluding padding. Subtract each pair to obtain twelve candidate semantic directions.</li>
<li>Use singular value decomposition of the stacked differences to obtain an orthonormal basis. Keep the leading one or four directions.</li>
<li>Project the selected directions out of the conditioning projection, at strengths α = 0.5 and α = 1.0.</li></ol>
<pre>W′ = W − α (W Uᵣ) Uᵣᵀ

W     original ctx_proj.weight
Uᵣ    orthonormal conditioning directions, rank r
α     suppression strength</pre>
<p>This compiles a small semantic-subspace ablation into one matrix. It is not automatic synthesis of a program in weight space, a demonstration of a localized typography circuit, or a comprehensive replication of WeightCode.</p>
<h2>Protocol and controls</h2>
<p>Eight development captions × two seeds × five conditions produced 80 comparison images. Every condition used 256px, 25 sampling steps, CFG 3 and the same pretrained TAESD decoder. Conditions were the original baseline, rank 1 at 0.5 and 1.0 strength, and rank 4 at 0.5 and 1.0 strength. Evaluation captions were not used to select the typography directions.</p>
"""
    weight += '<div class="scroll"><table><thead><tr><th>Condition</th><th>Rank</th><th>Strength</th><th>Weight-change Frobenius norm</th></tr></thead><tbody>'
    for label, variant in edit_variants:
        row = next(r for r in edit_rows if r["variant"] == variant)
        weight += f"<tr><td>{esc(label)}</td><td>{row['rank']}</td><td>{row['strength']}</td><td>{row['edit_frobenius_norm']:.5f}</td></tr>"
    weight += "</tbody></table></div><p class=muted>The matrix-change norm measures edit magnitude, not image quality.</p>"
    weight += f"<p>After all edits, the original weight tensor was restored exactly. Regenerating the first ship control produced the same PNG bytes as the baseline. SHA-256: <code>{original}</code>. This checks reversibility and pipeline consistency; it is not evidence that the edit improved anything.</p>"
    weight += "<h2>Every comparison</h2><p>Both seeds and all eight development captions are included. Click a preview to inspect it. These are compressed public-report previews; original PNGs remain in local evidence.</p>"
    weight += report.gallery(edit_rows, edit_variants)
    weight += """
<h2>Result and interpretation</h2>
<p>Visual inspection did not find a convincing, consistent reduction in unwanted lettering or an improvement in scene coherence. Some images changed, which shows that the projection affects generation; image change alone is not a successful repair. No edited variant was selected or deployed.</p>
<p>The phrase contrasts also change subject and visual semantics: “a newspaper headline” versus “a landscape painting” differs in more than typography. Averaged text embeddings are a crude proxy for the internal representation that drives pseudo-text. Suppressing such a subspace can remove useful conditioning along with the target concept.</p>
<p>We did not run OCR-based scoring, a blinded preference study, a causal localization study, a multi-layer search or gradient-based model editing. The sample is too small to quantify general behavior. Pseudo-lettering may arise from the generator’s learned image distribution and decoder interaction rather than one separable conditioning direction.</p>
<h2>What would make a stronger follow-up?</h2>
<p>Use carefully matched text/no-text contrasts, localize effects with activation interventions, measure both lettering and prompt adherence, and validate on new captions before committing to a permanent edit. That would be a new experiment. This cleanup/report task did not modify or retrain the model.</p>
<h2>Reproduction and sources</h2>
<p>Implementation: <a href="source/supra_edit.py">supra_edit.py</a>. The committed <a href="weight-editing-data.json">protocol and compact image records</a> include model revisions, phrase pairs and original image hashes. The local <code>edit.safetensors</code> retains the basis, singular values and original projection tensor.</p>
<p>Motivation: <a href="https://rikiyat.github.io/blog/weight-space-programming/">weight-space programming experiments</a> and <a href="https://arxiv.org/abs/2611.25476">WeightCode</a>. These sources motivated investigation; their reported results do not establish that this image-generator repair should succeed.</p>
"""
    report.page("weight-editing.html", "Editing Supra2’s conditioning weights",
                "A targeted, gradient-free typography-subspace experiment—with unchanged controls and a negative result.", weight)
    save_json(destination / "weight-editing-data.json", dict(
        protocol=protocol, restored_sha256=restored, baseline_sha256=original,
        interpretation="No convincing visual improvement; one targeted hypothesis, not a general verdict on weight programming",
        records=[{**{k: v for k, v in r.items() if k != "image"},
                  "original_image_sha256": sha256(r["image"]), "preview": report.preview(r["image"])} for r in edit_rows],
    ))

    evolution = """
<div class="flow"><div><strong>1 · Original Inko</strong>532M generator + Qwen text encoder + Qwen VAE. Good scene quality; too much joint residency.</div><div><strong>2 · Stream the weights</strong>Mixed 4/8-bit matrices, mapped embeddings and separate native CPU stages.</div><div><strong>3 · Distill the trajectory</strong>24 guided teacher updates become eight single-pass student updates.</div><div><strong>4 · Distill the decoder</strong>New 16-channel tiny decoder replaces the heavy original decoder.</div></div>
<h2>Start with the full pipeline, not only the generator</h2>
<p>Starting points: the pinned <a href="https://huggingface.co/kaupane/Inko/tree/2f690976f5ce9aadad7c76f786d3c503f86795e4">Inko checkpoint</a> and <a href="https://github.com/Mtrya/inko/tree/9f8f79cd7491bf96b13d43f290cc9a281afd5787">upstream implementation</a>. These are upstream components, not models trained from scratch in this project.</p>
<p>At BF16, the original generator stored about 1015.66 MiB for 532,496,992 parameters. The 596M-parameter Qwen text encoder added 1136.88 MiB and the 126.9M-parameter VAE added 242.03 MiB. Their sum, 2394.56 MiB, excludes activations and runtime workspaces. Reducing output resolution alone could not solve this weight residency problem.</p>
<p>The released Inko recipe used 20 Euler steps and CFG 4.5. We used its existing first-20-layer Qwen conditioning convention, rather than inventing a new text representation. Caption preparation remains separate from model generation.</p>
<h2>Pack weights and make residency explicit</h2>
<p>Transformer matrices are quantized in groups of 64 input columns. Most block matrices use four bits; sensitive modulation matrices use eight. Small projections, biases and norms stay FP32. Groupwise Hessian error feedback, calibrated on real development trajectories, compensates some activation error. This is a bounded groupwise adaptation of <a href="https://arxiv.org/abs/2210.17323">GPTQ</a>, not full-Hessian GPTQ.</p>
<p>The packed transformer occupies about 361.7 MiB on disk. The Qwen text assets occupy about 626.6 MiB, including a 296.75 MiB mapped embedding table from which only requested rows are gathered. Disk size is not peak process RAM.</p>
<p>At runtime we map the packed arrays, expand only the current matrix, execute optimized FP32 NCNN GEMM, and release unused workspace/mapped pages. This saves residency; it does not imply int4 arithmetic or a guaranteed speedup. Chunked attention avoids materializing all heads’ complete attention matrices together.</p>
<h2>Keep a faithful decoder route first</h2>
<p>For one image frame, a causal temporal convolution uses only its final temporal slice. We specialized the Qwen decoder to the first frame’s exact 2D operations and skipped temporal resampling that the original first-frame path also skips. This retained about 25.3M effective parameters, stored in 48.3 MiB FP16 NCNN weights.</p>
<p>Split graphs and complete row halos bound local convolution workspaces. Global attention still sees the full frame. This differs from approximate tiling of an entire attention-containing VAE. This decoder remains in the bundle as the quality fallback.</p>
<h2>Teach eight steps to imitate twenty-four</h2>
<p>We captured 192 short public captions from a pinned revision of <a href="https://huggingface.co/datasets/kaupane/inko-rollout-25k">Inko’s rollout caption dataset</a>, added 144 handwritten scene combinations and eight development captions: 344 prompts total. No external image dataset was downloaded. The teacher generated all training trajectories locally at 384px.</p>
<p>The teacher took 24 steps with CFG 4.5. Each group of three teacher intervals supplied an average velocity target for one of eight coarse student intervals. Rank-32 adapters were trained across the transformer’s linear layers: 19,752,960 trainable parameters, merged into the dense model after training.</p>
<p>After 3000 updates, mean rollout latent relative RMSE on four development captions fell from 0.6934 to 0.4775. The adapter was selected on that development evidence, then quantized and calibrated again for its own activations. The numerical loss does not measure coherence. The fixed-grid teacher-state regression also does not guarantee the student behaves well on its own off-trajectory states.</p>
<p>The run resumed after checkpoint 1500 with a fresh optimizer following a corrected inference-cache bug. This is recorded explicitly rather than presented as one uninterrupted training run.</p>
<h2>Train a decoder for Inko’s actual latent space</h2>
<p>Stock TAESD expects four-channel Stable Diffusion latents, while Inko uses sixteen channels with different semantics. It is not a drop-in substitute. We expanded the input convolution, initialized from TAESD, and trained all decoder weights against images from the original Qwen VAE.</p>
<p>The 1,229,443-parameter decoder completed 6000 updates using L1, local-window SSIM and finite-difference terms, with EMA weights. A 312/32 train/validation split gave mean PSNR 25.05 dB and SSIM 0.7996. It stores about 2.35 MiB in NCNN FP16 weights. Fine texture and colors can still change, especially beyond its small 384px teacher-data distribution.</p>
<pre>Transformer output: normalized Inko latent z, shape [1, 16, H/8, W/8]

Exact Qwen decoder input: z × original_std + original_mean
Tiny Qwen decoder input:  z directly

Do not apply Stable Diffusion’s 0.18215 scale to this decoder.</pre>
<h2>What the final version actually is</h2>
<div class="scroll"><table><thead><tr><th>Component</th><th>Final fast candidate</th><th>Quality fallback</th></tr></thead><tbody>
<tr><td>Generator architecture</td><td>Dense 532M, merged 3000-update adapter</td><td>Original dense 532M</td></tr>
<tr><td>Weight storage</td><td>Mixed 4/8-bit with activation-aware correction</td><td>Same quantization method, original weights</td></tr>
<tr><td>Default sampling</td><td>384px, 8 steps, CFG 1</td><td>384px, 20 steps, CFG 4.5</td></tr>
<tr><td>Evaluations per image</td><td>8</td><td>40</td></tr>
<tr><td>Text encoder</td><td colspan=2>Qwen layer-20 features; 8-bit weights and gathered mapped embeddings</td></tr>
<tr><td>Decoder</td><td>Locally trained 16-channel Tiny Qwen</td><td>Exact first-frame Qwen specialization</td></tr>
<tr><td>Execution</td><td colspan=2>Fresh sequential NumPy/NCNN processes; no Torch at runtime</td></tr>
</tbody></table></div>
<p>The architecture did not become a 244M model: that pruning attempt failed. “Smaller” here means packed storage, lower residency and fewer forward evaluations, plus a much smaller trained decoder. The standalone bundle is still about 1.1 GiB on disk because it includes text assets and exact decoder graphs for multiple sizes.</p>
<h2>CFG is a choice still under evaluation</h2>
<p>The distilled student runs at CFG 1 because its targets already contain the teacher’s CFG-4.5 guidance. Additional CFG 2, 3 and 4.5 were tested afterwards; they require a second sequential model evaluation each step. Drew preferred CFG 2 in roughly 60% of cases and CFG 1 in roughly 40%. CFG 1.5 is an untested suggestion. The default and weights remain unchanged.</p>
<h2>What is worth publishing?</h2>
<ul><li><strong>The dense eight-step adapter and merged checkpoint:</strong> publish both for convenient use and reproducibility, with the exact base revision, eight-step training recipe and limitations. The adapter is about 75.4 MiB locally; it is smaller than the merged checkpoint but does not shrink runtime parameter count.</li>
<li><strong>Tiny Qwen decoder as a separate artifact:</strong> its sixteen-channel normalized-latent contract and training recipe are independently useful. Clearly identify it as a lossy, locally trained TAESD-derived decoder for this Inko latent space, rather than a general Qwen or SD replacement.</li>
<li><strong>The packed weights and portable runtime:</strong> the weight format and NCNN/NumPy runner belong together. GitHub is suitable for code, reports and compact metrics; a model host is suitable for larger arrays and checkpoint files. Label current CPU results as laptop measurements until ARMv7 validation is complete.</li>
<li><strong>The exact decoder export and negative results:</strong> the one-frame specialization, workspace work and failed pruning/weight-edit experiments are useful engineering documentation. Rejected model checkpoints need not be promoted as releases.</li></ul>
<p>No Hugging Face upload was made. Before a weight release, retain the base/encoder/decoder revisions and attribution, model-card settings, checkpoint hashes and evaluation limitations. The pinned local Inko card does not specify a license tag; resolve and document the upstream weight terms rather than assuming a source-code license covers every component. <a href="https://huggingface.co/madebyollin/taesd/blob/main/README.md">TAESD’s card</a> declares MIT, while the other components have their own notices. Caption-data provenance is recorded; no public caption corpus is redistributed in this Git report.</p>
<h2>Reproduce or resume without the clutter</h2>
<p>The final dense checkpoint, all dense adapter checkpoints, final optimizer state, tiny decoder checkpoints, packed models, training trajectories, and exact runtime graphs remain local. Intermediate dense merges can be reconstructed from the original base plus each adapter; discarded exporter scratch can be regenerated from source. Retained final artifacts and original downloads remain outside Git.</p>
<p>Follow the commands in <a href="methods.txt">the experiment README</a>. The current offline bundle includes a checksummed default recipe and the exact decoder fallback. The next deployment gate is a completed image on the actual Pi under its real memory and storage constraints, followed by worker integration and HDMI verification.</p>
"""
    report.page("inko-evolution.html", "From original Inko to the eight-pass candidate",
                "The chain of changes, what each one saves, and which compromises remain visible.", evolution)

    files = {str(p.relative_to(destination)): dict(bytes=p.stat().st_size, sha256=sha256(p))
             for p in sorted(destination.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    if any(meta["bytes"] > 200_000 for meta in files.values()):
        raise ValueError("A report file exceeds the 200 kB commit budget")
    if sum(meta["bytes"] for meta in files.values()) > 3_000_000:
        raise ValueError("Reports exceed the 3 MB total commit budget")
    save_json(destination / "manifest.json", dict(
        format="news-art-research-reports-v1", report_date="2026-11-26",
        evidence=report.assets, files=files, total_bytes=sum(m["bytes"] for m in files.values()),
        settings_changed=False, model_weights_included=False,
    ))
    print(json.dumps(dict(destination=str(destination),files=len(files),
                          total_bytes=sum(m["bytes"] for m in files.values()),
                          largest_file_bytes=max(m["bytes"] for m in files.values()))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
