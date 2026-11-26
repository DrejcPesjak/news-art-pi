"""Build a local comparison report from saved images and measured records."""
import argparse
import html
import json
import os
from pathlib import Path

from .common import OUTPUT,prompts,save_json


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--student-variant")
    args=parser.parse_args()
    destination=OUTPUT/"index.html"
    def link(path):
        return html.escape(os.path.relpath(path,destination.parent),quote=True)
    def e(value):
        return html.escape(str(value))
    records=read(OUTPUT/"inko/records.json",[])
    supra=read(OUTPUT/"supra-evaluation/records.json",[])
    selected=[]
    for row in supra:
        if row["variant"]=="baseline":
            selected.append(dict(row,variant="Supra2 / 256px / 25 steps / CFG 3"))
    variants=["bf16-384px-20s-cfg4.5","q4-gptq-384px-20s-cfg4.5"]
    if args.student_variant:
        variants.append(args.student_variant)
    selected.extend(row for row in records if row["split"]=="evaluation" and row["variant"] in variants)
    variants=["Supra2 / 256px / 25 steps / CFG 3"]+variants
    rows=["<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>Inko on a small Pi — laptop research</title><style>body{font:16px system-ui;max-width:1400px;margin:28px auto;padding:18px;background:#eee9df;color:#20201d}h1,h2{line-height:1.2}a{color:#214e6c}table{border-collapse:collapse}th,td{padding:9px;vertical-align:top;border-bottom:1px solid #ccc}th{font-size:13px;text-align:left}.scroll{overflow-x:auto}td img{width:224px;height:224px;object-fit:contain;background:#ddd}.prompt{min-width:160px;max-width:210px}figure{margin:20px 0}figure img{max-width:100%;height:auto}.metric{background:#faf7f0;padding:16px;border-radius:8px}pre{white-space:pre-wrap;font-size:13px}small{display:block;max-width:224px}</style>",
        "<h1>Inko on a small Pi</h1><p>Laptop experiments started 25 November 2026. The Pi was disconnected. Production services, configuration and models were not changed.</p>",
        "<p>Compare actual outputs below. These are small engineering samples, with two seeds; source models and resolutions use different compute. Exact evaluation captions were excluded from training; their subject families appear in synthetic training captions. Same seed does not mean the same noise across architectures. Subject omissions, anatomy errors and pseudo-lettering remain.</p>",
        f"<p><a href='{link(Path(__file__).parent/'README.md')}'>Methods and reproduction</a> · <a href='{link(OUTPUT/'inko/records.json')}'>Image records</a> · <a href='{link(OUTPUT/'fallback-selection.json')}'>Frozen fallback recipe</a></p>"]
    rows.append("<p>The selected fast candidate uses the dense 3000-update checkpoint, eight conditional passes, mixed 4/8-bit weights and a separately trained 6000-update tiny decoder. The 244M pruned model failed recovery and is rejected. Guided quantized Inko retains more texture and remains the quality fallback.</p>")
    for seed in (0,1):
        rows.append(f"<p><a href='{link(OUTPUT/f'final-evaluation-seed-{seed}.jpg')}'>Four-way Inko comparison, seed {seed}, including exact and tiny student decoders</a></p>")
    if (OUTPUT/"cfg-sweep-summary.json").exists():
        rows.append("<h2>Additional CFG sweep</h2><p>The same eight-step student and tiny decoder at CFG 1, 2, 3 and 4.5. CFG 2 restores contrast/color in many samples; higher settings do not reliably repair coherence. These are evaluation-caption diagnostics, not a new independent test set.</p>")
        for seed in (0,1):
            rows.append(f"<p><a href='{link(OUTPUT/f'cfg-sweep-seed-{seed}.jpg')}'>CFG comparison, seed {seed}</a></p>")
    for seed in (0,1):
        rows.append(f"<h2>Evaluation scenes — seed {seed}</h2><div class='scroll'><table><tr><th>Prompt</th>"+"".join(f"<th>{e(v)}</th>" for v in variants)+"</tr>")
        for scene in prompts("evaluation"):
            rows.append(f"<tr><td class='prompt'>{e(scene['prompt'])}</td>")
            for variant in variants:
                record=next((r for r in selected if r["seed"]==seed and r["id"]==scene["id"] and r["variant"]==variant),None)
                if record:
                    path=Path(record["image"])
                    rows.append(f"<td><a href='{link(path)}'><img loading='lazy' src='{link(path)}' alt='{e(scene['id'])}'></a></td>")
                else:
                    rows.append("<td>Not evaluated</td>")
            rows.append("</tr>")
        rows.append("</table></div>")
    rows.append("<h2>Complete bounded CPU runs</h2><p>Each listed run completed all stages with no PyTorch, Transformers or Diffusers loaded. RSS is per process; the 480 MiB allocation limit is not an RSS or swap cap. Laptop timings were collected during other experiments and do not predict Pi speed.</p><div class='scroll'><table><tr><th>Run</th><th>Image</th><th>Text RSS</th><th>Sample RSS</th><th>Decode RSS</th><th>Total time</th></tr>")
    native=[]
    for directory in sorted(OUTPUT.iterdir()):
        if not directory.is_dir():
            continue
        report=read(directory/"result.json")
        if not report or "total_seconds" not in report or not all(stage in report.get("stages",{}) for stage in ("encode","sample","decode")):
            continue
        stages=report["stages"]
        if any(stages[stage].get("allocation_limit_mib")!=480 or stages[stage].get("forbidden_imports_loaded") for stage in stages):
            continue
        image=directory/"image.png"
        if not image.exists():
            continue
        native.append(dict(name=directory.name,report=str(directory/"result.json"),**report))
        cells="".join(f"<td>{stages[stage]['peak_rss_mib']:.1f} MiB</td>" for stage in ("encode","sample","decode"))
        rows.append(f"<tr><td><a href='{link(directory/'result.json')}'>{e(directory.name)}</a><small>{e(report.get('variant'))}; {report.get('size')}px; {report.get('steps')} steps; CFG {report.get('guidance')}</small></td><td><a href='{link(image)}'><img loading='lazy' src='{link(image)}' alt='CPU output'></a></td>{cells}<td>{report['total_seconds']:.1f} s</td></tr>")
    rows.append("</table></div>")
    rows.append("<h2>Weight-space edit experiment</h2><p>Typography-subspace edits to Supra2 did not convincingly repair generation. Restoring the original weights reproduced the original image byte-for-byte. This tests one specific edit, not the limits of all direct weight programming.</p>")
    for seed in (0,1):
        path=OUTPUT/"supra-edit"/f"comparison-seed-{seed}.jpg"
        if path.exists():
            rows.append(f"<p><a href='{link(path)}'>All Supra2 edit variants, seed {seed}</a></p>")
    rows.append("<h2>Distillation checkpoints</h2><p>Checkpoint figures use development scenes, not the final evaluation set. A lower velocity or latent error alone does not establish image quality.</p>")
    for experiment in ("distill","small","tiny-qwen"):
        directory=OUTPUT/experiment/"evaluation"
        metrics=[]
        for path in sorted(directory.glob("checkpoint-*.json"),key=lambda p:int(p.stem.split('-')[-1])):
            metric=read(path)
            metrics.append(metric)
            picture=path.with_suffix(".jpg")
            scalar={k:v for k,v in metric.items() if k not in ("per_prompt","examples")}
            rows.append(f"<p><a href='{link(picture)}'>{e(experiment)} checkpoint {metric['step']}</a> — {e(scalar)}</p>")
        if metrics:
            save_json(OUTPUT/f"{experiment}-checkpoint-summary.json",metrics)
    rows.append("<h2>Numerical validation</h2><p>Portable transformer and text-encoder math were checked against independently generated FP32 fixtures. The exact one-frame Qwen decoder retains full spatial attention and complete local convolution halos. A trained tiny decoder is a separate lossy option.</p><ul>")
    validation_paths=list(OUTPUT.glob("array-validation-*.json"))+list((OUTPUT/"decoder").glob("bounded-validation-*.json"))
    validation_paths.append(OUTPUT/"text/q8/validation-ncnn.json")
    for path in sorted(validation_paths):
        if path.exists():
            rows.append(f"<li><a href='{link(path)}'>{e(path.name)}</a></li>")
    rows.append("</ul><p>Physical Pi checks remain: ARMv7 dependencies, complete generation under memory pressure, actual latency, power, services and HDMI. No Pi deployment success is claimed here.</p></html>")
    destination.write_text("\n".join(rows))
    save_json(OUTPUT/"report-data.json",dict(evaluation_variants=variants,native_runs=native))
    print(destination,flush=True)


if __name__=="__main__":
    main()
