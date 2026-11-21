import argparse
import gc
import importlib.metadata
import json
import logging
from pathlib import Path
import platform

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from .assets import ROOT, REVISIONS
from .pipeline import Pipeline


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["generate", "benchmark"])
    parser.add_argument("--precision", choices=["baseline", "bf16"], default="baseline")
    parser.add_argument("--prompt", default="A sea jellyfish floating in the pitch-black ocean depths")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--output", type=Path, default=ROOT/"artifacts/supra")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "generate":
        pipeline = Pipeline(args.precision)
        _, metrics = pipeline.generate(args.prompt, args.steps, args.seed, args.guidance, args.output/"sample.png")
        metrics["load_seconds"] = pipeline.load_seconds
        metrics["components"] = pipeline.components
        save_json(args.output/"sample.json", metrics)
        print(json.dumps(metrics, indent=2), flush=True)
        return
    prompts = json.loads((ROOT/"research/supra/prompts.json").read_text())
    report = {"device": torch.cuda.get_device_name(), "python": platform.python_version(),
              "torch": torch.__version__, "revisions": REVISIONS,
              "versions": {name: importlib.metadata.version(name) for name in ["transformers", "diffusers", "numpy", "Pillow"]},
              "notes": ["Warm GPU runs; separate cold model load reported per precision.",
                        "Four fixed prompts, seed 0, CFG 3; no timing claim for ARM.",
                        "Pixel RMSE is a difference measure, not semantic image quality.",
                        "Process RSS is the process-lifetime high-water mark, not per-run RAM."],
              "loads": {}, "runs": []}
    variants = [("baseline", 50), ("bf16", 50), ("bf16", 25), ("bf16", 10), ("bf16", 5)]
    current_precision, pipeline = None, None
    for precision, steps in variants:
        if current_precision != precision:
            if pipeline is not None:
                del pipeline
                gc.collect()
                torch.cuda.empty_cache()
            pipeline = Pipeline(precision)
            current_precision = precision
            report["loads"][precision] = {"seconds": pipeline.load_seconds, "components": pipeline.components}
            pipeline.generate(prompts[0]["prompt"], steps=2)  # CUDA kernels/allocations warmup
        for prompt in prompts:
            filename = f'{precision}-{steps}-{prompt["id"]}.png'
            image, metrics = pipeline.generate(prompt["prompt"], steps=steps, output=args.output/filename)
            metrics.update(prompt_id=prompt["id"], prompt=prompt["prompt"], image=filename)
            if precision != "baseline":
                reference = np.asarray(Image.open(args.output/f'baseline-50-{prompt["id"]}.png')).astype(float)
                metrics["pixel_rmse_vs_baseline"] = float(np.sqrt(np.mean((np.asarray(image).astype(float)-reference)**2)))
            report["runs"].append(metrics)
            save_json(args.output/"benchmark.json", report)
            logging.info("%s %s: %.3fs, %.0f MiB CUDA", filename, prompt["id"], metrics["total_seconds"], metrics["peak_cuda_allocated_mib"])
    # Scientific comparison sheet; source images and full-resolution outputs stay local.
    sheet = Image.new("RGB", (len(variants)*270, len(prompts)*300+35), "#f3efe6")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    for column, (precision, steps) in enumerate(variants):
        draw.text((column*270+8, 8), f"{precision} / {steps} steps", fill="black", font=font)
        for row, prompt in enumerate(prompts):
            image = Image.open(args.output/f'{precision}-{steps}-{prompt["id"]}.png')
            sheet.paste(image, (column*270+7, row*300+35))
            draw.text((column*270+8, row*300+294), prompt["id"], fill="black", font=font)
    sheet.save(args.output/"comparison.jpg", quality=92)
    save_json(ROOT/"research/results/laptop.json", report)
    sheet.save(ROOT/"research/results/comparison.jpg", quality=90)


if __name__ == "__main__":
    main()
