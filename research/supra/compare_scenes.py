"""Compare actual automatic scene prompts with old prompts on saved headlines.

Laptop CUDA experiment only. Fixed seeds within each row isolate prompt changes;
CUDA seeds do not reproduce the Pi's NumPy initial noise.
"""
import argparse
import html
import json
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import AutoencoderTiny

from news_art.config import load
from news_art.generator import prompt_for
from news_art.scenes import scene_for
from .assets import MODEL_DIR
from .pipeline import Pipeline


def overview(output, records):
    selected = [r for r in records if r["variant"] == 2 and r["seed"] == records[0]["seed"]]
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    sheet = Image.new("RGB", (1200, 40+((len(selected)+3)//4)*350), "#f3efe6")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), "Automatically converted scenes / seed 0 / Supra2 + TAESD".replace("seed 0", f"seed {records[0]['seed']}"), font=font, fill="black")
    for i, record in enumerate(selected):
        x, y = (i%4)*300+12, (i//4)*350+40
        with Image.open(output/record["image"]) as image:
            sheet.paste(image, (x, y))
        caption = f"#{record['sequence']} / {record['rule']}\n"+"\n".join(textwrap.wrap(record["headline"], width=37))
        draw.multiline_text((x, y+262), caption, font=font, fill="black", spacing=1)
    sheet.save(output/"overview.jpg", quality=95)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headlines", type=Path, required=True, help="Saved list with sequence and headline fields")
    parser.add_argument("--sequences", type=int, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--output", type=Path, default=Path("artifacts/scene-comparison"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    all_stories = json.loads(args.headlines.read_text())
    by_sequence = {s["sequence"]: s for s in all_stories}
    stories = [by_sequence[n] for n in args.sequences]
    style = load()["generation"]["style"]
    coverage = [{**s, "rule": scene_for(s["headline"])[0],
                 "scene": scene_for(s["headline"])[1]} for s in all_stories]
    (args.output/"all-headline-scenes.json").write_text(json.dumps(coverage, indent=2)+"\n")
    pipeline = Pipeline("baseline")
    tiny = AutoencoderTiny.from_pretrained(MODEL_DIR/"tiny", local_files_only=True,
                                          torch_dtype=torch.float32).to("cuda").eval()
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 13)
    bold = ImageFont.truetype(font_path.replace(".ttf", "-Bold.ttf"), 14)
    labels = ["Old production prompt", "Headline + new style", "Automatic scene + new style"]
    records, sheets = [], []
    # Two pages per seed keep twelve cases legible at native image size.
    with torch.inference_mode():
        for seed in args.seeds:
            for start in range(0, len(stories), 6):
                page = stories[start:start+6]
                sheet = Image.new("RGB", (900, 65+len(page)*340), "#f3efe6")
                draw = ImageDraw.Draw(sheet)
                draw.text((12, 8), f"25-step Supra2 + TAESD / seed {seed}".replace("25-step", f"{args.steps}-step"), font=bold, fill="black")
                for col, label in enumerate(labels):
                    draw.text((col*300+12, 35), label, font=font, fill="black")
                for row, story in enumerate(page):
                    rule, _ = scene_for(story["headline"])
                    old = ("A realistic editorial illustration, natural lighting, detailed composition.\n"
                           f"Illustrate this news headline: {story['headline']}\nNo text or lettering.")
                    prompts = [old, story["headline"]+".\n"+style, prompt_for(story, style)]
                    for col, prompt in enumerate(prompts):
                        context, mask = pipeline.encode(prompt, 3)
                        latent = pipeline.sample(context, mask, args.steps, seed, 3)
                        decoded = tiny.decode(latent).sample
                        pixels = np.rint((decoded[0].float().cpu().numpy().clip(-1, 1)+1)*127.5)
                        image = Image.fromarray(pixels.transpose(1, 2, 0).astype(np.uint8))
                        name = f"poster-{story['sequence']:04d}-seed{seed}-variant{col}.png"
                        image.save(args.output/name)
                        x, y = col*300+12, 65+row*340
                        sheet.paste(image, (x, y))
                        caption = f"#{story['sequence']} / {rule}\n"+"\n".join(textwrap.wrap(story["headline"], width=37))
                        draw.multiline_text((x, y+262), caption, font=font, fill="black", spacing=1)
                        records.append({"sequence": story["sequence"], "headline": story["headline"],
                                        "rule": rule, "variant": col, "prompt": prompt,
                                        "seed": seed, "steps": args.steps, "guidance": 3,
                                        "decoder": "TAESD", "image": name,
                                        "tokens": len(pipeline.tokenizer.encode(prompt))})
                    print(f"seed={seed} poster={story['sequence']} rule={rule}", flush=True)
                filename = f"grid-seed{seed}-page{start//6+1}.jpg"
                sheet.save(args.output/filename, quality=95)
                sheets.append(filename)
    (args.output/"results.json").write_text(json.dumps(records, indent=2)+"\n")
    overview(args.output, records)
    gallery = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
               '<title>Automatic scene comparison</title><style>body{font:16px system-ui;background:#f3efe6;max-width:1100px;margin:30px auto;padding:16px}img{max-width:100%}pre{white-space:pre-wrap}article{padding:15px;background:white;margin:15px 0}</style>',
               '<h1>Automatic headline-to-scene conversion</h1><p>Same model, decoder, steps, guidance and initial seed across each row. The left-to-middle change isolates the prompt wrapper/style; middle-to-right isolates automatic conversion. These are laptop CUDA samples, not exact reproductions of Pi noise.</p>',
               '<p>Known topics use generic scenes: names, locations and event details may be lost. No additional model or network call. Unknown topics keep a cleaned headline. Two seeds and twelve selected cases are a small qualitative sample.</p>']
    gallery.append('<h2>New automatic scenes overview</h2><a href="overview.jpg"><img src="overview.jpg" alt="Twelve automatically converted scenes"></a>')
    for filename in sheets:
        gallery.append(f'<h2>{filename}</h2><a href="{filename}"><img src="{filename}" alt="Prompt comparison grid"></a>')
    gallery.append('<h2>Exact automatically generated prompts</h2>')
    for r in records:
        if r["variant"] == 2 and r["seed"] == args.seeds[0]:
            gallery.append(f'<article><b>#{r["sequence"]}: {html.escape(r["headline"])}</b><p>Rule: {r["rule"]}</p><pre>{html.escape(r["prompt"])}</pre></article>')
    gallery.append('<p><a href="results.json">All prompts and settings</a> · <a href="all-headline-scenes.json">Rule coverage for all supplied headlines</a></p></html>')
    (args.output/"index.html").write_text("\n".join(gallery))


if __name__ == "__main__":
    main()
