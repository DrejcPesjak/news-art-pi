"""Controlled local Supra2 comparison of brief oil-painting captions."""
import html
import json
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import AutoencoderTiny

from .assets import MODEL_DIR
from .pipeline import Pipeline

# Explicitly handwritten controls, never used by the automatic converter.
REFERENCES = {
    0: "a Mexican drug lord on FBI's 10 most wanted list. oil painting",
    2: "a president and government officials. oil painting",
    5: "a rainstorm on a coast. oil painting",
    10: "a Russian cargo ship hit by drone in Baltic Sea. oil painting",
    11: "a swimmer in the sea. oil painting",
    13: "a pile of coal. oil painting",
    18: "a mobile phone with a colorful screen. oil painting",
    22: "a couple getting married. oil painting",
    27: "a soccer player holding a trophy. oil painting",
    30: "a spacecraft orbiting a planet. oil painting",
    31: "a space capsule above Earth. oil painting",
    37: "a bottle of medicine. oil painting",
}


def main():
    root = Path("artifacts/supra-brief")
    rows = json.loads((root/"prompts.json").read_text())
    labels = ["Full headline", "Auto: short, names kept", "Auto: short, names removed", "Handwritten control"]
    pipe = Pipeline("baseline")
    tiny = AutoencoderTiny.from_pretrained(MODEL_DIR/"tiny", local_files_only=True,
                                          torch_dtype=torch.float32).to("cuda").eval()
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    results, sheets = [], []
    with torch.inference_mode():
        for seed in [0, 1]:
            for start in range(0, len(rows), 4):
                sheet = Image.new("RGB", (1200, 60+4*430), "#f3efe6")
                draw = ImageDraw.Draw(sheet)
                draw.text((12, 8), f"Supra2 + TAESD / oil painting / 25 steps / CFG 3 / seed {seed}", font=font, fill="black")
                for col, label in enumerate(labels):
                    draw.text((12+300*col, 32), label, font=font, fill="black")
                for rownum, row in enumerate(rows[start:start+4]):
                    prompts = [row["headline"].rstrip(". ")+". oil painting",
                               row["short_named"]["prompt"], row["short_generic"]["prompt"],
                               REFERENCES[row["feed_index"]]]
                    for col, prompt in enumerate(prompts):
                        tokens = len(pipe.tokenizer.encode(prompt))
                        if tokens > pipe.context_length:
                            raise ValueError(f"Prompt exceeds context length: {prompt}")
                        context, mask = pipe.encode(prompt, 3)
                        latent = pipe.sample(context, mask, 25, seed, 3)
                        decoded = tiny.decode(latent).sample
                        pixels = np.rint((decoded[0].float().cpu().numpy().clip(-1, 1)+1)*127.5)
                        image = Image.fromarray(pixels.transpose(1, 2, 0).astype(np.uint8))
                        filename = f'article-{row["feed_index"]:02d}-seed{seed}-variant{col}.png'
                        image.save(root/filename)
                        x, y = 12+col*300, 60+rownum*430
                        sheet.paste(image, (x, y))
                        caption = f'#{row["feed_index"]}\n'+"\n".join(textwrap.wrap(prompt, width=35))
                        draw.multiline_text((x, y+264), caption, font=font, fill="black", spacing=2)
                        results.append(dict(feed_index=row["feed_index"], headline=row["headline"],
                                            variant=col, label=labels[col], prompt=prompt, tokens=tokens,
                                            seed=seed, steps=25, guidance=3, decoder="TAESD", image=filename))
                    print(f'seed {seed}: article {row["feed_index"]} complete', flush=True)
                filename = f'grid-seed{seed}-page{start//4+1}.jpg'
                sheet.save(root/filename, quality=95)
                sheets.append(filename)
    (root/"results.json").write_text(json.dumps(results, indent=2)+"\n")
    gallery = ['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
               '<title>Brief oil painting captions — Supra2</title><style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:15px;background:#f3efe6}img{max-width:100%}pre{white-space:pre-wrap}article{background:white;padding:16px;margin:20px 0}</style>',
               '<h1>Short scene descriptions / Supra2</h1><p>12 articles × 4 variants × 2 seeds = 96 images. Every column uses oil painting. Same seed, 256×256, 25 steps, CFG 3 and TAESD throughout. Laptop CUDA inference; not a Pi benchmark.</p>',
               '<p>Columns: full headline; automatic short caption retaining names; automatic short caption stripping identifying names; handwritten control. The last column includes your two original prompts unchanged. Other handwritten controls can add inferred illustrative objects and are NOT automatic outputs.</p>',
               '<p>The automatic prototype uses POS noun chunks, NER and WordNet, retaining at most two subjects/settings. It can lose actions and misread concrete nouns. It is a research prototype, not deployed. Article context was available for nine of twelve articles; the rest use headlines alone.</p>']
    for filename in sheets:
        gallery.append(f'<h2>{filename}</h2><a href="{filename}"><img src="{filename}" alt="Four by four controlled prompt comparison"></a>')
    gallery.append('<h2>Exact prompts and source text</h2>')
    for row in rows:
        gallery.append('<article><h3>'+html.escape(row['headline'])+'</h3>')
        gallery.append('<details><summary>Source context and automatic diagnostics</summary><pre>'+html.escape(json.dumps({k:row[k] for k in ('context','short_named','short_generic')}, indent=2))+'</pre></details>')
        for result in [r for r in results if r['feed_index']==row['feed_index'] and r['seed']==0]:
            gallery.append('<p><b>'+result['label']+'</b></p><pre>'+html.escape(result['prompt'])+'</pre>')
        gallery.append('</article>')
    gallery.append('<p><a href="results.json">Image settings / prompts</a> · <a href="prompts.json">Full converter diagnostics</a></p>')
    (root/'index.html').write_text('\n'.join(gallery))


if __name__ == '__main__':
    main()
