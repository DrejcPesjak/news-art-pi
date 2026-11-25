"""Generate laptop grids from real publisher-context/NLP probe outputs."""
import argparse
import html
import json
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import AutoencoderTiny

from news_art.scenes import scene_for
from .assets import MODEL_DIR
from .nlp_scenes import STYLE
from .pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("artifacts/context-nlp/nlp-prompts.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/context-nlp/images"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = json.loads(args.input.read_text())
    labels = ["Headline + photo style", "Topic template", "NLP: headline only", "NLP: article context"]
    pipeline = Pipeline("baseline")
    tiny = AutoencoderTiny.from_pretrained(MODEL_DIR/"tiny", local_files_only=True,
                                          torch_dtype=torch.float32).to("cuda").eval()
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    records, sheets = [], []
    with torch.inference_mode():
        for seed in args.seeds:
            for start in range(0, len(rows), 4):
                page = rows[start:start+4]
                sheet = Image.new("RGB", (1200, 60+len(page)*385), "#f3efe6")
                draw = ImageDraw.Draw(sheet)
                draw.text((12, 8), f"Supra2 + TAESD / 25 steps / CFG 3 / seed {seed}", font=font, fill="black")
                for col,label in enumerate(labels):
                    draw.text((col*300+12, 32), label, font=font, fill="black")
                for i,row in enumerate(page):
                    prompts = [row["headline"]+".\n"+STYLE, scene_for(row["headline"])[1]+"\n"+STYLE,
                               row["nlp_headline"]["prompt"], row["nlp_context"]["prompt"]]
                    for col,prompt in enumerate(prompts):
                        tokens = len(pipeline.tokenizer.encode(prompt))
                        if tokens > pipeline.context_length:
                            raise ValueError(f'Prompt {row["feed_index"]}/{col} exceeds {pipeline.context_length} tokens: {tokens}')
                        context,mask = pipeline.encode(prompt,3)
                        latent = pipeline.sample(context,mask,25,seed,3)
                        decoded = tiny.decode(latent).sample
                        pixels = np.rint((decoded[0].float().cpu().numpy().clip(-1,1)+1)*127.5)
                        image = Image.fromarray(pixels.transpose(1,2,0).astype(np.uint8))
                        filename = f'article-{row["feed_index"]:02d}-seed{seed}-variant{col}.png'
                        image.save(args.output/filename)
                        x,y = col*300+12,60+i*385
                        sheet.paste(image,(x,y))
                        caption = f'#{row["feed_index"]}: {row["context_status"]}\n'+"\n".join(textwrap.wrap(row["headline"],width=36))
                        draw.multiline_text((x,y+262),caption,font=font,fill="black",spacing=1)
                        records.append({"feed_index":row["feed_index"],"headline":row["headline"],"source":row["source"],
                                        "publisher_url":row.get("publisher_url"),"context_status":row["context_status"],
                                        "variant":col,"label":labels[col],"prompt":prompt,"tokens":tokens,
                                        "seed":seed,"steps":25,"guidance":3,"decoder":"TAESD","image":filename})
                    print(f'seed={seed} article={row["feed_index"]} complete',flush=True)
                filename = f'grid-seed{seed}-page{start//4+1}.jpg'
                sheet.save(args.output/filename,quality=95)
                sheets.append(filename)
    (args.output/"results.json").write_text(json.dumps(records,indent=2)+"\n")
    gallery = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
               '<title>Article context and NLP experiment</title><style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:15px;background:#f3efe6}img{max-width:100%}pre{white-space:pre-wrap}article{background:white;padding:18px;margin:20px 0}a{color:#934526}</style>',
               '<h1>Does article context help Supra2?</h1><p>Twelve real feed articles, four prompt variants, two seeds: 96 images. Same initial noise, model, steps, guidance and decoder across each row. GPU noise differs from the Pi. NLP selects a source sentence; it does not generate a new scene. NER entities and SVO diagnostics are retained in the source report.</p>',
               '<p>Google RSS descriptions contained related headlines. Publisher summaries/opening text were available for 9 of 12 attempts. Rows marked failed use headline fallback in the context column; they are not successful context extractions. Topic templates are a generic baseline, not the NLP converter.</p>']
    for filename in sheets:
        gallery.append(f'<h2>{filename}</h2><a href="{filename}"><img src="{filename}" alt="Four-column NLP/context comparison"></a>')
    gallery.append('<h2>Source text and actual prompts</h2>')
    for row in rows:
        url=row.get("publisher_url",row["url"])
        gallery.append(f'<article><h3>#{row["feed_index"]}: {html.escape(row["headline"])}</h3><p><a href="{html.escape(url,quote=True)}">{html.escape(row["source"])}</a> / {row["context_status"]}</p>')
        if row.get("error"):
            gallery.append('<p>'+html.escape(row["error"])+'</p>')
        gallery.append('<details><summary>Extracted subtitle/summary and opening paragraph</summary><pre>'+html.escape(row["context"] or "No article context available.")+'</pre></details>')
        for key,label in [("nlp_headline","NLP headline prompt"),("nlp_context","NLP context prompt")]:
            report=row[key]
            gallery.append('<h4>'+label+'</h4><pre>'+html.escape(report["prompt"])+'</pre>')
            gallery.append('<p>'+html.escape(f'Headline fallback: {report["headline_fallback"]}; context supplied: {report["context_used"]}')+'</p>')
        gallery.append('<details><summary>Named entities (diagnostics)</summary><pre>'+html.escape(json.dumps(row['nlp_context']['entities'],indent=2))+'</pre></details></article>')
    gallery.append('<p><a href="results.json">Every image prompt and setting</a> · <a href="../nlp-prompts.json">Full NLP/source report</a> · <a href="../nlp-metrics.json">Laptop timing and memory</a></p></html>')
    (args.output/"index.html").write_text("\n".join(gallery))


if __name__ == "__main__":
    main()
