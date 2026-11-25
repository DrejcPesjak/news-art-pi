"""Index exact prompts, original outputs and four-model by four-prompt grids."""
import argparse
from collections import defaultdict
import html
import json
from pathlib import Path
import shutil
import textwrap

from PIL import Image, ImageDraw, ImageFont
from .alternative_compare import OUTPUT, key


MODELS = [("supra", "Supra2 + TAESD", "256px / 25 steps / CFG 3"),
          ("flux", "FLUX Turbo Alpha", "512px / NF4 / 8 steps"),
          ("inko", "Inko", "896px / 20 steps / CFG 4.5"),
          ("surjo", "Surjo Preview", "256px / 50 steps / CFG 3")]
LABELS = ["Headline + photo style", "Topic template", "NLP headline", "NLP article context"]


def contacts(rows, model, label, recipe, font):
    distinct = {}
    for row in rows:
        if model in row["outputs"]:
            distinct.setdefault(key(row["prompt"]), row)
    cases = list(distinct.values())
    filenames = []
    for start in range(0,len(cases),16):
        page = cases[start:start+16]
        image = Image.new("RGB",(1120,1360),"#f3efe6")
        draw = ImageDraw.Draw(image)
        draw.text((10,8),f'{label} / {recipe} / seed 0 / {len(cases)} of 32 distinct prompts available',font=font,fill="black")
        for i,row in enumerate(page):
            x,y = (i%4)*280+10,(i//4)*330+40
            with Image.open(OUTPUT/row["outputs"][model]) as source:
                image.paste(source.convert("RGB").resize((256,256),Image.Resampling.LANCZOS),(x,y))
            text = f'Article {row["feed_index"]} / {LABELS[row["variant"]]}\n'+"\n".join(textwrap.wrap(row["headline"],width=38)[:3])
            draw.multiline_text((x,y+260),text,font=font,fill="black",spacing=1)
        filename = f'{model}-contact-{start//16+1}.jpg'
        image.save(OUTPUT/filename,quality=95)
        filenames.append(filename)
    return filenames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    rows = json.loads((OUTPUT/"prompts.json").read_text())
    stories = defaultdict(list)
    (OUTPUT/"supra").mkdir(exist_ok=True)
    missing = []
    for row in rows:
        source = Path(row["baseline_image"])
        baseline = OUTPUT/"supra"/f'article-{row["feed_index"]:02d}-v{row["variant"]}.png'
        shutil.copyfile(source,baseline)
        row["outputs"] = {"supra":str(baseline.relative_to(OUTPUT))}
        for model,_,_ in MODELS[1:]:
            path = OUTPUT/model/(key(row["prompt"])+".png")
            if path.exists():
                row["outputs"][model] = str(path.relative_to(OUTPUT))
            else:
                missing.append(f'{model}/{row["feed_index"]}/{row["variant"]}')
        stories[row["feed_index"]].append(row)
    if missing and not args.partial:
        raise ValueError(f"Cannot publish complete gallery: {len(missing)} cells missing: {missing[:8]}")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",13)
    title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",19)
    pages = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>Four-model news prompt comparison</title><style>body{font:16px system-ui;background:#f3efe6;color:#202620;max-width:1300px;margin:30px auto;padding:18px}img{max-width:100%}a{color:#934526}section{margin:50px 0}pre{white-space:pre-wrap}table{width:100%;table-layout:fixed}td{vertical-align:top}small{font-size:12px}details{background:white;padding:15px;margin:10px 0}</style>',
             '<h1>News prompts: four image generators</h1><p>12 articles × 4 unchanged prompt formats × 4 models = 192 comparison cells, seed 0. There are 32 distinct prompt strings; identical prompt/seed results are reused within each model. Each plot is a 4×4 grid: models are rows, prompt formats are columns.</p>',
             '<p>Supra2 outputs come from the previous experiment. Surjo uses its original decoder, 50 steps at 256px; Inko uses 20 steps at its default 896px. FLUX Turbo is the requested adapter on a Diffusers NF4 FLUX.1-dev base, 8 steps at 512px to fit laptop memory, with NF4 T5 conditioning. FLUX is not a Pi-sized model. These are different recommended/practical recipes, not an equal-compute benchmark. Same seed does not mean identical noise across architectures.</p>',
             '<p>Grid tiles are resized to 256px for display. Click individual images below each grid for original resolution. All prompts and sampling settings are saved; no model or prompt was substituted silently. No Pi deployment.</p>']
    pages.append('<p>Publisher context was unavailable for three articles; those context-column prompts use the previous headline fallback. NLP parsing can also fall back even when context is available. Exact strings are preserved from the previous experiment.</p>')
    pages.append('<p>Model sources: <a href="https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha">FLUX Turbo Alpha</a> · <a href="https://huggingface.co/kaupane/Inko">Inko</a> · <a href="https://huggingface.co/SurjoLabs/Surjo-Image-Preview">Surjo Image Preview</a> · <a href="https://huggingface.co/SupraLabs/Supra2-IMG">Supra2 baseline</a>.</p>')
    summary_path = OUTPUT/"summary.json"
    if summary_path.exists() and not missing:
        summary = json.loads(summary_path.read_text())
        pages.append('<details><summary>Measured laptop timings and GPU memory</summary><table><tr><th>Model</th><th>Resolution</th><th>Median inference</th><th>Allocated GPU peak</th></tr>')
        for model,label,_ in MODELS[1:]:
            stats = summary["models"][model]
            pages.append(f'<tr><td>{label}</td><td>{stats["resolution"]}²</td><td>{stats["median_seconds"]:.2f} s</td><td>{stats["max_cuda_allocated_mib"]:.0f} MiB</td></tr>')
        pages.append('</table><p>Different resolutions and recipes; not a controlled speed ranking. The GPU warmed and reduced clocks during sustained runs. Model loading is excluded. FLUX stages run separately; the GPU peak shown is sampling. GPU memory is not a Pi RAM estimate.</p><a href="summary.json">Full measurement summary</a></details>')
    if missing:
        pages.append(f'<p><b>IN PROGRESS: {len(missing)} model/prompt cells still missing.</b></p>')
    pages.append('<nav><b>Articles:</b> '+" · ".join(f'<a href="#article-{i}">#{i}</a>' for i in stories)+'</nav>')
    pages.append('<details><summary>4×4 contact sheets grouped by model (distinct prompts)</summary>')
    for model,label,recipe in MODELS:
        for filename in contacts(rows,model,label,recipe,font):
            pages.append(f'<h3>{label}</h3><a href="{filename}"><img src="{filename}" alt="Sixteen prompts for {html.escape(label)}"></a>')
    pages.append('</details>')
    for number,cases in stories.items():
        cases.sort(key=lambda row:row["variant"])
        headline=cases[0]["headline"]
        grid=Image.new("RGB",(1280,1340),"#f3efe6")
        draw=ImageDraw.Draw(grid)
        draw.multiline_text((12,8),f'#{number}: '+"\n".join(textwrap.wrap(headline,width=95)),font=title_font,fill="black")
        context_status = cases[0]["context_status"]
        draw.text((12,65),f'{cases[0]["source"]} / publisher context: {context_status} / seed 0',font=font,fill="#657066")
        for col,label in enumerate(LABELS):
            if col == 3 and context_status != "available":
                label = "Context unavailable: headline"
            draw.text((160+col*280,85),label,font=font,fill="black")
        for r,(model,label,recipe) in enumerate(MODELS):
            y=115+r*300
            draw.multiline_text((10,y+70),"\n".join(textwrap.wrap(label,18))+"\n\n"+"\n".join(textwrap.wrap(recipe,19)),font=font,fill="black")
            for col,case in enumerate(cases):
                x=160+col*280
                path=case["outputs"].get(model)
                if path:
                    with Image.open(OUTPUT/path) as source:
                        grid.paste(source.convert("RGB").resize((256,256),Image.Resampling.LANCZOS),(x,y))
                else:
                    draw.rectangle((x,y,x+256,y+256),fill="#dedbd3")
                    draw.text((x+70,y+120),"PENDING",font=font,fill="black")
        filename=f'article-{number:02d}-grid.jpg'
        grid.save(OUTPUT/filename,quality=95)
        pages.append(f'<section id="article-{number}"><h2>#{number}: {html.escape(headline)}</h2><a href="{filename}"><img src="{filename}" alt="4 by 4 comparison grid"></a>')
        pages.append(f'<p>Publisher context: {context_status}.</p>')
        pages.append('<details><summary>Original-resolution images and recorded settings</summary><table><tr><th>Model</th>'+"".join('<th>'+label+'</th>' for label in LABELS)+'</tr>')
        for model,label,_ in MODELS:
            pages.append('<tr><th>'+label+'</th>')
            for case in cases:
                path=case['outputs'].get(model)
                record=OUTPUT/model/(key(case['prompt'])+'.json')
                if path:
                    pages.append(f'<td><a href="{path}"><img src="{path}" alt="{html.escape(label)} output"></a>')
                    if record.exists():
                        relative=record.relative_to(OUTPUT)
                        pages.append(f'<small><a href="{relative}">Settings / timing</a></small>')
                    pages.append('</td>')
                else:
                    pages.append('<td>Pending</td>')
            pages.append('</tr>')
        pages.append('</table></details><details><summary>Exact prompts (unchanged across models)</summary>')
        for case in cases:
            pages.append(f'<h3>{LABELS[case["variant"]]}</h3><pre>{html.escape(case["prompt"])}</pre>')
        pages.append('</details></section>')
    pages.append('<p><a href="comparison.json">Complete comparison index</a></p></html>')
    (OUTPUT/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
    (OUTPUT/'index.html').write_text("\n".join(pages))
    print(f"Built {len(stories)} four-by-four grids; {len(missing)} pending cells")


if __name__ == '__main__':
    main()
