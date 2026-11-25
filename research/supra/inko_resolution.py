"""Local Inko art-style/size comparison; does not change production settings."""
import html
import json
from pathlib import Path
import statistics
import textwrap
import time

from PIL import Image, ImageDraw, ImageFont
import torch

from .alternative_compare import Inko, key

OUTPUT = Path("artifacts/inko-resolution")
STYLE = "An expressive oil painting, visible brushstrokes, rich colors."


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    source = Path("artifacts/model-comparison")
    rows = json.loads((source/"prompts.json").read_text())
    selected = [next(r for r in rows if r["feed_index"]==i and r["variant"]==0) for i in [30,27,22,10]]
    torch.set_num_threads(4)
    model = Inko()
    components = {}
    for name,module in [("generator",model.pipe.transformer),("text_encoder",model.pipe.text_encoder),("vae",model.pipe.vae)]:
        components[name] = {"parameters":sum(p.numel() for p in module.parameters()),
                            "weight_mib":sum(p.numel()*p.element_size() for p in module.parameters())/2**20}
    (OUTPUT/"components.json").write_text(json.dumps(components,indent=2)+"\n")
    print(json.dumps(components),flush=True)
    records=[]
    for size in [896,512]:
        for row in selected:
            prompt = row["prompt"].rsplit("\n",1)[0]+"\n"+STYLE
            filename = f'article-{row["feed_index"]:02d}-art-{size}.png'
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            image = model.pipe(prompt,width=size,height=size,num_inference_steps=20,guidance_scale=4.5,
                               generator=torch.Generator(device="cpu").manual_seed(0)).images[0]
            torch.cuda.synchronize()
            record = {"feed_index":row["feed_index"],"headline":row["headline"],"prompt":prompt,
                      "size":size,"steps":20,"guidance":4.5,"seed":0,"image":filename,
                      "seconds":time.perf_counter()-started,
                      "peak_cuda_allocated_mib":torch.cuda.max_memory_allocated()/2**20}
            image.save(OUTPUT/filename);records.append(record)
            (OUTPUT/"results.json").write_text(json.dumps(records,indent=2)+"\n")
            print(f'{size}px / article {row["feed_index"]}: {record["seconds"]:.2f}s, {record["peak_cuda_allocated_mib"]:.0f}MiB',flush=True)
    summary = {str(s):{"median_seconds":statistics.median(r["seconds"] for r in records if r["size"]==s),
                       "peak_cuda_allocated_mib":max(r["peak_cuda_allocated_mib"] for r in records if r["size"]==s)} for s in [896,512]}
    (OUTPUT/"summary.json").write_text(json.dumps({"components":components,"sizes":summary},indent=2)+"\n")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",13)
    grid = Image.new("RGB",(1280,1120),"#f3efe6");draw=ImageDraw.Draw(grid)
    for c,row in enumerate(selected):
        draw.multiline_text((160+c*280,10),"\n".join(textwrap.wrap(row["headline"],35)),font=font,fill="black")
    labels=["896px photo\nprevious test","896px oil painting","512px oil painting"]
    for r,label in enumerate(labels):
        y=150+r*310
        draw.multiline_text((10,y+75),label,font=font,fill="black")
        for c,row in enumerate(selected):
            if r==0:
                path=source/"inko"/(key(row["prompt"])+".png")
            else:
                size=896 if r==1 else 512
                path=OUTPUT/next(x["image"] for x in records if x["feed_index"]==row["feed_index"] and x["size"]==size)
            with Image.open(path) as image:
                grid.paste(image.resize((256,256),Image.Resampling.LANCZOS),(160+c*280,y))
    grid.save(OUTPUT/"comparison.jpg",quality=95)
    page=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Inko size and art style</title><style>body{font:17px system-ui;max-width:1280px;margin:30px auto;background:#f3efe6;padding:16px}img{max-width:100%}pre{white-space:pre-wrap}</style>',
          '<h1>Inko: 896px versus 512px, with a painting style</h1><p>Four headlines, seed 0, BF16, 20 steps, CFG 4.5. All components stay on the GPU. Top row reuses the previous photo-style outputs; the other rows share the same oil-painting prompts. Noise dimensions and resolution-dependent sampling differ between sizes, so compositions need not match.</p><a href="comparison.jpg"><img src="comparison.jpg" alt="Inko style and resolution comparison"></a>',
          '<h2>Measured laptop results</h2><pre>'+html.escape(json.dumps(summary,indent=2))+'</pre><h2>Model weights</h2><pre>'+html.escape(json.dumps(components,indent=2))+'</pre>']
    for row in selected:
        page.append('<h2>'+html.escape(row["headline"])+'</h2>')
        for record in records:
            if record["feed_index"]==row["feed_index"]:
                page.append(f'<a href="{record["image"]}">{record["size"]}px original</a> · ')
        page.append('<pre>'+html.escape(next(r["prompt"] for r in records if r["feed_index"]==row["feed_index"]))+'</pre>')
    page.append('<p><a href="results.json">All per-image measurements</a> · <a href="summary.json">Summary</a></p></html>')
    (OUTPUT/"index.html").write_text("\n".join(page))


if __name__=="__main__":main()
