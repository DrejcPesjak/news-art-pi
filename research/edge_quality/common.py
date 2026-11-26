"""Experiment records and static comparison sheets; no production imports."""
import hashlib
import json
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/edge-quality"


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def prompts(split=None):
    rows = json.loads((Path(__file__).parent / "prompts.json").read_text())
    return [r for r in rows if split is None or r["split"] == split]


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sheet(records, destination, variants=None, seed=0, size=192):
    selected = [r for r in records if r["seed"] == seed]
    variants = variants or list(dict.fromkeys(r["variant"] for r in selected))
    ids = list(dict.fromkeys(r["id"] for r in selected))
    lookup = {(r["id"], r["variant"]): r for r in selected}
    width = 160 + len(variants) * (size + 12)
    canvas = Image.new("RGB", (width, 65 + len(ids) * (size + 20)), "#eee9df")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    for col, variant in enumerate(variants):
        draw.multiline_text((160 + col * (size + 12), 8), "\n".join(textwrap.wrap(variant, 24)), font=font, fill="black")
    for row, identifier in enumerate(ids):
        y = 65 + row * (size + 20)
        prompt = next(r["prompt"] for r in selected if r["id"] == identifier)
        draw.multiline_text((5, y + 10), "\n".join(textwrap.wrap(identifier + ": " + prompt, 21)), font=font, fill="black")
        for col, variant in enumerate(variants):
            record = lookup.get((identifier, variant))
            if record:
                with Image.open(record["image"]) as im:
                    canvas.paste(im.convert("RGB").resize((size, size), Image.Resampling.LANCZOS), (160 + col * (size + 12), y))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
