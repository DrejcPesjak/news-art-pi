import logging
from PIL import Image, ImageOps
from .config import ROOT
from .captions import prepare

log = logging.getLogger(__name__)


def prompt_for(story, style, fetch_context=False):
    report = prepare(story, fetch_context=fetch_context)
    scene = report["scene"].rstrip(". ")
    prompt = f"{scene}. {style.strip()}".strip()
    story["caption"] = {**report, "prompt": prompt}
    log.info("Caption (%s): %s", report["article"]["status"], prompt)
    return prompt


def generate(story, config, index, state_dir=None):
    """Return an RGB square and its prompt, using the selected local backend."""
    if config.get("backend", "demo") == "supra":
        prompt = prompt_for(story, config["style"], fetch_context=True)
    else:
        prompt = f'{story["headline"]}. {config["style"].strip()}'
    if config.get("backend", "demo") == "supra":
        from .supra import generate as generate_supra
        if state_dir is None:
            raise ValueError("Supra generation requires a writable state directory")
        return generate_supra(prompt, config, index, state_dir), prompt
    paths = sorted(p for p in (ROOT / "imgs").iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not paths:
        raise ValueError("No demo images found in imgs/")
    with Image.open(paths[index % len(paths)]) as original:
        image = ImageOps.pad(ImageOps.exif_transpose(original).convert("RGB"),
                             (config["image_size"], config["image_size"]),
                             method=Image.Resampling.LANCZOS, color="#e8e3d9")
    return image, prompt

