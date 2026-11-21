from PIL import Image, ImageOps
from .config import ROOT


def prompt_for(story, style):
    return f"{style.strip()}\nIllustrate this news headline: {story['headline']}\nNo text or lettering."


def generate(story, config, index):
    """Return an RGB square and its prompt. Replace this backend with Supra later."""
    paths = sorted(p for p in (ROOT / "imgs").iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not paths:
        raise ValueError("No demo images found in imgs/")
    with Image.open(paths[index % len(paths)]) as original:
        image = ImageOps.pad(ImageOps.exif_transpose(original).convert("RGB"),
                             (config["image_size"], config["image_size"]),
                             method=Image.Resampling.LANCZOS, color="#e8e3d9")
    return image, prompt_for(story, config["style"])

