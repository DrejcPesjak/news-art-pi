"""Compare full and distilled decoding of identical Supra2 latent samples."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import AutoencoderTiny

from .assets import MODEL_DIR, ROOT, REVISIONS
from .model import VAE_SCALE
from .pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/supra/decoder-comparison")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline("baseline")
    tiny = AutoencoderTiny.from_pretrained(MODEL_DIR / "tiny", local_files_only=True,
                                           torch_dtype=torch.float32).to("cuda").eval()
    prompts = json.loads((ROOT / "research/supra/prompts.json").read_text())
    sheet = Image.new("RGB", (540, len(prompts) * 300 + 35), "#f3efe6")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    for column, label in enumerate(["Original SD VAE", "Tiny distilled decoder"]):
        draw.text((column * 270 + 8, 8), f"{label} / {args.steps} steps", font=font, fill="black")
    with torch.inference_mode():
        for row, prompt in enumerate(prompts):
            context, mask = pipeline.encode(prompt["prompt"])
            latent = pipeline.sample(context, mask, args.steps, args.seed, 3)
            np.save(args.output / (prompt["id"] + "-latent.npy"), latent.cpu().numpy())
            original = pipeline.vae.decode(latent / VAE_SCALE).sample
            approximate = tiny.decode(latent).sample
            for column, decoded in enumerate([original, approximate]):
                pixels = np.rint((decoded[0].float().cpu().numpy().clip(-1, 1) + 1) * 127.5)
                image = Image.fromarray(pixels.transpose(1, 2, 0).astype(np.uint8))
                image.save(args.output / f'{prompt["id"]}-{"original" if column == 0 else "tiny"}.png')
                sheet.paste(image, (column * 270 + 7, row * 300 + 35))
                draw.text((column * 270 + 8, row * 300 + 294), prompt["id"], font=font, fill="black")
    sheet.save(args.output / "comparison.jpg", quality=90)
    metadata = {"steps": args.steps, "seed": args.seed, "guidance": 3,
                "same_latents_for_both_decoders": True, "revisions": REVISIONS,
                "original_decoder_parameters": sum(p.numel() for p in pipeline.vae.decoder.parameters()),
                "tiny_decoder_parameters": sum(p.numel() for p in tiny.decoder.parameters()),
                "note": "A lossy pretrained decoder substitution, not training or a second image generator."}
    (args.output / "comparison.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
