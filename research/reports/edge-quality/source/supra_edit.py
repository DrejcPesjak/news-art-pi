"""Gradient-free semantic-subspace weight edit, with unedited/restore controls.

This tests a specific hypothesis: typography-related conditioning directions
contribute to unwanted lettering. It does not claim to locate a text circuit.
The edit is compiled into ctx_proj.weight; no encoder or runtime change is needed.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import AutoencoderTiny
from safetensors.torch import save_file

from research.supra.assets import MODEL_DIR, REVISIONS
from research.supra.pipeline import Pipeline
from .common import OUTPUT, prompts, save_json, sheet, sha256

PAIRS = [
    ("printed letters", "plain colors"),
    ("written words", "simple shapes"),
    ("a newspaper headline", "a landscape painting"),
    ("typography and lettering", "texture and brushstrokes"),
    ("a poster covered with text", "a canvas covered with paint"),
    ("a page of writing", "a view of the sea"),
    ("a sign with printed words", "a tree with green leaves"),
    ("a logo and a caption", "a cloud and a mountain"),
    ("rows of alphabet characters", "rows of colorful flowers"),
    ("a title and subtitles", "a sunrise and shadows"),
    ("a magazine cover with lettering", "a garden filled with flowers"),
    ("black text on a white background", "blue water under a clear sky"),
]


@torch.inference_mode()
def directions(pipe):
    vectors = []
    for positive, negative in PAIRS:
        means = []
        for phrase in (positive, negative):
            context, mask = pipe.encode("An image of " + phrase + ".", guidance=1)
            # Include all valid tokens; exclude padding, without selecting image cases.
            means.append((context.float() * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True))
        vectors.append((means[0] - means[1])[0])
    _, singular, vh = torch.linalg.svd(torch.stack(vectors).float(), full_matrices=False)
    return vh.T.contiguous(), singular


@torch.inference_mode()
def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    pipe = Pipeline("bf16")
    tiny = AutoencoderTiny.from_pretrained(MODEL_DIR / "tiny", local_files_only=True,
                                           torch_dtype=torch.bfloat16).to("cuda").eval()
    tiny.encoder = None
    basis, singular = directions(pipe)
    original = pipe.model.ctx_proj.weight.detach().clone()
    save_file({"basis": basis.cpu(), "singular_values": singular.cpu(),
               "original_ctx_proj_weight": original.cpu()}, str(output / "edit.safetensors"))
    variants = [("baseline", 0, 0), ("erase-r1-a0.5", 1, .5), ("erase-r1-a1", 1, 1),
                ("erase-r4-a0.5", 4, .5), ("erase-r4-a1", 4, 1)]
    if args.baseline_only:
        variants=variants[:1]
    records = json.loads((output / "records.json").read_text()) if (output / "records.json").exists() else []
    rows = prompts(args.split)
    for variant, rank, strength in variants:
        u = basis[:, :rank]
        edited = original.float() - strength * (original.float() @ u) @ u.T
        pipe.model.ctx_proj.weight.copy_(edited.to(original.dtype))
        delta = (pipe.model.ctx_proj.weight.float() - original.float()).norm().item()
        for row in rows:
            for seed in args.seeds:
                image_path = output / f'{row["id"]}-{seed}-{variant}.png'
                if any(r["image"] == str(image_path) for r in records):
                    continue
                torch.cuda.synchronize()
                start = time.perf_counter()
                context, mask = pipe.encode(row["prompt"], 3)
                latent = pipe.sample(context, mask, 25, seed, 3)
                with pipe.autocast():
                    decoded = tiny.decode(latent.to(torch.bfloat16)).sample
                from PIL import Image
                pixels = ((decoded[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
                Image.fromarray(pixels).save(image_path)
                record = dict(row, seed=seed, variant=variant, image=str(image_path),
                              seconds=time.perf_counter()-start, edit_frobenius_norm=delta,
                              rank=rank, strength=strength, decoder="TAESD", steps=25, guidance=3)
                records.append(record)
                save_json(output / "records.json", records)
                print(json.dumps({k:record[k] for k in ("id", "seed", "variant", "seconds")}), flush=True)
    pipe.model.ctx_proj.weight.copy_(original)
    assert torch.equal(pipe.model.ctx_proj.weight, original)
    row = rows[0]
    context, mask = pipe.encode(row["prompt"], 3)
    latent = pipe.sample(context, mask, 25, args.seeds[0], 3)
    with pipe.autocast():
        decoded = tiny.decode(latent.to(torch.bfloat16)).sample
    from PIL import Image
    pixels = ((decoded[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    restored = output / "restored-control.png"
    Image.fromarray(pixels).save(restored)
    baseline = output / f'{row["id"]}-{args.seeds[0]}-baseline.png'
    assert sha256(restored) == sha256(baseline), "Restored control must reproduce baseline"
    save_json(output / "protocol.json", {"revisions": REVISIONS, "pairs": PAIRS,
        "definition": "W_new = W - alpha * W * U_rank * U_rank.T",
        "target": "ctx_proj.weight", "no_gradients": True, "restore_byte_identical": True,
        "edit_sha256": sha256(output / "edit.safetensors"), "split":args.split,
        "variants":[variant[0] for variant in variants],
        "warning":"Semantic-subspace ablation; does not demonstrate a localized typography circuit."})
    for seed in args.seeds:
        sheet(records, output / f"comparison-seed-{seed}.jpg", seed=seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["development", "evaluation"], default="development")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--output",type=Path,default=OUTPUT/"supra-edit")
    parser.add_argument("--baseline-only",action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
