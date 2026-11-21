from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "supra"
REVISIONS = {
    "supra": ("SupraLabs/Supra2-IMG", "b6186a6e8fa9a846c44b804f42639ce4228e3b94"),
    "text": ("google/flan-t5-base", "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"),
    "vae": ("stabilityai/sd-vae-ft-mse", "31f26fdeee1355a5c34592e401dd41e45d25a493"),
}


def download():
    # The HTTP path is reliable on this laptop; Xet stalled without transferring.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download
    patterns = {
        "supra": ["model_final_ema.pt", "config.json", "README.md", "inference.py"],
        "text": ["*.json", "spiece.model", "model.safetensors", "README.md"],
        "vae": ["config.json", "diffusion_pytorch_model.safetensors", "README.md"],
    }
    for name, (repo, revision) in REVISIONS.items():
        snapshot_download(repo, revision=revision, allow_patterns=patterns[name],
                          local_dir=MODEL_DIR / name)


if __name__ == "__main__":
    download()
