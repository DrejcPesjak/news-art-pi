"""Production adapter for sequential, bounded NumPy/NCNN Inko stages."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

from PIL import Image

from .supra import run_pipeline, runtime_environment


def bundle_dir():
    return Path(os.environ.get("NEWS_ART_INKO_DIR", "/var/lib/news-art-pi/inko/current"))


def check_runtime():
    directory = bundle_dir()
    manifest = json.loads((directory / "bundle.json").read_text())
    recipe = json.loads((directory / "models/recipe.json").read_text())
    if manifest.get("format") != "news-art-inko-research-v1":
        raise ValueError("Install the verified Inko bundle")
    if recipe.get("decoder") != "tiny-qwen" or recipe.get("text_bits") != 8:
        raise ValueError("Inko requires the tested Q8 text / tiny-Qwen decoder recipe")
    for name, metadata in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid Inko bundle filename")
        path = directory / relative
        if not path.is_file() or path.stat().st_size != metadata["bytes"]:
            raise ValueError(f"Missing or incomplete Inko asset: {path}")
    import subprocess
    subprocess.run([sys.executable, "-c", "import ncnn, numpy, regex; from PIL import Image"],
                   env=runtime_environment(), check=True, timeout=30, stdin=subprocess.DEVNULL)


def generate(prompt, config, sequence, state_dir):
    models = bundle_dir() / "models"
    recipe = json.loads((models / "recipe.json").read_text())
    parent = Path(state_dir) / "generation"
    parent.mkdir(parents=True, exist_ok=True)
    size = config["image_size"]
    seed = (config.get("seed", 0) + sequence) % 2**32
    timeout = config.get("timeout_seconds", 7200)
    started = time.monotonic()
    environment = runtime_environment()
    environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS=str(config.get("threads", 4)),
                       MALLOC_ARENA_MAX="1")
    stages = {}
    with tempfile.TemporaryDirectory(prefix="inko-", dir=parent) as temporary:
        for stage in ("encode", "sample", "decode"):
            # Each stage gets its own process group, so the existing supervisor
            # can terminate every descendant on timeout or worker shutdown.
            command = [sys.executable, "-m", "research.edge_quality.staged", stage,
                       "--models", str(models), "--work", temporary, "--prompt", prompt,
                       "--variant", recipe["variant"], "--text-bits", "8",
                       "--decoder", "tiny-qwen", "--size", str(size),
                       "--steps", str(config.get("steps", 8)),
                       "--guidance", str(config.get("guidance", 1.0)),
                       "--seed", str(seed), "--threads", str(config.get("threads", 4)),
                       "--limit-mib", str(config.get("memory_mib", 480))]
            nice = shutil.which("nice")
            if nice:
                command = [nice, "-n", "10"] + command
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError(f"Inko exceeded {timeout}s")
            run_pipeline(command, remaining, environment)
            report = json.loads((Path(temporary) / f"{stage}.json").read_text())
            if report.get("forbidden_imports_loaded"):
                raise ValueError("Heavy ML dependencies loaded in the Inko runtime")
            stages[stage] = report
        with Image.open(Path(temporary) / "image.png") as source:
            image = source.convert("RGB")
        if image.size != (size, size):
            raise ValueError(f"Inko returned unexpected image size {image.size}")
        image.info["inference"] = dict(variant=recipe["variant"], decoder="tiny-qwen",
                                       guidance=config.get("guidance", 1.0),
                                       stages=stages, seconds=time.monotonic()-started)
        return image
