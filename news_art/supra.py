"""Production adapter for the tested, staged NCNN CPU pipeline."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile

from PIL import Image
from .config import ROOT

REQUIRED = ("dit.ncnn.param", "dit.ncnn.bin", "text.ncnn.param", "text.ncnn.bin",
            "tiny.ncnn.param", "tiny.ncnn.bin", "embedding.npy", "spiece.model",
            "unconditional.npz", "pipeline.json")


def model_dir():
    return Path(os.environ.get("NEWS_ART_MODEL_DIR", "/var/lib/news-art-pi/models/supra"))


def runtime_environment():
    environment = os.environ.copy()
    runtime = os.environ.get("NEWS_ART_RUNTIME_DIR", "/opt/news-art-pi/runtime")
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [runtime, environment.get("PYTHONPATH")]))
    environment["PYTHONUNBUFFERED"] = "1"
    environment["LC_ALL"] = "C.UTF-8"
    return environment


def check_runtime():
    directory = model_dir()
    manifest = json.loads((directory / "manifest.json").read_text())
    for name in REQUIRED:
        path = directory / name
        if name not in manifest or not path.is_file() or path.stat().st_size != manifest[name]["bytes"]:
            raise ValueError(f"Missing or incomplete model asset: {path}; run deployment/install-models.sh")
    pipeline = json.loads((directory / "pipeline.json").read_text())
    if pipeline.get("generator") != "Supra2-IMG" or pipeline.get("decoder") != "tiny" or pipeline.get("resolution") != [256, 256]:
        raise ValueError("Install the verified Supra2 bundle with the tiny decoder")
    subprocess.run([sys.executable, "-c", "import ncnn, numpy, sentencepiece; from PIL import Image"],
                   env=runtime_environment(), check=True, timeout=30, stdin=subprocess.DEVNULL)


def stop_group(process):
    # Terminate the entire staged pipeline, including a currently active child.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # A supervisor may exit before its child. Always clear the remaining group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_pipeline(command, timeout, environment):
    process = subprocess.Popen(command, cwd=ROOT, env=environment,
                               stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        result = process.wait(timeout=timeout)
        if result != 0:
            raise subprocess.CalledProcessError(result, command)
    except BaseException:
        stop_group(process)
        raise


def generate(prompt, config, sequence, state_dir):
    parent = Path(state_dir) / "generation"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="job-", dir=parent) as temporary:
        command = [sys.executable, "-m", "research.supra.staged", "generate",
                   "--model-dir", str(model_dir()), "--output", temporary,
                   "--prompt", prompt, "--steps", str(config.get("steps", 25)),
                   "--guidance", str(config.get("guidance", 3.0)),
                   "--seed", str((config.get("seed", 0) + sequence) % 2**32),
                   "--threads", str(config.get("threads", 4)),
                   "--memory-mib", str(config.get("memory_mib", 480)),
                   "--decoder", "tiny", "--packed-decoder"]
        nice = shutil.which("nice")
        if nice:
            command = [nice, "-n", "10"] + command
        run_pipeline(command, config.get("timeout_seconds", 900), runtime_environment())
        with Image.open(Path(temporary) / "image.png") as source:
            image = source.convert("RGB")
        if image.size != (256, 256):
            raise ValueError(f"Supra2 returned unexpected image size {image.size}")
        return image
