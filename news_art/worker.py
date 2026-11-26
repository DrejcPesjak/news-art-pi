import argparse
from datetime import datetime, timezone
import fcntl
import json
import logging
import os
from pathlib import Path
import random
import tempfile
import time

from .config import ROOT, load
from .generator import generate
from .news import fetch

log = logging.getLogger(__name__)


def read_state(directory):
    try:
        return json.loads((directory / "state.json").read_text())
    except (FileNotFoundError, ValueError):
        return {}


def write_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        temporary = stream.name
    os.replace(temporary, path)


def cycle(config, directory, stories=None):
    previous = read_state(directory)
    stories = fetch(config["news"]) if stories is None else stories
    if not stories:
        raise ValueError("No stories available")
    history = previous.get("history", [])
    candidates = [story for story in stories if story["id"] not in history]
    if not candidates:
        candidates = [story for story in stories if story["id"] != previous.get("story", {}).get("id")] or stories
        history = []
    story = random.choice(candidates)
    sequence = previous.get("sequence", 0)
    started = time.monotonic()
    write_json(directory / "job.json", {"status": "generating", "story": story,
               "backend": config["generation"]["backend"],
               "started_at": datetime.now(timezone.utc).isoformat()})
    try:
        picture, prompt = generate(story, config["generation"], sequence, state_dir=directory)
    except Exception as error:
        write_json(directory / "job.json", {"status": "failed", "story": story,
                   "error": str(error), "finished_at": datetime.now(timezone.utc).isoformat()})
        raise
    # Alternate slots: never overwrite the image referenced by the current state.
    filename = f"image-{sequence % 2}.png"
    temporary = directory / (filename + ".tmp")
    picture.save(temporary, format="PNG")
    os.replace(temporary, directory / filename)
    state = {"sequence": sequence + 1, "image": filename, "story": story,
             "prompt": prompt, "generated_at": datetime.now(timezone.utc).isoformat(),
             "title": config["display"]["title"], "backend": config["generation"]["backend"],
             "generation_seconds": time.monotonic()-started,
             "seed": (config["generation"].get("seed", 0)+sequence) % 2**32,
             "steps": config["generation"].get("steps") if config["generation"]["backend"] in ("supra", "inko") else None,
             "history": (history + [story["id"]])[-500:]}
    if "inference" in picture.info:
        state["inference"] = picture.info["inference"]
    write_json(directory / "state.json", state)
    write_json(directory / "job.json", {"status": "idle", "finished_at": state["generated_at"],
               "sequence": state["sequence"], "generation_seconds": state["generation_seconds"]})
    log.info("Poster %s: %s — %s", sequence + 1, story["source"], story["headline"])
    return state


def self_check(config):
    # Offline release validation: exercises assets and configuration without network.
    sample = {"id": "check", "headline": "Display ready", "source": "Demo",
              "url": "https://example.com", "published": datetime.now(timezone.utc).isoformat()}
    if config["generation"]["backend"] in ("supra", "inko"):
        return  # Target runtime is checked separately; never infer during validation.
    for index in range(3):
        image, _ = generate(sample, config["generation"], index)
        if image.size != (config["generation"]["image_size"],) * 2:
            raise ValueError("Generator returned a non-square image")


def refresh_delay(interval, elapsed):
    return max(1, interval-elapsed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.toml")
    parser.add_argument("--state-dir", type=Path, default=ROOT / "state")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--check-runtime", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load(args.config)
    self_check(config)
    if config["generation"]["backend"] in ("supra", "inko") and (args.check_runtime or not args.check):
        from .captions import check_runtime as check_captions
        check_captions()
        if config["generation"]["backend"] == "inko":
            from .inko import check_runtime
        else:
            from .supra import check_runtime
        check_runtime()
    if args.check or args.check_runtime:
        log.info("Configuration and selected assets are valid")
        return
    args.state_dir.mkdir(parents=True, exist_ok=True)
    with open(args.state_dir / "worker.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            started = time.monotonic()
            try:
                config = load(args.config)
                cycle(config, args.state_dir)
                delay = refresh_delay(config["display"]["refresh_seconds"], time.monotonic()-started)
            except Exception:
                log.exception("Refresh failed; retaining the last successful poster")
                if args.once:
                    raise
                delay = 60
            if args.once:
                return
            time.sleep(delay)


if __name__ == "__main__":
    main()

