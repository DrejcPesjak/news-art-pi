"""Run v3 name-stripped caption preparation before loading the image model."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .config import ROOT


def data_dir():
    configured = os.environ.get("NEWS_ART_NLTK_DATA")
    if configured:
        return Path(configured)
    production = Path("/var/lib/news-art-pi/models/nltk")
    return production if production.is_dir() else Path.home()/".cache/news-art-nltk"


def prepare(story, fetch_context=False):
    # A fresh process releases NER/WordNet memory before NCNN inference.
    result = subprocess.run(
        [sys.executable, "-m", "news_art.captions", "--child"]
        + (["--fetch-context"] if fetch_context else []),
        input=json.dumps(story), text=True, capture_output=True, cwd=ROOT,
        timeout=120, check=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    return json.loads(result.stdout)


def check_runtime():
    result = prepare({"headline": "A Russian cargo ship hit by a drone"})
    if result["scene"] != "a Russian cargo ship hit by a drone":
        raise ValueError(f"Caption runtime produced an unexpected result: {result['scene']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--fetch-context", action="store_true")
    args = parser.parse_args()
    if not args.child:
        check_runtime()
        print("Caption runtime ready")
        return
    from .brief_scenes import Pruner
    story = json.load(sys.stdin)
    context = {"status": "not-requested", "context": story.get("context", "")}
    if args.fetch_context:
        from .article import context_for
        context = context_for(story)
    report = Pruner(data_dir()).convert(story["headline"], context["context"], generic=True)
    print(json.dumps({**report, "converter": "brief-v3", "article": context}))


if __name__ == "__main__":
    main()
