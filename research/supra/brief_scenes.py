"""Laptop experiment entry point; production shares the tested converter."""
from pathlib import Path
from news_art.brief_scenes import Pruner

def main():
    import argparse
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=Path("artifacts/context-nlp/nlp-prompts.json"))
    parser.add_argument("--output",type=Path,default=Path("artifacts/supra-brief/prompts.json"))
    args=parser.parse_args()
    pruner=Pruner(Path.home()/".cache/news-art-nltk");rows=[]
    for row in json.loads(args.input.read_text()):
        row["short_named"]=pruner.convert(row["headline"],row["context"],False)
        row["short_generic"]=pruner.convert(row["headline"],row["context"],True)
        rows.append(row)
        print(row["feed_index"],row["short_generic"]["prompt"],flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(rows,indent=2)+"\n")


if __name__=="__main__":
    main()
