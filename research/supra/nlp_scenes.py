"""Experimental extractive scene captions using UDPipe, NLTK NER and WordNet.

No full-scene topic templates or generative language model. Grammar extraction
can still lose qualifiers and misread headlines; keep its evidence in reports.
Not imported by the production application.
"""
from collections import defaultdict
from pathlib import Path
import re
import time


STYLE = "A detailed color photograph, natural lighting."
REPORTING = {"say", "tell", "report", "announce", "suggest", "know", "think",
             "claim", "believe", "mean", "show", "expect", "consider", "mention"}
NOUN_CLASSES = {"noun.person", "noun.animal", "noun.artifact", "noun.body", "noun.food",
                "noun.location", "noun.object", "noun.plant", "noun.phenomenon"}
VERB_CLASSES = {"verb.motion", "verb.contact", "verb.creation", "verb.body",
                "verb.competition", "verb.weather", "verb.change"}


class Extractor:
    def __init__(self, model=Path("models/nlp/english-ewt.udpipe"), data=None):
        import nltk
        from nltk.corpus import wordnet
        from ufal.udpipe import Model, Pipeline
        started = time.perf_counter()
        self.nltk, self.wordnet = nltk, wordnet
        nltk.data.path[:] = [str(data or Path.home()/".cache/news-art-nltk")]
        loaded = Model.load(str(model))
        if loaded is None:
            raise ValueError(f"Cannot load UDPipe model: {model}")
        self.model = loaded  # Native Pipeline borrows the model; keep it alive.
        self.pipeline = Pipeline(loaded, "horizontal", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")
        # Warm the statistical tagger/chunker and lexical lookup once.
        nltk.ne_chunk(nltk.pos_tag(["A", "scientist", "works", "in", "London", "."]))
        wordnet.synsets("ship", pos="n")
        self.load_seconds = time.perf_counter()-started

    def parse(self, text):
        from ufal.udpipe import ProcessingError
        originals = self.nltk.sent_tokenize(text)
        horizontal = "\n".join(" ".join(self.nltk.word_tokenize(s, preserve_line=True)) for s in originals)+"\n"
        error = ProcessingError()
        conllu = self.pipeline.process(horizontal, error)
        if error.occurred():
            raise ValueError(error.message)
        sentences = []
        for block in conllu.strip().split("\n\n"):
            nodes, original = [], ""
            for line in block.splitlines():
                if line.startswith("# text = "):
                    original = line[9:]
                elif line and not line.startswith("#"):
                    cells = line.split("\t")
                    if cells[0].isdigit():
                        nodes.append(dict(id=int(cells[0]), text=cells[1], lemma=cells[2].lower(),
                                          pos=cells[3], features=cells[5], head=int(cells[6]), rel=cells[7]))
            if nodes:
                sentences.append((original, nodes))
        if len(sentences) != len(originals):
            raise ValueError("Parser changed presegmented sentence boundaries")
        return [(original, nodes) for original, (_, nodes) in zip(originals, sentences)]

    def concrete(self, node):
        return any(s.lexname() in NOUN_CLASSES for s in self.wordnet.synsets(node["lemma"], pos="n")[:3])

    def caption(self, original, nodes):
        children = defaultdict(list)
        for n in nodes:
            children[n["head"]].append(n)

        def phrase(head, depth=0):
            members = [head]
            def modifiers(n):
                for child in children[n["id"]]:
                    if child["rel"] in {"det", "amod", "compound", "flat", "flat:name", "fixed"}:
                        members.append(child)
                        modifiers(child)
            modifiers(head)
            common = [n for n in members if n["pos"] == "NOUN"]
            # A role noun from the actual text can stand in for a person's name.
            # Keep proper names where there is no supplied common-noun identity.
            if common:
                members = [n for n in members if n["pos"] != "PROPN"]
            text = " ".join(n["text"] for n in sorted(members, key=lambda n:n["id"]))
            if depth == 0:
                for child in children[head["id"]]:
                    case = [c["text"] for c in children[child["id"]] if c["rel"] == "case"]
                    if child["rel"] == "nmod" and case and case[0].lower() in {"of", "in", "on", "around", "with"}:
                        text += " " + " ".join(case) + " " + phrase(child, 1)
                        break
            return text

        candidates = []
        for verb in nodes:
            if verb["pos"] != "VERB" or verb["lemma"] in REPORTING:
                continue
            subjects = [n for n in children[verb["id"]] if n["rel"].startswith("nsubj") and n["pos"] in {"NOUN", "PROPN"}]
            if not subjects:
                continue
            subject = subjects[0]
            objects = [n for n in children[verb["id"]] if n["rel"] in {"obj", "iobj"} and n["pos"] in {"NOUN", "PROPN"}]
            settings = []
            for n in children[verb["id"]]:
                if n["rel"] != "obl" or n["pos"] not in {"NOUN", "PROPN"}:
                    continue
                case = [c["text"] for c in children[n["id"]] if c["rel"] == "case"]
                if case and case[0].lower() in {"in", "on", "at", "above", "around", "through", "across", "into", "from", "to", "with"}:
                    settings.append((n, " ".join(case)))
            auxiliaries = [n for n in children[verb["id"]] if n["rel"].startswith("aux")]
            # Keep negation and modal auxiliaries instead of turning a possibility
            # or failed attempt into an asserted action.
            negatives = [n for n in children[verb["id"]] if n["lemma"] in {"not", "never", "n't"}]
            action = " ".join(n["text"] for n in sorted(auxiliaries+negatives+[verb], key=lambda n:n["id"]))
            parts = [phrase(subject), action]+[phrase(n) for n in objects[:1]]
            parts += [case+" "+phrase(n) for n,case in settings[:2]]
            text = " ".join(parts)
            terms = [subject]+objects+[n for n,_ in settings]
            concrete = sum(self.concrete(n) for n in terms)
            motion = any(s.lexname() in VERB_CLASSES for s in self.wordnet.synsets(verb["lemma"], pos="v")[:3])
            score = concrete*2 + motion*2 + (subject["pos"] == "NOUN") + (verb["rel"] == "root")
            if any(n["lemma"] in {"will", "would", "could", "may", "might", "should"} for n in auxiliaries):
                score -= 3
            # Prefer the first equally scored proposition; do not join unrelated
            # facts into a scene that the article never describes.
            # Early SVO-only rewrites dropped coordinated verbs and misread
            # irregular forms (e.g. "has swum" -> "has"). Use the actual selected
            # sentence for the image prompt; retain the core as diagnostics only.
            caption = re.sub(r"\([^)]*\)", "", original)
            caption = " ".join(caption.split()).strip().rstrip(".!?")+"."
            candidates.append({"caption": caption, "core": text[:1].upper()+text[1:]+".", "score": score,
                               "source_sentence": original, "subject": phrase(subject),
                               "verb": action, "objects": [phrase(n) for n in objects],
                               "settings": [case+" "+phrase(n) for n,case in settings],
                               "concrete_terms": concrete})
        return candidates

    def convert(self, headline, context=""):
        started = time.perf_counter()
        text = re.sub(r"^(?:exclusive|breaking news)\s*[:|]\s*", "", headline, flags=re.I)
        # Context first: complete sentences usually parse better than headlines.
        material = context.strip() if context.strip() else text
        parsed = self.parse(material[:2000])
        candidates = [c for original,nodes in parsed for c in self.caption(original,nodes)]
        if not candidates and context:
            candidates = [c for original,nodes in self.parse(text) for c in self.caption(original,nodes)]
        best = max(candidates, key=lambda c:c["score"]) if candidates else None
        caption = best["caption"] if best else text.rstrip(".!?")+"."
        tokens = self.nltk.tokenize.wordpunct_tokenize(material[:2000])
        tree = self.nltk.ne_chunk(self.nltk.pos_tag(tokens))
        entities = [{"text": " ".join(w for w,_ in chunk.leaves()), "label": chunk.label()}
                    for chunk in tree if hasattr(chunk, "label")]
        return {"prompt": caption+"\n"+STYLE, "caption": caption, "selected": best,
                "candidates": candidates, "entities": entities,
                "context_used": bool(context.strip()), "headline_fallback": best is None,
                "seconds": time.perf_counter()-started}


def main():
    import argparse
    import json
    import resource
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("artifacts/context-nlp"))
    args = parser.parse_args()
    extractor = Extractor()
    rows = json.loads((args.directory/"articles.json").read_text())
    results = []
    for row in rows:
        parts = list(dict.fromkeys(s for s in [row["summary"], row["opening"]] if s))
        context = "\n\n".join(p.rstrip(".!?")+"." for p in parts)
        head = extractor.convert(row["headline"])
        enriched = extractor.convert(row["headline"], context)
        results.append({**row, "context": context, "nlp_headline": head, "nlp_context": enriched})
        print(f'{row["feed_index"]}: {enriched["caption"]}', flush=True)
    (args.directory/"nlp-prompts.json").write_text(json.dumps(results, indent=2)+"\n")
    metrics = {"load_seconds": extractor.load_seconds,
               "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
               "conversions": len(rows)*2,
               "total_conversion_seconds": sum(r[k]["seconds"] for r in results for k in ["nlp_headline", "nlp_context"])}
    (args.directory/"nlp-metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")


if __name__ == "__main__":
    main()
