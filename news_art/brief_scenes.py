"""Experimental extractive pruning: one subject, at most one object/setting.

NER removes identifying names; POS chunks and WordNet retain concrete role
words. No whole-scene topic templates or generative language model.
"""
from pathlib import Path
import re

CONCRETE = {"noun.person", "noun.animal", "noun.artifact", "noun.body", "noun.food",
            "noun.location", "noun.object", "noun.plant", "noun.phenomenon", "noun.substance"}
SKIP = set("list number day week year month time history title study report journey odyssey front part side area growth change case record debut draft talk rule enhancement customization threat attention inquiry communication mission orbit ministerial knot warming order".split())
TYPES = set("sea ocean strait river lake coast beach mountain planet moon earth spacecraft capsule ship boat station screen phone airport hospital stadium".split())
PLACES = set("sea ocean strait river lake coast beach mountain planet moon earth airport hospital stadium".split())
COLORS = set("red blue green white black gray grey brown orange yellow silver golden wooden metallic".split())


class Pruner:
    def __init__(self, data_dir=None):
        import nltk
        from nltk.corpus import wordnet
        self.nltk,self.wn=nltk,wordnet
        if data_dir is not None:
            nltk.data.path[:]=[str(data_dir)]
        self.ner=nltk.chunk.ne_chunker()
        self.chunker=nltk.RegexpParser(r"NP: {<DT|PRP\$>?<JJ.*|NN.*|CD>*<NN.*>+}")

    def concrete(self,word):
        lemma=self.wn.morphy(word.lower(),self.wn.NOUN) or word.lower()
        return lemma not in SKIP and (lemma in {"couple","crowd","team","people"} or
               any(s.lexname() in CONCRETE for s in self.wn.synsets(lemma,pos="n")[:3]))

    def chunks(self,text,generic):
        # Keep sentence boundaries, but stop after the short supplied context.
        tokens=self.nltk.word_tokenize(text[:1800],preserve_line=True)
        tagged=self.nltk.pos_tag(tokens)
        # Common role words can be mistagged as adjectives before a verb.
        tagged=[(w,"NN" if w.lower()=="couple" else t) for w,t in tagged]
        # Headline POS tagging can absorb "swimmer ... swims" or "couple
        # walks" into one noun chunk. Split these only when WordNet supports
        # a verb and the token has no concrete noun reading.
        tagged=[(w,"VBZ" if i and t in {"NN","NNS"} and w.lower().endswith("s")
                 and tagged[i-1][1].startswith("NN") and not self.concrete(w)
                 and self.wn.synsets(w.lower(),pos="v") else t)
                for i,(w,t) in enumerate(tagged)]
        entities=self.ner.parse(tagged)
        named=set();labels=[];offset=0
        for entity in entities:
            if hasattr(entity,"label"):
                leaves=entity.leaves();labels.append({"text":" ".join(w for w,_ in leaves),"label":entity.label()})
                named.update(range(offset,offset+len(leaves)));offset+=len(leaves)
            else:offset+=1
        chunks=[];offset=0
        for part in self.chunker.parse(tagged):
            if not hasattr(part,"label"):
                offset+=1;continue
            leaves=part.leaves();start=offset;offset+=len(leaves)
            common=[(i,w,t) for i,(w,t) in enumerate(leaves) if t in {"NN","NNS"}]
            if common:
                _,head,tag=common[-1]
            else:
                acceptable=[(i,w,t) for i,(w,t) in enumerate(leaves) if w.lower() in TYPES]
                if not acceptable:continue
                _,head,tag=acceptable[-1]
            lemma=self.wn.morphy(head.lower(),self.wn.NOUN) or head.lower()
            if not self.concrete(lemma):continue
            words=[]
            for i,(word,pos) in enumerate(leaves):
                low=word.lower()
                if low in {"first","new","latest","more","most","last","final","big","huge"} or pos in {"DT","CD","PRP$"}:continue
                if generic:
                    if pos.startswith("NNP") or start+i in named:
                        # A named phrase's physical type can survive: "Strait of
                        # Hormuz" -> "sea", while the identifying name is dropped.
                        if low not in TYPES and not (pos=="JJ" and re.search(r"(ian|ican|ish|ese|ean)$",low)):continue
                    if pos.startswith("JJ") and low not in COLORS and not re.search(r"(ian|ican|ish|ese|ean)$",low):continue
                    if pos in {"NN","NNS"} and low!=head.lower() and not self.concrete(low):continue
                words.append(word)
            if not words:continue
            if generic:
                words=[w if w.lower() not in TYPES else w.lower() for w in words]
            phrase=" ".join(words[-6:])
            if generic:
                phrase=re.sub(r"\bstrait\b","sea",phrase,flags=re.I)
                phrase=re.sub(r"\bearth\b","planet",phrase,flags=re.I)
            if lemma=="rain":phrase="rainstorm"
            chunks.append({"phrase":phrase,"head":lemma,"place":lemma in PLACES,
                           "person":any(s.lexname()=="noun.person" for s in self.wn.synsets(lemma,pos="n")[:2]) or lemma in {"couple","people"},
                           "start":start,"end":offset,"plural":tag=="NNS"})
        return chunks,labels

    def convert(self,headline,context="",generic=True):
        headline=re.sub(r"^(?:exclusive|breaking news|live updates)\s*[:|]\s*","",headline,flags=re.I)
        headline=re.sub(r"^[‘'\"].*?[’'\"]\s*:\s*","",headline)
        heading,entities=self.chunks(headline,generic)
        background,more=self.chunks(context,generic) if context else ([],[])
        candidates=heading+background
        primary=next((c for c in heading if not c["place"]),None)
        if primary is None or primary["head"] in {"athlete","person","contractor"}:
            primary=next((c for c in background if not c["place"]),primary)
        if primary is None:primary=next(iter(candidates),None)
        if primary is None:
            return {"scene":"a scene","prompt":"a scene. oil painting","fallback":True,"entities":entities+more}
        # Prefer a source-mentioned object after "hit by"; otherwise retain one
        # setting or second role. Do not add a complete invented scene.
        text=headline+" "+context
        hit=re.search(r"\bhit\s+by\b",text,re.I)
        secondary=None
        if hit:
            after,_=self.chunks(text[hit.end():hit.end()+160],generic)
            secondary=next((c for c in after if not c["place"] and c["head"]!=primary["head"]),None)
        if secondary is None:
            others=[c for c in candidates if c["head"]!=primary["head"]]
            secondary=next((c for c in others if c["place"]),None)
            if secondary is None:secondary=next((c for c in others if c["person"]==primary["person"]),None)
        phrase=primary["phrase"]
        if primary["plural"]:
            # A singular subject lets every caption start naturally with "a".
            phrase=re.sub(r"\b"+re.escape(phrase.split()[-1])+r"$",primary["head"],phrase)
        scene="a "+phrase
        if re.search(r"\b(married|marries|marry|wedding|wed|knot)\b",text,re.I) and (primary["person"] or primary["head"]=="couple"):
            scene+=" getting married";secondary=None
        if secondary:
            other=secondary["phrase"]
            if hit and not secondary["place"]:scene+=" hit by a "+other
            elif secondary["head"] in {"planet","earth","moon"} and re.search(r"\borbit\b",text,re.I):scene+=" orbiting a "+other
            elif secondary["place"]:
                scene+=(" at sea" if generic and secondary["head"] in {"sea","strait","ocean"} else " near a "+other)
            else:scene+=" and "+("" if secondary["plural"] else "a ")+other
        # Deliberately cap the scene to a short caption, not the article length.
        scene=" ".join(scene.split()[:20]).rstrip(" ,;:")
        return {"scene":scene,"prompt":scene+". oil painting","fallback":False,
                "primary":primary,"secondary":secondary,"entities":entities+more,
                "generic":generic}

