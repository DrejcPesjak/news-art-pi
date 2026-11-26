"""Qwen's released byte-level BPE tokenizer using Python + regex only.

Reads the pinned tokenizer.json rather than maintaining a second vocabulary.
No model code, Transformers, Rust tokenizers wheel or network is required.
"""
from functools import lru_cache
import json
from pathlib import Path
import unicodedata

import regex


SYSTEM_PROMPT="Describe the image, focusing on its content, artistic style, composition, lighting, color, texture, and the spatial relationships between objects and the background:"
PROMPT_TEMPLATE="<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"


class QwenTokenizer:
    def __init__(self,path):
        source=json.loads(Path(path).read_text())
        self.vocab=source["model"]["vocab"]
        self.ranks={tuple(pair):i for i,pair in enumerate(source["model"]["merges"])}
        self.special={token["content"]:token["id"] for token in source["added_tokens"]}
        self.split_special=regex.compile("("+"|".join(regex.escape(t) for t in sorted(self.special,key=len,reverse=True))+")")
        pattern=source["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
        self.pattern=regex.compile(pattern)
        values=list(range(ord("!"),ord("~")+1))+list(range(ord("¡"),ord("¬")+1))+list(range(ord("®"),ord("ÿ")+1))
        characters=values.copy()
        n=0
        for value in range(256):
            if value not in values:
                values.append(value)
                characters.append(256+n)
                n+=1
        self.byte_encoder=dict(zip(values,map(chr,characters)))

    @lru_cache(maxsize=2048)
    def bpe(self,word):
        pieces=list(word)
        while len(pieces)>1:
            choices=[(self.ranks.get(pair,float("inf")),pair) for pair in zip(pieces,pieces[1:])]
            rank,pair=min(choices,key=lambda p:p[0])
            if rank==float("inf"):
                break
            merged=[]
            index=0
            while index<len(pieces):
                if index+1<len(pieces) and (pieces[index],pieces[index+1])==pair:
                    merged.append(pieces[index]+pieces[index+1])
                    index+=2
                else:
                    merged.append(pieces[index])
                    index+=1
            pieces=merged
        return tuple(self.vocab[piece] for piece in pieces)

    def encode(self,text):
        ids=[]
        for part in self.split_special.split(text):
            if part in self.special:
                ids.append(self.special[part])
                continue
            # Added tokens are isolated before the model's NFC normalizer.
            part=unicodedata.normalize("NFC",part)
            words=self.pattern.findall(part)
            if "".join(words)!=part:
                raise ValueError("Tokenizer split omitted input characters")
            for word in words:
                encoded="".join(self.byte_encoder[b] for b in word.encode("utf-8"))
                ids.extend(self.bpe(encoded))
        return ids

    def prompt(self,text,max_tokens=2086):
        return self.encode(PROMPT_TEMPLATE.format(system_prompt=SYSTEM_PROMPT,user_prompt=text))[:max_tokens]
