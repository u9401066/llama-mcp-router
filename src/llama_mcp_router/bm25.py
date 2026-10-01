"""Tiny dependency-free BM25 with CJK-aware tokenisation."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence

_WORD = re.compile(r"[a-z0-9]+")
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]+")
_STOP = {"the", "a", "an", "of", "to", "and", "or", "in", "for", "on", "is", "are", "with", "by", "from", "this", "that", "it", "as", "be", "use", "args", "returns"}


def tokenize(text: str) -> List[str]:
    text = text.lower()
    toks = [w for w in _WORD.findall(text) if w not in _STOP and len(w) > 1]
    for run in _CJK.findall(text):
        toks.extend(run)  # unigrams
        toks.extend(run[i : i + 2] for i in range(len(run) - 1))
    return toks


class BM25:
    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 0.0
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(self.tf)
        self.idf: Dict[str, float] = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: str) -> List[float]:
        q = tokenize(query)
        out = []
        for c, ln in zip(self.tf, self.len):
            s = 0.0
            for t in q:
                f = c.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / (self.avg or 1)))
            out.append(s)
        return out
