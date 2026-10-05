"""Tool retrieval for large pools: BM25 plus (optional) embeddings from any OpenAI-compatible
``POST /v1/embeddings`` endpoint (llama-server --embedding, TEI, vLLM, Ollama, OpenAI ...).

No extra dependencies: numpy is used when installed, plain Python otherwise.
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional, Sequence

import httpx

from .bm25 import BM25
from .tools import Tool, first_sentence, tool_description, tool_name

try:  # optional speed-up
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None


def tool_text(t: Tool, chars: int = 300) -> str:
    """What a tool is matched on: its name in words + the first sentence of its description."""
    return "%s: %s" % (tool_name(t).replace("_", " "), first_sentence(tool_description(t), chars))


def is_cjk(text: str) -> bool:
    return any("\u3040" <= ch <= "\u9fff" or "\uac00" <= ch <= "\ud7af" for ch in text)


def _normalize(v: Sequence[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class HTTPEmbedder:
    """Embeddings from an OpenAI-compatible endpoint. Document vectors are cached by text."""

    def __init__(self, url: str, model: Optional[str] = None, batch: int = 64, timeout: float = 60.0,
                 query_prefix: str = "", doc_prefix: str = "", transport: Optional[httpx.AsyncBaseTransport] = None):
        self.client = httpx.AsyncClient(base_url=url.rstrip("/"), timeout=timeout, transport=transport)
        self.model, self.batch = model, batch
        self.query_prefix, self.doc_prefix = query_prefix, doc_prefix
        self._cache: Dict[str, List[float]] = {}

    async def aclose(self) -> None:
        await self.client.aclose()

    async def _embed(self, texts: Sequence[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch):
            body = {"input": list(texts[i:i + self.batch])}
            if self.model:
                body["model"] = self.model
            r = await self.client.post("/v1/embeddings", json=body)
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda d: d.get("index", 0))
            out += [_normalize(d["embedding"]) for d in data]
        return out

    async def query(self, text: str) -> List[float]:
        return (await self._embed([self.query_prefix + text]))[0]

    async def documents(self, texts: Sequence[str]) -> List[List[float]]:
        keys = [hashlib.sha1((self.doc_prefix + t).encode()).hexdigest() for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if k not in self._cache]
        if missing:
            vecs = await self._embed([self.doc_prefix + t for _, t in missing])
            for (k, _), v in zip(missing, vecs):
                self._cache[k] = v
        return [self._cache[k] for k in keys]


def _dots(q: List[float], docs: List[List[float]]) -> List[float]:
    if _np is not None:
        return list(_np.asarray(docs, dtype=_np.float32) @ _np.asarray(q, dtype=_np.float32))
    return [sum(a * b for a, b in zip(q, d)) for d in docs]


class Retriever:
    """Ranks tools for a request. BM25 and embedding rankings are fused by reciprocal rank;
    for CJK requests only the embedding ranking is used (BM25 against English descriptions hurts there).
    Without an embedder it is plain BM25."""

    def __init__(self, embedder: Optional[HTTPEmbedder] = None, rrf_k: int = 20, cjk_embedding_only: bool = True):
        self.embedder, self.rrf_k, self.cjk_embedding_only = embedder, rrf_k, cjk_embedding_only
        self._bm_key: Optional[tuple] = None
        self._bm: Optional[BM25] = None

    async def aclose(self) -> None:
        if self.embedder:
            await self.embedder.aclose()

    def _bm25_rank(self, query: str, tools: Sequence[Tool]) -> List[str]:
        key = tuple(tool_name(t) for t in tools)
        if key != self._bm_key:
            self._bm_key, self._bm = key, BM25([tool_text(t) for t in tools])
        s = self._bm.scores(query)  # type: ignore[union-attr]
        order = sorted(range(len(tools)), key=lambda i: -s[i])  # stable: unmatched tools keep pool order, last
        return [tool_name(tools[i]) for i in order]

    async def _emb_rank(self, query: str, tools: Sequence[Tool]) -> List[str]:
        docs = await self.embedder.documents([tool_text(t) for t in tools])  # type: ignore[union-attr]
        s = _dots(await self.embedder.query(query), docs)  # type: ignore[union-attr]
        return [tool_name(tools[i]) for i in sorted(range(len(tools)), key=lambda i: -s[i])]

    async def rank(self, query: str, tools: Sequence[Tool]) -> List[str]:
        if not tools:
            return []
        if self.embedder is None:
            return self._bm25_rank(query, tools)
        emb = await self._emb_rank(query, tools)
        if self.cjk_embedding_only and is_cjk(query):
            return emb
        score: Dict[str, float] = {}
        for ranking in (emb, self._bm25_rank(query, tools)):
            for i, n in enumerate(ranking[:200]):
                score[n] = score.get(n, 0.0) + 1.0 / (self.rrf_k + i)
        fused = sorted(score, key=lambda n: -score[n])
        return fused + [n for n in emb if n not in score]
