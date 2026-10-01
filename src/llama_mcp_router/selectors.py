"""Pluggable tool selectors.

A selector turns ``(user query, candidate tools)`` into the subset of tool names that
should be sent to the model. Write your own by implementing :class:`Selector`
(one ``async select`` method) and pass it as ``--selector package.module:Class`` or
register it under the ``llama_mcp_router.selectors`` entry-point group.
"""
from __future__ import annotations

import fnmatch
import hashlib
import importlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import httpx

from .bm25 import BM25
from .tools import Tool, first_sentence, tool_description, tool_name


@dataclass
class Selection:
    names: List[str]
    info: Dict[str, Any] = field(default_factory=dict)


class Selector:
    """Base class. ``select`` must return tool names that exist in ``tools``."""

    name = "selector"

    async def select(self, query: str, tools: Sequence[Tool]) -> Selection:  # pragma: no cover
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


class AllSelector(Selector):
    """Send every tool (the llama.cpp default behaviour). Useful as a baseline."""

    name = "all"

    async def select(self, query: str, tools: Sequence[Tool]) -> Selection:
        return Selection([tool_name(t) for t in tools])


class BM25Selector(Selector):
    """Lexical top-k over tool names and descriptions. No services needed, English-centric."""

    name = "bm25"

    def __init__(self, top_k: int = 8, min_score: float = 0.0, doc_chars: int = 1500):
        self.top_k, self.min_score, self.doc_chars = top_k, min_score, doc_chars
        self._key: Optional[Tuple[str, ...]] = None
        self._index: Optional[BM25] = None

    def _bm25(self, tools: Sequence[Tool]) -> BM25:
        key = tuple(tool_name(t) for t in tools)
        if key != self._key:
            docs = [(tool_name(t).replace("_", " ") + " ") * 3 + tool_description(t)[: self.doc_chars] for t in tools]
            self._key, self._index = key, BM25(docs)
        return self._index  # type: ignore[return-value]

    async def select(self, query: str, tools: Sequence[Tool]) -> Selection:
        if not tools:
            return Selection([])
        scores = self._bm25(tools).scores(query)
        order = sorted(range(len(tools)), key=lambda i: -scores[i])
        picked = [i for i in order[: self.top_k] if scores[i] > self.min_score]
        return Selection([tool_name(tools[i]) for i in picked], {"scores": {tool_name(tools[i]): round(scores[i], 3) for i in picked}})


# --------------------------------------------------------------------------- groups

@dataclass
class Group:
    name: str
    description: str
    tools: List[str]


def load_groups_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {"always": [], "groups": {}}
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("always", [])
    cfg.setdefault("groups", {})
    return cfg


def build_groups(cfg: Dict[str, Any], tools: Sequence[Tool]) -> List[Group]:
    """Groups from config; tools the config does not mention become one-tool groups,
    so newly added tools are never invisible to the router."""
    by_name = {tool_name(t): t for t in tools}
    seen: set = set()
    groups: List[Group] = []
    for gname, g in cfg.get("groups", {}).items():
        members = [n for n in by_name if any(fnmatch.fnmatch(n, p) for p in g.get("tools", []))]
        members = [n for n in members if n not in seen]
        if members:
            seen.update(members)
            groups.append(Group(gname, g.get("description", gname), members))
    for n, t in by_name.items():
        if n not in seen:
            groups.append(Group(n, first_sentence(tool_description(t)) or n, [n]))
    return groups


# --------------------------------------------------------------------------- Laya

class LayaSelector(Selector):
    """Use a `Laya <https://huggingface.co/convaiinnovations/laya>`_ server as a tool router.

    Laya is a non-autoregressive decision model (~15-100 ms per call). The tool groups
    become the options of a single ``choice`` question; the probabilities decide how
    many groups to keep. ``mode="noul"`` asks one yes/no question per group instead.
    """

    name = "laya"

    def __init__(
        self,
        url: str = "http://127.0.0.1:8000",
        groups: Optional[Dict[str, Any]] = None,
        mode: str = "choice",
        top_p: float = 0.95,
        min_groups: int = 1,
        max_groups: int = 4,
        threshold: float = 0.5,
        instructions: str = "Which kind of tool is needed to handle this request?",
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        if mode not in ("choice", "noul"):
            raise ValueError("mode must be 'choice' or 'noul'")
        self.cfg = groups or {"always": [], "groups": {}}
        self.mode, self.top_p, self.min_groups, self.max_groups = mode, top_p, min_groups, max_groups
        self.threshold, self.instructions = threshold, instructions
        headers = {"Authorization": "Bearer " + api_key} if api_key else {}
        self.client = httpx.AsyncClient(base_url=url.rstrip("/"), headers=headers, timeout=timeout, transport=transport)
        self._cache: Dict[str, Tuple[List[Group], Dict[str, Any]]] = {}

    async def aclose(self) -> None:
        await self.client.aclose()

    def _prepare(self, tools: Sequence[Tool]) -> Tuple[List[Group], Dict[str, Any]]:
        key = hashlib.sha1("\0".join(sorted(tool_name(t) for t in tools)).encode()).hexdigest()
        if key not in self._cache:
            groups = build_groups(self.cfg, tools)
            if self.mode == "choice":
                questions = {"tool_group": {"type": "choice", "instructions": self.instructions, "criteria": {g.name: g.description for g in groups}}}
            else:
                questions = {g.name: {"type": "noul", "instructions": "Does handling this request require: %s?" % g.description} for g in groups}
            self._cache = {key: (groups, questions)}
        return self._cache[key]

    async def select(self, query: str, tools: Sequence[Tool]) -> Selection:
        groups, questions = self._prepare(tools)
        r = await self.client.post("/v1/systemone", json={"state": query, "questions": questions})
        r.raise_for_status()
        answers = r.json()["answers"]
        if self.mode == "choice":
            probs = answers["tool_group"]["probabilities"]
            ranked = sorted(probs.items(), key=lambda kv: -kv[1])
            kept, mass = [], 0.0
            for name, p in ranked:
                if len(kept) >= self.max_groups or (len(kept) >= self.min_groups and mass >= self.top_p):
                    break
                kept.append(name)
                mass += p
        else:
            probs = {g.name: answers[g.name]["noul"] for g in groups}
            ranked = sorted(probs.items(), key=lambda kv: -kv[1])
            kept = [n for n, p in ranked if p >= self.threshold][: self.max_groups]
            if len(kept) < self.min_groups:
                kept = [n for n, _ in ranked[: self.min_groups]]
        members = {g.name: g.tools for g in groups}
        names: List[str] = []
        for g in kept:
            names.extend(n for n in members[g] if n not in names)
        return Selection(names, {"groups": {g: round(probs[g], 3) for g in kept}})


# --------------------------------------------------------------------------- combinators

class UnionSelector(Selector):
    """Union of several selectors, in order. A selector that raises is skipped."""

    def __init__(self, selectors: Sequence[Selector]):
        self.selectors = list(selectors)
        self.name = "+".join(s.name for s in self.selectors)

    async def aclose(self) -> None:
        for s in self.selectors:
            await s.aclose()

    async def select(self, query: str, tools: Sequence[Tool]) -> Selection:
        names: List[str] = []
        info: Dict[str, Any] = {}
        ok = 0
        for s in self.selectors:
            try:
                sel = await s.select(query, tools)
            except Exception as e:  # one broken selector must not take the router down
                info[s.name + "_error"] = type(e).__name__
                continue
            ok += 1
            info[s.name] = sel.info
            names.extend(n for n in sel.names if n not in names)
        if not ok:
            raise RuntimeError("all selectors failed")
        return Selection(names, info)


# --------------------------------------------------------------------------- registry

def _entry_points() -> Dict[str, Any]:
    try:
        from importlib.metadata import entry_points

        eps = entry_points()
        group = eps.select(group="llama_mcp_router.selectors") if hasattr(eps, "select") else eps.get("llama_mcp_router.selectors", [])
        return {ep.name: ep for ep in group}
    except Exception:
        return {}


BUILTIN: Dict[str, Callable[..., Selector]] = {"all": AllSelector, "bm25": BM25Selector, "laya": LayaSelector}


def load_selector(spec: str, **options: Any) -> Selector:
    """Build a selector from a spec such as ``laya``, ``laya+bm25`` or ``pkg.mod:Class``.

    ``options`` maps selector name -> kwargs, e.g. ``{"laya": {"url": ...}, "bm25": {"top_k": 5}}``.
    """
    parts = [p.strip() for p in spec.split("+") if p.strip()]
    built: List[Selector] = []
    for p in parts:
        kwargs = options.get(p, {})
        if p in BUILTIN:
            built.append(BUILTIN[p](**kwargs))
            continue
        eps = _entry_points()
        if p in eps:
            built.append(eps[p].load()(**kwargs))
        elif ":" in p:
            mod, _, cls = p.partition(":")
            built.append(getattr(importlib.import_module(mod), cls)(**kwargs))
        else:
            raise ValueError("unknown selector %r (builtin: %s)" % (p, ", ".join(BUILTIN)))
    return built[0] if len(built) == 1 else UnionSelector(built)
