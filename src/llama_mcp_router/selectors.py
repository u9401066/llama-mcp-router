"""Pluggable tool selectors.

A selector turns ``(user query, candidate tools)`` into the subset of tool names that
should be sent to the model. Write your own by implementing :class:`Selector`
(one ``async select`` method) and pass it as ``--selector package.module:Class`` or
register it under the ``llama_mcp_router.selectors`` entry-point group.
"""
from __future__ import annotations

import asyncio
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
    abstain: bool = False  # "no tool is needed for this request"


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
    label: str = ""  # short, front-loaded text shown to Laya (falls back to description)
    auto_label: str = ""  # built from the member tools' own descriptions


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
            desc = g.get("description", gname)
            auto = "; ".join(first_sentence(tool_description(by_name[m]), 90) for m in members[:3])
            groups.append(Group(gname, desc, members, g.get("label") or desc, auto or desc))
    for n, t in by_name.items():
        if n not in seen:
            d = first_sentence(tool_description(t)) or n
            groups.append(Group(n, d, [n], d, d))
    return groups


# --------------------------------------------------------------------------- Laya

NONE_LABEL = "no tool: small talk, general knowledge, explanation, writing, translation, maths"
NONE = "none"


@dataclass(frozen=True)
class View:
    """One way of asking Laya. Several views are averaged (``LayaSelector(views="ensemble")``)."""

    state_mode: str = "json"  # "json": {"request": q}; "raw": q
    labels: str = "label"  # "label" | "description" | "auto"
    model: Optional[str] = None  # Laya checkpoint ("english", "multilingual", ...); None lets Laya route by language


VIEW_PRESETS: Dict[str, List[View]] = {
    "single": [View()],
    # Different framings make different mistakes; the multilingual checkpoint covers non-English requests.
    "ensemble": [View("json", "label"), View("raw", "auto"), View("raw", "description", "multilingual")],
}


def _as_view(v: Any) -> View:
    return v if isinstance(v, View) else View(**v)


class LayaSelector(Selector):
    """Use a `Laya <https://huggingface.co/convaiinnovations/laya>`_ server as a tool router.

    Laya is a non-autoregressive decision model (~15-100 ms per call). The tool groups
    become the options of a single ``choice`` question; the probabilities decide how
    many groups to keep. ``mode="noul"`` asks one yes/no question per group instead.

    How the question is *asked* matters (see benchmarks/HARNESS.md): the default wraps the
    request as ``{"request": ...}`` and names that field in the instruction (+8 points top-1
    over passing the bare string). Laya reads at most ~11 tokens of each option when there are
    14 of them (``head_max_len`` is 192 tokens in total), so ``label`` texts must be short and
    front-loaded, and *raising* ``head_max_len`` makes it worse.

    ``views``: ``"ensemble"`` (the default: three differently-framed views averaged; on the tuning set
    +6 points top-2 group recall over a single view for ~20 ms more), ``"single"``, or a list of
    :class:`View` / dicts. Passing ``state_mode`` / ``labels`` / ``model`` instead builds one view.

    ``state_mode``: ``"json"`` or ``"raw"``.
    ``labels``: ``"label"`` (group ``label`` or description), ``"description"`` or ``"auto"``
    (the first sentences of the member tools' own descriptions; needs no hand-written text).
    """

    name = "laya"

    def __init__(
        self,
        url: str = "http://127.0.0.1:8000",
        groups: Optional[Dict[str, Any]] = None,
        mode: str = "choice",
        top_p: float = 1.0,
        min_groups: int = 1,
        max_groups: int = 2,
        threshold: float = 0.5,
        instructions: Optional[str] = None,
        state_mode: Optional[str] = None,
        labels: Optional[str] = None,
        model: Optional[str] = None,
        views: Any = None,
        max_query_chars: int = 1500,
        none_threshold: Optional[float] = None,
        none_label: str = NONE_LABEL,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        if mode not in ("choice", "noul"):
            raise ValueError("mode must be 'choice' or 'noul'")
        if state_mode not in (None, "json", "raw"):
            raise ValueError("state_mode must be 'json' or 'raw'")
        if labels not in (None, "label", "description", "auto"):
            raise ValueError("labels must be 'label', 'description' or 'auto'")
        self.cfg = groups or {"always": [], "groups": {}}
        self.mode, self.top_p, self.min_groups, self.max_groups = mode, top_p, min_groups, max_groups
        self.threshold = threshold
        self.max_query_chars = max_query_chars
        if views is None and state_mode is None and labels is None and model is None:
            views = "ensemble"
        if views is None:
            self.views = [View(state_mode or "json", labels or "label", model)]
        elif isinstance(views, str):
            if views not in VIEW_PRESETS:
                raise ValueError("unknown views preset %r (use %s)" % (views, ", ".join(VIEW_PRESETS)))
            self.views = list(VIEW_PRESETS[views])
        else:
            self.views = [_as_view(v) for v in views]
        for v in self.views:
            if v.state_mode not in ("json", "raw") or v.labels not in ("label", "description", "auto"):
                raise ValueError("invalid view %r" % (v,))
        self._custom_instructions = instructions
        if none_threshold is not None and mode != "choice":
            raise ValueError("none_threshold needs mode='choice'")
        self.none_threshold, self.none_label = none_threshold, none_label
        headers = {"Authorization": "Bearer " + api_key} if api_key else {}
        self.client = httpx.AsyncClient(base_url=url.rstrip("/"), headers=headers, timeout=timeout, transport=transport)
        self._cache: Dict[str, Tuple[List[Group], List[Dict[str, Any]]]] = {}

    async def aclose(self) -> None:
        await self.client.aclose()

    def _instruction(self, view: View) -> str:
        if self._custom_instructions:
            return self._custom_instructions
        return "Which kind of tool does `request` need?" if view.state_mode == "json" else "Which kind of tool is needed to handle this request?"

    def _noul_text(self, view: View) -> str:
        return "Does `request` require: {d}?" if view.state_mode == "json" else "Does handling this request require: {d}?"

    def _prepare(self, tools: Sequence[Tool]) -> Tuple[List[Group], List[Dict[str, Any]]]:
        key = hashlib.sha1("\0".join(sorted(tool_name(t) for t in tools)).encode()).hexdigest()
        if key not in self._cache:
            groups = build_groups(self.cfg, tools)
            qsets: List[Dict[str, Any]] = []
            for v in self.views:
                text = {g.name: {"label": g.label, "description": g.description, "auto": g.auto_label}[v.labels] for g in groups}
                if self.mode == "choice":
                    if self.none_threshold is not None:
                        text = dict(text, **{NONE: self.none_label})
                    qsets.append({"tool_group": {"type": "choice", "instructions": self._instruction(v), "criteria": text}})
                else:
                    qsets.append({g.name: {"type": "noul", "instructions": self._noul_text(v).format(d=text[g.name])} for g in groups})
            self._cache = {key: (groups, qsets)}
        return self._cache[key]

    async def _ask(self, view: View, questions: Dict[str, Any], query: str, groups: List[Group]) -> Dict[str, float]:
        state = {"request": query} if view.state_mode == "json" else query
        body: Dict[str, Any] = {"state": state, "questions": questions}
        if view.model:
            body["model"] = view.model
        r = await self.client.post("/v1/systemone", json=body)
        r.raise_for_status()
        answers = r.json()["answers"]
        if self.mode == "choice":
            probs = {k: float(p) for k, p in answers["tool_group"]["probabilities"].items()}
        else:
            probs = {g.name: float(answers[g.name]["noul"]) for g in groups}
        z = sum(probs.values()) or 1.0
        return {k: p / z for k, p in probs.items()}

    async def select(self, query: str, tools: Sequence[Tool]) -> Selection:
        groups, qsets = self._prepare(tools)
        query = query[-self.max_query_chars:]
        per_view = await asyncio.gather(*[self._ask(v, q, query, groups) for v, q in zip(self.views, qsets)])
        probs = {g.name: sum(p.get(g.name, 0.0) for p in per_view) / len(per_view) for g in groups}
        none_p = sum(p.get(NONE, 0.0) for p in per_view) / len(per_view) if self.none_threshold is not None else None
        if none_p is not None and none_p >= self.none_threshold:
            return Selection([], {"none": round(none_p, 3)}, abstain=True)
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])
        if self.mode == "choice":
            kept, mass = [], 0.0
            for name, p in ranked:
                if len(kept) >= self.max_groups or (len(kept) >= self.min_groups and mass >= self.top_p):
                    break
                kept.append(name)
                mass += p
        else:
            kept = [n for n, p in ranked if p >= self.threshold][: self.max_groups]
            if len(kept) < self.min_groups:
                kept = [n for n, _ in ranked[: self.min_groups]]
        members = {g.name: g.tools for g in groups}
        names: List[str] = []
        for g in kept:
            names.extend(n for n in members[g] if n not in names)
        info: Dict[str, Any] = {"groups": {g: round(probs[g], 3) for g in kept}}
        if none_p is not None:
            info["none"] = round(none_p, 3)
        return Selection(names, info)


# --------------------------------------------------------------------------- combinators

class UnionSelector(Selector):
    """Union of several selectors, in order. A selector that raises is skipped.

    ``abstain``: what a selector saying "no tool needed" means for the union --
    ``"all"`` (default): abstain only if every selector abstains or finds nothing,
    ``"any"``: one abstaining selector is enough (a veto).
    """

    def __init__(self, selectors: Sequence[Selector], abstain: str = "all"):
        if abstain not in ("all", "any"):
            raise ValueError("abstain must be 'all' or 'any'")
        self.abstain = abstain
        self.selectors = list(selectors)
        self.name = "+".join(s.name for s in self.selectors)

    async def aclose(self) -> None:
        for s in self.selectors:
            await s.aclose()

    async def select(self, query: str, tools: Sequence[Tool]) -> Selection:
        names: List[str] = []
        info: Dict[str, Any] = {}
        ok = 0
        vetoes = 0
        for s in self.selectors:
            try:
                sel = await s.select(query, tools)
            except Exception as e:  # one broken selector must not take the router down
                info[s.name + "_error"] = type(e).__name__
                continue
            ok += 1
            info[s.name] = sel.info
            vetoes += bool(sel.abstain)
            names.extend(n for n in sel.names if n not in names)
        if not ok:
            raise RuntimeError("all selectors failed")
        if vetoes and (self.abstain == "any" or not names):
            return Selection([], info, abstain=True)
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

    ``options`` maps selector name -> kwargs, e.g. ``{"laya": {"url": ...}, "bm25": {"top_k": 5}}``;
    the key ``"union"`` holds kwargs for :class:`UnionSelector` (e.g. ``{"abstain": "any"}``).
    """
    union_kwargs = options.pop("union", {})
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
    return built[0] if len(built) == 1 else UnionSelector(built, **union_kwargs)
