"""Tool definitions: normalisation and where they come from."""
from __future__ import annotations

import fnmatch
import json
import re
import time
from typing import Any, Dict, Iterable, List, Optional

import httpx

Tool = Dict[str, Any]


def normalize_tool(t: Tool) -> Tool:
    """Return an OpenAI ``{"type":"function","function":{...}}`` tool.

    Accepts OpenAI tools, llama-server ``GET /tools`` entries (``{"definition": ...}``)
    and raw MCP tool descriptors (``{"name", "description", "inputSchema"}``).
    """
    if "function" in t:
        return t
    if "definition" in t:
        return normalize_tool(t["definition"])
    if "name" in t:
        return {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("inputSchema") or t.get("parameters") or {"type": "object", "properties": {}},
            },
        }
    raise ValueError("unrecognised tool definition: %r" % (list(t)[:5],))


_DEF_KEYS = ("$defs", "definitions")
_BIG_LIMITS = ("maxLength", "minLength", "maxItems", "minItems", "maxProperties", "minProperties")


def _collect_defs(o: Any, out: Dict[str, Any]) -> None:
    if isinstance(o, dict):
        for k in _DEF_KEYS:
            if isinstance(o.get(k), dict):
                for name, d in o[k].items():
                    out.setdefault(name, d)
        for v in o.values():
            _collect_defs(v, out)
    elif isinstance(o, list):
        for v in o:
            _collect_defs(v, out)


def sanitize_schema(schema: Any, max_limit: int = 64, max_depth: int = 8) -> Any:
    """Make an MCP input schema digestible for llama.cpp's JSON-schema-to-grammar converter.

    * inlines local ``$ref``s (``#/$defs/X``, ``#/definitions/X``), wherever the ``$defs`` block sits -- some
      servers nest it inside a property while referencing it from the root, which llama.cpp rejects
      ("Error resolving ref"); recursive refs stop at ``max_depth`` and become unconstrained
    * drops length / count limits above ``max_limit`` (``maxLength: 2000`` makes llama.cpp unroll a huge
      repetition and fail with "failed to parse grammar"); small limits are kept
    Semantics are otherwise unchanged; the result only loosens what the grammar enforces.
    """
    defs: Dict[str, Any] = {}
    _collect_defs(schema, defs)

    def walk(o: Any, stack: tuple) -> Any:
        if isinstance(o, list):
            return [walk(v, stack) for v in o]
        if not isinstance(o, dict):
            return o
        ref = o.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/"):
            name = ref.rsplit("/", 1)[-1]
            rest = {k: v for k, v in o.items() if k != "$ref"}
            if name in defs and name not in stack and len(stack) < max_depth:
                target = walk(defs[name], stack + (name,))
                return dict(target, **walk(rest, stack)) if isinstance(target, dict) else target
            return walk(rest, stack)  # unresolvable or recursive: leave it unconstrained
        out = {}
        for k, v in o.items():
            if k in _DEF_KEYS:
                continue
            if k in _BIG_LIMITS and isinstance(v, int) and v > max_limit:
                continue
            out[k] = walk(v, stack)
        return out

    return walk(schema, ())


def sanitize_tool(t: Tool) -> Tool:
    f = normalize_tool(t)["function"]
    params = f.get("parameters")
    if not isinstance(params, dict):
        return normalize_tool(t)
    clean = sanitize_schema(params)
    if clean == params:
        return normalize_tool(t)
    return {"type": "function", "function": dict(f, parameters=clean)}


_NOISE_KEYS = ("title", "examples", "$comment", "pattern", "minLength", "maxLength", "readOnly", "writeOnly", "deprecated",
               "discriminator", "contentMediaType", "contentEncoding", "contentSchema")
_DECORATION = re.compile(r"[═─━]{4,}|[=\-_*~#]{8,}")


def compact_text(text: str, limit: Optional[int] = None) -> str:
    """A tool / parameter description with less layout: common indentation, decoration-only lines (``═══``), trailing
    blanks and runs of empty lines removed; cut at a line boundary after ``limit`` characters (with "…")."""
    lines = []
    for ln in str(text or "").expandtabs(4).splitlines():
        if _DECORATION.search(ln):
            body = " ".join(_DECORATION.sub(" ", ln).split())
            if not body:
                continue
            ln = ln[: len(ln) - len(ln.lstrip())] + body
        lines.append(ln.rstrip())
    indents = [len(ln) - len(ln.lstrip()) for ln in lines[1:] if ln.strip()]
    cut = min(indents) if indents else 0
    lines = [lines[0].strip()] + [ln[cut:] for ln in lines[1:]] if lines else []
    out: List[str] = []
    for ln in lines:
        if not ln.strip() and (not out or not out[-1].strip()):
            continue
        out.append(ln)
    s = "\n".join(out).strip()
    if limit and len(s) > limit:
        head = s[:limit]
        nl = head.rfind("\n")
        s = (head[:nl] if nl > limit // 2 else head).rstrip() + "\n…"
    return s


def _is_null(o: Any) -> bool:
    return isinstance(o, dict) and o.get("type") == "null" and len(o) == 1


def _is_free_string(o: Any) -> bool:
    return isinstance(o, dict) and o.get("type") == "string" and not (set(o) - {"type", "description"})


def _encoded_alternative(o: Any) -> bool:
    """A string branch that only re-encodes a structured branch (JSON text, a fenced JSON block...)."""
    return isinstance(o, dict) and o.get("type") == "string" and any(
        k in ("contentSchema", "contentMediaType", "contentEncoding") or k.startswith("x-") for k in o)


def _structured(o: Any) -> bool:
    return isinstance(o, dict) and (o.get("type") in ("object", "array") or any(k in o for k in ("$ref", "oneOf", "anyOf", "properties")))


def _union(branches: List[Any]) -> List[Any]:
    """Simplify anyOf/oneOf branches: drop ``null`` (the parameter can be left out), drop free-form string branches
    next to constants or typed scalars (alternative spellings: ``{"const": "x"} | {"type": "string", "pattern": "(?i)x"}``,
    ``{"type": "integer"} | {"type": "string", "description": "decimal integer"}``) and fold several constants into one
    enum. Every call valid for the result is valid for the original."""
    out = [b for b in branches if not _is_null(b)]
    if any(isinstance(b, dict) and ("const" in b or "enum" in b or b.get("type") in ("integer", "number", "boolean")) for b in out):
        out = [b for b in out if not _is_free_string(b)]  # "or the same value as text": the typed branch is enough
    consts = [b for b in out if isinstance(b, dict) and "const" in b and not (set(b) - {"const", "type", "description"})]
    if len(consts) > 1 and len(consts) == len(out):
        values = [b["const"] for b in consts]
        merged: Dict[str, Any] = {"enum": values}
        if all(isinstance(v, str) for v in values):
            merged["type"] = "string"
        out = [merged]
    return out or branches


def compact_schema(schema: Any, description_limit: Optional[int] = 300) -> Any:
    """A smaller JSON schema for the model's prompt (tool schemas are a large share of an agent's prompt: the pubmed MCP's
    41 tools are ~60k tokens). Drops what does not help the model pick values -- titles, examples, regex patterns,
    string length limits, ``"default": null`` -- simplifies Optional/alias unions (see ``_union``) and shortens
    property descriptions. Types, enums, consts, required, defaults, numeric ranges and ``$ref``/``$defs`` are kept,
    so the schema stays valid; it only enforces less (the MCP server still validates the call)."""

    def walk(o: Any) -> Any:
        if isinstance(o, list):
            return [walk(v) for v in o]
        if not isinstance(o, dict):
            return o
        for u in ("anyOf", "oneOf"):
            if isinstance(o.get(u), list) and any(_structured(b) for b in o[u]):
                o = dict(o, **{u: [b for b in o[u] if not _encoded_alternative(b)]})
        out: Dict[str, Any] = {}
        for k, v in o.items():
            if k in _NOISE_KEYS or k.startswith("x-"):
                continue
            if k == "default" and v is None:
                continue
            if k == "description" and isinstance(v, str):
                v = compact_text(v, description_limit)
                if not v:
                    continue
                out[k] = v
                continue
            if k in ("properties", "$defs", "definitions") and isinstance(v, dict):
                out[k] = {name: walk(sub) for name, sub in v.items()}
                continue
            out[k] = walk(v)
        for u in ("anyOf", "oneOf"):
            if isinstance(out.get(u), list):
                branches = _union(out[u])
                if len(branches) == 1 and isinstance(branches[0], dict):
                    rest = {k: v for k, v in out.items() if k != u}
                    out = dict(branches[0], **rest)
                else:
                    out[u] = branches
        return out

    return walk(schema)


def compact_tool(t: Tool, description_limit: Optional[int] = None, property_limit: Optional[int] = 300) -> Tool:
    """``compact_schema`` for the parameters plus ``compact_text`` for the tool's description (no length limit by
    default: MCP servers often document their parameters there)."""
    f = normalize_tool(t)["function"]
    out = dict(f, description=compact_text(f.get("description") or "", description_limit))
    if isinstance(f.get("parameters"), dict):
        out["parameters"] = compact_schema(f["parameters"], property_limit)
    return {"type": "function", "function": out}


def tool_name(t: Tool) -> str:
    return normalize_tool(t)["function"]["name"]


def tool_description(t: Tool) -> str:
    return normalize_tool(t)["function"].get("description") or ""


def first_sentence(text: str, limit: int = 200) -> str:
    """A short human description: first meaningful line/sentence of a (often huge) tool doc."""
    for line in text.splitlines():
        line = re.sub(r"^[\W_]+", "", line).strip()  # leading emoji, box-drawing, markdown
        if len(line) > 8:
            text = line
            break
    text = " ".join(text.split())
    cut = text.find(". ")
    if 20 < cut < limit:
        text = text[: cut + 1]
    return text[:limit]


def exclude_tools(tools: Iterable[Tool], patterns: Iterable[str]) -> List[Tool]:
    pats = list(patterns)
    return [t for t in tools if not any(fnmatch.fnmatch(tool_name(t), p) for p in pats)]


class ServerToolSource:
    """Tools exposed by llama-server's ``GET /tools`` (built-in tools and ``--mcp-servers-config``).

    Results are cached for ``ttl`` seconds. Servers without ``/tools`` yield an empty list.
    """

    def __init__(self, client: httpx.AsyncClient, ttl: float = 60.0):
        self.client = client
        self.ttl = ttl
        self._cache: Optional[List[Tool]] = None
        self._at = 0.0

    async def fetch(self) -> List[Tool]:
        now = time.monotonic()
        if self._cache is not None and now - self._at < self.ttl:
            return self._cache
        try:
            r = await self.client.get("/tools", timeout=15)
            r.raise_for_status()
            data = r.json()
            tools = [normalize_tool(t) for t in data] if isinstance(data, list) else []
        except (httpx.HTTPError, ValueError):
            tools = self._cache or []
        self._cache, self._at = tools, now
        return tools


class FileToolSource:
    """Tools from a JSON file (OpenAI tools list, MCP ``tools/list`` result, or llama-server ``/tools``)."""

    def __init__(self, path: str):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("tools", [])
        self._tools = [normalize_tool(t) for t in data]

    async def fetch(self) -> List[Tool]:
        return self._tools
