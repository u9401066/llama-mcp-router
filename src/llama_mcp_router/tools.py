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
