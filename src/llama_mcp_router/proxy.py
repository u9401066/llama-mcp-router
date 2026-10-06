"""OpenAI-compatible proxy that sends the model only the tools it needs."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Set, Tuple

import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from .agent import AcpError, AgentConfig, AgentManager, sum_timings
from .sentinel import SentinelConfig, SentinelRun, SystemOneCritic
from .selectors import AllSelector, Selector
from .tools import ServerToolSource, Tool, exclude_tools, first_sentence, normalize_tool, sanitize_tool, tool_description, tool_name

log = logging.getLogger("llama_mcp_router")

_HOP = {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "te", "upgrade"}


@dataclass
class RouterConfig:
    backend: str = "http://127.0.0.1:8080"
    selector: Selector = field(default_factory=AllSelector)
    always: List[str] = field(default_factory=list)  # tool names always sent
    max_tools: int = 12  # cap on selector output (always/continuation tools are extra)
    mode: str = "inject"  # "inject": client runs tool calls; "agent": router runs llama-server tools
    max_iterations: int = 8
    use_server_tools: bool = True  # merge llama-server's GET /tools into the candidate pool
    exclude: List[str] = field(default_factory=list)  # fnmatch patterns of tools never offered
    fallback: str = "all"  # when the selector fails: "all" tools or "none"
    apply: str = "select"  # "select": send only the selection; "reorder": send ALL tools, most relevant first; "all": send all tools unchanged
    hint: bool = False  # append the selector's routing hint to the last user message
    escalate: bool = False  # add a catalog meta-tool; if the model calls it, rerun with the tools it asked for
    max_escalations: int = 1
    catalog_max: int = 80  # list left-out tools by name up to this many; beyond it the meta-tool takes a search query
    search_k: int = 8  # tools loaded per escalation search query
    sanitize: bool = True  # inline $refs / drop huge length limits so llama.cpp can build a grammar for every tool
    agent: Optional[AgentConfig] = None  # run an ACP agent per conversation for model=agent.model_id / trigger prefix
    sentinel: Optional[SentinelConfig] = None  # step-checked reasoning for model=sentinel.model_id / trigger prefix (e.g. "med:")
    sticky: bool = False  # opt-in: keep a conversation's tool list append-only so llama-server's prompt cache can keep hitting
    sticky_conversations: int = 512  # how many conversations to remember
    tools_ttl: float = 60.0
    request_timeout: float = 600.0


def last_user_query(messages: Sequence[Dict[str, Any]]) -> str:
    """Text the selector sees: the last user turn (plus the one before it if that is very short)."""
    texts: List[str] = []
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        c = m.get("content")
        text = c if isinstance(c, str) else " ".join(p.get("text", "") for p in (c or []) if isinstance(p, dict) and p.get("type") == "text")
        texts.append(text.strip())
        if len(text.strip()) >= 20 or len(texts) >= 2:
            break
    return "\n".join(reversed(texts))


def conversation_key(messages: Sequence[Dict[str, Any]]) -> Optional[str]:
    """Identify a conversation by its system prompt and first user message (stateless clients resend both)."""
    parts: List[str] = []
    for m in messages:
        if m.get("role") in ("system", "developer", "user"):
            parts.append("%s:%s" % (m["role"], json.dumps(m.get("content"), sort_keys=True)))
        if m.get("role") == "user":
            return hashlib.sha1("\0".join(parts).encode()).hexdigest()
    return None


def arrange(pool: Sequence[Tool], sel: Any, mode: str, max_tools: int, always: Sequence[str] = (), extra: Set[str] = frozenset()) -> List[Tool]:
    """Turn a Selection into the tool list that is actually sent (shared with the benchmarks)."""
    if mode == "all":
        return list(pool)
    if mode == "reorder":
        order = {n: i for i, n in enumerate(list(sel.ranking) or list(sel.names))}
        return sorted(pool, key=lambda t: order.get(tool_name(t), len(order)))
    keep = set(sel.names[:max_tools]) | set(always) | set(extra)
    return [t for t in pool if tool_name(t) in keep]


META_TOOL = "router_load_tools"


def _server_summary(tools: Sequence[Tool], limit: int = 40) -> str:
    counts: Dict[str, int] = {}
    for t in tools:
        p = tool_name(t).split("_", 1)[0]
        counts[p] = counts.get(p, 0) + 1
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    out = ", ".join("%s (%d)" % kv for kv in items[:limit])
    return out + (", ..." if len(items) > limit else "")


def catalog_tool(pool: Sequence[Tool], sent: Sequence[Tool], desc_chars: int = 80, max_listed: Optional[int] = None) -> Optional[Tool]:
    """A meta-tool for the tools that were *not* sent, so the model can ask for them.

    Up to ``max_listed`` left-out tools: they are listed by name + one line (~15 tokens each) and the model
    loads them by name. Beyond that the listing itself would be too big, so the meta-tool takes a free-text
    ``query`` (the router searches the pool for it) and only summarises which servers are available.
    It turns a selector miss into one extra round-trip instead of a wrong tool call.
    """
    sent_names = {tool_name(t) for t in sent}
    rest = [t for t in pool if tool_name(t) not in sent_names]
    if not rest:
        return None
    if max_listed is not None and len(rest) > max_listed:
        return {
            "type": "function",
            "function": {
                "name": META_TOOL,
                "description": (
                    "Only the tools most likely needed are loaded. If none of the loaded tools fits the request, call this FIRST "
                    "with a short description of the tool you need; matching tools are loaded and you can call them right after. "
                    "%d more tools are available, prefixed by server: %s." % (len(rest), _server_summary(rest))
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "what the needed tool should do, e.g. 'convert a docx file to pdf'"},
                        "names": {"type": "array", "items": {"type": "string"}, "description": "exact tool names, if you know them"},
                    },
                    "required": ["query"],
                },
            },
        }
    lines = "\n".join("- %s: %s" % (tool_name(t), first_sentence(tool_description(t), desc_chars)) for t in rest)
    return {
        "type": "function",
        "function": {
            "name": META_TOOL,
            "description": (
                "Only the tools most likely needed are loaded. If none of the loaded tools fits the request, "
                "call this FIRST to load the right ones by name; you can call them right after. Not loaded yet:\n" + lines
            ),
            "parameters": {
                "type": "object",
                "properties": {"names": {"type": "array", "items": {"type": "string", "enum": [tool_name(t) for t in rest]}, "description": "tool names to load"}},
                "required": ["names"],
            },
        },
    }


def _args(arguments: Any) -> Dict[str, Any]:
    try:
        a = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
        return a if isinstance(a, dict) else {}
    except ValueError:
        return {}


def requested_tools(arguments: Any) -> List[str]:
    names = _args(arguments).get("names") or []
    return [n for n in names if isinstance(n, str)] if isinstance(names, list) else []


def requested_query(arguments: Any) -> str:
    q = _args(arguments).get("query")
    return q.strip() if isinstance(q, str) else ""


def expand_tools(pool: Sequence[Tool], sent: Sequence[Tool], names: Sequence[str]) -> List[Tool]:
    """Tools for the rerun: what was sent plus what the model asked for (all tools if it asked for nothing valid)."""
    known = {tool_name(t) for t in pool}
    want = set(names) & known
    if not want:
        return list(pool)
    keep = {tool_name(t) for t in sent if tool_name(t) != META_TOOL} | want
    return [t for t in pool if tool_name(t) in keep]


def add_hint(messages: List[Dict[str, Any]], hint: str) -> List[Dict[str, Any]]:
    """Append a routing hint to the last user message (late in the prompt, so the cached prefix is untouched)."""
    if not hint:
        return messages
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m.get("role") == "user":
            note = "\n\n[Routing hint from a fast classifier: this request most likely concerns: %s. Prefer tools for that; ignore the hint if it clearly does not fit.]" % hint
            c = m.get("content")
            if isinstance(c, str):
                m["content"] = c + note
            elif isinstance(c, list):
                m["content"] = list(c) + [{"type": "text", "text": note}]
            break
    return out


def _msg_text(m: Dict[str, Any]) -> str:
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _strip_prefix(messages: List[Dict[str, Any]], prefixes: Sequence[str]) -> None:
    """Remove an opt-out prefix (e.g. 'chat:') from the first user message before it reaches the model."""
    for m in messages:
        if m.get("role") != "user":
            continue
        c = m.get("content")
        for p in prefixes:
            if isinstance(c, str) and c.lstrip().lower().startswith(p.lower()):
                m["content"] = c.lstrip()[len(p):].lstrip()
        return


def _agent_summary(ev: Dict[str, Any]) -> str:
    sid = ev.get("session")
    base = "/agent/sessions/%s" % sid
    lines = ["", "", "---"]
    if ev.get("changed"):
        lines.append("**Files changed** (commit `%s`):" % ev.get("commit"))
        for c in ev["changed"]:
            mark = {"A": "added", "M": "modified", "D": "deleted"}.get(c["status"], c["status"])
            lines.append("- [%s](%s/files/%s) — %s" % (c["path"], base, c["path"], mark) if c["status"] != "D" else "- %s — deleted" % c["path"])
    else:
        lines.append("No files changed.")
    lines.append("[Workspace](%s) · [Download zip](%s/archive.zip)" % (base, base))
    return "\n".join(lines)


def called_tool_names(messages: Sequence[Dict[str, Any]]) -> Set[str]:
    names: Set[str] = set()
    for m in messages:
        for tc in m.get("tool_calls") or []:
            n = (tc.get("function") or {}).get("name")
            if n:
                names.add(n)
    return names


def _forward_headers(request: Request) -> Dict[str, str]:
    """Client headers minus hop-by-hop ones. Accept-Encoding is passed through as sent (or 'identity'), so the
    backend compresses exactly as it would for the client; raw-proxied bodies then keep their Content-Encoding."""
    out = {k: v for k, v in request.headers.items() if k.lower() not in _HOP}
    if not any(k.lower() == "accept-encoding" for k in out):
        out["accept-encoding"] = "identity"
    return out


def _result_text(res: Any) -> str:
    if isinstance(res, dict):
        if "plain_text_response" in res:
            return str(res["plain_text_response"])
        if "error" in res and len(res) == 1:
            return "Error: %s" % res["error"]
    return res if isinstance(res, str) else json.dumps(res, ensure_ascii=False)


class AgentStream:
    """A streamed agent turn that outlives its HTTP connection and can be replayed from a byte offset: the resumable-stream
    protocol of llama-server's Web UI (POST /v1/streams/lookup, GET /v1/stream?conv_id=&from=, DELETE /v1/stream = Stop)."""

    KEEP_S = 900
    MAX = 256

    def __init__(self, sid: str, owner: str):
        self.id, self.owner = sid, owner
        self.buf = bytearray()
        self.done = False
        self.started, self.completed = int(time.time()), 0
        self.task: Optional["asyncio.Future[None]"] = None
        self._changed = asyncio.Event()

    def append(self, piece: str) -> None:
        self.buf += piece.encode("utf-8")
        self._changed.set()

    def finish(self) -> None:
        self.done, self.completed = True, int(time.time())
        self._changed.set()

    async def follow(self, start: int = 0) -> AsyncIterator[bytes]:
        pos = max(0, start)
        while True:
            if pos < len(self.buf):
                piece = bytes(self.buf[pos:])
                pos += len(piece)
                yield piece
            elif self.done:
                return
            else:
                self._changed.clear()
                await self._changed.wait()

    def info(self) -> Dict[str, Any]:
        return {"conversation_id": self.id, "is_done": self.done, "total_bytes": len(self.buf), "started_at": self.started, "completed_at": self.completed}


class Router:
    def __init__(self, config: RouterConfig, transport: Optional[httpx.AsyncBaseTransport] = None):
        self.cfg = config
        self.client = httpx.AsyncClient(base_url=config.backend.rstrip("/"), timeout=config.request_timeout, transport=transport)
        self.source = ServerToolSource(self.client, ttl=config.tools_ttl)
        self._sticky: "OrderedDict[str, List[str]]" = OrderedDict()
        self._fallback_retriever: Any = None
        self._clean: Dict[str, Tool] = {}
        self.agents: Optional[AgentManager] = AgentManager(config.agent) if config.agent else None
        self.streams: "OrderedDict[str, AgentStream]" = OrderedDict()
        self.critic: Optional[SystemOneCritic] = SystemOneCritic(config.sentinel) if config.sentinel else None

    async def aclose(self) -> None:
        for st in self.streams.values():
            if st.task and not st.task.done():
                st.task.cancel()
        await self.client.aclose()
        await self.cfg.selector.aclose()
        if self.critic:
            await self.critic.aclose()
        if self.agents:
            await self.agents.aclose()

    # ------------------------------------------------------------------ selection
    async def pool(self, client_tools: Sequence[Tool]) -> List[Tool]:
        pool = [normalize_tool(t) for t in client_tools]
        known = {tool_name(t) for t in pool}
        if self.cfg.use_server_tools:
            pool += [t for t in await self.source.fetch() if tool_name(t) not in known]
        return exclude_tools(pool, self.cfg.exclude)

    async def choose(self, query: str, pool: Sequence[Tool], extra: Set[str], conv_key: Optional[str] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        info: Dict[str, Any] = {}
        hint = ""
        try:
            sel = await self.cfg.selector.select(query, pool)
            info = sel.info
            chosen = arrange(pool, sel, self.cfg.apply, self.cfg.max_tools, self.cfg.always, extra)
            hint = sel.hint if self.cfg.hint else ""
        except Exception as e:
            log.warning("selector %s failed (%s: %s); fallback=%s", self.cfg.selector.name, type(e).__name__, e, self.cfg.fallback)
            keep = set(self.cfg.always) | extra
            chosen = list(pool) if self.cfg.fallback == "all" else [t for t in pool if tool_name(t) in keep]
            info = {"error": type(e).__name__}
        if conv_key and self.cfg.sticky and self.cfg.apply == "select":
            chosen = self._stabilise(conv_key, chosen, {tool_name(t): t for t in pool})
        return {"tools": chosen, "info": info, "hint": hint, "ms": round((time.perf_counter() - t0) * 1000, 1), "pool": len(pool)}

    def _stabilise(self, key: str, chosen: List[Tool], lookup: Dict[str, Tool]) -> List[Tool]:
        """Reuse the previous turn's tool list when it already covers this turn's choice.

        The tool schemas sit at the very start of the prompt, so any change to them forces llama-server to
        re-read the whole conversation. A list that only ever grows (new tools appended, order kept) lets
        the prompt cache keep hitting on follow-up turns.
        """
        by_name = {tool_name(t): t for t in chosen}
        prev = self._sticky.pop(key, [])
        new = set(by_name)
        final = prev if new <= set(prev) else prev + [n for n in by_name if n not in prev]
        final = final[-(self.cfg.max_tools * 2 + len(self.cfg.always)):]
        self._sticky[key] = final
        while len(self._sticky) > self.cfg.sticky_conversations:
            self._sticky.popitem(last=False)
        return [lookup[n] for n in final if n in lookup]

    # ------------------------------------------------------------------ chat
    async def chat(self, request: Request) -> Response:
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"error": {"message": "invalid JSON body"}}, status_code=400)
        if self.critic and self._sentinel_trigger(body) is not None:
            return await self._sentinel_chat(body)
        if self.agents and self._agent_trigger(body) is not None:
            return await self._agent_chat(body, request)
        if self.agents and self.cfg.agent and self.cfg.agent.default:
            _strip_prefix(body.get("messages") or [], self.cfg.agent.optout)
        bypass = request.headers.get("x-router-bypass") or body.pop("router", None) is False
        headers = _forward_headers(request)
        if bypass:
            return await self._relay(body, headers)

        client_names = {tool_name(t) for t in body.get("tools") or []}
        pool = await self.pool(body.get("tools") or [])
        if not pool:
            return await self._relay(body, headers)

        extra = called_tool_names(body.get("messages") or [])
        tc = body.get("tool_choice")
        if isinstance(tc, dict) and (tc.get("function") or {}).get("name"):
            extra.add(tc["function"]["name"])
        choice = await self.choose(last_user_query(body.get("messages") or []), pool, extra, conversation_key(body.get("messages") or []))
        if choice["hint"]:
            body["messages"] = add_hint(body.get("messages") or [], choice["hint"])
        if self.cfg.sanitize:
            pool = self._sanitized(pool)
            choice["tools"] = self._sanitized(choice["tools"])
        meta_tool = self._catalog(pool, choice["tools"]) if self.cfg.escalate else None
        sent = choice["tools"] + ([meta_tool] if meta_tool else [])
        if sent:
            body["tools"] = sent
        else:  # nothing selected: send a tool-free prompt (also drop tool_choice, which would force a call)
            body.pop("tools", None)
            body.pop("tool_choice", None)
        meta = {"x-router-tools": ",".join(tool_name(t) for t in choice["tools"])[:1000], "x-router-pool": str(choice["pool"]), "x-router-select-ms": str(choice["ms"])}
        log.info("selected %d/%d tools in %sms: %s", len(choice["tools"]), choice["pool"], choice["ms"], meta["x-router-tools"][:200])

        if self.cfg.mode == "agent":
            server_names = {tool_name(t) for t in await self.source.fetch()} - client_names
            return await self._agent(body, headers, server_names, meta, choice, pool)
        if meta_tool:
            return await (self._escalating_stream(body, headers, meta, pool) if body.get("stream") else self._escalating_json(body, headers, meta, pool))
        return await self._relay(body, headers, meta)

    # ------------------------------------------------------------------ escalation (catalog meta-tool)
    def _retriever(self) -> Any:
        sel = self.cfg.selector
        for s in [sel] + list(getattr(sel, "selectors", [])):
            if getattr(s, "retriever", None) is not None:
                return s.retriever
        if self._fallback_retriever is None:
            from .retrieval import Retriever

            self._fallback_retriever = Retriever()
        return self._fallback_retriever

    def _sanitized(self, tools: Sequence[Tool]) -> List[Tool]:
        out = []
        for t in tools:
            key = json.dumps(t, sort_keys=True)
            if key not in self._clean:
                if len(self._clean) > 5000:
                    self._clean.clear()
                self._clean[key] = sanitize_tool(t)
            out.append(self._clean[key])
        return out

    def _catalog(self, pool: Sequence[Tool], sent: Sequence[Tool]) -> Optional[Tool]:
        return catalog_tool(pool, sent, max_listed=self.cfg.catalog_max)

    async def _rerun_body(self, body: Dict[str, Any], pool: Sequence[Tool], calls: Sequence[Dict[str, Any]], rounds_left: int) -> Tuple[Dict[str, Any], List[str]]:
        sent = [t for t in body.get("tools") or [] if tool_name(t) != META_TOOL]
        names = [n for c in calls for n in requested_tools(c.get("arguments"))]
        queries = [q for c in calls for q in [requested_query(c.get("arguments"))] if q]
        known = {tool_name(t) for t in pool}
        found: List[str] = [n for n in names if n in known]
        for q in queries:
            try:
                found += [n for n in (await self._retriever().rank(q, pool))[: self.cfg.search_k] if n not in found]
            except Exception as e:  # noqa: BLE001
                log.warning("escalation search failed (%s: %s)", type(e).__name__, e)
        if found or len(pool) <= self.cfg.catalog_max:
            tools = expand_tools(pool, sent, found)  # nothing valid in a small pool -> everything
        else:  # nothing usable and the pool is too big to send whole: search with the user's request instead
            q = last_user_query(body.get("messages") or [])
            found = (await self._retriever().rank(q, pool))[: self.cfg.search_k * 2]
            tools = expand_tools(pool, sent, found)
        meta_tool = self._catalog(pool, tools) if rounds_left > 0 else None
        out = dict(body, tools=tools + ([meta_tool] if meta_tool else []))
        if isinstance(out.get("tool_choice"), dict) and (out["tool_choice"].get("function") or {}).get("name") == META_TOOL:
            out.pop("tool_choice")
        return out, found

    async def _escalating_json(self, body: Dict[str, Any], headers: Dict[str, str], meta: Dict[str, str], pool: Sequence[Tool]) -> Response:
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        loaded: List[str] = []
        for rnd in range(self.cfg.max_escalations + 1):
            r = await self.client.post("/v1/chat/completions", json=body, headers=headers)
            if r.status_code != 200:
                return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))
            data = r.json()
            for k in usage:
                usage[k] += (data.get("usage") or {}).get(k, 0)
            calls = ((data.get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or []
            metas = [c["function"] for c in calls if (c.get("function") or {}).get("name") == META_TOOL]
            if not metas or rnd == self.cfg.max_escalations:
                break
            body, found = await self._rerun_body(body, pool, metas, self.cfg.max_escalations - rnd - 1)
            loaded += found
            log.info("model asked for %s; loaded %s; rerunning", [m.get("arguments") for m in metas], found)
        data["usage"] = usage
        if loaded:
            meta = dict(meta, **{"x-router-loaded": ",".join(loaded)[:1000]})
        return JSONResponse(data, headers=meta)

    async def _escalating_stream(self, body: Dict[str, Any], headers: Dict[str, str], meta: Dict[str, str], pool: Sequence[Tool]) -> Response:
        """Stream text/reasoning through immediately; hold tool-call chunks until the end of each round,
        and if the model called the meta-tool, swallow that round's tail and stream a rerun instead."""

        async def gen():
            nonlocal body
            for rnd in range(self.cfg.max_escalations + 1):
                last = rnd == self.cfg.max_escalations
                held: List[str] = []
                calls: Dict[int, Dict[str, str]] = {}
                async with self.client.stream("POST", "/v1/chat/completions", json=body, headers=headers) as r:
                    if r.status_code != 200:
                        yield await r.aread()
                        return
                    async for line in r.aiter_lines():
                        if not line:
                            continue
                        out = line + "\n\n"
                        if line.startswith("data:") and line[5:].strip() != "[DONE]":
                            try:
                                chunk = json.loads(line[5:])
                            except ValueError:
                                chunk = {}
                            for ch in chunk.get("choices") or []:
                                for tc in (ch.get("delta") or {}).get("tool_calls") or []:
                                    c = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                                    f = tc.get("function") or {}
                                    c["name"] += f.get("name") or ""
                                    c["arguments"] += f.get("arguments") or ""
                        if calls and not last:
                            held.append(out)
                        else:
                            yield out.encode()
                metas = [c for c in calls.values() if c["name"] == META_TOOL]
                if last or not metas:
                    for h in held:
                        yield h.encode()
                    return
                body, found = await self._rerun_body(body, pool, metas, self.cfg.max_escalations - rnd - 1)
                log.info("model asked for %s; loaded %s; rerunning (stream)", [m["arguments"] for m in metas], found)
                note = {"choices": [{"index": 0, "delta": {"reasoning_content": "\n[router: loading tools %s]\n" % (", ".join(found) or "all")}, "finish_reason": None}], "object": "chat.completion.chunk"}
                yield ("data: %s\n\n" % json.dumps(note)).encode()

        return StreamingResponse(gen(), media_type="text/event-stream", headers=meta)

    async def _relay(self, body: Dict[str, Any], headers: Dict[str, str], meta: Optional[Dict[str, str]] = None) -> Response:
        req = self.client.build_request("POST", "/v1/chat/completions", json=body, headers=headers)
        r = await self.client.send(req, stream=True)
        out = {k: v for k, v in r.headers.items() if k.lower() not in _HOP}
        out.update(meta or {})
        return StreamingResponse(r.aiter_raw(), status_code=r.status_code, headers=out, background=BackgroundTask(r.aclose))

    async def _agent(self, body: Dict[str, Any], headers: Dict[str, str], server_names: Set[str], meta: Dict[str, str], choice: Dict[str, Any], pool: Sequence[Tool] = ()) -> Response:
        want_stream = bool(body.pop("stream", False))
        body.pop("stream_options", None)
        messages = list(body.get("messages") or [])
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        data: Dict[str, Any] = {}
        iterations = 0
        escalations = 0
        for iterations in range(1, self.cfg.max_iterations + 1):
            r = await self.client.post("/v1/chat/completions", json={**body, "messages": messages}, headers=headers)
            if r.status_code != 200:
                return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))
            data = r.json()
            for k in usage:
                usage[k] += (data.get("usage") or {}).get(k, 0)
            msg = data["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            metas = [c["function"] for c in calls if c["function"]["name"] == META_TOOL]
            if metas and escalations < self.cfg.max_escalations:
                escalations += 1
                body, _ = await self._rerun_body(body, pool, metas, self.cfg.max_escalations - escalations)
                continue
            if not calls or any(c["function"]["name"] not in server_names for c in calls):
                break  # final answer, or the client has to run one of its own tools
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
            for c in calls:
                messages.append({"role": "tool", "tool_call_id": c.get("id"), "content": await self._run_tool(c["function"])})
        data["usage"] = usage
        data["router"] = {"selected": [tool_name(t) for t in choice["tools"]], "pool": choice["pool"], "select_ms": choice["ms"], "iterations": iterations, "escalations": escalations, "info": choice["info"]}
        if not want_stream:
            return JSONResponse(data, headers=meta)
        return StreamingResponse(self._fake_stream(data), media_type="text/event-stream", headers=meta)

    async def _run_tool(self, fn: Dict[str, Any]) -> str:
        try:
            args = json.loads(fn.get("arguments") or "{}")
            r = await self.client.post("/tools", json={"tool": fn["name"], "params": args}, timeout=self.cfg.request_timeout)
            return _result_text(r.json())
        except Exception as e:
            return "Error running %s: %s: %s" % (fn.get("name"), type(e).__name__, e)

    @staticmethod
    async def _fake_stream(data: Dict[str, Any]):
        msg = data["choices"][0]["message"]
        delta = {k: v for k, v in msg.items() if v is not None}
        base = {"id": data.get("id", "router"), "object": "chat.completion.chunk", "created": data.get("created", int(time.time())), "model": data.get("model", "")}
        yield "data: %s\n\n" % json.dumps({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}, ensure_ascii=False)
        yield "data: %s\n\n" % json.dumps({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": data["choices"][0].get("finish_reason")}], "usage": data.get("usage")}, ensure_ascii=False)
        yield "data: [DONE]\n\n"

    # ------------------------------------------------------------------ agent bridge
    def _sentinel_trigger(self, body: Dict[str, Any]) -> Optional[str]:
        """'' when the request is for the sentinel by model name, the matched prefix when by trigger, else None."""
        cfg = self.cfg.sentinel
        if cfg is None:
            return None
        if body.get("model") == cfg.model_id:
            return ""
        first = next((_msg_text(m) for m in body.get("messages") or [] if m.get("role") == "user"), "")
        return next((t for t in cfg.triggers if first.lstrip().lower().startswith(t.lower())), None)

    async def _sentinel_chat(self, body: Dict[str, Any]) -> Response:
        """Answer with step-checked reasoning (see sentinel.py). No tools: the client's tool list is ignored."""
        assert self.critic and self.cfg.sentinel
        messages = [dict(m) for m in body.get("messages") or [] if m.get("role") in ("system", "user", "assistant")]
        for m in messages:
            m.pop("reasoning_content", None)
            m.pop("tool_calls", None)
        _strip_prefix(messages, self.cfg.sentinel.triggers)
        options = {k: body[k] for k in ("temperature", "top_p", "chat_template_kwargs") if k in body}
        run = SentinelRun(self.cfg.sentinel, self.client, self.critic, messages, options)
        model = body.get("model") or self.cfg.sentinel.model_id
        created = int(time.time())
        cid = "chatcmpl-sentinel-%d" % created

        def chunk(delta: Dict[str, Any], finish: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> str:
            data: Dict[str, Any] = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            data.update(extra or {})
            return "data: %s\n\n" % json.dumps(data, ensure_ascii=False)

        def report(ev: Dict[str, Any]) -> Dict[str, Any]:
            return {"sentinel": {k: ev[k] for k in ("interventions", "flagged", "steps_checked", "critic_ms", "attempts")}, "timings": sum_timings(ev["timings"]) or {}}

        if body.get("stream"):
            async def gen():
                yield chunk({"role": "assistant", "content": ""})
                try:
                    async for ev in run.events():
                        if ev["type"] in ("reasoning", "note"):
                            yield chunk({"reasoning_content": ev["text"]})
                        elif ev["type"] == "content":
                            yield chunk({"content": ev["text"]})
                        elif ev["type"] == "done":
                            yield chunk({}, "stop", report(ev))
                except httpx.HTTPError as e:
                    log.warning("sentinel answer failed: %s: %s", type(e).__name__, e)
                    yield chunk({"content": "\n\n**Sentinel error:** %s" % e}, "stop")
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream", headers={"x-router-sentinel": "1"})
        content, reasoning, final = "", "", {}
        async for ev in run.events():
            if ev["type"] in ("reasoning", "note"):
                reasoning += ev["text"]
            elif ev["type"] == "content":
                content += ev["text"]
            elif ev["type"] == "done":
                final = report(ev)
        msg: Dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning:
            msg["reasoning_content"] = reasoning
        t = final.get("timings") or {}
        usage = {"prompt_tokens": int(t.get("prompt_n", 0) + t.get("cache_n", 0)), "completion_tokens": int(t.get("predicted_n", 0))}
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        return JSONResponse({"id": cid, "object": "chat.completion", "created": created, "model": model,
                             "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}], "usage": usage, **final})

    def _agent_trigger(self, body: Dict[str, Any]) -> Optional[str]:
        """'' when the request is for the agent by model name, the matched prefix when by trigger, else None."""
        cfg = self.cfg.agent
        if cfg is None:
            return None
        if body.get("model") == cfg.model_id:
            return ""
        first = next((_msg_text(m) for m in body.get("messages") or [] if m.get("role") == "user"), "")
        for t in cfg.triggers:
            if first.lstrip().lower().startswith(t.lower()):
                return t
        if cfg.default and first.strip() and not any(first.lstrip().lower().startswith(o.lower()) for o in cfg.optout):
            return ""
        return None

    def _agent_owner(self, request: Request) -> Optional[str]:
        """The user an agent request belongs to: 'anonymous' without a users map, None if a required key is missing/wrong."""
        users = self.cfg.agent.users if self.cfg.agent else {}
        if not users:
            return "anonymous"
        auth = request.headers.get("authorization") or ""
        key = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        return users.get(key)

    @staticmethod
    def _conversation_id(request: Request) -> Optional[str]:
        """The Web UI sends 'X-Conversation-Id: <conversation>::<model>' with every streamed chat request."""
        raw = (request.headers.get("x-conversation-id") or "").split("::", 1)[0].strip()
        return raw if raw and len(raw) <= 200 and re.fullmatch(r"[A-Za-z0-9_.:@-]+", raw) else None

    async def _agent_chat(self, body: Dict[str, Any], request: Request) -> Response:
        assert self.agents and self.cfg.agent
        owner = self._agent_owner(request)
        if owner is None:
            return JSONResponse({"error": {"message": "This agent needs a valid API key: set it in the Web UI (Settings → API key) or send "
                                                      "'Authorization: Bearer <key>'.", "type": "authentication_error"}}, status_code=401)
        messages = body.get("messages") or []
        trigger = self._agent_trigger(body) or ""
        users = [m for m in messages if m.get("role") == "user"]
        text = _msg_text(users[-1]) if users else ""
        if trigger and text.lstrip().lower().startswith(trigger.lower()):
            text = text.lstrip()[len(trigger):].strip()
        conv = self._conversation_id(request)
        key = "agent:%s:%s" % (owner, ("c:" + conv) if conv else ("h:" + (conversation_key(messages) or "default")))
        model = body.get("model") or self.cfg.agent.model_id
        created = int(time.time())
        cid = "chatcmpl-agent-%d" % created

        def chunk(delta: Dict[str, Any], finish: Optional[str] = None, timings: Optional[Dict[str, Any]] = None,
                  progress: Optional[Dict[str, Any]] = None) -> str:
            data: Dict[str, Any] = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if timings:  # llama-server's per-request stats, summed over the agent's model calls: the Web UI shows tokens + speed
                data["timings"] = timings
            if progress:  # prompt processing of the agent's current model call
                data["prompt_progress"] = progress
            return "data: %s\n\n" % json.dumps(data, ensure_ascii=False)

        async def events():
            if not text:
                yield {"type": "message", "text": "Send a task after `%s`, e.g. `%s write a script that ...`." % (trigger or "/agent", trigger or "/agent")}
                return
            try:
                async for ev in self.agents.turn(key, text, owner):  # type: ignore[union-attr]
                    yield ev
            except (AcpError, OSError, RuntimeError, asyncio.TimeoutError) as e:
                log.warning("agent turn failed: %s: %s", type(e).__name__, e)
                yield {"type": "message", "text": "\n\n**Agent error:** %s" % e}

        def render(ev: Dict[str, Any]) -> Dict[str, Any]:
            t = ev["type"]
            if t == "message":
                return {"content": ev["text"]}
            if t == "thought":
                return {"reasoning_content": ev["text"]}
            if t == "status":
                return {"reasoning_content": "[agent: %s]\n" % ev["text"]}
            if t == "tool":
                label = ev.get("title") or ev.get("kind") or "tool"
                if ev.get("new"):
                    return {"reasoning_content": "\n▶ %s\n" % label}
                if ev.get("status") in ("completed", "failed"):
                    return {"reasoning_content": "%s %s\n" % ("✓" if ev["status"] == "completed" else "✗", label)}
                return {}
            if t == "plan":
                return {"reasoning_content": "\nPlan:\n" + "".join("- %s\n" % e for e in ev.get("entries") or [])}
            if t == "permission":
                return {"reasoning_content": "\n[permission %s: %s]\n" % ("granted" if ev.get("granted") else "denied", ev.get("title"))}
            if t == "done":
                return {"content": _agent_summary(ev)}
            return {}

        if body.get("stream"):
            async def gen():
                yield chunk({"role": "assistant", "content": ""})
                last: Optional[Dict[str, Any]] = None
                async for ev in events():
                    if ev["type"] == "timings":
                        last = ev.get("timings") or last
                        yield chunk({}, timings=ev.get("timings"), progress=ev.get("prompt_progress"))
                        continue
                    last = ev.get("timings") or last
                    d = render(ev)
                    if d:
                        yield chunk(d)
                yield chunk({}, "stop", timings=last)
                yield "data: [DONE]\n\n"

            headers = {"x-router-agent": self.cfg.agent.name}
            sid = self._stream_id(request)
            if sid is None:  # no conversation id: the turn lives and dies with this connection
                return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)
            st = self._new_stream(sid, owner)

            async def run() -> None:
                try:
                    async for piece in gen():
                        st.append(piece)
                except Exception as e:  # pragma: no cover - gen() already turns agent failures into messages
                    log.warning("agent stream %s failed: %s: %s", sid, type(e).__name__, e)
                finally:
                    st.finish()

            # The turn keeps running if the browser reloads or the network drops; the Web UI reattaches via
            # /v1/streams/lookup + GET /v1/stream, and its Stop button sends DELETE /v1/stream.
            st.task = asyncio.ensure_future(run())
            return StreamingResponse(st.follow(0), media_type="text/event-stream", headers=headers)
        content, reasoning, session = "", "", None
        timings: Optional[Dict[str, Any]] = None
        async for ev in events():
            timings = ev.get("timings") or timings
            d = render(ev)
            content += d.get("content", "")
            reasoning += d.get("reasoning_content", "")
            session = ev.get("session", session)
        msg = {"role": "assistant", "content": content}
        if reasoning:
            msg["reasoning_content"] = reasoning
        t = timings or {}
        prompt_tokens = int(t.get("prompt_n", 0) + t.get("cache_n", 0))
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": int(t.get("predicted_n", 0)), "total_tokens": prompt_tokens + int(t.get("predicted_n", 0))}
        out: Dict[str, Any] = {"id": cid, "object": "chat.completion", "created": created, "model": model,
                               "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}], "usage": usage, "router": {"agent_session": session}}
        if timings:
            out["timings"] = timings
        return JSONResponse(out)

    async def agent_llm(self, request: Request) -> Response:
        """/agent/llm/<session>/<path>: the agent's own model requests (point its provider's base URL here). Passed to
        llama-server unchanged, except that streams ask for per-token timings and prompt progress; timings are summed per turn
        and sent along with the agent's answer (progress as it happens), so the Web UI shows prompt processing, tokens, time
        and speed for agent turns as it does for plain chats."""
        s = self.agents.live_session(request.path_params["sid"]) if self.agents else None
        if s is None:
            return JSONResponse({"error": {"message": "no running agent session with this id", "type": "not_found_error"}}, status_code=404)
        url = "/" + request.path_params["path"] + (("?" + request.url.query) if request.url.query else "")
        body = await request.body()
        streaming = False
        if request.method == "POST" and body:
            try:
                data = json.loads(body)
            except ValueError:
                data = None
            if isinstance(data, dict) and data.get("stream"):
                streaming = True
                data["timings_per_token"] = True
                data["return_progress"] = True
                body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        headers = dict(_forward_headers(request), **{"accept-encoding": "identity"})
        r = await self.client.send(self.client.build_request(request.method, url, content=body, headers=headers), stream=True)
        out = {k: v for k, v in r.headers.items() if k.lower() not in _HOP and k.lower() != "content-encoding"}
        rid = s.begin_request()
        if not streaming or "text/event-stream" not in r.headers.get("content-type", ""):
            content = await r.aread()
            await r.aclose()
            try:
                t = json.loads(content).get("timings")
            except (ValueError, AttributeError):
                t = None
            s.end_request(rid, t if isinstance(t, dict) else None)
            return Response(content, status_code=r.status_code, headers=out)

        async def relay() -> AsyncIterator[bytes]:
            buf, last = b"", None
            try:
                async for piece in r.aiter_raw():
                    yield piece
                    buf += piece
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if line.startswith(b"data: {") and (b'"timings"' in line or b'"prompt_progress"' in line):
                            try:
                                d = json.loads(line[6:])
                            except ValueError:
                                continue
                            t, pp = d.get("timings"), d.get("prompt_progress")
                            t = t if isinstance(t, dict) else None
                            last = t or last
                            s.update_request(rid, t, pp if isinstance(pp, dict) else None)
            finally:
                await r.aclose()
                s.end_request(rid, last)

        return StreamingResponse(relay(), status_code=r.status_code, headers=out)

    @staticmethod
    def _stream_id(request: Request) -> Optional[str]:
        raw = (request.headers.get("x-conversation-id") or "").strip()
        return raw if raw and len(raw) <= 300 and re.fullmatch(r"[A-Za-z0-9_.:@-]+", raw) else None

    def _new_stream(self, sid: str, owner: str) -> AgentStream:
        now = time.time()
        for k, old in list(self.streams.items()):
            if old.done and now - old.completed > AgentStream.KEEP_S:
                del self.streams[k]
        while len(self.streams) >= AgentStream.MAX:
            k = next((k for k, v in self.streams.items() if v.done), next(iter(self.streams)))
            del self.streams[k]
        st = self.streams[sid] = AgentStream(sid, owner)
        self.streams.move_to_end(sid)
        return st

    def _own_stream(self, sid: Optional[str], request: Request) -> Optional[AgentStream]:
        st = self.streams.get(sid or "")
        if st is not None and self.cfg.agent and self.cfg.agent.users and self._agent_owner(request) != st.owner:
            return None
        return st

    async def streams_lookup(self, request: Request) -> Response:
        """POST /v1/streams/lookup: agent streams from the router, the rest from llama-server."""
        try:
            ids = [str(i) for i in (json.loads(await request.body() or b"{}").get("conversation_ids") or [])]
        except (ValueError, AttributeError, TypeError):
            return await self.passthrough(request)
        mine = [st.info() for st in (self._own_stream(i, request) for i in ids) if st is not None]
        if not mine:
            return await self.passthrough(request)
        rest = [i for i in ids if self._own_stream(i, request) is None]
        theirs: List[Any] = []
        if rest:
            headers = dict(_forward_headers(request), **{"accept-encoding": "identity", "content-type": "application/json"})
            try:
                r = await self.client.post(request.url.path, content=json.dumps({"conversation_ids": rest}), headers=headers)
                data = r.json() if r.status_code == 200 else []
                theirs = data if isinstance(data, list) else []
            except (httpx.HTTPError, ValueError):
                theirs = []
        return JSONResponse(mine + theirs)

    async def stream(self, request: Request) -> Response:
        """GET /v1/stream?conv_id=&from=<byte offset> (resume) and DELETE /v1/stream?conv_id= (Stop) for agent turns."""
        st = self._own_stream(request.query_params.get("conv_id"), request)
        if st is None:
            return await self.passthrough(request)
        if request.method == "DELETE":
            if st.task is not None and not st.task.done():
                st.task.cancel()
            return JSONResponse({"success": True})
        try:
            start = int(request.query_params.get("from") or 0)
        except ValueError:
            start = 0
        return StreamingResponse(st.follow(start), media_type="text/event-stream")

    async def agent_sessions(self, request: Request) -> Response:
        """GET /agent/sessions: the caller's sessions (only with a users map; without one there is no identity to list by)."""
        if not self.agents or not self.cfg.agent or not self.cfg.agent.users:
            return JSONResponse({"error": "session listing needs per-user API keys (agent config 'users')"}, status_code=404)
        owner = self._agent_owner(request)
        if owner is None:
            return JSONResponse({"error": "invalid API key"}, status_code=401)
        return JSONResponse({"user": owner, "sessions": self.agents.list_for(owner)})

    async def agent_session(self, request: Request) -> Response:
        if not self.agents:
            return JSONResponse({"error": "agent bridge disabled"}, status_code=404)
        info = await self.agents.describe(request.path_params["sid"])
        return JSONResponse(info) if info else JSONResponse({"error": "no such session"}, status_code=404)

    async def agent_file(self, request: Request) -> Response:
        path = self.agents.file_path(request.path_params["sid"], request.path_params["path"]) if self.agents else None
        if not path:
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(path, filename=os.path.basename(path), content_disposition_type="inline")

    async def agent_archive(self, request: Request) -> Response:
        data = await self.agents.archive(request.path_params["sid"]) if self.agents else None
        if data is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return Response(data, media_type="application/zip", headers={"content-disposition": "attachment; filename=session-%s.zip" % request.path_params["sid"][:8]})

    async def models(self, request: Request) -> Response:
        """llama-server's model list, plus the agent as a selectable model."""
        r = await self.client.get("/v1/models", headers={"accept-encoding": "identity"})
        try:
            data = r.json()
        except ValueError:
            return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))
        if self.cfg.sentinel and isinstance(data, dict):
            sid = self.cfg.sentinel.model_id
            if isinstance(data.get("data"), list) and not any(m.get("id") == sid for m in data["data"]):
                data["data"].append({"id": sid, "object": "model", "owned_by": "llama-mcp-router", "created": 0})
            if isinstance(data.get("models"), list) and not any(m.get("model") == sid for m in data["models"]):
                data["models"].append({"name": sid, "model": sid, "type": "model", "description": "reasoning checked step by step by a decision model"})
        if self.cfg.agent and isinstance(data, dict):
            mid = self.cfg.agent.model_id
            if isinstance(data.get("data"), list) and not any(m.get("id") == mid for m in data["data"]):
                data["data"].append({"id": mid, "object": "model", "owned_by": "llama-mcp-router", "created": 0})
            if isinstance(data.get("models"), list) and not any(m.get("model") == mid for m in data["models"]):
                data["models"].append({"name": mid, "model": mid, "type": "model", "description": "ACP agent (%s) with a per-session workspace" % self.cfg.agent.name})
        return JSONResponse(data, status_code=r.status_code)

    # ------------------------------------------------------------------ other routes
    async def select_endpoint(self, request: Request) -> Response:
        """POST /router/select {"query": "...", "tools": [...optional]} -> what would be sent."""
        body = await request.json()
        pool = await self.pool(body.get("tools") or [])
        choice = await self.choose(body.get("query", ""), pool, set())
        return JSONResponse({"selected": [tool_name(t) for t in choice["tools"]], "pool": choice["pool"], "select_ms": choice["ms"], "info": choice["info"]})

    async def health(self, request: Request) -> Response:
        return JSONResponse({"status": "ok", "selector": self.cfg.selector.name, "mode": self.cfg.mode, "backend": self.cfg.backend})

    async def passthrough(self, request: Request) -> Response:
        url = request.url.path + (("?" + request.url.query) if request.url.query else "")
        req = self.client.build_request(request.method, url, content=await request.body(), headers=_forward_headers(request))
        r = await self.client.send(req, stream=True)
        out = {k: v for k, v in r.headers.items() if k.lower() not in _HOP}
        return StreamingResponse(r.aiter_raw(), status_code=r.status_code, headers=out, background=BackgroundTask(r.aclose))


def create_app(config: RouterConfig, transport: Optional[httpx.AsyncBaseTransport] = None) -> Starlette:
    router = Router(config, transport)

    @asynccontextmanager
    async def lifespan(app: Starlette):
        yield
        await router.aclose()

    extra = [
        Route("/v1/models", router.models, methods=["GET"]),
        Route("/models", router.models, methods=["GET"]),
        Route("/v1/streams/lookup", router.streams_lookup, methods=["POST"]),
        Route("/v1/stream", router.stream, methods=["GET", "DELETE"]),
    ] if config.agent else ([Route("/v1/models", router.models, methods=["GET"]), Route("/models", router.models, methods=["GET"])] if config.sentinel else [])
    app = Starlette(
        routes=extra + [
            Route("/v1/chat/completions", router.chat, methods=["POST"]),
            Route("/router/select", router.select_endpoint, methods=["POST"]),
            Route("/router/health", router.health, methods=["GET"]),
            Route("/agent/sessions", router.agent_sessions, methods=["GET"]),
            Route("/agent/sessions/{sid}", router.agent_session, methods=["GET"]),
            Route("/agent/sessions/{sid}/archive.zip", router.agent_archive, methods=["GET"]),
            Route("/agent/sessions/{sid}/files/{path:path}", router.agent_file, methods=["GET"]),
            Route("/agent/llm/{sid}/{path:path}", router.agent_llm, methods=["GET", "POST"]),
            Route("/{path:path}", router.passthrough, methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]),
        ],
        lifespan=lifespan,
    )
    app.state.router = router
    return app
