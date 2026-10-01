"""OpenAI-compatible proxy that sends the model only the tools it needs."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from .selectors import AllSelector, Selector
from .tools import ServerToolSource, Tool, exclude_tools, normalize_tool, tool_name

log = logging.getLogger("llama_mcp_router")

_HOP = {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "te", "upgrade", "accept-encoding", "content-encoding"}


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


def called_tool_names(messages: Sequence[Dict[str, Any]]) -> Set[str]:
    names: Set[str] = set()
    for m in messages:
        for tc in m.get("tool_calls") or []:
            n = (tc.get("function") or {}).get("name")
            if n:
                names.add(n)
    return names


def _forward_headers(request: Request) -> Dict[str, str]:
    return {k: v for k, v in request.headers.items() if k.lower() not in _HOP}


def _result_text(res: Any) -> str:
    if isinstance(res, dict):
        if "plain_text_response" in res:
            return str(res["plain_text_response"])
        if "error" in res and len(res) == 1:
            return "Error: %s" % res["error"]
    return res if isinstance(res, str) else json.dumps(res, ensure_ascii=False)


class Router:
    def __init__(self, config: RouterConfig, transport: Optional[httpx.AsyncBaseTransport] = None):
        self.cfg = config
        self.client = httpx.AsyncClient(base_url=config.backend.rstrip("/"), timeout=config.request_timeout, transport=transport)
        self.source = ServerToolSource(self.client, ttl=config.tools_ttl)
        self._sticky: "OrderedDict[str, List[str]]" = OrderedDict()

    async def aclose(self) -> None:
        await self.client.aclose()
        await self.cfg.selector.aclose()

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
        if choice["tools"]:
            body["tools"] = choice["tools"]
        else:  # nothing selected: send a tool-free prompt (also drop tool_choice, which would force a call)
            body.pop("tools", None)
            body.pop("tool_choice", None)
        meta = {"x-router-tools": ",".join(tool_name(t) for t in choice["tools"])[:1000], "x-router-pool": str(choice["pool"]), "x-router-select-ms": str(choice["ms"])}
        log.info("selected %d/%d tools in %sms: %s", len(choice["tools"]), choice["pool"], choice["ms"], meta["x-router-tools"][:200])

        if self.cfg.mode == "agent":
            server_names = {tool_name(t) for t in await self.source.fetch()} - client_names
            return await self._agent(body, headers, server_names, meta, choice)
        return await self._relay(body, headers, meta)

    async def _relay(self, body: Dict[str, Any], headers: Dict[str, str], meta: Optional[Dict[str, str]] = None) -> Response:
        req = self.client.build_request("POST", "/v1/chat/completions", json=body, headers=headers)
        r = await self.client.send(req, stream=True)
        out = {k: v for k, v in r.headers.items() if k.lower() not in _HOP}
        out.update(meta or {})
        return StreamingResponse(r.aiter_raw(), status_code=r.status_code, headers=out, background=BackgroundTask(r.aclose))

    async def _agent(self, body: Dict[str, Any], headers: Dict[str, str], server_names: Set[str], meta: Dict[str, str], choice: Dict[str, Any]) -> Response:
        want_stream = bool(body.pop("stream", False))
        body.pop("stream_options", None)
        messages = list(body.get("messages") or [])
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        data: Dict[str, Any] = {}
        iterations = 0
        for iterations in range(1, self.cfg.max_iterations + 1):
            r = await self.client.post("/v1/chat/completions", json={**body, "messages": messages}, headers=headers)
            if r.status_code != 200:
                return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))
            data = r.json()
            for k in usage:
                usage[k] += (data.get("usage") or {}).get(k, 0)
            msg = data["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            if not calls or any(c["function"]["name"] not in server_names for c in calls):
                break  # final answer, or the client has to run one of its own tools
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
            for c in calls:
                messages.append({"role": "tool", "tool_call_id": c.get("id"), "content": await self._run_tool(c["function"])})
        data["usage"] = usage
        data["router"] = {"selected": [tool_name(t) for t in choice["tools"]], "pool": choice["pool"], "select_ms": choice["ms"], "iterations": iterations, "info": choice["info"]}
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

    app = Starlette(
        routes=[
            Route("/v1/chat/completions", router.chat, methods=["POST"]),
            Route("/router/select", router.select_endpoint, methods=["POST"]),
            Route("/router/health", router.health, methods=["GET"]),
            Route("/{path:path}", router.passthrough, methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]),
        ],
        lifespan=lifespan,
    )
    app.state.router = router
    return app
