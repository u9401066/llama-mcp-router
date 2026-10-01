"""Command line: ``llama-mcp-router serve | select | tools``."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import httpx

from . import __version__
from .proxy import RouterConfig, create_app
from .selectors import load_groups_config, load_selector
from .tools import FileToolSource, ServerToolSource, tool_name


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get("LLAMA_ROUTER_" + name, default)


def _selector_options(a: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    return {
        "laya": {
            "url": a.laya_url,
            "groups": load_groups_config(a.groups),
            "mode": a.laya_mode,
            "top_p": a.top_p,
            "max_groups": a.max_groups,
            "threshold": a.threshold,
            "api_key": a.laya_api_key,
            "state_mode": a.laya_state,
            "labels": a.laya_labels,
            "views": a.laya_views,
            "model": a.laya_model,
        },
        "bm25": {"top_k": a.top_k},
    }


def _add_selector_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--selector", default=_env("SELECTOR", "laya+bm25"), help="all | bm25 | laya | a+b (union) | pkg.mod:Class  [laya+bm25]")
    p.add_argument("--groups", default=_env("GROUPS"), help="JSON file defining tool groups (see examples/pubmed_groups.json)")
    p.add_argument("--laya-url", default=_env("LAYA_URL", "http://127.0.0.1:8000"))
    p.add_argument("--laya-api-key", default=_env("LAYA_API_KEY"))
    p.add_argument("--laya-mode", choices=["choice", "noul"], default=_env("LAYA_MODE", "choice"))
    p.add_argument("--laya-state", choices=["json", "raw"], default=_env("LAYA_STATE"), help="how the request is passed to Laya")
    p.add_argument("--laya-labels", choices=["label", "description", "auto"], default=_env("LAYA_LABELS"), help="text Laya sees for each group")
    p.add_argument("--laya-views", choices=["single", "ensemble"], default=_env("LAYA_VIEWS"), help="ensemble (default): average three differently-framed questions, 3 Laya calls per request; single: one")
    p.add_argument("--laya-model", default=_env("LAYA_MODEL"), help="force a Laya checkpoint (english | multilingual | typed-decisions); default: Laya routes by language")
    p.add_argument("--top-p", type=float, default=float(_env("TOP_P", "1.0")), help="keep groups until this much probability mass (choice mode)")
    p.add_argument("--max-groups", type=int, default=int(_env("MAX_GROUPS", "2")))
    p.add_argument("--threshold", type=float, default=float(_env("THRESHOLD", "0.5")), help="noul mode: min probability to keep a group")
    p.add_argument("--top-k", type=int, default=int(_env("TOP_K", "3")), help="bm25: number of tools")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llama-mcp-router", description="Send llama-server only the MCP tools a request needs.")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the proxy")
    s.add_argument("--backend", default=_env("BACKEND", "http://127.0.0.1:8080"), help="llama-server (or any OpenAI-compatible) base URL")
    s.add_argument("--host", default=_env("HOST", "127.0.0.1"))
    s.add_argument("--port", type=int, default=int(_env("PORT", "8090")))
    s.add_argument("--mode", choices=["inject", "agent"], default=_env("MODE", "inject"), help="inject: client executes tool calls; agent: router executes llama-server tools")
    s.add_argument("--max-tools", type=int, default=int(_env("MAX_TOOLS", "12")))
    s.add_argument("--always", default=_env("ALWAYS", ""), help="comma-separated tool names always sent")
    s.add_argument("--exclude", default=_env("EXCLUDE", ""), help="comma-separated fnmatch patterns of tools never offered")
    s.add_argument("--no-server-tools", action="store_true", help="only route tools the client sends, ignore llama-server's /tools")
    s.add_argument("--sticky", action="store_true", help="keep each conversation's tool list append-only so llama-server's prompt cache can hit on follow-up turns (helps on-topic chats, hurts topic-hopping ones; see README)")
    s.add_argument("--fallback", choices=["all", "none"], default=_env("FALLBACK", "all"), help="if the selector fails")
    s.add_argument("--log-level", default=_env("LOG_LEVEL", "info"))
    _add_selector_args(s)

    q = sub.add_parser("select", help="show which tools would be sent for a query")
    q.add_argument("query")
    q.add_argument("--backend", default=_env("BACKEND", "http://127.0.0.1:8080"))
    q.add_argument("--tools-file", help="read tools from a JSON file instead of the backend's /tools")
    q.add_argument("--max-tools", type=int, default=int(_env("MAX_TOOLS", "12")))
    _add_selector_args(q)

    t = sub.add_parser("tools", help="list the tools the backend exposes (and their rough size)")
    t.add_argument("--backend", default=_env("BACKEND", "http://127.0.0.1:8080"))
    t.add_argument("--json", action="store_true", help="dump full definitions (OpenAI format)")
    return p


async def _cmd_select(a: argparse.Namespace) -> int:
    async with httpx.AsyncClient(base_url=a.backend) as client:
        source = FileToolSource(a.tools_file) if a.tools_file else ServerToolSource(client)
        tools = await source.fetch()
    sel = load_selector(a.selector, **_selector_options(a))
    try:
        res = await sel.select(a.query, tools)
    finally:
        await sel.aclose()
    print(json.dumps({"selected": res.names[: a.max_tools], "pool": len(tools), "info": res.info}, ensure_ascii=False, indent=2))
    return 0


async def _cmd_tools(a: argparse.Namespace) -> int:
    async with httpx.AsyncClient(base_url=a.backend) as client:
        tools = await ServerToolSource(client).fetch()
    if a.json:
        print(json.dumps(tools, ensure_ascii=False, indent=1))
    else:
        for t in tools:
            print("%6d B  %s" % (len(json.dumps(t)), tool_name(t)))
        print("%d tools, %d bytes of schema" % (len(tools), len(json.dumps(tools))))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(getattr(a, "log_level", "info")).upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if a.cmd == "select":
        return asyncio.run(_cmd_select(a))
    if a.cmd == "tools":
        return asyncio.run(_cmd_tools(a))

    import uvicorn

    cfg = RouterConfig(
        backend=a.backend,
        selector=load_selector(a.selector, **_selector_options(a)),
        always=[x for x in a.always.split(",") if x],
        max_tools=a.max_tools,
        mode=a.mode,
        use_server_tools=not a.no_server_tools,
        exclude=[x for x in a.exclude.split(",") if x],
        fallback=a.fallback,
        sticky=a.sticky,
    )
    uvicorn.run(create_app(cfg), host=a.host, port=a.port, log_level=a.log_level.lower())
    return 0


if __name__ == "__main__":
    sys.exit(main())
