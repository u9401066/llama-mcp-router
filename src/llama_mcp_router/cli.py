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
from .agent import AgentConfig
from .proxy import RouterConfig, create_app, load_routes
from .selectors import load_groups_config, load_selector
from .tools import FileToolSource, ServerToolSource, tool_name


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get("LLAMA_ROUTER_" + name, default)


def _selector_options(a: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    keep = a.keep if a.keep is not None else 5
    also = a.also if a.also is not None else 5
    opts: Dict[str, Dict[str, Any]] = {
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
        "retrieve": {"top_k": keep + also, "embed_url": a.embed_url, "embed_model": a.embed_model},
        "laya-rerank": {"url": a.laya_url, "embed_url": a.embed_url, "embed_model": a.embed_model, "shortlist": a.shortlist,
                        "keep": keep, "also": also, "api_key": a.laya_api_key, "model": a.laya_model},
        "2pass": {"url": a.laya_url, "groups": load_groups_config(a.groups), "max_groups": a.draft_groups, "keep": keep,
                  "also": a.also if a.also is not None else 3, "views": a.laya_views or "single", "model": a.laya_model,
                  "api_key": a.laya_api_key},
    }
    if a.laya_none is not None:
        opts["laya"]["none_threshold"] = a.laya_none
        opts["union"] = {"abstain": "any"}  # a "no tool needed" verdict from Laya overrides BM25's lexical hits
    return opts


def _add_selector_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--selector", default=_env("SELECTOR", "laya+bm25"), help="all | none | bm25 | laya | retrieve | laya-rerank | 2pass | a+b (union) | pkg.mod:Class  [laya+bm25]. "
                   "Small pools (<~80 tools): laya+bm25 with --groups; with a stronger decision model at --laya-url (e.g. Clef-Flash): 2pass. "
                   "Hundreds of tools: laya-rerank with --embed-url. 'none --escalate' = catalog mode")
    p.add_argument("--groups", default=_env("GROUPS"), help="JSON file defining tool groups (see examples/pubmed_groups.json)")
    p.add_argument("--laya-url", default=_env("LAYA_URL", "http://127.0.0.1:8000"), help="SystemOne decision-model server (POST /v1/systemone): "
                   "laya-serve, or llama-server with Cloudflare Clef / Clef-Flash")
    p.add_argument("--laya-api-key", default=_env("LAYA_API_KEY"))
    p.add_argument("--laya-mode", choices=["choice", "noul"], default=_env("LAYA_MODE", "choice"))
    p.add_argument("--laya-state", choices=["json", "raw"], default=_env("LAYA_STATE"), help="how the request is passed to Laya")
    p.add_argument("--laya-labels", choices=["label", "description", "auto"], default=_env("LAYA_LABELS"), help="text Laya sees for each group")
    p.add_argument("--laya-views", choices=["single", "ensemble"], default=_env("LAYA_VIEWS"), help="ensemble (default): average three differently-framed questions, 3 Laya calls per request; single: one")
    p.add_argument("--laya-model", default=_env("LAYA_MODEL"), help="force a Laya checkpoint (english | multilingual | typed-decisions); default: Laya routes by language")
    p.add_argument("--laya-none", type=float, default=(float(_env("LAYA_NONE")) if _env("LAYA_NONE") else None), help="add a 'no tool needed' option; send no tools when its probability >= this (0.7 recommended; costs ~2 points recall)")
    p.add_argument("--top-p", type=float, default=float(_env("TOP_P", "1.0")), help="keep groups until this much probability mass (choice mode)")
    p.add_argument("--max-groups", type=int, default=int(_env("MAX_GROUPS", "2")))
    p.add_argument("--threshold", type=float, default=float(_env("THRESHOLD", "0.5")), help="noul mode: min probability to keep a group")
    p.add_argument("--embed-url", default=_env("EMBED_URL"), help="OpenAI-compatible /v1/embeddings base URL for retrieve / laya-rerank (e.g. llama-server --embedding with bge-m3); without it they use BM25 only")
    p.add_argument("--embed-model", default=_env("EMBED_MODEL"))
    p.add_argument("--shortlist", type=int, default=int(_env("SHORTLIST", "24")), help="laya-rerank: tools retrieved for Laya to rank")
    p.add_argument("--keep", type=int, default=(int(_env("KEEP")) if _env("KEEP") else None), help="laya-rerank / 2pass: the decision model's top tools sent [5]")
    p.add_argument("--also", type=int, default=(int(_env("ALSO")) if _env("ALSO") else None), help="laya-rerank: retriever's top tools sent in addition [5]; 2pass: the draft's [3]")
    p.add_argument("--draft-groups", type=int, default=int(_env("DRAFT_GROUPS", "3")), help="2pass: groups the decision model picks for the draft")
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
    s.add_argument("--apply", choices=["select", "reorder", "all"], default=_env("APPLY", "select"), help="select: send only the selection (default); reorder: send all tools, most relevant first; all: leave tools unchanged (use with --hint)")
    s.add_argument("--escalate", action="store_true", help="add a catalog meta-tool listing the tools that were not sent; if the model calls it, rerun once with the tools it asked for")
    s.add_argument("--catalog-max", type=int, default=int(_env("CATALOG_MAX", "80")), help="--escalate lists left-out tools by name up to this many; beyond, the meta-tool takes a search query")
    s.add_argument("--no-sanitize", action="store_true", help="send tool schemas unchanged (default: inline $refs and drop huge length limits that llama.cpp cannot turn into a grammar)")
    s.add_argument("--agent-config", default=_env("AGENT_CONFIG"), help="JSON file enabling the ACP agent bridge (see examples/agent-dsh.json)")
    s.add_argument("--routes", default=_env("ROUTES"), help="JSON file: whole chats to other OpenAI-compatible services by model name or message prefix "
                   "(see examples/routes.json)")
    s.add_argument("--hint", action="store_true", help="append the selector's routing hint to the last user message")
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

    d = sub.add_parser("install-dsh-plugin", help="copy the DeepSeek Harness tool-routing plugin into a dsh install (<dir>/plugins/)")
    d.add_argument("dsh_dir", help="directory whose node_modules contains @deepseek-ai/dsh (e.g. ~/agent-runtimes/dsh)")

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


def _install_dsh_plugin(dsh_dir: str) -> int:
    import shutil

    root = os.path.abspath(os.path.expanduser(dsh_dir))
    if not os.path.isdir(os.path.join(root, "node_modules", "@deepseek-ai", "dsh-tools")):
        print("%s does not look like a dsh install (no node_modules/@deepseek-ai/dsh-tools)" % root, file=sys.stderr)
        return 1
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "integrations", "dsh-plugin.mjs")
    dst = os.path.join(root, "plugins", "llama-mcp-router.mjs")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    print(dst)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(getattr(a, "log_level", "info")).upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if a.cmd == "select":
        return asyncio.run(_cmd_select(a))
    if a.cmd == "tools":
        return asyncio.run(_cmd_tools(a))
    if a.cmd == "install-dsh-plugin":
        return _install_dsh_plugin(a.dsh_dir)

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
        apply=a.apply,
        hint=a.hint,
        escalate=a.escalate,
        catalog_max=a.catalog_max,
        sanitize=not a.no_sanitize,
        agent=AgentConfig.load(a.agent_config) if a.agent_config else None,
        routes=load_routes(a.routes) if a.routes else [],
    )
    uvicorn.run(create_app(cfg), host=a.host, port=a.port, log_level=a.log_level.lower())
    return 0


if __name__ == "__main__":
    sys.exit(main())
