"""Does a selector help a model pick the right tool?

Phase 1 (always, no LLM):   selector recall, tools sent, schema size, selector latency.
Phase 2 (--llm URL):        ask a model (llama-server / any OpenAI API) to answer each query with
                            the selected tools and check its FIRST tool call against the expected tools.

    python benchmarks/run_bench.py --laya-url http://127.0.0.1:8000 \
        --llm http://127.0.0.1:8001 --effort medium --concurrency 2
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import httpx

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))

from llama_mcp_router import AllSelector, BM25Selector, LayaSelector, UnionSelector  # noqa: E402
from llama_mcp_router.selectors import build_groups, load_groups_config  # noqa: E402
from llama_mcp_router.proxy import add_hint, arrange  # noqa: E402
from llama_mcp_router.tools import tool_name  # noqa: E402

SYSTEM = ("You are a biomedical literature research assistant with access to tools. "
          "Always handle the user's request by calling the single most appropriate tool right away. "
          "Never ask clarifying questions; use the information in the request, and if an input such as a "
          "session or previous search is implied, just call the tool.")


class OracleSelector:
    """Upper bound: exactly the group(s) that contain the expected tool (set per query)."""

    name = "oracle"

    def __init__(self, cfg, hint=False):
        self.cfg, self.expect, self.use_hint = cfg, [], hint

    async def select(self, query, tools):
        from llama_mcp_router.selectors import Selection

        names, hint = [], ""
        for g in build_groups(self.cfg, tools):
            if any(e in g.tools for e in self.expect):
                names += g.tools
                hint = hint or "%s (100%%)" % g.description
        by = [n for n in (tool_name(t) for t in tools)]
        return Selection(names, abstain=not self.expect, ranking=names + [n for n in by if n not in names], hint=hint if self.use_hint else hint)

    async def aclose(self):
        return None


class Cfg:
    """A selector plus how its output is applied (see RouterConfig.apply / hint)."""

    def __init__(self, selector, apply="select", hint=False):
        self.selector, self.apply, self.hint = selector, apply, hint
        self.name = selector.name

    async def select(self, query, tools):
        return await self.selector.select(query, tools)

    async def aclose(self):
        return await self.selector.aclose()

    @property
    def expect(self):
        return getattr(self.selector, "expect", None)

    @expect.setter
    def expect(self, v):
        self.selector.expect = v


def make_selectors(a, cfg):
    U = a.laya_url
    old = LayaSelector(url=U, groups=cfg, state_mode="raw", labels="description", top_p=0.95, max_groups=4)  # what v0.1.0 did

    def v2(**kw):
        return UnionSelector([LayaSelector(url=a.laya_url if not kw.get("url") else kw["url"], groups=cfg, top_p=a.top_p, max_groups=a.max_groups, **{k: v for k, v in kw.items() if k != "url"}), BM25Selector(top_k=3)], **({"abstain": "any"} if kw.get("none_threshold") else {}))

    sels = {
        "all": Cfg(AllSelector()),
        "bm25": Cfg(BM25Selector(top_k=a.top_k)),
        "v0.1 laya+bm25": Cfg(UnionSelector([old, BM25Selector(top_k=3)])),
        "v0.2 laya+bm25": Cfg(v2()),
        "v0.2 top3+bm25": Cfg(UnionSelector([LayaSelector(url=U, groups=cfg, top_p=1.0, max_groups=3), BM25Selector(top_k=3)])),
        "v0.2 +none(0.7)": Cfg(v2(none_threshold=0.7)),
        "v0.2 reorder": Cfg(v2(), "reorder"),
        "v0.2 all+hint": Cfg(v2(), "all", True),
        "v0.2 select+hint": Cfg(v2(), "select", True),
        "oracle": Cfg(OracleSelector(cfg)),
        "oracle all+hint": Cfg(OracleSelector(cfg, hint=True), "all", True),
    }
    return sels


async def phase1(selectors, tools, queries):
    by_name = {tool_name(t): t for t in tools}
    out: Dict[str, List[Dict[str, Any]]] = {}
    for name, sel in selectors.items():
        rows = []
        for q in queries:
            if name.startswith("oracle"):
                sel.expect = q["expect"]
            t0 = time.perf_counter()
            err = None
            try:
                res = await sel.select(q["query"], tools)
                names, err, abst = res.names, None, bool(res.abstain)
            except Exception as e:  # noqa: BLE001
                names, err, abst = [], repr(e), False
            ms = (time.perf_counter() - t0) * 1000
            rows.append({"id": q["id"], "ranking": (res.ranking if not err else []), "hint": (res.hint if not err else ""), "names": names, "hit": (any(e in names for e in q["expect"]) if q["expect"] else not names), "bytes": sum(len(json.dumps(by_name[n])) for n in names if n in by_name), "ms": ms, "err": err, "abstain": abst})
        out[name] = rows
    return out


async def ask(client, url, q, tools, effort, max_tokens, hint=""):
    body = {"model": "m", "messages": add_hint([{"role": "system", "content": SYSTEM}, {"role": "user", "content": q["query"]}], hint),
            "temperature": 0, "max_tokens": max_tokens, "chat_template_kwargs": {"reasoning_effort": effort}}
    if tools:
        body["tools"] = tools
    t0 = time.perf_counter()
    r = await client.post(url + "/v1/chat/completions", json=body)
    dt = time.perf_counter() - t0
    d = r.json()
    msg = d["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    first = calls[0]["function"]["name"] if calls else None
    try:
        json.loads(calls[0]["function"]["arguments"]) if calls else None
        valid = bool(calls)
    except ValueError:
        valid = False
    return {"id": q["id"], "first": first, "ok": (first in q["expect"]) if q["expect"] else first is None, "neg": not q["expect"], "valid_args": valid, "prompt_tokens": d["usage"]["prompt_tokens"], "completion_tokens": d["usage"]["completion_tokens"], "s": dt, "finish": d["choices"][0]["finish_reason"]}


async def phase2(a, selectors, tools, queries, p1):
    by_name = {tool_name(t): t for t in tools}
    results: Dict[str, List[Dict[str, Any]]] = {}
    sem = asyncio.Semaphore(a.concurrency)
    async with httpx.AsyncClient(timeout=1800) as client:
        for name in selectors:
            sel_by_id = {r["id"]: r for r in p1[name]}

            async def one(q):
                async with sem:
                    row = sel_by_id[q["id"]]
                    c = selectors[name]
                    from llama_mcp_router import Selection

                    sent = arrange(tools, Selection(row["names"], ranking=row["ranking"]), c.apply, 10 ** 6)
                    if not sent and not name.startswith("oracle") and not row.get("abstain"):
                        sent = tools  # selector returned nothing and did not abstain: fail open like the router
                    return await ask(client, a.llm, q, sent, a.effort, a.max_tokens, row["hint"] if c.hint else "")

            t0 = time.time()
            results[name] = await asyncio.gather(*[one(q) for q in queries])
            ok = sum(r["ok"] for r in results[name])
            print("  %-18s %d/%d correct  (%.0fs)" % (name, ok, len(queries), time.time() - t0), flush=True)
            (Path(a.out) / "llm_results.json").write_text(json.dumps(results, indent=1))
    return results


async def prefill_one(client, url, q, tools, cold):
    body = {"model": "m", "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q["query"]}],
            "temperature": 0, "max_tokens": 1, "cache_prompt": not cold}
    if tools:
        body["tools"] = tools
    t0 = time.perf_counter()
    d = (await client.post(url + "/v1/chat/completions", json=body)).json()
    wall = (time.perf_counter() - t0) * 1000
    tm = d.get("timings") or {}
    return {"prompt_n": tm.get("prompt_n", 0), "cache_n": tm.get("cache_n", 0), "prompt_ms": tm.get("prompt_ms", wall), "wall_ms": wall, "tokens": d["usage"]["prompt_tokens"]}


async def phase3(a, selectors, tools, queries, p1):
    """Prefill cost: cold (cache_prompt=false: new conversation / evicted cache / other client) and warm (sequential, cache on)."""
    by_name = {tool_name(t): t for t in tools}
    res: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    async with httpx.AsyncClient(timeout=1800) as client:
        for name in selectors:
            sel_by_id = {r["id"]: r for r in p1[name]}
            res[name] = {"cold": [], "warm": []}
            for mode in ("cold", "warm"):
                for q in queries:
                    sent = [by_name[n] for n in sel_by_id[q["id"]]["names"] if n in by_name]
                    if not sent and name != "oracle" and not sel_by_id[q["id"]].get("abstain"):
                        sent = tools
                    r = await prefill_one(client, a.llm, q, sent, mode == "cold")
                    r["select_ms"] = sel_by_id[q["id"]]["ms"]
                    res[name][mode].append(r)
            c = res[name]["cold"]
            print("  %-18s cold prefill %.0f ms avg, %.0f tokens" % (name, statistics.mean(r["prompt_ms"] for r in c), statistics.mean(r["tokens"] for r in c)), flush=True)
    return res


def prefill_report(p3):
    base = statistics.mean(r["prompt_ms"] + r["select_ms"] for r in p3["all"]["cold"]) if "all" in p3 else None
    out = ["### Prefill cost (time the server spends reading the prompt before it can answer)", "", "| selector | prompt tokens (avg) | selector ms | cold prefill ms (avg) | cold total ms (select+prefill) | cold speed-up vs all | warm prefill ms (avg, prefix cache on) |", "|---|---|---|---|---|---|---|"]
    for name, d in p3.items():
        c, w = d["cold"], d["warm"]
        tot = statistics.mean(r["prompt_ms"] + r["select_ms"] for r in c)
        out.append("| %s | %.0f | %.0f | %.0f | %.0f | %s | %.0f |" % (name, statistics.mean(r["tokens"] for r in c), statistics.mean(r["select_ms"] for r in c), statistics.mean(r["prompt_ms"] for r in c), tot, ("%.1fx" % (base / tot)) if base else "-", statistics.mean(r["prompt_ms"] for r in w)))
    return "\n".join(out)


def pct(x, n):
    return "%.1f%%" % (100.0 * x / n)


def report(queries, p1, p2):
    qs = {q["id"]: q for q in queries}
    npos = sum(bool(q["expect"]) for q in queries)
    nneg = len(queries) - npos
    lines = ["| selector | recall on tool requests | tools sent (tool requests) | tools sent (no-tool requests) | no-tool requests sent zero tools | schema KB (avg, all) | selector ms (median) |", "|---|---|---|---|---|---|---|"]
    for name, rows in p1.items():
        pos = [r for r in rows if qs[r["id"]]["expect"]]
        neg = [r for r in rows if not qs[r["id"]]["expect"]]
        lines.append("| %s | %s | %.1f | %s | %s | %.1f | %.0f |" % (
            name, pct(sum(r["hit"] for r in pos), len(pos)), statistics.mean(len(r["names"]) for r in pos),
            ("%.1f" % statistics.mean(len(r["names"]) for r in neg)) if neg else "-",
            ("%d/%d" % (sum(not r["names"] for r in neg), len(neg))) if neg else "-",
            statistics.mean(r["bytes"] for r in rows) / 1000, statistics.median(r["ms"] for r in rows)))
    out = ["### Selection (no LLM): %d tool requests, %d no-tool requests" % (npos, nneg), ""] + lines
    if p2:
        out += ["", "### First tool call by the 27B model", "", "| selector | correct (all) | tool requests | no-tool requests (answered without a tool call) | EN | 中文 | prompt tokens (avg) | seconds/query (avg) |", "|---|---|---|---|---|---|---|---|"]
        for name, rows in p2.items():
            pos = [r for r in rows if not r["neg"]]
            neg = [r for r in rows if r["neg"]]
            en = [r for r in rows if qs[r["id"]]["lang"] == "en"]
            zh = [r for r in rows if qs[r["id"]]["lang"] == "zh"]
            out.append("| %s | %s | %s | %s | %s | %s | %.0f | %.1f |" % (name, pct(sum(r["ok"] for r in rows), len(rows)), pct(sum(r["ok"] for r in pos), len(pos)), (pct(sum(r["ok"] for r in neg), len(neg)) if neg else "-"), pct(sum(r["ok"] for r in en), len(en)), pct(sum(r["ok"] for r in zh), len(zh)), statistics.mean(r["prompt_tokens"] for r in rows), statistics.mean(r["s"] for r in rows)))
    return "\n".join(out)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", default=str(ROOT / "data/pubmed_tools.json"))
    ap.add_argument("--queries", default=str(ROOT / "data/queries.jsonl"))
    ap.add_argument("--groups", default=str(ROOT.parent / "examples/pubmed_groups.json"))
    ap.add_argument("--laya-url", default="http://127.0.0.1:8000")
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-groups", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--llm", help="OpenAI-compatible base URL, enables phase 2")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--distractors", action="store_true", help="add ~100 synthetic tools from other domains (bigger pool)")
    ap.add_argument("--prefill", action="store_true", help="with --llm: measure prefill time (max_tokens=1) instead of answer accuracy")
    ap.add_argument("--sample", type=int, help="evenly sample this many queries (prefill runs)")
    ap.add_argument("--only", help="comma-separated selector names")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)

    tools = json.load(open(a.tools))
    queries = [json.loads(l) for l in open(a.queries) if l.strip()][: a.limit]
    if a.sample and a.sample < len(queries):
        queries = [queries[int(i * len(queries) / a.sample)] for i in range(a.sample)]
    cfg = load_groups_config(a.groups)
    if a.distractors:
        tools += json.load(open(ROOT / "data/distractor_tools.json"))
        cfg["groups"].update(json.load(open(ROOT / "data/distractor_groups.json"))["groups"])
    selectors = make_selectors(a, cfg)
    if a.only:
        selectors = {k: v for k, v in selectors.items() if k in a.only.split(",")}

    print("%d tools, %d queries" % (len(tools), len(queries)))
    p1 = await phase1(selectors, tools, queries)
    p2 = await phase2(a, selectors, tools, queries, p1) if a.llm and not a.prefill else None
    p3 = await phase3(a, selectors, tools, queries, p1) if a.llm and a.prefill else None
    (Path(a.out) / "selection.json").write_text(json.dumps(p1, indent=1))
    md = report(queries, p1, p2)
    if p3:
        md += "\n\n" + prefill_report(p3)
    (Path(a.out) / "report.md").write_text(md + "\n")
    print("\n" + md)
    for s in selectors.values():
        await s.aclose()


if __name__ == "__main__":
    asyncio.run(main())
