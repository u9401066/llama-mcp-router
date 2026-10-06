"""Selector lab: tool-selection recall, tools sent and latency for new selector ideas (no LLM; extends phase 1 of run_bench).

Pools and queries are the ones of the README accuracy table: 41 PubMed tools, or 133 with --distractors; queries_heldout.jsonl
(52, never used for tuning) and queries_test.jsonl (68).

    python benchmarks/selector_lab.py --rerank-url http://127.0.0.1:8085 [--clef-url http://127.0.0.1:8084] [--distractors]

Selectors:
  laya+bm25            what the router ships: Laya ensemble picks <= 2 groups, plus BM25's top 3
  laya 2-pass          Laya twice: groups (<= 3) + BM25 top 8 as a draft, then Laya ranks those tools one by one (chunks of 12)
  bm25 -> rerank       BM25's top 20, re-scored by a cross-encoder reranker (llama-server --rerank), top k kept
  laya+bm25 -> rerank  draft = Laya groups (<= 3) + BM25 top 10, verified by the reranker
  rerank all           the reranker scores every tool
  clef ...             the same with a Clef / Clef-Flash decision model (same /v1/systemone API as Laya);
                       "clef 2pass (router)" is the router's own --selector 2pass
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import httpx

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))

from llama_mcp_router import BM25Selector, LayaSelector, UnionSelector  # noqa: E402
from llama_mcp_router.retrieval import tool_text  # noqa: E402
from llama_mcp_router.selectors import LayaRerankSelector, Selection, Selector, TwoPassSelector, load_groups_config  # noqa: E402
from llama_mcp_router.tools import tool_name  # noqa: E402


class Reranker:
    """Cross-encoder served by llama-server --rerank (POST /v1/rerank)."""

    def __init__(self, url: str, chars: int = 300):
        self.client = httpx.AsyncClient(base_url=url.rstrip("/"), timeout=60)
        self.chars = chars

    async def order(self, query: str, names: List[str], by: Dict[str, Any]) -> List[str]:
        if not names:
            return []
        docs = [tool_text(by[n], self.chars) for n in names]
        r = await self.client.post("/v1/rerank", json={"query": query, "documents": docs, "top_n": len(docs)})
        r.raise_for_status()
        scores = {it["index"]: it["relevance_score"] for it in r.json()["results"]}
        return [names[i] for i in sorted(range(len(names)), key=lambda i: -scores.get(i, -1e9))]

    async def aclose(self) -> None:
        await self.client.aclose()


class DraftThenRerank(Selector):
    """Draft (any selector, or every tool) -> cross-encoder verifies -> keep top_k."""

    def __init__(self, reranker: Reranker, draft: Optional[Selector], top_k: int, name: str):
        self.reranker, self.draft, self.top_k, self.name = reranker, draft, top_k, name

    async def select(self, query: str, tools: Sequence[Any]) -> Selection:
        by = {tool_name(t): t for t in tools}
        if self.draft is None:
            cands = list(by)
        else:
            d = await self.draft.select(query, tools)
            cands = list(dict.fromkeys(d.names))
        order = await self.reranker.order(query, cands, by)
        return Selection(order[: self.top_k], ranking=order)

    async def aclose(self) -> None:
        return None


class DraftRanking:
    """Adapter so LayaRerankSelector re-ranks a selector's draft (its 'retriever')."""

    def __init__(self, draft: Selector):
        self.draft = draft

    async def rank(self, query: str, tools: Sequence[Any]) -> List[str]:
        d = await self.draft.select(query, tools)
        return list(dict.fromkeys(list(d.names) + list(d.ranking or [])))

    async def aclose(self) -> None:
        return None


def laya_groups(url: str, cfg: Dict[str, Any], max_groups: int, views: Any = None) -> LayaSelector:
    kw = {"views": views} if views else {}
    return LayaSelector(url=url, groups=cfg, top_p=1.0, max_groups=max_groups, **kw)


def build(a, cfg) -> Dict[str, Selector]:
    sels: Dict[str, Selector] = {"laya+bm25": UnionSelector([laya_groups(a.laya_url, cfg, 2), BM25Selector(top_k=3)])}
    draft8 = UnionSelector([laya_groups(a.laya_url, cfg, 3), BM25Selector(top_k=8)])
    sels["laya 2-pass"] = LayaRerankSelector(url=a.laya_url, retriever=DraftRanking(draft8), shortlist=24, keep=5, also=3, chunk=12)
    if a.rerank_url:
        rr = Reranker(a.rerank_url)
        for k in (3, 5, 8):
            sels["bm25 -> rerank top%d" % k] = DraftThenRerank(rr, BM25Selector(top_k=20), k, "bm25-rr")
            sels["laya+bm25 -> rerank top%d" % k] = DraftThenRerank(rr, UnionSelector([laya_groups(a.laya_url, cfg, 3), BM25Selector(top_k=10)]), k, "lb-rr")
            sels["rerank all top%d" % k] = DraftThenRerank(rr, None, k, "rr-all")
    if a.clef_url:
        sels["clef groups+bm25"] = UnionSelector([laya_groups(a.clef_url, cfg, 2, views="single"), BM25Selector(top_k=3)])
        sels["clef groups(1)+bm25"] = UnionSelector([laya_groups(a.clef_url, cfg, 1, views="single"), BM25Selector(top_k=3)])
        sels["clef 2-pass"] = LayaRerankSelector(url=a.clef_url, retriever=DraftRanking(UnionSelector([laya_groups(a.clef_url, cfg, 3, views="single"),
                                                                                                         BM25Selector(top_k=8)])),
                                                 shortlist=24, keep=5, also=3, chunk=24, label_chars=120)
        sels["clef 2pass (router)"] = TwoPassSelector(url=a.clef_url, groups=cfg)
        sels["clef tools top5"] = LayaRerankSelector(url=a.clef_url, retriever=DraftRanking(BM25Selector(top_k=200)), shortlist=200, keep=5, also=0,
                                                     chunk=24, label_chars=120)
    if a.only:
        sels = {k: v for k, v in sels.items() if any(o in k for o in a.only.split(","))}
    return sels


async def run(sels: Dict[str, Selector], tools: List[Any], queries: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for name, sel in sels.items():
        rows = []
        for q in queries:
            t0 = time.perf_counter()
            try:
                res = await sel.select(q["query"], tools)
                names, err = list(res.names), None
                info = res.info or {}
                if any(k.endswith("error") for k in info):  # a union / rerank that silently fell back
                    err = "fallback: " + json.dumps({k: v for k, v in info.items() if k.endswith("error")})
            except Exception as e:  # noqa: BLE001
                names, err = [], repr(e)
            rows.append({"id": q["id"], "lang": q.get("lang", "en"), "hit": any(e in names for e in q["expect"]), "n": len(names),
                         "ms": (time.perf_counter() - t0) * 1000, "err": err, "names": names, "expect": q["expect"]})
        out[name] = rows
        errs = sum(1 for r in rows if r["err"])
        print("  %-28s recall %5.1f%%  tools %4.1f  %5.0f ms%s" % (name, 100 * sum(r["hit"] for r in rows) / len(rows), statistics.mean(r["n"] for r in rows),
                                                                   statistics.median(r["ms"] for r in rows), "  (%d errors)" % errs if errs else ""), flush=True)
    return out


def table(res: Dict[str, List[Dict[str, Any]]], title: str) -> str:
    lines = ["### %s" % title, "", "| selector | recall | EN | 中文 | tools sent | median ms |", "|---|---|---|---|---|---|"]
    for name, rows in res.items():
        en = [r for r in rows if r["lang"] != "zh"]
        zh = [r for r in rows if r["lang"] == "zh"]
        pct = lambda rs: "%.1f%%" % (100 * sum(r["hit"] for r in rs) / len(rs)) if rs else "–"  # noqa: E731
        lines.append("| %s | %s | %s | %s | %.1f | %.0f |" % (name, pct(rows), pct(en), pct(zh), statistics.mean(r["n"] for r in rows), statistics.median(r["ms"] for r in rows)))
    return "\n".join(lines) + "\n"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--laya-url", default="http://127.0.0.1:8000")
    ap.add_argument("--rerank-url")
    ap.add_argument("--clef-url")
    ap.add_argument("--groups", default=str(ROOT.parent / "examples/pubmed_groups.json"))
    ap.add_argument("--queries", default="queries_heldout.jsonl,queries_test.jsonl")
    ap.add_argument("--distractors", action="store_true")
    ap.add_argument("--only")
    ap.add_argument("--out")
    a = ap.parse_args()
    tools = json.load(open(ROOT / "data/pubmed_tools.json"))
    cfg = load_groups_config(a.groups)
    if a.distractors:
        tools += json.load(open(ROOT / "data/distractor_tools.json"))
        cfg["groups"].update(json.load(open(ROOT / "data/distractor_groups.json"))["groups"])
    queries = []
    for f in a.queries.split(","):
        queries += [dict(json.loads(l), id="%s:%s" % (f, json.loads(l)["id"])) for l in open(ROOT / "data" / f) if l.strip()]
    queries = [q for q in queries if q.get("expect")]
    print("%d tools, %d queries" % (len(tools), len(queries)))
    res = await run(build(a, cfg), tools, queries)
    md = table(res, "%d tools, %d queries (%s)" % (len(tools), len(queries), a.queries))
    print(md)
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
