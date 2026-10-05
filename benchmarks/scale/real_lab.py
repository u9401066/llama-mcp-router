"""Tool selection over the real 639-tool / 19-server pool (selector level, no LLM).

recall = an acceptable tool is among the tools that would be sent.
    PRIVATE_TOOLS_DIR=... python benchmarks/scale/real_lab.py [--only "a;b"]
Needs emb.npz from embed.py and a laya-serve on :8000.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import httpx
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))
from pool import load  # noqa: E402
from llama_mcp_router.bm25 import BM25  # noqa: E402
from llama_mcp_router.tools import first_sentence, tool_description, tool_name  # noqa: E402

TOOLS, SERVER_OF, SERVERS = load()
NAMES = [tool_name(t) for t in TOOLS]
import os  # noqa: E402
E = np.load(HERE / os.environ.get("EMBED_OUT", "emb.npz"))
IDX = json.load(open(HERE / "emb_index.json"))
assert IDX["tools"] == NAMES, "re-run embed.py (pool changed)"
QROW = {q: i for i, q in enumerate(IDX["queries"])}
SNAMES = IDX["servers"]
LABEL = {n: first_sentence(tool_description(t), 90) for n, t in zip(NAMES, TOOLS)}
LABEL_NAME = {n: "%s: %s" % (n.split("_", 1)[1].replace("_", " "), first_sentence(tool_description(t), 70)) for n, t in zip(NAMES, TOOLS)}
LABEL_SRV = {n: "%s %s: %s" % (n.split("_", 1)[0], n.split("_", 1)[1].replace("_", " "), first_sentence(tool_description(t), 60)) for n, t in zip(NAMES, TOOLS)}
BM = BM25(["%s %s" % (n.replace("_", " "), first_sentence(tool_description(t), 300)) for n, t in zip(NAMES, TOOLS)])
SEM = asyncio.Semaphore(4)
CLIENT: httpx.AsyncClient


async def laya_choice(query, options, model=None, raw=False):
    if len(options) <= 1:
        return {k: 1.0 for k in options}
    body = {"state": query if raw else {"request": query}, "questions": {"g": {"type": "choice", "instructions": "Which tool is needed to handle this request?" if raw else "Which tool does `request` need?", "criteria": options}}}
    if model:
        body["model"] = model
    async with SEM:
        r = await CLIENT.post("/v1/systemone", json=body)
    r.raise_for_status()
    return {k: float(v) for k, v in r.json()["answers"]["g"]["probabilities"].items()}


def top(d, k):
    return [n for n, _ in sorted(d.items(), key=lambda kv: -kv[1])[:k]]


# ---------------------------------------------------------------- retrieval (no Laya)
def s_bm25(q):
    return BM.scores(q)


def s_emb(q):
    return E["t"] @ E["q"][QROW[q]]


def rank(scores, allowed=None):
    order = np.argsort(-np.asarray(scores))
    return [NAMES[i] for i in order if allowed is None or SERVER_OF[NAMES[i]] in allowed]


def rrf(*rankings, k0=20):
    s = {}
    for r in rankings:
        for i, n in enumerate(r[:200]):
            s[n] = s.get(n, 0.0) + 1.0 / (k0 + i)
    return top(s, len(s))


def hybrid(q, allowed=None):
    return rrf(rank(s_emb(q), allowed), rank(s_bm25(q), allowed))


# ---------------------------------------------------------------- server gating
async def gate_laya(q, n):
    return top(await laya_choice(q, {s: SERVERS[s]["label"] for s in SNAMES}), n)


def gate_emb(q, n):
    return [SNAMES[i] for i in np.argsort(-(E["s"] @ E["q"][QROW[q]]))[:n]]


def gate_tools(q, n):
    """Servers ranked by their best tools in the hybrid ranking."""
    out = []
    for t in hybrid(q):
        if SERVER_OF[t] not in out:
            out.append(SERVER_OF[t])
        if len(out) == n:
            break
    return out


# ---------------------------------------------------------------- Laya reranking of a shortlist
async def laya_rerank(q, cands, keep, chunk=12, labels=None, model=None, raw=False):
    """Laya ranks a shortlist; >chunk candidates are split into chunks (Laya reads ~11 tokens per option)."""
    L = labels or LABEL
    chunks = [cands[i:i + chunk] for i in range(0, len(cands), chunk)]
    if len(chunks) == 1:
        return top(await laya_choice(q, {c: L[c] for c in cands}, model, raw), keep)
    per = await asyncio.gather(*[laya_choice(q, {c: L[c] for c in ch}, model, raw) for ch in chunks])
    finalists = [n for p in per for n in top(p, max(2, keep // len(chunks) + 1))]
    return top(await laya_choice(q, {c: L[c] for c in finalists}, model, raw), keep)


async def laya_scores(q, cands, chunk=12, labels=None, model=None, raw=False):
    """Laya probabilities for every candidate (per chunk, renormalised within the chunk)."""
    L = labels or LABEL
    chunks = [cands[i:i + chunk] for i in range(0, len(cands), chunk)]
    per = await asyncio.gather(*[laya_choice(q, {c: L[c] for c in ch}, model, raw) for ch in chunks])
    out = {}
    for p in per:
        out.update(p)
    return out


def is_cjk(q):
    return any("\u3040" <= ch <= "\u9fff" or "\uac00" <= ch <= "\ud7af" for ch in q)


def hybrid_cjk(q, allowed=None):
    """BM25 cannot tokenise mixed CJK well against English descriptions: embeddings only for CJK requests."""
    return rank(s_emb(q), allowed) if is_cjk(q) else hybrid(q, allowed)


def union(*lists):
    out = []
    for l in lists:
        out += [x for x in l if x not in out]
    return out


async def fused(q, n_short=24, keep=8, w_laya=1.0, **kw):
    h = hybrid_cjk(q)[:n_short]
    ls = await laya_scores(q, h, **kw)
    lr = top(ls, len(ls))
    s = {}
    for i, n in enumerate(h):
        s[n] = s.get(n, 0) + 1.0 / (20 + i)
    for i, n in enumerate(lr):
        s[n] = s.get(n, 0) + w_laya / (20 + i)
    return top(s, keep)


async def ensemble_rerank(q, n_short=24, keep=5):
    h = hybrid_cjk(q)[:n_short]
    views = await asyncio.gather(laya_scores(q, h), laya_scores(q, h, labels=LABEL_NAME, raw=True), laya_scores(q, h, model="multilingual"))
    avg = {n: sum(v.get(n, 0) for v in views) / len(views) for n in h}
    return top(avg, keep)


STRATS = {
    "bm25 top-10": lambda q: rank(s_bm25(q))[:10],
    "emb top-10": lambda q: rank(s_emb(q))[:10],
    "hybrid top-5": lambda q: hybrid(q)[:5],
    "hybrid top-10": lambda q: hybrid(q)[:10],
    "hybrid top-20": lambda q: hybrid(q)[:20],
    "laya flat over 19 servers (top-3) -> hybrid top-10": None,
    "emb gate 3 servers -> hybrid top-10": lambda q: hybrid(q, set(gate_emb(q, 3)))[:10],
    "hybrid top-12 -> laya keep 5": None,
    "hybrid top-12 -> laya keep 5 | hybrid top-5": None,
    "hybrid top-24 -> laya (2x12) keep 8": None,
    "hybrid top-24 -> laya keep 5 | hybrid top-5": None,
    "hybrid-cjk top-8": lambda q: hybrid_cjk(q)[:8],
    "hybrid-cjk top-10": lambda q: hybrid_cjk(q)[:10],
    "hybrid-cjk top-20": lambda q: hybrid_cjk(q)[:20],
    "cjk: top-24 -> laya keep 5 | top-5": None,
    "cjk: top-24 -> laya(name labels) keep 5 | top-5": None,
    "cjk: top-24 -> laya(server+name labels) keep 5 | top-5": None,
    "cjk: top-24 -> laya(multilingual) keep 5 | top-5": None,
    "cjk: top-24 -> laya ensemble(3 views) keep 5 | top-5": None,
    "cjk: fuse(retrieval rank, laya rank) top-8": None,
    "cjk: fuse(retrieval, 2x laya) top-8": None,
    "cjk: fuse(retrieval, laya) top-10": None,
}


async def strat(name, q):
    if STRATS[name] is not None:
        return STRATS[name](q)
    if name.startswith("laya flat over 19 servers"):
        return hybrid(q, set(await gate_laya(q, 3)))[:10]
    if name == "hybrid top-12 -> laya keep 5":
        return await laya_rerank(q, hybrid(q)[:12], 5)
    if name == "hybrid top-12 -> laya keep 5 | hybrid top-5":
        h = hybrid(q)
        return union(await laya_rerank(q, h[:12], 5), h[:5])
    if name == "hybrid top-24 -> laya (2x12) keep 8":
        return await laya_rerank(q, hybrid(q)[:24], 8)
    if name == "hybrid top-24 -> laya keep 5 | hybrid top-5":
        h = hybrid(q)
        return union(await laya_rerank(q, h[:24], 5), h[:5])
    if name.startswith("cjk: top-24 -> laya"):
        h = hybrid_cjk(q)
        kw = {"labels": LABEL_NAME} if "(name labels)" in name else {"labels": LABEL_SRV} if "server+name" in name else {"model": "multilingual"} if "multilingual" in name else {}
        if "ensemble" in name:
            return union(await ensemble_rerank(q), h[:5])
        return union(await laya_rerank(q, h[:24], 5, **kw), h[:5])
    if name == "cjk: fuse(retrieval rank, laya rank) top-8":
        return await fused(q)
    if name == "cjk: fuse(retrieval, 2x laya) top-8":
        return await fused(q, w_laya=2.0)
    if name == "cjk: fuse(retrieval, laya) top-10":
        return await fused(q, keep=10)
    raise KeyError(name)


def report(name, rows):
    pos = [(q, n, ms) for q, n, ms in rows if q["expect"]]

    def hit(sub):
        return 100.0 * sum(any(e in n for e in q["expect"]) for q, n, _ in sub) / len(sub) if sub else float("nan")

    pz = [r for r in pos if SERVER_OF[r[0]["expect"][0]] in ("pubmed", "zotero")]
    other = [r for r in pos if SERVER_OF[r[0]["expect"][0]] not in ("pubmed", "zotero")]
    srv = 100.0 * sum(any(SERVER_OF[x] in {SERVER_OF[e] for e in q["expect"]} for x in n) for q, n, _ in pos) / len(pos)
    return "| %s | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f | %.0f |" % (
        name, hit(pos), hit(pz), hit(other), hit([r for r in pos if r[0]["lang"] == "en"]), hit([r for r in pos if r[0]["lang"] == "zh"]),
        srv, statistics.mean(len(n) for _, n, _ in pos), statistics.median(ms for *_, ms in pos))


async def main():
    global CLIENT
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--laya-url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", default=str(HERE / "real_lab_report.md"))
    a = ap.parse_args()
    CLIENT = httpx.AsyncClient(base_url=a.laya_url, timeout=60)
    queries = [json.loads(l) for l in open(HERE / "queries_real.jsonl", encoding="utf-8")]
    names = [n for n in STRATS if not a.only or n in a.only.split(";")]
    pos = [q for q in queries if q["expect"]]
    head = ["%d tools, %d servers, %d tool requests (%d PubMed/Zotero, %d other servers)" % (len(TOOLS), len(SERVERS), len(pos), sum(SERVER_OF[q['expect'][0]] in ('pubmed', 'zotero') for q in pos), sum(SERVER_OF[q['expect'][0]] not in ('pubmed', 'zotero') for q in pos)), "",
            "| strategy | recall | PubMed+Zotero | other servers | EN | 中文 | right server present | tools sent | ms (median) |", "|---|---|---|---|---|---|---|---|---|"]
    print("\n".join(head), flush=True)
    lines = []
    for name in names:
        async def one(q):
            t0 = time.perf_counter()
            n = await strat(name, q["query"])
            return q, n, (time.perf_counter() - t0) * 1000
        rows = await asyncio.gather(*[one(q) for q in queries])
        line = report(name, rows)
        lines.append(line)
        print(line, flush=True)
    Path(a.out).write_text("\n".join(head + lines) + "\n")
    await CLIENT.aclose()


if __name__ == "__main__":
    asyncio.run(main())
