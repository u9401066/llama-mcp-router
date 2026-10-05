"""End-to-end over the real pool: the client sends ALL tools; the router picks, the model answers.
Correct = the model's first real tool call is acceptable (or no call for chit-chat), after any router escalation.
    python benchmarks/scale/e2e.py --router name=http://127.0.0.1:8093 [--router ...] [--sample N]
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from pool import load  # noqa: E402

SYSTEM = ("You are a research assistant with access to tools. Always handle the user's request by calling the single most "
          "appropriate tool right away. Never ask clarifying questions; use the information in the request.")


async def one(client, url, q, tools, effort):
    body = {"model": "m", "temperature": 0, "max_tokens": 4096, "tools": tools, "chat_template_kwargs": {"reasoning_effort": effort},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q["query"]}]}
    t0 = time.perf_counter()
    r = await client.post(url + "/v1/chat/completions", json=body)
    dt = time.perf_counter() - t0
    if r.status_code != 200:
        return {"id": q["id"], "ok": False, "first": "HTTP %d" % r.status_code, "s": dt, "prompt_tokens": 0, "escalated": False, "sent": 0}
    d = r.json()
    calls = d["choices"][0]["message"].get("tool_calls") or []
    first = calls[0]["function"]["name"] if calls else None
    ok = (first in q["expect"]) if q["expect"] else first is None
    return {"id": q["id"], "ok": ok, "first": first, "s": dt, "prompt_tokens": d["usage"]["prompt_tokens"],
            "escalated": bool(r.headers.get("x-router-loaded")), "sent": len([x for x in r.headers.get("x-router-tools", "").split(",") if x])}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", action="append", required=True, help="name=url")
    ap.add_argument("--sample", type=int)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--out", default=str(HERE / "e2e_results.json"))
    a = ap.parse_args()
    tools, server_of, _ = load()
    qs = [json.loads(l) for l in open(HERE / "queries_real.jsonl", encoding="utf-8")]
    if a.sample:
        qs = [qs[int(i * len(qs) / a.sample)] for i in range(a.sample)]
    sem = asyncio.Semaphore(a.concurrency)
    results = {}
    async with httpx.AsyncClient(timeout=1800) as client:
        for spec in a.router:
            name, url = spec.split("=", 1)

            async def run(q):
                async with sem:
                    return await one(client, url, q, tools, a.effort)

            t0 = time.time()
            results[name] = await asyncio.gather(*[run(q) for q in qs])
            print("  %-28s %d/%d correct (%.0fs)" % (name, sum(r["ok"] for r in results[name]), len(qs), time.time() - t0), flush=True)
            Path(a.out).write_text(json.dumps(results, indent=1))
    by = {q["id"]: q for q in qs}
    print("\n%d tools offered by the client, %d queries\n" % (len(tools), len(qs)))
    print("| router | correct | tool requests | chit-chat | PubMed+Zotero | other servers | 中文 | escalated | tools sent (1st round) | prompt tokens | s/query |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, rows in results.items():
        pos = [r for r in rows if by[r["id"]]["expect"]]
        neg = [r for r in rows if not by[r["id"]]["expect"]]
        pz = [r for r in pos if server_of[by[r["id"]]["expect"][0]] in ("pubmed", "zotero")]
        ot = [r for r in pos if r not in pz]
        zh = [r for r in rows if by[r["id"]]["lang"] == "zh"]
        p = lambda xs: "%.1f%%" % (100.0 * sum(x["ok"] for x in xs) / len(xs)) if xs else "-"  # noqa: E731
        print("| %s | %s | %s | %s | %s | %s | %s | %d/%d | %.1f | %.0f | %.1f |" % (
            name, p(rows), p(pos), p(neg), p(pz), p(ot), p(zh), sum(r["escalated"] for r in rows), len(rows),
            statistics.mean(r["sent"] for r in rows), statistics.mean(r["prompt_tokens"] for r in rows), statistics.mean(r["s"] for r in rows)))


asyncio.run(main())
