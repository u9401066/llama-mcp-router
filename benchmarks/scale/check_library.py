"""Selector-level recall of the shipped selectors (retrieve, laya-rerank) on the real pool, via live services:
embeddings from $EMBED_URL (default llama-server --embedding with bge-m3 on :8082) and Laya on :8000."""
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))
from pool import load  # noqa: E402
from llama_mcp_router import HTTPEmbedder, LayaRerankSelector, Retriever, RetrieverSelector  # noqa: E402

EMBED = os.environ.get("EMBED_URL", "http://127.0.0.1:8082")


async def main():
    tools, server_of, _ = load()
    qs = [json.loads(l) for l in open(HERE / "queries_real.jsonl", encoding="utf-8")]
    pos = [q for q in qs if q["expect"]]
    shared = Retriever(HTTPEmbedder(EMBED))
    t0 = time.time()
    await shared.rank("warm up", tools)
    print("%d tools embedded in %.1fs" % (len(tools), time.time() - t0))
    configs = {
        "retrieve top-8 (BM25 only)": RetrieverSelector(Retriever(), top_k=8),
        "retrieve top-8 (bge-m3 + BM25)": RetrieverSelector(shared, top_k=8),
        "retrieve top-20 (bge-m3 + BM25)": RetrieverSelector(shared, top_k=20),
        "laya-rerank 24 -> keep 5 + also 5": LayaRerankSelector(retriever=shared),
    }
    print("| selector | recall | 中文 | tools sent | ms median (sequential) |\n|---|---|---|---|---|")
    for name, sel in configs.items():
        hits, n, ms, zh = 0, 0, [], []
        for q in pos:
            t = time.perf_counter()
            r = await sel.select(q["query"], tools)
            ms.append((time.perf_counter() - t) * 1000)
            ok = any(e in r.names for e in q["expect"])
            hits += ok
            n += len(r.names)
            if q["lang"] == "zh":
                zh.append(ok)
        print("| %s | %.1f%% | %.1f%% | %.1f | %.0f |" % (name, 100 * hits / len(pos), 100 * sum(zh) / len(zh), n / len(pos), statistics.median(ms)), flush=True)


asyncio.run(main())
