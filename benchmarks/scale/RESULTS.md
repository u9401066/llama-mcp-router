# 600+ tools: results

Pool: **639 real tools from 19 MCP servers** (`servers.json`; 17 public servers are in `real/`, 2 private ones are not published,
so the public pool has 553 tools). Their schemas are 992 KB ≈ 247k tokens, i.e. they do not even fit the model's 131k context.
Queries: `queries_real.jsonl`, 156 tool requests (98 PubMed/Zotero, 58 for the other 17 servers; several tools accepted where servers
overlap) + 12 chit-chat, 44 in Chinese. Model: Bonsai 27B (llama.cpp, PrismML fork), reasoning_effort=medium, temperature 0.

## 1. Selection recall (no LLM; `real_lab.py`)

multilingual-e5-small embeddings (`real_lab_report.md`):

| strategy | recall | PubMed+Zotero | other servers | EN | 中文 | right server present | tools sent | ms (median) |
|---|---|---|---|---|---|---|---|---|
| bm25 top-10 | 61.5% | 51.0% | 79.3% | 75.7% | 22.0% | 77.6% | 10.0 | 0 |
| emb top-10 | 71.2% | 62.2% | 86.2% | 79.1% | 48.8% | 87.2% | 10.0 | 5 |
| hybrid top-5 | 69.2% | 61.2% | 82.8% | 79.1% | 41.5% | 84.6% | 5.0 | 6 |
| hybrid top-10 | 75.0% | 69.4% | 84.5% | 84.3% | 48.8% | 90.4% | 10.0 | 6 |
| hybrid top-20 | 79.5% | 75.5% | 86.2% | 88.7% | 53.7% | 91.7% | 20.0 | 6 |
| laya flat over 19 servers (top-3) -> hybrid top-10 | 64.1% | 55.1% | 79.3% | 72.2% | 41.5% | 75.0% | 10.0 | 1016 |
| emb gate 3 servers -> hybrid top-10 | 60.3% | 51.0% | 75.9% | 66.1% | 43.9% | 68.6% | 10.0 | 7 |
| hybrid top-12 -> laya keep 5 | 73.7% | 68.4% | 82.8% | 84.3% | 43.9% | 88.5% | 5.0 | 1492 |
| hybrid top-12 -> laya keep 5 | hybrid top-5 | 75.6% | 70.4% | 84.5% | 85.2% | 48.8% | 88.5% | 6.5 | 1374 |
| hybrid top-24 -> laya (2x12) keep 8 | 76.9% | 71.4% | 86.2% | 87.8% | 46.3% | 89.1% | 8.0 | 4848 |
| hybrid top-24 -> laya keep 5 | hybrid top-5 | 77.6% | 71.4% | 87.9% | 87.0% | 51.2% | 89.1% | 8.0 | 4762 |

The "ms" column above was measured with 4 concurrent requests and includes queueing; sequential Laya time is ~28 ms (12-tool shortlist, 1 call) and ~100 ms (24-tool shortlist, 3 calls).

BAAI/bge-m3 embeddings (`real_lab_bge-m3.md`); `cjk:` = embeddings only for CJK requests:

| strategy | recall | PubMed+Zotero | other servers | EN | 中文 | right server present | tools sent | ms (median) |
|---|---|---|---|---|---|---|---|---|
| hybrid top-10 | 82.1% | 75.5% | 93.1% | 88.7% | 63.4% | 91.7% | 10.0 | 14 |
| hybrid top-24 -> laya keep 5 | hybrid top-5 | 85.9% | 80.6% | 94.8% | 93.9% | 63.4% | 93.6% | 8.0 | 5595 |
| hybrid-cjk top-8 | 82.7% | 75.5% | 94.8% | 87.8% | 68.3% | 93.6% | 8.0 | 15 |
| hybrid-cjk top-10 | 83.3% | 76.5% | 94.8% | 88.7% | 68.3% | 94.2% | 10.0 | 15 |
| hybrid-cjk top-20 | 88.5% | 81.6% | 100.0% | 93.0% | 75.6% | 96.8% | 20.0 | 14 |
| cjk: top-24 -> laya keep 5 | top-5 | 87.2% | 81.6% | 96.6% | 93.9% | 68.3% | 94.2% | 8.0 | 5955 |
| cjk: top-24 -> laya(name labels) keep 5 | top-5 | 87.8% | 81.6% | 98.3% | 93.0% | 73.2% | 95.5% | 8.1 | 5937 |
| cjk: top-24 -> laya(server+name labels) keep 5 | top-5 | 86.5% | 81.6% | 94.8% | 93.0% | 68.3% | 94.9% | 8.1 | 5961 |
| cjk: top-24 -> laya(multilingual) keep 5 | top-5 | 84.6% | 78.6% | 94.8% | 90.4% | 68.3% | 93.6% | 8.1 | 5192 |
| cjk: top-24 -> laya ensemble(3 views) keep 5 | top-5 | 85.9% | 80.6% | 94.8% | 92.2% | 68.3% | 94.2% | 8.6 | 8227 |
| cjk: fuse(retrieval rank, laya rank) top-8 | 86.5% | 80.6% | 96.6% | 93.0% | 68.3% | 94.2% | 8.0 | 3196 |
| cjk: fuse(retrieval, 2x laya) top-8 | 85.3% | 78.6% | 96.6% | 93.0% | 63.4% | 92.9% | 8.0 | 3345 |
| cjk: fuse(retrieval, laya) top-10 | 86.5% | 80.6% | 96.6% | 93.0% | 68.3% | 94.2% | 10.0 | 3180 |

Shipped selectors through live services (`check_library.py`: bge-m3 Q8_0 GGUF served by llama-server `--embedding` on CPU, laya-serve on GPU; sequential):

| selector | recall | 中文 | tools sent | ms median |
|---|---|---|---|---|
| retrieve top-8 (BM25 only) | 62.2% | 22.0% | 8.0 | 0 |
| retrieve top-8 (bge-m3 + BM25) | 82.7% | 68.3% | 8.0 | 171 |
| retrieve top-20 (bge-m3 + BM25) | 89.7% | 75.6% | 20.0 | 174 |
| laya-rerank 24 → keep 5 + also 5 | 86.5% | 70.7% | 8.1 | 168 |

## 2. Is Laya's confidence useful? (12-tool shortlist, 156 tool requests)

| Laya top-1 probability | n | top-1 right | top-3 right | answer in shortlist |
|---|---|---|---|---|
| ≥ 0.9 | 96 | 76.0% | 85.4% | 91.7% |
| 0.7 – 0.9 | 27 | 48.1% | 70.4% | 70.4% |
| 0.5 – 0.7 | 15 | 6.7% | 46.7% | 60.0% |
| 0.3 – 0.5 | 17 | 17.6% | 58.8% | 76.5% |

Informative but over-confident (≥ 0.9 is right 76% of the time), as Laya's README warns for the shipped checkpoints.

## 3. End to end (`e2e.py`): the client sends all 639 tools, the router picks, the 27B model answers

Routers ran with `--escalate --no-server-tools --catalog-max 80`; with 600+ tools the meta-tool takes a search query.
"all tools" is not a baseline here: 247k tokens of schema exceed the 131k context.

| router | correct | tool requests | chit-chat | PubMed+Zotero | other servers | 中文 | escalated | tools sent (1st round) | prompt tokens | s/query |
|---|---|---|---|---|---|---|---|---|---|---|
| retrieve top-8 + search | 82.1% | 80.8% | 100.0% | 81.6% | 79.3% | 72.7% | 23/168 | 8.0 | 6812 | 11.3 |
| laya-rerank + search | 78.6% | 77.6% | 91.7% | 78.6% | 75.9% | 72.7% | 26/168 | 6.4 | 5895 | 11.3 |
| search only (no tools) | 71.4% | 69.2% | 100.0% | 74.5% | 60.3% | 68.2% | 140/168 | 0.0 | 4658 | 11.6 |
| laya-rerank (also 5) + search | 81.0% | 79.5% | 100.0% | 83.7% | 72.4% | 75.0% | 23/168 | 8.1 | 6860 | 12.0 |

(`laya-rerank + search` used `--keep 5 --also 3`, 6.4 tools; `(also 5)` is the lab recipe, 8.1 tools. Raw rows: `e2e_results_main.json`, `e2e_results_also5.json`.)
Before schema sanitising, 7 of 168 requests failed with HTTP 400 from llama-server (3 of the 639 tools have schemas llama.cpp cannot turn into a grammar); after it, 0.
