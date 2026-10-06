# Decision models for tool selection: Laya twice, a reranker, Cloudflare Clef-Flash

Can tool selection get closer to "send every tool" accuracy without sending every tool? Measured on the pools and queries of the
README accuracy table (41 PubMed tools; 133 with `--distractors`), all models local on one RTX 4090.

* `selector_lab.py` — selection only (no LLM): **recall** (the expected tool is among those sent), tools sent, latency;
  108 queries = `queries_heldout.jsonl` (52, never used for tuning) + the tool requests of `queries_test.jsonl` (56); 33 are Chinese.
* `run_bench.py --clef-url …` — the 27B model (Bonsai, llama-server) answers each of the 52 held-out queries with the selected tools;
  **correct** = its first tool call is the expected one.

## Selection (108 queries)

| selector | 41 tools: recall | 中文 | tools sent | 133 tools: recall | 中文 | tools sent | median ms |
|---|---|---|---|---|---|---|---|
| `laya+bm25` (the router's default) | 93.5% | 84.8% | 7.7 | 81.5% | 60.6% | 9.1 | 35 |
| Laya twice (groups, then tool by tool) | 83.3% | 66.7% | 6.3 | 79.6% | 63.6% | 6.5 | 70 |
| BM25 top 20 → reranker top 5 | 74.1% | 48.5% | 4.2 | 73.1% | 48.5% | 4.2 | 100 |
| Laya+BM25 draft → reranker top 5 | 90.7% | 84.8% | 5.0 | 83.3% | 69.7% | 5.0 | 150 |
| reranker over every tool, top 8 | 90.7% | 93.9% | 8.0 | 89.8% | 90.9% | 8.0 | 290 / 840 |
| **Clef-Flash groups + BM25** | **100%** | **100%** | 7.6 | **100%** | **100%** | 8.9 | 430 |
| **Clef-Flash, two passes (`2pass`)** | **100%** | **100%** | **5.8** | **99.1%** | **100%** | **5.9** | 860–950 |
| Clef-Flash groups (1) + BM25 | 94.4% | 93.9% | 4.8 | 94.4% | 97.0% | 4.9 | 400 |
| Clef-Flash tool by tool over every tool | 98.1% | 97.0% | 5.0 | 95.4% | 90.9% | 5.0 | 1,300 / 3,300 |

Reranker: Qwen3-Reranker-0.6B (Q8_0) on llama-server `--rerank`. Clef-Flash: `ggml-org/Clef-Flash-GGUF` Q4_K_M on upstream
llama.cpp b11433, **partly on the CPU** (Bonsai and Laya keep the GPU), hence the latencies; fully on a GPU it is several times faster.

## First tool call of the 27B model (52 held-out queries)

| pool | selector | correct | EN | 中文 | prompt tokens | s / query |
|---|---|---|---|---|---|---|
| 41 | `laya+bm25` | 92.3% | 96.9% | 85.0% | 6,241 | 7.3 |
| 41 | **`2pass` with Clef-Flash** | **98.1%** | 96.9% | **100%** | **4,415** | 6.0 |
| 133 | `laya+bm25` | 84.6% | 100% | 60.0% | 5,118 | 7.0 |
| 133 | Clef-Flash groups + BM25 | 98.1% | 96.9% | 100% | 5,183 | 6.5 |
| 133 | **`2pass` with Clef-Flash** | **98.1%** | 96.9% | **100%** | **4,058** | 5.1 |

For comparison (README table): every tool sent = 96.2% on both pools; the oracle (exactly the right group) = 96.2% / 98.1%.
Two passes of Clef-Flash match the oracle and beat sending everything, with ~30% fewer prompt tokens than Laya+BM25.

## What this says

* **Running Laya twice does not help.** Laya reads the question and *all* options through a 192-token head (HARNESS.md §2) and
  scores 12.7 nDCG@10 on ToolRet in Cloudflare's comparison (Clef-Flash 66.4): its second opinion repeats its first mistake. Running
  it several times only pays off as an ensemble of different framings (what the router already does, ~30 ms).
* **"Fast draft, strong verifier" works** — like speculative decoding with a draft model: a cheap pass narrows the tools, a model
  that can actually tell tools apart decides. The verifier has to be stronger than the draft: a 0.6B cross-encoder helps mainly on
  Chinese at 133 tools; Clef-Flash (9B, the same `/v1/systemone` API as Laya) gets ~100%.
* `laya-typed-decisions` (Sept 2026) is English-only and fine-tuned on four workflow types (invoices, customer service, security
  incidents, agent traces), not tool routing.

## Reproduce

```bash
# reranker (any llama.cpp):  llama-server -m qwen3-reranker-0.6b-q8_0.gguf --rerank -c 8192 -np 8 -b 2048 -ub 2048 --cache-ram 0 --port 8085
# Clef-Flash (llama.cpp >= b11433, PR #29831):  llama-server -m Clef-Flash-Q4_K_M.gguf -c 8192 -b 2048 -ub 2048 -np 1 --cache-ram 0 --port 8084
python benchmarks/selector_lab.py --rerank-url http://127.0.0.1:8085 --clef-url http://127.0.0.1:8084 [--distractors]
python benchmarks/run_bench.py --queries benchmarks/data/queries_heldout.jsonl --clef-url http://127.0.0.1:8084 \
    --only "v0.2 laya+bm25,clef 2-pass" --llm http://127.0.0.1:8081 [--distractors]
```

`-ub` must cover the whole decision request (Clef reads it in one batch); `--cache-ram 0` keeps llama-server's host prompt cache
(8 GB by default) from filling a small machine's RAM.
