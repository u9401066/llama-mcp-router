# llama-mcp-router

**Send `llama-server` only the MCP tools a request needs — not all of them.**

`llama-server` (llama.cpp) can expose MCP tools (`--mcp-servers-config`), but a client then puts **every** tool schema in the prompt.
One real MCP server (41 tools) is ~137 KB of JSON ≈ **30,000 prompt tokens** before the user has said a word. The model has to
read all of it on every cold start (new chat, different client, evicted prompt cache), which costs seconds of prefill and a lot of context.

`llama-mcp-router` is a small OpenAI-compatible proxy that sits **in front of llama-server**. Before the prompt reaches the model, a
pluggable *selector* picks the few tools that fit the request, and only those go into the prompt:

```
client ──► llama-mcp-router ──► llama-server (+ MCP servers)
              │  1. read GET /tools (cached)
              │  2. selector(query, tools) -> a handful of tool names
              │        e.g. Laya (15 ms), BM25, or your own
              └─ 3. forward the request with only those tools
```

The default selector asks [Laya](https://huggingface.co/convaiinnovations/laya), a tiny non-autoregressive decision model, which tool
group the request belongs to (≈ 12 ms per call), and unions that with a lexical BM25 pick so that either one's mistakes are covered.

* Works with **any OpenAI-compatible backend** (llama-server, vLLM, Ollama's `/v1`, …); `GET /tools` is used when present.
* **Fails open**: if the selector is down the router falls back to sending all tools (or none).
* **Python ≥ 3.9**, three small dependencies (`httpx`, `starlette`, `uvicorn`). No GPU, no model download for the router itself.
* Extensible: selectors are plain classes, registered by name, entry-point or `pkg.mod:Class`.

> **Read [the benchmark](#benchmark) before you adopt this.** It is a *latency / context* optimisation, not an accuracy booster:
> on a 41-tool MCP server the router cut cold prefill from **11.4 s to 2.9 s (3.9×)** and prompt tokens by 73%, but a 27B model with
> *all* tools already chose the right tool 96% of the time, so the selector costs a few points of accuracy (92% here). And when
> llama-server's prompt cache is warm, sending *all* tools is faster (0.3 s) than a changing selection (2.9 s).

## Install

```bash
pip install git+https://github.com/u9401066/llama-mcp-router      # or: pipx install git+https://github.com/u9401066/llama-mcp-router
```

(Not on PyPI yet.)

## Quick start

1. Run `llama-server` with your MCP servers ([examples/llama-server-mcp.json](examples/llama-server-mcp.json)):

   ```bash
   llama-server -m model.gguf --jinja --port 8080 --mcp-servers-config mcp.json
   ```

2. Run a Laya server (optional but recommended for non-English requests): `pip install "laya[serve]" && laya-serve` (port 8000).

3. Tell the router what your tool groups are ([examples/pubmed_groups.json](examples/pubmed_groups.json) is a complete example):

   ```bash
   llama-mcp-router serve --backend http://127.0.0.1:8080 --port 8090 \
       --groups examples/pubmed_groups.json --selector laya+bm25 --laya-url http://127.0.0.1:8000
   ```

4. Point your client (Web UI, Open WebUI, VS Code, your own code) at `http://127.0.0.1:8090/v1` instead of `:8080`.

Check what would be sent for a query, without calling the model:

```bash
llama-mcp-router select "把 PMID 12345678 匯出成 BibTeX" --backend http://127.0.0.1:8080 --groups examples/pubmed_groups.json
```

```
{"selected": ["pubmed_prepare_export", "pubmed_save_literature_notes", "pubmed_fetch_article_details", ...], "pool": 41, ...}
```

## How requests are handled

For `POST /v1/chat/completions`:

1. **Candidate pool** = tools in the request (`tools`) ∪ llama-server's `GET /tools` (unless `--no-server-tools`). Client tools win name collisions.
2. **Query** = the last user message (plus the previous one if the last is very short).
3. **Selection** = selector output (capped by `--max-tools`) ∪ `--always` ∪ tools already called in this conversation ∪ the tool named by `tool_choice`.
4. (`--sticky`, opt-in) the conversation's previous tool list is reused when it already covers this turn's choice, otherwise new tools are appended.
5. The request is forwarded with `tools` replaced by the selection (streaming is passed through untouched).
   Response headers `X-Router-Tools`, `X-Router-Pool`, `X-Router-Select-Ms` show what happened.

Skip the router per request with header `X-Router-Bypass: 1` or body field `"router": false`. All other paths (`/health`, `/props`, `/tools`, …) are proxied unchanged.

### Modes

| `--mode` | behaviour |
|---|---|
| `inject` (default) | The client runs tool calls, exactly as with plain llama-server. |
| `agent` | The router also **executes** llama-server tools (`POST /tools`) and loops until the model answers (`--max-iterations`). Tool calls for client-declared tools are returned to the client. Streaming is emulated (the full answer arrives as one chunk). |

### Tool groups

Groups give the selector a small, meaningful label set instead of 40 raw tools:

```json
{
  "always": [],
  "groups": {
    "export": {"description": "exporting citations to RIS / BibTeX / EndNote", "tools": ["pubmed_prepare_export", "pubmed_save_*"]}
  }
}
```

`tools` accept `fnmatch` patterns. **Tools not mentioned in any group become one-tool groups**, so a newly added tool is never invisible.
Keep group descriptions short and about the *user's intent*; with Laya they are the answer options of a `choice` question.

## Selectors

| name | what it does | needs |
|---|---|---|
| `all` | every tool (baseline, same as plain llama-server) | – |
| `bm25` | lexical top-k over tool names + descriptions; CJK-aware tokeniser | – |
| `laya` | Laya `choice` over tool groups; keeps the top `--max-groups` (default 3) groups (or fewer once `--top-p` of the probability mass is covered; default 1.0 = always 3, which measured better than adaptive cut-offs). `--laya-state json\|raw`, `--laya-labels label\|description\|auto` control how the question is asked (see [benchmarks/HARNESS.md](benchmarks/HARNESS.md)) | a `laya-serve` instance |
| `laya+bm25` | union (default) | both |

Write your own:

```python
from llama_mcp_router import Selector, Selection

class MySelector(Selector):
    name = "mine"
    async def select(self, query, tools):      # tools are OpenAI-format dicts
        return Selection(["my_tool_a", "my_tool_b"])
```

```bash
llama-mcp-router serve --selector my_pkg.my_mod:MySelector          # or register it under the "llama_mcp_router.selectors" entry-point group
```

Selectors combine with `+` (`my_pkg.my_mod:MySelector+bm25`). A selector that raises is skipped; if all fail the router falls back (`--fallback all|none`).

Library use:

```python
from llama_mcp_router import LayaSelector, BM25Selector, UnionSelector
sel = UnionSelector([LayaSelector(url="http://127.0.0.1:8000", groups=groups_cfg), BM25Selector(top_k=3)])
names = (await sel.select("export these to bibtex", tools)).names
```

## CLI

```
llama-mcp-router serve  --backend URL --port 8090 --selector laya+bm25 --groups FILE [--mode inject|agent] [--max-tools 12] [--always a,b] [--exclude 'fs_*']
llama-mcp-router select "query" --backend URL --groups FILE      # what would be sent
llama-mcp-router tools  --backend URL [--json]                   # tools + schema size
```

Every option also reads an environment variable `LLAMA_ROUTER_<NAME>` (e.g. `LLAMA_ROUTER_BACKEND`, `LLAMA_ROUTER_LAYA_URL`).
`POST /router/select {"query": "..."}` returns the selection over HTTP; `GET /router/health` shows the config.

## Compatibility

* Python 3.9 – 3.13 (CI matrix). `starlette >= 0.27`, `httpx >= 0.24`.
* llama.cpp: any build whose `llama-server` serves `/v1/chat/completions`. The MCP integration (`GET /tools`, `POST /tools`,
  `--mcp-servers-config`) needs a recent build; on older builds the router still routes the tools your *client* sends, or a JSON file
  (`llama-mcp-router select --tools-file tools.json`). Tested with PrismML's llama.cpp fork `prism-b10743` (upstream-based).
* Tool-calling must work in the model's chat template (`--jinja`).

## Benchmark

Question asked: *if llama-server only receives the tools a request needs, how much prompt-reading time do we save, and what does it cost in tool-choice accuracy?*

**Setup** — model: Ternary-Bonsai-2-27B (a 27B reasoning model, PQ2_0, MTP) on one RTX 4090 via PrismML's llama.cpp fork `prism-b10743`
(`--jinja`, 131k context, q8_0 KV cache, `reasoning_effort=medium`, temperature 0). Tools: the 41 tools of
[pubmed-search-mcp](https://github.com/u9401066/pubmed-search-mcp) (137 KB of schemas ≈ 30.6k tokens); the "133-tool pool" adds 92 synthetic
tools from other domains (files, GitHub, calendar, mail, Kubernetes, …, see `benchmarks/make_distractors.py`). Laya: `laya-serve` 0.3.22 on the same GPU, zero-shot, no fine-tuning.
Queries: 52 **held-out** requests (EN + 繁體中文, one or two per tool, written before any selector was run on them); the 83 "dev" queries in `data/queries.jsonl`
were used to pick the defaults (`top_p=0.95`, `max_groups=4`, BM25 `top_k=3` inside the union). A query is *correct* if the model's first tool call is an acceptable tool.
Everything is reproducible with `benchmarks/run_bench.py`; raw reports are in [benchmarks/RESULTS.md](benchmarks/RESULTS.md).

### Prefill: what the router saves (26 held-out queries, nothing cached)

| pool | selector | tools sent | prompt tokens | selector | **cold prefill** | speed-up |
|---|---|---|---|---|---|---|
| 41 | all tools (llama.cpp default) | 41 | 30,635 | – | 11,359 ms | 1.0× |
| 41 | `bm25` (k=5) | 4 | 4,880 | 0 ms | 1,807 ms | 6.3× |
| 41 | **`laya+bm25`** | 10 | 8,267 | 12 ms | **2,933 ms** | **3.9×** |
| 41 | oracle (the right group) | 3 | 3,089 | – | 1,192 ms | 9.5× |
| 133 | all tools | 133 | 37,805 | – | 14,469 ms | 1.0× |
| 133 | **`laya+bm25`** | 11 | 7,011 | 12 ms | **2,523 ms** | **5.7×** |

Selecting costs ~12 ms (a Laya call). The savings grow with the pool size because the selection does not.

### Accuracy: does a better-informed prompt help the 27B model? No — it already copes (52 held-out queries)

| pool | selector | tool offered (recall) | correct first call | EN | 中文 |
|---|---|---|---|---|---|
| 41 | all tools | 100% | **96.2%** | 93.8% | 100% |
| 41 | `bm25` | 75.0% | 75.0% | 90.6% | 50.0% |
| 41 | `laya` (groups only) | 75.0% | 73.1% | 75.0% | 70.0% |
| 41 | **`laya+bm25`** | 96.2% | **92.3%** | 93.8% | 90.0% |
| 41 | oracle | 100% | 96.2% | 96.9% | 95.0% |
| 133 | all tools | 100% | **96.2%** | 96.9% | 95.0% |
| 133 | `bm25` | 69.2% | 69.2% | 84.4% | 45.0% |
| 133 | `laya+bm25` | 80.8% | 78.8% | 90.6% | 60.0% |
| 133 | oracle | 100% | 98.1% | 96.9% | 100% |

What this says:

* **Laya cannot make this 27B model choose better** — with every tool in the prompt it was already right 96% of the time, even among 133 tools,
  and the oracle (perfect selection) only reaches the same ceiling. A selector can only *lose* accuracy relative to that, never add it.
* **Selector recall is the whole game.** The 4 points lost at 41 tools (92.3 vs 96.2) are queries where the right tool was not offered.
  At 133 tools Laya's group classification got noticeably worse (recall 96% → 81%), so the router lost 17 points. Zero-shot Laya alone
  is only about as good as BM25 (75%), but they fail on *different* queries: BM25 is strong on English with tool-specific words and collapses on 中文 (50%),
  Laya is language-agnostic. The union fixes most of both.
* **Laya is a poor judge of fine-grained, overlapping groups.** Typical misses: "Search the NCBI Gene database for BRCA1" → the generic *search* group;
  "Search PubChem for …" → *search*. Per-tool classification without groups was far worse (17–22% recall), so write groups.
  Fine-tuning Laya on your tool-routing data (it supports that) is the obvious next step and is **not** covered here.

### When the router wins, and when it loses

A 5-turn conversation that hops between tool groups (`benchmarks/multiturn_prefill.py`, one run):

| | turn 1 | turns 2–5 | total prefill |
|---|---|---|---|
| all tools | 11.2 s | 0.4 s each (30k tokens reused from the prompt cache) | **12.7 s** |
| router (default) | 2.0 s | 0.8–4.0 s each (the tool list changes → nothing reusable) | **10.1 s** |
| router `--sticky` | 2.1 s | 1.2–6.2 s (list grows; one eviction re-read 17k tokens) | 14.3 s |

llama-server caches the prompt prefix, and the tool schemas sit at the very *start* of the prompt, so changing the tool list invalidates the cache for
the whole conversation. Therefore:

* ✅ **Router wins** when prefill is mostly *cold*: short chats or one-shot calls, many users/clients sharing few slots, a small `--cache-ram`
  (a cached prompt entry was ~0.6–0.8 GiB in my runs, so a 1 GiB cache holds about one), slower GPUs/CPUs/Macs, or when 30k tokens of schemas is eating your context window.
* ❌ **Plain llama-server wins** for a single user in one long, warm, on-topic conversation: after the first 11 s every turn costs ~0.4 s.
  Break-even in this test was roughly 6 turns of topic-hopping.
* `--sticky` keeps the list append-only to protect the cache, but with topic changes it grows toward "all tools" and is worse (14.3 s); it only pays off for on-topic chats. It is off by default.

### Limitations of this benchmark

One model, one hardware setup, one MCP server (plus synthetic distractors), one run per cell at temperature 0, and 52 queries — differences of a few points are noise
(2 queries = 3.8 points). The queries were written by the maintainer, in the style of the tool descriptions. Laya was used zero-shot. Prefill numbers are
llama-server's own `timings.prompt_ms` at max_tokens=1 with `cache_prompt=false`; "warm" numbers assume an identical prompt prefix. Please run `benchmarks/run_bench.py` on your own tools.

## Development

```bash
git clone https://github.com/u9401066/llama-mcp-router && cd llama-mcp-router
pip install -e ".[test]" && pytest
python benchmarks/run_bench.py                      # selector recall only (needs a laya-serve on :8000)
python benchmarks/run_bench.py --llm http://127.0.0.1:8080 --queries benchmarks/data/queries_heldout.jsonl            # accuracy
python benchmarks/run_bench.py --llm http://127.0.0.1:8080 --prefill --distractors --limit 26 --only all,laya-groups+bm25   # prefill, 133-tool pool
```

## Credits and license

MIT. Not affiliated with llama.cpp, Laya / Convai Innovations, or pubmed-search-mcp. `benchmarks/data/pubmed_tools.json` is a snapshot of the tool
definitions of [pubmed-search-mcp](https://github.com/u9401066/pubmed-search-mcp) (Apache-2.0) used as realistic benchmark input.
