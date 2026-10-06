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

> **Best measured setup (v0.4): `--escalate`.** The router adds a small *catalog* meta-tool listing every tool it did not send
> (≈ 1.5k tokens for 61 tools). If the model needs one of them it calls the catalog, and the router transparently reruns with those tools.
> On 110 queries over 61 PubMed + Zotero tools with a 27B model: **all tools 91.8%** (36.5k prompt tokens) → **Laya + catalog 94.5%**
> (6.7k) → **catalog only, no Laya at all: 96.4%** (3.1k tokens, fastest). Without the catalog the router *lost* accuracy (89.1%).
> So for a model of this size the catalog is what matters and Laya is optional; see [the v0.4 section](#v04-the-catalog-meta-tool---escalate).
>
> **Hundreds of tools / many MCP servers (v0.5): `--selector retrieve --embed-url … --escalate`.** With 639 real tools from 19 MCP
> servers (247k tokens of schema, more than the model's context) a bge-m3 retriever picks 8 tools and the meta-tool becomes a search;
> the 27B model chose the right tool **82%** of the time. Laya cannot read such a pool directly (see [600+ tools](#v05-600-tools-from-19-mcp-servers)).

## Install

```bash
pip install git+https://github.com/u9401066/llama-mcp-router      # or: pipx install git+https://github.com/u9401066/llama-mcp-router
```

(Not on PyPI yet.)

## Quick start

1. Run `llama-server` with your MCP servers ([examples/llama-server-mcp.json](examples/llama-server-mcp.json)).
   Point `command` at an **installed** executable (e.g. `uv tool install pubmed-search-mcp`, then `~/.local/bin/pubmed-search-mcp`),
   not at `uvx …`: llama-server lists MCP tools only during a ~10 s warmup at startup, and a slow `uvx` start (e.g. at boot, while
   the model loads) silently leaves it with **0 MCP tools**. Check with `curl -s localhost:8080/tools`.

   ```bash
   llama-server -m model.gguf --jinja --port 8080 --mcp-servers-config mcp.json
   ```

2. Run a Laya server (optional but recommended for non-English requests): `pip install "laya[serve]" && laya-serve` (port 8000).

3. Tell the router what your tool groups are ([examples/pubmed_groups.json](examples/pubmed_groups.json) is a complete example):

   ```bash
   llama-mcp-router serve --backend http://127.0.0.1:8080 --port 8090 --escalate \
       --groups examples/pubmed_groups.json --selector laya+bm25 --laya-url http://127.0.0.1:8000
   # or, without Laya (measured best for a 27B model):
   llama-mcp-router serve --backend http://127.0.0.1:8080 --port 8090 --selector none --escalate
   ```

4. Point your client at the router instead of llama-server: `http://127.0.0.1:8090/v1` for API clients, or open `http://127.0.0.1:8090/` for llama-server's own Web UI (every non-chat path is proxied, so the Web UI lists and runs MCP tools as usual).

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

### What is sent: `--apply` and `--hint`

| flag | behaviour |
|---|---|
| `--apply select` (default) | send only the selected tools |
| `--apply reorder` | send **all** tools, most relevant first (no recall loss, no prefill saving) |
| `--apply all --hint` | send all tools unchanged and add a one-line routing hint to the last user message |
| `--hint` (with `select`) | selection + hint |
| `--escalate` | also send a `router_load_tools` meta-tool listing the tools *not* sent (name + one line); above `--catalog-max` left-out tools (default 80) it instead takes a free-text `query` and the router searches the pool. If the model calls it, the router reruns the request with the tools it asked for (once, `max_escalations=1`); the client never sees the meta-tool. Works with streaming (text/reasoning stream immediately, tool-call chunks are held until the round ends) and in agent mode. |

The hint is appended to the *last user message*, i.e. late in the prompt, so llama-server's cached prefix is not disturbed. See [benchmarks/HARNESS.md](benchmarks/HARNESS.md) for what each is worth.

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
| `none` | no tools; with `--escalate` = **catalog mode** (the model loads tools by name) | – |
| `bm25` | lexical top-k over tool names + descriptions; CJK-aware tokeniser | – |
| `laya` | Laya `choice` over tool groups; keeps the top `--max-groups` (default 2) groups (`--top-p` < 1 keeps fewer once that much probability mass is covered; fixed k measured better than adaptive cut-offs). `--laya-none 0.7` adds a *no tool needed* option (sends no tools for small talk; costs ~2 points recall, see HARNESS.md). By default it averages **three differently-framed questions** (`--laya-views ensemble`, ~30 ms); `--laya-views single --laya-state json\|raw --laya-labels label\|description\|auto --laya-model multilingual` pick one framing (see [benchmarks/HARNESS.md](benchmarks/HARNESS.md)) | a `laya-serve` instance |
| `laya+bm25` | union (default) | both |
| `retrieve` | top-k by BM25 + embeddings (`--embed-url`, any OpenAI-compatible `/v1/embeddings`), fused; embeddings only for CJK requests | an embedding endpoint (optional; BM25 alone otherwise) |
| `laya-rerank` | retriever shortlist (`--shortlist 24`) → Laya ranks it in chunks of ≤ 12 → Laya's `--keep` + retriever's `--also` | embedding endpoint + `laya-serve` |

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
llama-mcp-router serve  --backend URL --port 8090 --selector laya+bm25|none --groups FILE [--escalate] [--agent-config FILE] [--mode inject|agent] [--apply select|reorder|all] [--max-tools 12] [--always a,b] [--exclude 'fs_*']
llama-mcp-router select "query" --backend URL --groups FILE      # what would be sent
llama-mcp-router tools  --backend URL [--json]                   # tools + schema size
llama-mcp-router install-dsh-plugin DSH_DIR                     # copy the DSH tool-routing plugin into a dsh install
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

## v0.2 / v0.3 update: does a better Laya input harness help? (61 tools, 110 queries)

Details of the input experiments are in [benchmarks/HARNESS.md](benchmarks/HARNESS.md). Here is the end-to-end result.
Pool: 41 PubMed + 20 Zotero Keeper tools (their verbs overlap: *search*, *list*, *save*, *import*). Queries: 110 (98 tool requests, 12 chit-chat), EN + 中文, none used to tune anything.
Same 27B model and settings as above; one run per cell.

| what is sent | correct first call | tool requests | chit-chat (no tool call) | 中文 | prompt tokens | s / query |
|---|---|---|---|---|---|---|
| all 61 tools (llama.cpp default) | 91.8% | 90.8% | 100% | 96.8% | 36,481 | 6.9 |
| router v0.1 (bare string, long labels, top-p) + BM25 | 83.6% | 81.6% | 100% | 74.2% | 6,769 | 9.2 |
| **router v0.2** (json framing, short labels, 3-view ensemble, top-2) + BM25 | **89.1%** | 87.8% | 100% | 77.4% | **4,912** | 7.3 |
| v0.2 with top-3 groups | 90.0% | 88.8% | 100% | 83.9% | 6,472 | 8.8 |
| v0.2 selection **+ routing hint** in the user message | 89.1% | 87.8% | 100% | 77.4% | 4,980 | 8.2 |
| all 61 tools **+ routing hint** | 92.7% | 93.9% | 83.3% | 93.5% | 36,549 | 7.5 |
| all 61 tools **reordered**, most relevant first | **94.5%** | 93.9% | 100% | 96.8% | 36,481 | 44.6 |
| *oracle: perfect group only* | 98.2% | | | | | 3.6 |
| *oracle hint + all 61 tools* | 93.6% | | | | | |

On the 68-query set of the previous section (41 tools only) v0.2 was *not* better: 88.2% vs 92.6% for v0.1 and 94.1% for all tools, because v0.2 sends fewer tools (7.7 vs 10.4) and its recall is 92.9% vs 94.6%.
One query is 0.9 points, and running the same prompt twice gave 101 vs 102 correct, so only differences of ≈ 3 points or more mean anything.

* **A better-framed question gives a better ranking, not a free lunch.** Top-2 group recall on the tuning set rose from 64% to 81%. End-to-end that was +5 points over v0.1 on this pool and −4 on the other: at *equal tools sent* v0.2 is better, but v0.1 compensated with more tools.
* **The router still cannot beat "all tools" for this model.** Best router configuration: 90.0% (top-3) vs 91.8%. What you buy is prompt size (−85%) and cold prefill: **14.1 s → 2.0 s (7.1×, 20 sampled queries; v0.1: 2.6 s)**; selection costs ~32 ms (3 Laya calls) vs ~12 ms.
* **A routing hint is worth ≈ nothing for a 27B model.** Even a *perfect* hint (oracle) gave 103/110 vs 101–102 without; the router's own hint gave 102/110 and made two chit-chat requests call a tool (100% → 83.3%). Not recommended; it stays as an option for smaller models, which I did not test.
* **Reordering helped (+2.7 points, 104/110) but cannot be used for speed:** the tool order changes with every request, so llama-server's prompt cache never hits and each request re-reads 37k tokens (44.6 s/query with two parallel clients). It is only an accuracy option.
* **Non-English is the weak spot of every selector** (中文 74–84% vs 97% with all tools): zero-shot Laya and BM25 both do worse on Chinese than a 27B model reading all the tool descriptions.
* **The `none` option (`--laya-none`) lost accuracy** on the 41-tool test (88.2% → 82.4%); do not enable it unless you measure it on your workload.

**What is left on the table:** fine-tuning Laya on routing data (its README claims this is where most of the value is) – not attempted – and testing smaller/weaker models, where a hint or a short list should matter more than for a 27B.

## v0.4: the catalog meta-tool (`--escalate`)

Every miss of the selector above was the same failure: the right tool was not in the prompt, so the model called the closest one.
`--escalate` gives the model a way out: a compact catalog of the tools that were left out. Same 61-tool pool, 110 queries, model and settings as above:

| configuration | correct first call | tool requests | 中文 | chit-chat | prompt tokens (avg, all rounds) | s / query | needed a 2nd round |
|---|---|---|---|---|---|---|---|
| all 61 tools (llama.cpp default) | 91.8% | 90.8% | 96.8% | 100% | 36,481 | 6.9 | – |
| Laya v0.2 + BM25 (`--selector laya+bm25`) | 89.1% | 87.8% | 77.4% | 100% | 4,832 | 7.0 | – |
| **Laya v0.2 + BM25 + catalog** (`--selector laya+bm25 --escalate`) | **94.5%** | 93.9% | 93.5% | 100% | 6,707 | 8.5 | 8 / 110 |
| **catalog only** (`--selector none --escalate`) | **96.4%** | 95.9% | **100%** | 100% | **3,076** | **6.5** | 98 / 110 |

The same 30 queries (evenly sampled) at `reasoning_effort=xhigh`: all tools 25/30 (8.8 s), Laya + catalog 25/30 (9.4 s), **catalog only 26/30 (7.2 s)**.
At `medium` the same 30 were 27, 28 and 30/30 – for tool choice this model did better at medium than at xhigh in every configuration.

What it means:

* **The catalog fixes the router's accuracy problem.** Laya + catalog went from 89.1% to 94.5%, above sending all tools, with 82% fewer prompt tokens. The model asked for more tools in only 8 of 110 requests.
* **For a 27B model, Laya is not needed.** With nothing but the catalog the model chose best (96.4%, every Chinese query right) and was fastest, even though every tool request costs a second generation round (~+150 completion tokens).
  It is fast because the catalog is *identical for every request*, so llama-server's prompt cache always hits on it, and the contexts stay small. Laya's varying tool lists defeat that cache.
* **Where Laya should still pay off** (not measured here): models that are too weak to do this two-step tool search reliably, slow generation where a second round is expensive, or a fine-tuned Laya.
* Differences of 1–2 queries are noise (104 vs 106 is two queries). The 30-query xhigh sample is small.

Recommended: `llama-mcp-router serve --selector none --escalate` (no Laya needed), or `--selector laya+bm25 --escalate` if you want one round for most requests.

## v0.5: 600+ tools from 19 MCP servers

The question: with 10+ MCP servers and hundreds of tools, sending everything is impossible. How should Laya be used then?
Pool: **639 real tools** from 19 MCP servers (PubMed, Zotero, data analysis, statistics, pharmacy, FHIR, symbolic maths, LibreOffice,
REAPER, …; `benchmarks/scale/`), 992 KB of schema ≈ **247k tokens** (> the 131k context). 168 queries (44 in Chinese). Only the
servers' `tools/list` output is needed; nothing was executed. Full numbers: [benchmarks/scale/RESULTS.md](benchmarks/scale/RESULTS.md).

**1. Laya cannot be the first stage.** laya-serve rejects more than 100 options per question, and with its 192-token head each option is
read through ~11 tokens at 14 options, ~2 at 100. Asking Laya to pick 3 of 19 *servers* first lost recall (64% vs 75% for plain retrieval).
Something else has to shortlist; Laya can only rerank.

**2. The retriever decides almost everything.** Selector recall at ~8–10 tools:

| retriever | recall | 中文 |
|---|---|---|
| BM25 (built-in, no service) | 61.5% | 22.0% |
| multilingual-e5-small + BM25 | 75.0% | 48.8% |
| bge-m3 + BM25 | 82.1% | 63.4% |
| bge-m3 + BM25, embeddings only for CJK requests (shipped default) | 83.3% | 68.3% |

**3. Laya as a reranker helps recall per tool, but not the final answer.** Reranking a 24-tool bge-m3 shortlist (tool name + first sentence
as option labels, ~100 ms) reached 86.5–87.8% recall with 8 tools vs 82.7% for retrieval alone with 8 (retrieval needs ~20 tools for the same recall).
Other harness variants (3-view ensembles, the multilingual checkpoint, server-prefixed labels, rank fusion) did not beat that.
End to end, however, with the router's search escalation:

| router (client sends all 639 tools) | correct first call | 中文 | tools sent | needed a search round |
|---|---|---|---|---|
| **`retrieve` top-8 + search escalation** | **82.1%** | 72.7% | 8.0 | 23/168 |
| `laya-rerank` (keep 5 + also 5) + search escalation | 81.0% | 75.0% | 8.1 | 23/168 |
| `laya-rerank` (keep 5 + also 3) + search escalation | 78.6% | 72.7% | 6.4 | 26/168 |
| no tools, model searches itself (`none --escalate`) | 71.4% | 68.2% | 0 | 140/168 |

Laya's extra recall is offset by the plausible-but-wrong tools it puts in front of the model; 136 vs 138 correct is within noise.
And unlike at 61 tools (where a catalog alone was best, 96.4%), letting the model search from nothing is clearly worse at 639: the
retriever working on the user's own words is a better first guess than the model's search query.

**4. Real MCP schemas break llama.cpp.** 3 of the 639 tools were rejected by llama-server (HTTP 400: a `$defs` block nested inside a property
but referenced from the root; `maxLength: 2000` unrolled into an unparsable grammar), which failed 7 of 168 requests. The router now
sanitises schemas before forwarding (inlines local `$ref`s, drops length limits > 64; `--no-sanitize` to disable): 0 failures.

**5. Where Laya can still add value at scale** (not shipped, measured only partly):
* its confidence is informative (top-1 right 76% when p ≥ 0.9, 7% when 0.5–0.7) → send fewer tools when it is sure, more when it is not; needs calibrated temperatures
* fine-tuning Laya on routing data: the router sees which tool the LLM finally called for each request, i.e. free labels
* a cheap "no tool needed" gate before retrieval (tested at 41 tools in v0.2.2: saves prompt, cost 2 points)

**Recommended for many MCP servers:** run an embedding model (e.g. `llama-server -m bge-m3-Q8_0.gguf --embedding --pooling cls` on CPU, ~50 ms/query) and
`llama-mcp-router serve --selector retrieve --embed-url http://127.0.0.1:8082 --escalate`. `--selector laya-rerank` is available for experiments.

## v0.6: an existing agent per chat session (ACP agent bridge)

The router can host a full coding agent behind the ordinary chat API, so llama-server's own Web UI (or any OpenAI client) gets
an agent with **its own sandboxed, git-versioned workspace per conversation**, **skills**, and a history that is **not bounded by the
chat client's context window** (only the newest message is sent; the agent keeps and compacts its own history).

It speaks the [Agent Client Protocol](https://github.com/agentclientprotocol/agent-client-protocol) (ACP v1, JSON-RPC over stdio), so any
of the [~35 ACP agents](https://github.com/agentclientprotocol/registry) can be plugged in by configuration. Tested here with
**DeepSeek Harness** (`dsh --profile acp`, the default example) and **Qwen Code** (`qwen --acp`), both driving a local 27B model on llama-server.
The agent is a separate process, so its language (TypeScript, Rust, Go …) does not matter.

```
Web UI ── "/agent write a script that …" ──► llama-mcp-router ──ACP──► agent process (per session, in bubblewrap)
                                               │  stream: thoughts + tool steps → reasoning, answer → content      │
                                               │  after each turn: git commit, links to changed files               └─► llama-server (model)
```

Start a conversation with `agent:` (use this one in llama-server's Web UI, where a leading `/` goes to its command picker and `@` to its
file-mention picker), `/agent` or `@agent` from other clients, or request `model: "agent"`, which `/v1/models` also lists:

```bash
llama-mcp-router serve --backend http://127.0.0.1:8081 --port 8001 --agent-config examples/agent-dsh.json
```

Per conversation the bridge creates `<root>/<session-id>/` with `workspace/` (a git repo, the agent's cwd; skills from `skills_dirs`
are copied to `skills_target`), `home/` (the agent's private state, e.g. `DSH_HOME` / `QWEN_HOME`) and `agent.stderr.log`; it starts one
agent process (stopped after `idle_ttl`, resumed with `session/resume` / `session/load` later), and commits after every turn. Every answer
ends with links to the changed files; `GET /agent/sessions/<id>` lists files and history, `…/files/<path>` serves a file, `…/archive.zip` the workspace.

**Sandbox** (`"sandbox": "bwrap"`, agent-independent): the whole agent process runs under bubblewrap: the host filesystem is read-only,
`sandbox_hide` paths (default: your home, `/run` with the Docker socket, `/tmp`) are replaced by empty tmpfs, `sandbox_ro` paths (the agent's
runtime) are mounted back read-only, and only the session directory is writable. Agents' own sandboxes still apply inside (DSH's bash sandbox
nests fine; Qwen Code's refuses ACP mode, so the outer one is what isolates it). Permission requests from the agent (e.g. sandbox escalation)
are **denied** unless `"permissions": "allow"`. Verified: the agent cannot read `~/.ssh` or reach the Docker socket but can write its workspace.
Network is shared (the agent must reach llama-server); set `sandbox_network: false` for agents that do not need it.

**Skills.** [examples/skills](examples/skills) has three small skills aimed at a 27B model's weak spots: `work-in-files` (write results to files,
answer briefly), `session-notes` (keep `NOTES.md` as working memory across compaction), `deliverables` (edit deliverables in place, git keeps
versions, log them in `outputs/CHANGELOG.md`). In the tests the model loaded them on its own and followed them.

**Security.** Enabling the bridge lets anyone who can reach the router run (sandboxed) code and read the session files; session ids are
random 128-bit, but keep the router on a trusted network and/or give each person an API key (`users`, v0.8).

Configuration reference: `AgentConfig` in [agent.py](src/llama_mcp_router/agent.py); examples: [agent-dsh.json](examples/agent-dsh.json),
[agent-qwen-code.json](examples/agent-qwen-code.json).

## v0.7: one stack — llama.cpp + DeepSeek Harness + the router as a DSH plugin

```
Web UI / OpenAI client ──► llama-mcp-router :8001 ──ACP──► DeepSeek Harness (per chat, in bubblewrap)
                              │ default: every chat           │  skills, git workspace, compaction, sub-agents
                              │ "chat:" → plain llama.cpp      │  MCP servers (dsh-mcp-client, e.g. PubMed)
                              │                                │  plugin "llama-mcp-router": per turn, POST /router/select
                              │◄──────── /router/select ───────┘  (Laya+BM25) → ctx.tools.restrict(); `find_tools` to load more
                              └──► llama-server :8081 (model) ◄── DSH's model requests
```

* **`"default": true`** in the agent config sends every chat to the agent; a first message starting with `chat:` (configurable `optout`)
  goes to plain llama.cpp instead (fast Q&A; the prefix is stripped).
* **MCP lives in the agent.** Give DSH its MCP servers with `@deepseek-ai/dsh-mcp-client` entries in the per-session patch (see
  [examples/agent-dsh.json](examples/agent-dsh.json)); their tools appear as `mcp__<server>__<tool>`.
* **The tool router is a DSH plugin** ([integrations/dsh-plugin.mjs](src/llama_mcp_router/integrations/dsh-plugin.mjs), shipped in the
  wheel): `llama-mcp-router install-dsh-plugin ~/agent-runtimes/dsh` copies it into a DSH install; the patch mounts it. It asks
  the router which MCP tools fit the user's message and leaves the rest out of the agent's requests (since v0.9 directly in DSH's
  `system-prompt/assemble`, see below); DSH's own tools are not routed; `find_tools(query)` loads more; it fails open if the router is down. The selection logic (Laya groups, BM25, retriever, sanitizing)
  stays in one place, the router, and also serves plain chats and other harnesses.
* **Sandboxed MCP servers need their runtimes mounted**: e.g. a `uv tool install`ed server needs both its tool environment and
  `~/.local/share/uv/python` (the venv's interpreter is a symlink through uv's version-alias directory) in `sandbox_ro`.

Measured on the live stack (Bonsai 27B, 41 PubMed tools): "find 3 RCTs on remimazolam for ICU sedation and save them as CSV" → the plugin showed
9 of 41 PubMed tools, the agent searched via MCP and wrote the CSV; the follow-up "add the first paper's citation count" showed 8 of 41 (incl. the
citation tools) and was answered in 81 s with iCite data, editing the same file. With the MCP server broken (before the `uv/python` mount fix)
the agent fell back to PubMed's public E-utilities via web fetch: correct but 3–5× slower; the plugin now logs when no MCP tools appear.

## v0.8: one agent and sandbox per person and per conversation

Up to v0.7 an agent session was keyed by a hash of the system prompt and the first user message, so two people (or two chats) that both
opened with "hi" shared one workspace. Now:

* **Per conversation.** llama-server's Web UI sends `X-Conversation-Id: <conversation>::<model>` with every chat request (it uses it for
  resumable streams); the bridge keys agent sessions by that conversation id, so every Web UI chat gets its own agent, workspace and
  git history, whatever its first message. Clients without the header fall back to the first-message hash.
* **Per person** (optional). Give the agent config `"users": {"<api key>": "<name>"}` and/or `"users_file": "users.json"` (same mapping, keep it
  out of git). Then agent chats need `Authorization: Bearer <key>` — in the Web UI, *Settings → API Key* — and get `401` without a valid key;
  sessions are keyed by (user, conversation) and stored in `<root>/<user>/<session-id>/`, and `GET /agent/sessions` lists the caller's own
  sessions (id, title, turns). Plain `chat:` requests are not gated by these keys (llama-server ignores the header unless it has `--api-key`).
  File links stay unguessable capability URLs (`/agent/sessions/<random 128-bit id>/files/…`), so they open in a browser without a header.
* **`max_processes`** (default 4) caps live agent processes: a new session stops the least-recently-used idle agent first (its workspace stays;
  it resumes on the next message); if every slot is busy the request gets an error instead of exhausting RAM/VRAM. Size it to your memory:
  one DSH session with the PubMed MCP server measured ~290 MB RSS (DSH 184 MB + MCP 109 MB); all of them share the one llama-server, whose `--parallel` slots bound concurrent generation.

## v0.9: where an agent turn's time goes, and what fixed it

Measured on the live stack (RTX 4090, Bonsai 27B = Qwen3.5 *hybrid* architecture: Gated-DeltaNet layers plus attention every 4th layer,
41 PubMed MCP tools, DeepSeek Harness). llama-server prefills ~2,850 tokens/s and generates ~100 tokens/s (MTP speculative decoding), so
**re-processed prompt tokens** are what make a turn slow. A hybrid model can only reuse its cache for a byte-identical prefix (its recurrent
state cannot be rewound to an arbitrary point), and the tool list sits in the prompt *before* the conversation.

| what was wrong | effect | v0.9 |
|---|---|---|
| the plugin hid tools in `agent/pre-step`, after DSH had assembled the step | turn 1 went out without MCP tools; each turn's step 1 used the previous turn's tools, step 2 switched → full re-prefill (8–10 s per turn) | shape `system-prompt/assemble` itself: the selection applies to the request being built |
| MCP servers connect ~1 s after the agent starts | their tools + instructions appeared at step 2 → full re-prefill | the session's first request waits for `servers` and re-assembles |
| PubMed's MCP server instructions | 5,870 tokens in every prompt | `instructions: tool`: summary + one-line-per-tool catalog + `mcp_server_guide` tool |
| DSH tools a 27B model here cannot use (DeepSeek web search without a key, images without a vision projector, workflow/goal/sub-agent control) | ~2,400 tokens | `hide: [...]` |
| re-selecting tools every turn | each change re-processes the whole conversation: 49,367 tokens = 19.8 s for a "write it to paper.md" turn | `policy: session` (default): route once, then only `find_tools` changes the list |

Result on the same three turns ("what are the settings?" → "find a 2024 remimazolam RCT" → "write its PMID to paper.md"): turn 1 prefilled
6.3k + 22.5k tokens before (no MCP tools, then all of them), now 14.5k once (4.8 s); the third turn 25.7 s → 7.3 s (27–76 tokens re-processed per step instead of 49k); the model loaded the search tool
itself with `find_tools` when it needed it. Unchanged prefix ⇒ a new turn costs < 1 s of prefill.

Plugin options (`config:` of the plugin entry, see [examples/agent-dsh.json](examples/agent-dsh.json)): `routerUrl`, `policy`
(`session` | `grow` | `turn`), `servers` + `startupWaitMs`, `always` (MCP tools always visible), `hide`, `instructions`
(`keep` | `tool` | `drop`), `instructionsChars`, `catalogChars`, `searchK`, `timeoutMs`, `log`.

**Laya's "no tool needed" option does not help here**: on agent-style turns ("what are the settings?", "write a Python script", "write it to
paper.md") its probability was 0.05–0.17, below a real search query (0.32), so it cannot gate routing; letting the model ask (`find_tools`)
works better.

**llama-server flags** tried on the same prompt (each a restart): `-ub 256` 2,685 tok/s, `-ub 512` (default) 2,871, `-ub 1024` 2,718;
`--spec-draft-n-max 3` 97.9 tok/s vs 101.5 with 2. The defaults stay.

**Agent turns survive a reload.** llama-server's Web UI resumes streams after a reload (`POST /v1/streams/lookup`, `GET /v1/stream?conv_id=&from=<byte
offset>`) and its Stop button sends `DELETE /v1/stream`. The router now implements these for agent turns: a turn keeps running when the browser
drops the connection and can be reattached; Stop cancels it (`session/cancel`). A cancelled agent start no longer leaves its process behind.

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
