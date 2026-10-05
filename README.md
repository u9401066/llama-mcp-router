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
| `--escalate` | also send a `router_load_tools` meta-tool listing the tools *not* sent (name + one line). If the model calls it, the router reruns the request with the tools it asked for (once, `max_escalations=1`); the client never sees the meta-tool. Works with streaming (text/reasoning stream immediately, tool-call chunks are held until the round ends) and in agent mode. |

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
llama-mcp-router serve  --backend URL --port 8090 --selector laya+bm25|none --groups FILE [--escalate] [--mode inject|agent] [--apply select|reorder|all] [--max-tools 12] [--always a,b] [--exclude 'fs_*']
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
