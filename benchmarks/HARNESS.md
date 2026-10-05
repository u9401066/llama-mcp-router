# How to ask Laya: input-harness experiments

Laya answers *typed questions about a state*. For tool routing the state is the user's request and the question is "which tool group?".
Nothing in Laya fixes how that question is worded, so this is a harness-design problem: **what Laya sees can matter more than which checkpoint answers.**
All numbers below come from `benchmarks/laya_lab.py` (zero-shot `laya-serve` 0.3.22, the 14 PubMed tool groups).

**Method.** Tuning set = 135 tool requests (EN + 繁體中文) used while exploring. Metrics are budget-independent: *top-k* = the right group is within Laya's first *k*
ranked groups; *MRR* = mean reciprocal rank. A fresh set of 68 queries (`queries_test.jsonl`, 56 tool requests + 12 "no tool needed") was written **before** any tuning and only used in the final end-to-end runs.
Single run per cell, deterministic model; with 135 queries one query is 0.74 points, so treat differences under ~3 points as noise.

## 1. Findings that held up

| lever | top-1 | top-2 | top-3 | MRR | verdict |
|---|---|---|---|---|---|
| baseline: bare string, long descriptions, auto checkpoint | 52.6 | 64.4 | 74.8 | 0.664 | |
| wrap as `{"request": q}` + instruction names the field | 60.7 | 73.3 | 81.5 | 0.730 | **+8 top-1**. Cheapest win. `"User request: " + q` is equally good (60.7) |
| option text built from the tools' own first sentences (`labels=auto`) | 63.0 | 74.8 | 81.5 | 0.743 | **+10 top-1**, needs no hand-written text |
| short hand-written labels + `json` state | 64.4 | 74.8 | 80.0 | 0.749 | best single view |
| force `multilingual` checkpoint | 57.8 | 72.6 | 82.2 | 0.712 | +5 top-1; mostly fixes non-English |
| force `english` / `typed-decisions` | 45.9 / 34.8 | | | 0.616 / 0.561 | **worse** (typed-decisions is fine-tuned for four other workflows) |
| 3-view ensemble (json+short, raw+auto, raw+desc on multilingual) | 67.4 | 80.7 | 84.4 | 0.782 | **best**, ≈ +3 top-1 / +6 top-2 over the best single view, ~30 ms instead of ~12 |

At the policy level (union with BM25, tuning set): single-view top-2 + BM25 = 87.4% recall with 7.0 tools; ensemble top-2 + BM25 = **92.6% with 7.6 tools**; single-view top-3 + BM25 = 91.9% with 9.0 tools.

## 2. The hidden budget: `head_max_len`

Laya packs the question and *all* options into a head of `head_max_len` tokens (default 192). With 14 options each gets ≈ 11 tokens; the rest is cut off **silently**
(`usage.truncated_questions` stays empty). Consequences measured here:

* Long descriptions are mostly invisible: "both" / "examples" criteria gave **identical** results to the short description (54.8 → 54.8) because the extra text never reached the model.
* Putting the discriminative words **first** is what matters; hence short labels and `labels=auto` (a tool's first sentence) work.
* **Raising `head_max_len` made it worse** (top-1 52.6 → 45.2 at 384, 35–45 at 768) – the checkpoint was trained with the small head. Do not "fix" truncation by enlarging the budget.
* Pools with many options need a shortlist (Laya's own README says >20 options degrade); group your tools so there are ≲15 groups.

## 3. Things that did not help

| idea | result |
|---|---|
| per-group yes/no (`noul`) instead of one `choice` | 48.9 top-1 (worse; Laya's README warns noul follows its labels) |
| average `choice` and `noul` | 54.8 top-1 (cost 2×, ≈ no gain) |
| 3 random option permutations | 58.5 top-1, 3× the calls – the ensemble of *different framings* is better and cheaper in effect |
| two-stage (4 super-groups → groups) | 43.7 top-1 (errors multiply) |
| masking PMIDs/rsIDs/CIDs in the request | 54.1 (+1.5, noise) |
| appending detected-entity hints (`{"contains": ["PMID"]}`) | 62.2 top-1 but not above plain json (60.7 → 62.2 within noise) and worse after ensembling |
| unsupervised prior correction (divide by per-group mean probability) | 64.4 → 60.0 (worse) |
| lengthening options with examples, "short + auto" concatenations | 46.7 – 58.5 (worse or equal; truncated) |

## 4. A "no tool needed" option

Adding `none: "small talk, general knowledge, explanation, writing, translation, maths"` as an extra option lets Laya veto tool use:
with τ = 0.7 on the ensemble, 13 of 20 tuning-set small-talk requests got **zero** tools (6.8 → 2.0 tools on average) at the price of **−2.2 points recall** on real tool requests (92.6 → 90.4).
On the fresh test set it was worse end-to-end (see the README table): the 27B model with the tools present already declines to call a tool for most chit-chat, so the saving is mostly prompt size, not correctness. It is opt-in (`--laya-none 0.7`).

## 5. Where the confusions are

Most errors are *semantic overlap between groups*, not wording: "Search the NCBI Gene database for BRCA1" → generic *search*; "PubChem record of …" → *search*; "full text of PMC…" → *images*/*compound*; "save a pipeline" → *chronicle*/*icd*.
Zero-shot Laya has no way to know that in *this* MCP "search_gene" belongs to *gene*. Whatever the harness, ~35% of requests have the right group rank first; ~67% with the ensemble.
That is the ceiling for a zero-shot Laya on this tool set. Fine-tuning Laya on routing data (its README: "fine-tuning is where most of the value is") is the real next step; it is **not done here**.

## 6. Postscript (v0.4): the miss is better fixed outside Laya

The ~10% of requests whose group Laya ranks too low are recovered by the router's catalog meta-tool (`--escalate`): the model sees the names of the tools that were left out and loads them. End-to-end on 110 queries / 61 tools: Laya + catalog 94.5% vs Laya alone 89.1%. A catalog with *no* Laya at all scored 96.4% with the 27B model, so for strong models the harness work here matters less than giving the model a cheap way to ask. See the README.

## 7. Postscript (v0.5): 600+ tools

At 639 tools from 19 servers Laya can no longer see the pool at all (hard limit: 100 options per question; effective resolution
~11 tokens per option). The harness lessons that carried over: short, front-loaded option labels (tool name + first sentence beat
first sentence alone and server-prefixed names); chunks of ≤ 12 options; `{"request": …}` framing. What did not carry over: ensembles and the
multilingual checkpoint did not help as a reranker. End to end, a strong multilingual retriever (bge-m3) mattered far more than any Laya
harness, and Laya reranking did not beat retrieval once the model could search for missing tools. See the README section "v0.5" and
`benchmarks/scale/RESULTS.md`.

## 8. Reproduce

```bash
python benchmarks/laya_lab.py --set tune --variants "baseline,state=json+crit=short,state=raw+crit=auto,model_multi"
python benchmarks/laya_lab.py --set tune --variants "ens=state=json/crit=short|state=raw/crit=auto|state=raw/crit=desc/model=multilingual"
python benchmarks/laya_lab.py --set tune --variants "state=raw+crit=desc+head=384"          # the head budget experiment
```
