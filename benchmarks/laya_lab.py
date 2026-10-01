"""Laya input-harness lab: which way of *asking* Laya picks the right tool group?

Budget-independent metrics (they do not depend on a probability cut-off):
  top1/top2/top3 = the gold group is the 1st / within the first 2 / 3 groups ranked by Laya
  MRR            = mean reciprocal rank of the best gold group
  none@1         = (negatives only) Laya ranks the "no tool needed" option first

    python benchmarks/laya_lab.py --set tune --variants baseline,state_json,model_multi
    python benchmarks/laya_lab.py --set test  --variants baseline,best
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import httpx

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))
from llama_mcp_router.selectors import build_groups, load_groups_config  # noqa: E402
from llama_mcp_router.tools import first_sentence, tool_description, tool_name  # noqa: E402

NONE_DESC = "no tool is needed: small talk, general knowledge, explanation, writing, translation or maths that can be answered directly"
INSTR = "Which kind of tool is needed to handle this request?"


class Lab:
    def __init__(self, url: str, tools: List[dict], cfg: Dict[str, Any], extra: Optional[Dict[str, Any]] = None):
        self.client = httpx.AsyncClient(base_url=url, timeout=60)
        self.tools, self.cfg = tools, cfg
        self.groups = build_groups(cfg, tools)
        self.names = [g.name for g in self.groups]
        self.desc = {g.name: g.description for g in self.groups}
        self.members = {g.name: g.tools for g in self.groups}
        self.extra = extra or {}
        self.by_tool = {t: g.name for g in self.groups for t in g.tools}

    def gold(self, expect: Sequence[str]) -> set:
        return {self.by_tool[e] for e in expect if e in self.by_tool}

    async def call(self, state: Any, questions: Dict[str, Any], **kw) -> Dict[str, Any]:
        body = {"state": state, "questions": questions, **kw}
        r = await self.client.post("/v1/systemone", json=body)
        r.raise_for_status()
        return r.json()["answers"]

    async def choice(self, state: Any, criteria: Dict[str, str], instr: str = INSTR, **kw) -> Dict[str, float]:
        a = await self.call(state, {"g": {"type": "choice", "instructions": instr, "criteria": criteria}}, **kw)
        return dict(a["g"]["probabilities"])

    async def noul(self, state: Any, criteria: Dict[str, str], tmpl: str = "Does handling this request require: {d}?", **kw) -> Dict[str, float]:
        qs = {k: {"type": "noul", "instructions": tmpl.format(d=d)} for k, d in criteria.items()}
        a = await self.call(state, qs, **kw)
        s = {k: float(a[k]["noul"]) for k in criteria}
        z = sum(s.values()) or 1.0
        return {k: v / z for k, v in s.items()}


def norm(p: Dict[str, float]) -> Dict[str, float]:
    z = sum(p.values()) or 1.0
    return {k: v / z for k, v in p.items()}


def mean_probs(ps: Sequence[Dict[str, float]]) -> Dict[str, float]:
    keys = ps[0].keys()
    return {k: sum(p.get(k, 0.0) for p in ps) / len(ps) for k in keys}


# ----------------------------------------------------------------------------- state encodings
ENT = [
    (re.compile(r"\bPMC\d+\b", re.I), "<PMCID>"),
    (re.compile(r"\bPMID[:\s]*\d{5,9}\b", re.I), "<PMID>"),
    (re.compile(r"\brs\d{4,}\b", re.I), "<RSID>"),
    (re.compile(r"\bCID\s*\d+\b", re.I), "<CID>"),
    (re.compile(r"\b(?:NCBI\s+)?gene\s+(?:id\s+)?\d+\b", re.I), "<GENEID>"),
    (re.compile(r"\b\d{7,9}\b"), "<PMID>"),
    (re.compile(r"\b[A-Za-z]+_[A-Za-z0-9_]+\b"), "<NAME>"),
]


def mask_entities(q: str) -> str:
    for rx, tag in ENT:
        q = rx.sub(tag, q)
    return q


def entity_hints(q: str) -> List[str]:
    hints = []
    for rx, tag in ENT:
        if rx.search(q):
            hints.append(tag.strip("<>"))
    return sorted(set(hints))


# ----------------------------------------------------------------------------- variants
# Each variant: async fn(lab, query) -> {group: prob}
VARIANTS: Dict[str, Callable] = {}


def variant(name):
    def deco(fn):
        VARIANTS[name] = fn
        return fn
    return deco


def crit(lab: Lab, none: bool = False, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    c = {n: (extra or {}).get(n, lab.desc[n]) for n in lab.names}
    if none:
        c["none"] = NONE_DESC
    return c


@variant("baseline")
async def v_baseline(lab, q):
    return await lab.choice(q, crit(lab))


@variant("state_json")
async def v_state_json(lab, q):
    return await lab.choice({"request": q}, crit(lab), "Which kind of tool does `request` need?")


@variant("state_prefix")
async def v_state_prefix(lab, q):
    return await lab.choice("User request: " + q, crit(lab))


@variant("instr_user")
async def v_instr_user(lab, q):
    return await lab.choice(q, crit(lab), "What is the user asking the assistant to do? Pick the kind of tool that can do it.")


@variant("masked")
async def v_masked(lab, q):
    return await lab.choice(mask_entities(q), crit(lab))


@variant("hints")
async def v_hints(lab, q):
    return await lab.choice({"request": q, "contains": entity_hints(q)}, crit(lab), "Which kind of tool does `request` need?")


@variant("model_multi")
async def v_multi(lab, q):
    return await lab.choice(q, crit(lab), model="multilingual")


@variant("model_en")
async def v_en(lab, q):
    return await lab.choice(q, crit(lab), model="english")


@variant("model_typed")
async def v_typed(lab, q):
    return await lab.choice(q, crit(lab), task="typed_decisions")


@variant("crit_names")
async def v_names(lab, q):
    ex = {n: "%s (%s)" % (lab.desc[n], ", ".join(t.replace("pubmed_", "").replace("_", " ") for t in lab.members[n][:4])) for n in lab.names}
    return await lab.choice(q, crit(lab, extra=ex))


@variant("crit_auto")
async def v_auto(lab, q):
    by = {tool_name(t): t for t in lab.tools}
    ex = {n: "; ".join(first_sentence(tool_description(by[t]), 90) for t in lab.members[n][:3]) for n in lab.names}
    return await lab.choice(q, crit(lab, extra=ex))


@variant("crit_examples")
async def v_examples(lab, q):
    ex = {n: lab.desc[n] + (" (e.g. " + "; ".join(lab.extra.get("examples", {}).get(n, [])) + ")" if lab.extra.get("examples", {}).get(n) else "") for n in lab.names}
    return await lab.choice(q, crit(lab, extra=ex))


@variant("none_opt")
async def v_none(lab, q):
    p = await lab.choice(q, crit(lab, none=True))
    return p


@variant("noul")
async def v_noul(lab, q):
    return await lab.noul(q, crit(lab))


@variant("ensemble_choice_noul")
async def v_ens(lab, q):
    a, b = await asyncio.gather(lab.choice(q, crit(lab)), lab.noul(q, crit(lab)))
    return mean_probs([norm(a), b])


@variant("perm3")
async def v_perm3(lab, q):
    rng = random.Random(hash(q) & 0xFFFF)
    outs = []
    for _ in range(3):
        names = lab.names[:]
        rng.shuffle(names)
        outs.append(await lab.choice(q, {n: lab.desc[n] for n in names}))
    return mean_probs(outs)


@variant("multi_cross")
async def v_cross(lab, q):
    a, b = await asyncio.gather(lab.choice(q, crit(lab), model="multilingual"), lab.choice(q, crit(lab), model="english"))
    return mean_probs([a, b])


# ----------------------------------------------------------------------------- composable variants
def parse_spec(spec: str) -> Dict[str, str]:
    return dict(kv.split("=", 1) for kv in spec.split(",") if "=" in kv)


def build_criteria(lab: Lab, mode: str) -> Dict[str, str]:
    by = {tool_name(t): t for t in lab.tools}
    auto = {n: "; ".join(first_sentence(tool_description(by[t]), 90) for t in lab.members[n][:3]) for n in lab.names}
    ex = lab.extra.get("examples", {})
    out = {}
    for n in lab.names:
        d = lab.desc[n]
        if mode == "desc":
            out[n] = d
        elif mode == "auto":
            out[n] = auto[n]
        elif mode == "both":
            out[n] = "%s -- %s" % (d, auto[n])
        elif mode == "examples":
            out[n] = d + (" (e.g. " + "; ".join(ex.get(n, [])) + ")" if ex.get(n) else "")
        elif mode == "short":
            out[n] = lab.extra["labels"].get(n, d)
        elif mode == "short_auto":
            out[n] = "%s: %s" % (lab.extra["labels"].get(n, d), auto[n])
        elif mode == "both_ex":
            out[n] = "%s -- %s" % (d + (" (e.g. " + "; ".join(ex.get(n, [])) + ")" if ex.get(n) else ""), auto[n])
        else:
            raise ValueError(mode)
    return out


async def composed(lab: Lab, q: str, spec: Dict[str, str]) -> Dict[str, float]:
    if "ens" in spec:  # views separated by '|', e.g. ens=state=json/crit=short|state=raw/crit=auto
        views = [parse_spec(v.replace("/", ",")) for v in spec["ens"].split("|")]
        ps = await asyncio.gather(*[composed(lab, q, v) for v in views])
        return mean_probs([norm(p) for p in ps])
    if spec.get("cal") == "1":  # unsupervised prior correction using stored per-group means
        base = dict(spec)
        base.pop("cal")
        p = await composed(lab, q, base)
        prior = lab.extra.get("prior", {}).get(spec.get("calkey", ""), {})
        return norm({k: v / max(prior.get(k, 1.0 / len(p)), 1e-3) for k, v in p.items()}) if prior else p

    st, model, cm = spec.get("state", "raw"), spec.get("model", "auto"), spec.get("crit", "desc")
    form, perm, none = spec.get("form", "choice"), int(spec.get("perm", "1")), spec.get("none", "0") == "1"
    kw = {} if model == "auto" else {"model": model}
    if "head" in spec:
        kw["head_max_len"] = int(spec["head"])
        kw["max_len"] = int(spec.get("max", "1536"))
    if st == "raw":
        state, instr = q, INSTR
    elif st == "prefix":
        state, instr = "User request: " + q, INSTR
    elif st == "json":
        state, instr = {"request": q}, "Which kind of tool does `request` need?"
    elif st == "hints":
        state, instr = {"request": q, "contains": entity_hints(q)}, "Which kind of tool does `request` need?"
    else:
        raise ValueError(st)
    crit_all = build_criteria(lab, cm)
    if none:
        crit_all["none"] = NONE_DESC
    names = list(crit_all)
    if form == "noul":
        return await lab.noul(state, crit_all, **kw)
    if form == "hier":
        sup = lab.extra["supers"]
        p1 = await lab.choice(state, {k: v["label"] for k, v in sup.items()}, instr, **kw)
        subs = await asyncio.gather(*[lab.choice(state, {g: crit_all[g] for g in v["groups"]}, instr, **kw) for v in sup.values()])
        return {g: p1[k] * sp.get(g, 0.0) for (k, v), sp in zip(sup.items(), subs) for g in v["groups"]}
    outs = []
    rng = random.Random(hash(q) & 0xFFFF)
    for i in range(perm):
        order = names[:]
        if perm > 1 and i > 0:
            rng.shuffle(order)
        outs.append(await lab.choice(state, {n: crit_all[n] for n in order}, instr, **kw))
    p = mean_probs(outs)
    if form == "ens":
        p = mean_probs([p, await lab.noul(state, crit_all, **kw)])
    return p


# ----------------------------------------------------------------------------- scoring
def rank_of(probs: Dict[str, float], gold: set) -> Optional[int]:
    order = [k for k, _ in sorted(probs.items(), key=lambda kv: -kv[1]) if k != "none"]
    for i, k in enumerate(order, 1):
        if k in gold:
            return i
    return None


async def evaluate(lab: Lab, name: str, queries: List[dict], conc: int = 4):
    if name in VARIANTS:
        fn = VARIANTS[name]
    else:
        spec = parse_spec(name.replace("+", ","))

        async def fn(lab, q, spec=spec):
            return await composed(lab, q, spec)

    sem = asyncio.Semaphore(conc)

    async def one(q):
        async with sem:
            t0 = time.perf_counter()
            try:
                p = await fn(lab, q["query"])
                err = None
            except Exception as e:  # noqa: BLE001
                p, err = {}, repr(e)
            return q, p, (time.perf_counter() - t0) * 1000, err

    rows = await asyncio.gather(*[one(q) for q in queries])
    pos = [(q, p, ms) for q, p, ms, e in rows if q["expect"]]
    neg = [(q, p, ms) for q, p, ms, e in rows if not q["expect"]]
    ranks = [rank_of(p, lab.gold(q["expect"])) for q, p, _ in pos]
    n = len(ranks)
    out = {
        "n": n,
        "top1": sum(r == 1 for r in ranks) / n,
        "top2": sum(r is not None and r <= 2 for r in ranks) / n,
        "top3": sum(r is not None and r <= 3 for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
        "ms": statistics.median(ms for _, _, ms in pos),
        "errors": sum(1 for *_, e in rows if e),
    }
    if neg and any("none" in p for _, p, _ in neg):
        out["none@1"] = sum(max(p, key=p.get) == "none" for _, p, _ in neg) / len(neg)
        out["none_fp"] = sum(max(p, key=p.get) == "none" for q, p, _ in pos if "none" in p) / n  # positives wrongly sent to none
    out["_rows"] = [(q["id"], q["query"], q["expect"], p) for q, p, _, _ in rows]
    return out


def load_sets(which: str) -> List[dict]:
    files = {"tune": ["queries.jsonl", "queries_heldout.jsonl", "queries_neg_tune.jsonl"], "test": ["queries_test.jsonl"], "dev": ["queries.jsonl"], "heldout": ["queries_heldout.jsonl"]}[which]
    out = []
    for f in files:
        for l in open(ROOT / "data" / f, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                d["id"] = "%s:%s" % (f[:3], d["id"])
                out.append(d)
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="tune", choices=["tune", "test", "dev", "heldout"])
    ap.add_argument("--variants", default="baseline")
    ap.add_argument("--laya-url", default="http://127.0.0.1:8000")
    ap.add_argument("--tools", default=str(ROOT / "data/pubmed_tools.json"))
    ap.add_argument("--groups", default=str(ROOT.parent / "examples/pubmed_groups.json"))
    ap.add_argument("--calibrate", help="';'-separated variant specs to compute priors for (use calkey=<spec> in a variant)")
    ap.add_argument("--dump", help="write per-query rows to this JSON file")
    a = ap.parse_args()
    cfg = load_groups_config(a.groups)
    lab = Lab(a.laya_url, json.load(open(a.tools)), cfg, extra={"examples": json.load(open(ROOT / "data/group_examples.json")), "labels": json.load(open(ROOT / "data/group_labels.json")), "supers": json.load(open(ROOT / "data/group_supers.json"))})
    queries = load_sets(a.set)
    if a.calibrate:  # per-group mean probability over UNLABELED tuning queries, per variant spec
        unl = load_sets("tune")
        for item in a.calibrate.split(";"):
            alias, spec = item.split("=", 1)
            sem = asyncio.Semaphore(4)

            async def cal_one(q, spec=spec):
                async with sem:
                    return await composed(lab, q["query"], parse_spec(spec.replace("+", ",")))

            ps = await asyncio.gather(*[cal_one(q) for q in unl])
            lab.extra.setdefault("prior", {})[alias] = {k: sum(norm(p)[k] for p in ps) / len(ps) for k in ps[0]}
    print("set=%s  %d queries (%d positive)  groups=%d" % (a.set, len(queries), sum(bool(q["expect"]) for q in queries), len(lab.names)))
    print("%-24s %6s %6s %6s %6s %7s %7s" % ("variant", "top1", "top2", "top3", "MRR", "ms", "none@1"))
    dump = {}
    for name in [v for v in a.variants.split(";")] if ";" in a.variants else a.variants.split(","):
        r = await evaluate(lab, name, queries)
        dump[name] = r.pop("_rows")
        print("%-24s %5.1f%% %5.1f%% %5.1f%% %6.3f %6.0f  %s%s" % (name, 100 * r["top1"], 100 * r["top2"], 100 * r["top3"], r["mrr"], r["ms"], ("%.0f%%" % (100 * r["none@1"])) if "none@1" in r else "-", ("  (positives sent to none: %.0f%%)" % (100 * r["none_fp"])) if "none_fp" in r else ""), flush=True)
    if a.dump:
        Path(a.dump).write_text(json.dumps(dump, ensure_ascii=False))
    await lab.client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
