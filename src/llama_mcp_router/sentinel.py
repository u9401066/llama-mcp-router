"""Sentinel: watch a reasoning model's chain of thought step by step and steer it when a step breaks a rule.

The generator is llama-server (``/apply-template`` + streamed ``/completion``); the critic is a SystemOne decision model
(Cloudflare Clef / Clef-Flash, Laya, Jev: ``POST /v1/systemone``) that answers one yes/no question per rule for every step.

For each reasoning step (a paragraph of the thinking):

* the critic scores it against every check (``noul`` probability that the rule is broken), with the case (the user's
  messages) and the end of the accepted reasoning as context;
* a step that passes is released to the client and becomes part of the accepted reasoning;
* a step that breaks a ``rollback`` check is discarded (never shown); generation restarts from the accepted reasoning plus a
  first-person correction (e.g. "Before any benign explanation I must first rule out ...");
* a step that breaks an ``inject`` check is kept, followed by a first-person reflection ("Wait, ..."), and generation goes on.

The final answer (after ``</think>``) is held until it is complete and checked paragraph by paragraph; if a paragraph breaks a
rule, the answer is discarded and the model goes back to thinking with the correction.

Rolling back is a new request with the accepted text as prompt: llama-server's prefix cache keeps it cheap (hybrid/recurrent
models such as Qwen3.5 cannot cut their state at an arbitrary token, so ``seq_rm``-style truncation is not used). Each check
fires at most ``max_per_check`` times and all checks at most ``max_interventions`` times per answer, so a disagreement between
critic and generator cannot loop. Interventions are annotated in the reasoning so they stay auditable.

This is a research tool for steering reasoning, not a validated clinical safety device.
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx

log = logging.getLogger("llama_mcp_router")

THINK_END = "</think>"


@dataclass
class Check:
    id: str
    question: str  # yes/no question to the decision model: "does this step break the rule?"
    threshold: float = 0.5
    action: str = "rollback"  # rollback | inject
    correction: str = ""  # first-person text placed into the reasoning when the check fires
    label: str = ""

    def __post_init__(self) -> None:
        if self.action not in ("rollback", "inject"):
            raise ValueError("check %s: action must be 'rollback' or 'inject'" % self.id)


@dataclass
class SentinelConfig:
    critic_url: str = "http://127.0.0.1:8000"
    critic_model: Optional[str] = None
    critic_timeout: float = 20.0
    model_id: str = "sentinel"
    triggers: List[str] = field(default_factory=lambda: ["med:"])
    system: Optional[str] = None  # system prompt used when the request has none
    temperature: float = 0.6
    top_p: float = 0.9
    max_tokens: int = 8192
    min_step_chars: int = 60
    max_step_chars: int = 1200
    check_answer: bool = True
    max_per_check: int = 2
    max_interventions: int = 4
    case_chars: int = 2000
    context_chars: int = 1200
    annotate: bool = True
    warning: str = "Sentinel: this answer may still contain: {labels}. The automatic corrections did not resolve it; review before acting."
    checks: List[Check] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.checks = [c if isinstance(c, Check) else Check(**c) for c in self.checks]
        if not self.checks:
            raise ValueError("sentinel config needs at least one check")

    @classmethod
    def load(cls, path: str) -> "SentinelConfig":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class SystemOneCritic:
    """One /v1/systemone request per step: a noul question per check -> probability that the rule is broken."""

    def __init__(self, cfg: SentinelConfig, transport: Optional[httpx.AsyncBaseTransport] = None):
        self.cfg = cfg
        self.client = httpx.AsyncClient(base_url=cfg.critic_url.rstrip("/"), timeout=cfg.critic_timeout, transport=transport)

    async def aclose(self) -> None:
        await self.client.aclose()

    async def judge(self, case: str, context: str, step: str) -> Dict[str, float]:
        state: Dict[str, Any] = {"case": case[-self.cfg.case_chars:]}
        if context.strip():
            state["previous_reasoning"] = context[-self.cfg.context_chars:]
        state["reasoning_step"] = step
        body: Dict[str, Any] = {"state": state, "questions": {c.id: {"type": "noul", "instructions": c.question} for c in self.cfg.checks}}
        if self.cfg.critic_model:
            body["model"] = self.cfg.critic_model
        r = await self.client.post("/v1/systemone", json=body)
        r.raise_for_status()
        answers = r.json()["answers"]
        out = {}
        for c in self.cfg.checks:
            a = answers.get(c.id) or {}
            p = a.get("noul", a.get("probability", (a.get("probabilities") or {}).get("true", 0.0)))
            out[c.id] = float(p)
        return out


class Segmenter:
    """Cut streamed text into reasoning steps (paragraphs), the end of thinking, and answer paragraphs."""

    _SENTENCE_END = re.compile(r"[。！？!?]|\.\s")

    def __init__(self, min_chars: int, max_chars: int, thinking: bool = True):
        self.min_chars, self.max_chars = min_chars, max_chars
        self.thinking = thinking
        self.buf = ""

    def feed(self, text: str) -> List[Tuple[str, str]]:
        self.buf += text
        out: List[Tuple[str, str]] = []
        while True:
            if self.thinking:
                i = self.buf.find(THINK_END)
                if i >= 0:
                    head, self.buf = self.buf[:i], self.buf[i + len(THINK_END):]
                    out += self._cut_all(head, "reasoning", final=True)
                    out.append(("think_end", ""))
                    self.thinking = False
                    continue
            step = self._cut_one("reasoning" if self.thinking else "content")
            if step is None:
                return out
            out.append(step)

    def flush(self) -> List[Tuple[str, str]]:
        rest, self.buf = self.buf, ""
        return self._cut_all(rest, "reasoning" if self.thinking else "content", final=True)

    def _cut_all(self, text: str, kind: str, final: bool) -> List[Tuple[str, str]]:
        saved, self.buf = self.buf, text
        out = []
        while True:
            step = self._cut_one(kind)
            if step is None:
                break
            out.append(step)
        if final and self.buf.strip():
            out.append((kind, self.buf))
        elif final and self.buf and out:
            out[-1] = (out[-1][0], out[-1][1] + self.buf)
        self.buf = saved
        return out

    def _held(self) -> int:
        """Length of a buffer tail that could be the start of "</think>" (kept back until the rest arrives)."""
        if not self.thinking:
            return 0
        for k in range(min(len(THINK_END) - 1, len(self.buf)), 0, -1):
            if THINK_END.startswith(self.buf[-k:]):
                return k
        return 0

    def _cut_one(self, kind: str) -> Optional[Tuple[str, str]]:
        b = self.buf[: len(self.buf) - self._held()]
        i = b.find("\n\n")
        while i >= 0 and not b[:i].strip():  # skip leading blank lines
            i = b.find("\n\n", i + 2)
        if i >= 0 and (i >= self.min_chars or kind == "content"):
            return self._take(i + 2, kind)
        j = b.find("\n")
        while j >= 0:
            if j >= self.min_chars:
                return self._take(j + 1, kind)
            j = b.find("\n", j + 1)
        if len(b) >= self.max_chars:
            ends = [m.end() for m in self._SENTENCE_END.finditer(b, 0, self.max_chars)]
            return self._take(ends[-1] if ends else self.max_chars, kind)
        return None

    def _take(self, n: int, kind: str) -> Tuple[str, str]:
        step, self.buf = self.buf[:n], self.buf[n:]
        return (kind, step)


def _user_case(messages: List[Dict[str, Any]]) -> str:
    parts = []
    for m in messages:
        if m.get("role") == "user":
            c = m.get("content")
            parts.append(c if isinstance(c, str) else " ".join(p.get("text", "") for p in c or [] if isinstance(p, dict)))
    return "\n".join(parts)


class SentinelRun:
    """One guarded answer. ``events()`` yields {"type": "reasoning"|"content"|"note"|"done", ...}."""

    def __init__(self, cfg: SentinelConfig, backend: httpx.AsyncClient, critic: SystemOneCritic, messages: List[Dict[str, Any]],
                 options: Optional[Dict[str, Any]] = None):
        self.cfg, self.backend, self.critic = cfg, backend, critic
        self.messages = messages
        if cfg.system and not any(m.get("role") == "system" for m in messages):
            self.messages = [{"role": "system", "content": cfg.system}] + messages
        self.options = options or {}
        self.case = _user_case(messages)
        self.timings: List[Dict[str, Any]] = []
        self.critic_ms = 0.0
        self.steps_checked = 0
        self.interventions: List[Dict[str, Any]] = []
        self.attempts: List[Dict[str, Any]] = []  # per generation request: tokens, why it stopped
        self.flagged: List[Dict[str, Any]] = []  # answer paragraphs released with a warning
        self.counts: Counter = Counter()

    async def _render(self) -> str:
        body: Dict[str, Any] = {"messages": self.messages}
        if self.options.get("chat_template_kwargs"):
            body["chat_template_kwargs"] = self.options["chat_template_kwargs"]
        r = await self.backend.post("/apply-template", json=body)
        r.raise_for_status()
        return r.json()["prompt"]

    async def _judge(self, context: str, step: str) -> Optional[Tuple[Check, float, bool]]:
        """The most likely broken check as (check, p, may_fire); may_fire is False once its limits are used up."""
        broken = await self._broken(context, step)
        if not broken:
            return None
        if len(self.interventions) < self.cfg.max_interventions:
            for c, p in broken:
                if self.counts[c.id] < self.cfg.max_per_check:
                    return c, p, True
        c, p = broken[0]
        return c, p, False

    async def _broken(self, context: str, step: str) -> List[Tuple[Check, float]]:
        """Every check the critic says this step breaks, most likely first ([] if the critic is unavailable)."""
        t0 = time.perf_counter()
        try:
            probs = await self.critic.judge(self.case, context, step)
        except (httpx.HTTPError, KeyError, ValueError) as e:  # fail open: an unavailable critic must not block the answer
            log.warning("sentinel critic failed (%s: %s); step passed unchecked", type(e).__name__, e)
            return []
        finally:
            self.critic_ms += (time.perf_counter() - t0) * 1000
            self.steps_checked += 1
        return sorted(((c, probs[c.id]) for c in self.cfg.checks if probs.get(c.id, 0.0) >= c.threshold), key=lambda x: -x[1])

    def _note(self, c: Check, p: float, what: str) -> Dict[str, Any]:
        self.counts[c.id] += 1
        self.interventions.append({"check": c.id, "p": round(p, 3), "action": what})
        return {"type": "note", "text": "\n[sentinel: %s (p=%.2f) – %s]\n" % (c.label or c.id, p, what)}

    async def events(self) -> AsyncIterator[Dict[str, Any]]:
        rendered = await self._render()
        thinking = rendered.rstrip().endswith("<think>")
        accepted = ""  # reasoning text the model continues from
        answer_prefix = ""  # "</think>" when the answer is requested explicitly
        forced = False
        produced = 0
        while True:
            seg = Segmenter(self.cfg.min_step_chars, self.cfg.max_step_chars, thinking=thinking)
            content: List[str] = []
            restart = False
            body = {"prompt": rendered + accepted + answer_prefix, "stream": True, "cache_prompt": True, "n_predict": max(1024 if answer_prefix else 64, self.cfg.max_tokens - produced),
                    "temperature": self.options.get("temperature", self.cfg.temperature), "top_p": self.options.get("top_p", self.cfg.top_p),
                    "timings_per_token": True}
            last_timings = None
            async with self.backend.stream("POST", "/completion", json=body) as r:
                r.raise_for_status()
                pieces = self._pieces(r)
                async for piece, timings, stop in pieces:
                    last_timings = timings or last_timings
                    steps = seg.feed(piece) + (seg.flush() if stop else [])
                    for kind, step in steps:
                        if kind == "reasoning":
                            if not step.strip():
                                accepted += step
                                continue
                            verdict = await self._judge(accepted, step)
                            if verdict is None or not verdict[2]:
                                accepted += step
                                yield {"type": "reasoning", "text": step}
                                if verdict is not None and self.cfg.annotate:  # limits used up: keep going, but say so
                                    yield {"type": "note", "text": "\n[sentinel: %s still flagged (p=%.2f), intervention limit reached]\n" % (verdict[0].label or verdict[0].id, verdict[1])}
                                continue
                            c, p, _ = verdict
                            if c.action == "inject":
                                yield {"type": "reasoning", "text": step}
                                if self.cfg.annotate:
                                    yield self._note(c, p, "reflection added")
                                else:
                                    self._note(c, p, "reflection added")
                                fix = "\n" + c.correction.strip() + "\n\n"
                                accepted += step + fix
                                yield {"type": "reasoning", "text": fix}
                            else:
                                note = self._note(c, p, "step withdrawn, thinking again")
                                if self.cfg.annotate:
                                    yield note
                                fix = c.correction.strip() + "\n\n"
                                accepted += fix
                                yield {"type": "reasoning", "text": fix}
                            restart = True
                            break
                        elif kind == "think_end":
                            thinking = False
                        else:
                            content.append(step)
                    if restart:
                        self.attempts.append({"tokens": (last_timings or {}).get("predicted_n"), "stop": "sentinel"})
                        break
                await pieces.aclose()
            if last_timings:
                self.timings.append(last_timings)
                produced += int(last_timings.get("predicted_n", 0))
            if restart:
                continue
            if not content and thinking and not forced:
                # generation ended inside the reasoning (end of sequence or token budget): close it and ask for the answer
                forced = True
                accepted += "\n"
                rendered_answer = THINK_END + "\n\n"
                thinking = False
                answer_prefix = rendered_answer
                continue
            warning = ""
            if content and self.cfg.check_answer:
                fire: Optional[Tuple[Check, float]] = None
                still: Dict[str, Tuple[Check, float]] = {}
                for para in [p for p in "".join(content).split("\n\n") if p.strip()]:
                    for c, p in await self._broken(accepted, para):
                        can = len(self.interventions) < self.cfg.max_interventions and self.counts[c.id] < self.cfg.max_per_check
                        if can and fire is None:
                            fire = (c, p)
                        if c.id not in still or p > still[c.id][1]:
                            still[c.id] = (c, p)
                    if fire:
                        break
                if fire:
                    c, p = fire
                    note = self._note(c, p, "answer withdrawn, thinking again")
                    if self.cfg.annotate:
                        yield note
                    fix = "\n" + c.correction.strip() + "\n\n"
                    accepted += fix
                    yield {"type": "reasoning", "text": fix}
                    thinking, answer_prefix = True, ""
                    continue
                if still:  # still flagged but the limits are used up: release it, visibly marked
                    items = sorted(still.values(), key=lambda x: -x[1])
                    self.flagged = [{"check": c.id, "p": round(p, 3)} for c, p in items]
                    warning = "\n\n> ⚠️ " + self.cfg.warning.format(labels=", ".join("%s (p=%.2f)" % (c.label or c.id, p) for c, p in items))
            if content:
                yield {"type": "content", "text": "".join(content).lstrip("\n") + warning}
            yield {"type": "done", "interventions": self.interventions, "steps_checked": self.steps_checked, "flagged": self.flagged,
                   "critic_ms": round(self.critic_ms, 1), "timings": self.timings, "attempts": self.attempts}
            return

    async def _pieces(self, r: httpx.Response) -> AsyncIterator[Tuple[str, Optional[Dict[str, Any]], bool]]:
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            try:
                d = json.loads(line[6:])
            except ValueError:
                continue
            if d.get("stop"):
                self.attempts.append({"tokens": d.get("tokens_predicted"), "stop": d.get("stop_type"), "n_predict": (d.get("generation_settings") or {}).get("n_predict")})
            yield d.get("content", ""), d.get("timings"), bool(d.get("stop"))
            if d.get("stop"):
                return
