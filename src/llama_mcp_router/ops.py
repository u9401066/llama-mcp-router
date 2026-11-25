"""Operations helpers: a slot-aware queue in front of llama-server, Prometheus metrics and prompt-prefix warm-up."""
from __future__ import annotations

import asyncio
import hashlib
import heapq
import itertools
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

import httpx

log = logging.getLogger("llama_mcp_router")

INTERACTIVE, BACKGROUND = 0, 1  # SlotGate priorities: lower goes first


class QueueFull(Exception):
    def __init__(self, waiting: int, retry_after: int):
        super().__init__("all model slots are busy and %d requests are waiting" % waiting)
        self.waiting, self.retry_after = waiting, retry_after


class SlotGate:
    """At most ``capacity`` model requests at llama-server at once; the rest wait here, by priority then arrival.

    llama-server queues requests beyond its slots too, but blindly: first come first served, no position to report.
    Waiting here lets interactive requests (a chat, an agent's next step) overtake background ones (context compaction),
    lets the caller tell its user where it stands, and lets new work be refused with 429 + Retry-After when the queue is
    long (``max_queue``). capacity 0 disables the gate.
    """

    def __init__(self, capacity: int = 0, max_queue: int = 0, max_hold: float = 3600.0):
        self.capacity = capacity
        self.max_queue = max_queue
        self.max_hold = max_hold  # a slot held longer than this is taken back (a holder that never released it)
        self.active = 0
        self._heap: List[Tuple[int, int, asyncio.Future]] = []
        self._seq = itertools.count()
        self._leases: Dict[int, Tuple[float, Callable[[], None]]] = {}
        self._hold_ewma = 30.0  # seconds a request keeps its slot (for Retry-After)
        self.waited_total = 0.0
        self.waited_count = 0
        self.reclaimed = 0

    @property
    def waiting(self) -> int:
        return sum(1 for _, _, f in self._heap if not f.done())

    def position(self, fut: asyncio.Future) -> int:
        ahead = sorted((p, s) for p, s, f in self._heap if not f.done())
        mine = next(((p, s) for p, s, f in self._heap if f is fut), None)
        return ahead.index(mine) + 1 if mine in ahead else 0

    def retry_after(self) -> int:
        return max(5, int(self._hold_ewma * (self.waiting + 1) / max(1, self.capacity)))

    def check_admission(self) -> None:
        """Raise QueueFull when new work should be refused (only callers starting new work use this)."""
        if self.capacity and self.max_queue and self.active >= self.capacity and self.waiting >= self.max_queue:
            raise QueueFull(self.waiting, self.retry_after())

    async def acquire(self, priority: int = INTERACTIVE, on_wait: Optional[Callable[[int], None]] = None) -> Callable[[], None]:
        """Wait for a slot; returns its release function (idempotent), for holders that outlive a ``with`` block
        (streamed responses release from their generator's ``finally``)."""
        if not self.capacity:
            return lambda: None
        self._expire()
        t0 = time.monotonic()
        if self.active < self.capacity and not self.waiting:
            self.active += 1
        else:
            fut: asyncio.Future = asyncio.get_running_loop().create_future()
            heapq.heappush(self._heap, (priority, next(self._seq), fut))
            if on_wait:
                on_wait(self.position(fut))
            try:
                await fut  # _release() counted us in self.active before resolving
            except asyncio.CancelledError:
                if fut.done() and not fut.cancelled():
                    self._release(0.0)
                self._heap = [x for x in self._heap if x[2] is not fut]
                heapq.heapify(self._heap)
                raise
            self.waited_total += time.monotonic() - t0
            self.waited_count += 1
        start = time.monotonic()
        lease = next(self._seq)
        released = False

        def release() -> None:
            nonlocal released
            if not released:
                released = True
                self._leases.pop(lease, None)
                self._release(time.monotonic() - start)

        self._leases[lease] = (start, release)
        return release

    def _expire(self) -> None:
        now = time.monotonic()
        for start, release in [v for v in self._leases.values() if now - v[0] > self.max_hold]:
            log.warning("slot gate: taking back a slot held for %.0f s", now - start)
            self.reclaimed += 1
            release()

    @asynccontextmanager
    async def slot(self, priority: int = INTERACTIVE, on_wait: Optional[Callable[[int], None]] = None) -> AsyncIterator[None]:
        release = await self.acquire(priority, on_wait)
        try:
            yield
        finally:
            release()

    def _release(self, held: float) -> None:
        if held:
            self._hold_ewma = 0.8 * self._hold_ewma + 0.2 * held
        self.active = max(0, self.active - 1)
        while self._heap and self.active < self.capacity:
            _, _, fut = heapq.heappop(self._heap)
            if not fut.done():
                self.active += 1
                fut.set_result(None)

    def info(self) -> Dict[str, Any]:
        return {"capacity": self.capacity, "active": self.active, "waiting": self.waiting, "max_queue": self.max_queue,
                "retry_after_s": self.retry_after() if self.capacity else 0, "waited": self.waited_count,
                "waited_s": round(self.waited_total, 1), "reclaimed": self.reclaimed}


class Metrics:
    """Counters for /metrics (Prometheus text format) and /router/health; no dependency on prometheus_client."""

    HELP = {
        "requests_total": ("counter", "Chat requests by path (chat, agent_turn, agent_llm, compaction, warmup)"),
        "agent_turns_total": ("counter", "Agent turns by result (ok, error, rejected)"),
        "agent_turn_seconds": ("summary", "Wall time of agent turns"),
        "prompt_tokens_total": ("counter", "Prompt tokens llama-server processed (not from cache) for agent model requests"),
        "cache_tokens_total": ("counter", "Prompt tokens llama-server reused from its cache for agent model requests"),
        "predicted_tokens_total": ("counter", "Tokens generated for agent model requests"),
        "prompt_seconds_total": ("counter", "Time llama-server spent processing prompts for agent model requests"),
        "predicted_seconds_total": ("counter", "Time llama-server spent generating for agent model requests"),
        "compactions_total": ("counter", "Agent context-compaction requests"),
        "tool_calls_total": ("counter", "Agent tool calls by status"),
        "tool_seconds": ("summary", "Agent tool call duration"),
        "queue_wait_seconds": ("summary", "Time requests waited for a model slot in the router"),
        "rejected_total": ("counter", "Requests refused (429 queue full, 503 draining)"),
    }

    def __init__(self) -> None:
        self.counters: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = {}

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        key = (name, tuple(sorted(labels.items())))
        self.counters[key] = self.counters.get(key, 0.0) + value

    def observe(self, name: str, value: float, **labels: str) -> None:
        self.inc(name + "_sum", value, **labels)
        self.inc(name + "_count", 1, **labels)

    def add_timings(self, t: Optional[Dict[str, Any]], compaction: bool = False) -> None:
        if not t:
            return
        kind = "compaction" if compaction else "agent_llm"
        self.inc("prompt_tokens_total", float(t.get("prompt_n") or 0), kind=kind)
        self.inc("cache_tokens_total", float(t.get("cache_n") or 0), kind=kind)
        self.inc("predicted_tokens_total", float(t.get("predicted_n") or 0), kind=kind)
        self.inc("prompt_seconds_total", float(t.get("prompt_ms") or 0) / 1000.0, kind=kind)
        self.inc("predicted_seconds_total", float(t.get("predicted_ms") or 0) / 1000.0, kind=kind)

    def get(self, name: str, **labels: str) -> float:
        return self.counters.get((name, tuple(sorted(labels.items()))), 0.0)

    def total(self, name: str) -> float:
        return sum(v for (n, _), v in self.counters.items() if n == name)

    def render(self, gauges: Dict[str, float], prefix: str = "llama_mcp_router_") -> str:
        lines: List[str] = []
        seen = set()
        for (name, labels), value in sorted(self.counters.items()):
            base = name[:-4] if name.endswith("_sum") else name[:-6] if name.endswith("_count") else name
            if base not in seen and base in self.HELP:
                kind, text = self.HELP[base]
                lines += ["# HELP %s%s %s" % (prefix, base, text), "# TYPE %s%s %s" % (prefix, base, kind)]
                seen.add(base)
            lab = ",".join('%s="%s"' % (k, str(v).replace('"', "'")) for k, v in labels)
            lines.append("%s%s%s %s" % (prefix, name, "{%s}" % lab if lab else "", _num(value)))
        for name, value in sorted(gauges.items()):
            lines += ["# TYPE %s%s gauge" % (prefix, name), "%s%s %s" % (prefix, name, _num(value))]
        return "\n".join(lines) + "\n"


def _num(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else "%.6g" % v


class Warmup:
    """Keeps llama-server's slots holding the agents' shared prompt prefix (tools + system prompt).

    The first model request of every agent session is remembered (``record``: the request without its user message, in
    ``<agent root>/warmup.json``). A watcher polls llama-server's /slots; a slot that is idle and has never processed a
    prompt (llama-server just started) gets that prefix with a one-word user message and max_tokens 1, pinned with
    ``id_slot``. The next agent session then finds the ~35k-token prefix cached instead of paying for it (~14 s on llm01).
    Slots that already hold a prompt are never touched, so a running session's cache is never overwritten.
    """

    def __init__(self, client: httpx.AsyncClient, path: str, gate: SlotGate, metrics: Metrics, interval: float = 30.0):
        self.client, self.path, self.gate, self.metrics, self.interval = client, path, gate, metrics, interval
        self._digest: Optional[str] = None
        self.task: Optional[asyncio.Task] = None
        self.last: Dict[str, Any] = {}
        try:
            with open(path, "rb") as f:
                self._digest = hashlib.sha256(f.read()).hexdigest()
        except OSError:
            pass

    def record(self, body: Dict[str, Any]) -> bool:
        """Remember ``body`` (an agent model request about to be sent) when it is a session's first request."""
        msgs = body.get("messages") or []
        if len(msgs) != 2 or msgs[0].get("role") != "system" or msgs[1].get("role") != "user":
            return False
        tmpl = {k: v for k, v in body.items() if k not in ("messages", "stream", "stream_options", "timings_per_token", "return_progress")}
        tmpl["messages"] = [msgs[0]]
        data = json.dumps(tmpl, ensure_ascii=False, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        if digest == self._digest:
            return False
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, self.path)
        self._digest = digest
        log.info("warm-up prefix recorded (%d bytes)", len(data))
        return True

    def template(self) -> Optional[Dict[str, Any]]:
        try:
            with open(self.path, encoding="utf-8") as f:
                t = json.load(f)
            return t if isinstance(t, dict) and t.get("messages") else None
        except (OSError, ValueError):
            return None

    async def cold_slots(self) -> List[int]:
        """Idle slots that have never processed a prompt (llama-server started since)."""
        r = await self.client.get("/slots", timeout=10)
        r.raise_for_status()
        return [s["id"] for s in r.json() if not s.get("is_processing") and not s.get("n_prompt_tokens")]

    async def warm(self, slot: int, tmpl: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fill ``slot`` with the prefix, unless it took a request while this waited for a router slot (None)."""
        body = dict(tmpl, messages=tmpl["messages"] + [{"role": "user", "content": "ok"}], max_tokens=1, stream=False,
                    id_slot=slot, cache_prompt=True)
        t0 = time.monotonic()
        async with self.gate.slot(BACKGROUND):
            # checked again now: a request pinned to a busy slot would wait for it (holding the router slot) and then
            # replace that session's cache
            if slot not in await self.cold_slots():
                return None
            r = await self.client.post("/v1/chat/completions", json=body, timeout=600)
        self.metrics.inc("requests_total", path="warmup")
        t = (r.json() or {}).get("timings") if r.status_code == 200 else None
        res = {"slot": slot, "status": r.status_code, "s": round(time.monotonic() - t0, 1), "prompt_n": (t or {}).get("prompt_n")}
        log.info("warm-up slot %d: %s", slot, res)
        return res

    async def run_once(self) -> List[Dict[str, Any]]:
        tmpl = self.template()
        if not tmpl:
            return []
        done = []
        for slot in await self.cold_slots():
            res = await self.warm(slot, tmpl)
            if res:
                done.append(res)
        if done:
            self.last = {"at": time.time(), "slots": done}
        return done

    async def watch(self) -> None:
        while True:
            try:
                await self.run_once()
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
                log.debug("warm-up check failed: %s: %s", type(e).__name__, e)
            await asyncio.sleep(self.interval)

    def start(self) -> None:
        if self.task is None:
            self.task = asyncio.ensure_future(self.watch())

    async def stop(self) -> None:
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self.task = None
