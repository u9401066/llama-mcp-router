import asyncio
import json
import os
import socket
import subprocess
import time

import httpx
import pytest
from starlette.testclient import TestClient

from conftest import FakeBackend
from llama_mcp_router import __version__
from llama_mcp_router.agent import AgentConfig, AgentManager, bwrap_argv
from llama_mcp_router.ops import BACKGROUND, INTERACTIVE, Metrics, QueueFull, SlotGate, Warmup
from llama_mcp_router.proxy import RouterConfig, _cap_tool_results, create_app
from test_agent import NoTools, live, live_app, make, timing_backend


def free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


# ---------------------------------------------------------------------- slot gate
def test_slot_gate_lets_interactive_overtake_background_and_reports_positions():
    async def main():
        gate = SlotGate(capacity=1, max_queue=2)
        order, positions = [], {}
        first = await gate.acquire()

        async def wait(name, prio):
            async with gate.slot(prio, on_wait=lambda pos: positions.setdefault(name, pos)):
                order.append(name)
                await asyncio.sleep(0.01)

        bg = asyncio.ensure_future(wait("compaction", BACKGROUND))
        await asyncio.sleep(0.01)
        it = asyncio.ensure_future(wait("agent step", INTERACTIVE))
        await asyncio.sleep(0.01)
        assert gate.waiting == 2 and positions == {"compaction": 1, "agent step": 1}  # the agent step went ahead of compaction
        with pytest.raises(QueueFull) as e:
            gate.check_admission()
        assert e.value.retry_after >= 5 and "2 requests are waiting" in str(e.value)
        first()
        first()  # releasing twice is harmless
        await asyncio.gather(bg, it)
        assert order == ["agent step", "compaction"] and gate.active == 0 and gate.info()["waited"] == 2

    asyncio.run(main())


def test_slot_gate_cancelled_waiter_leaves_the_queue_and_stuck_slots_are_reclaimed():
    async def main():
        gate = SlotGate(capacity=1, max_hold=0.05)
        held = await gate.acquire()
        waiter = asyncio.ensure_future(gate.acquire())
        await asyncio.sleep(0.01)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert gate.waiting == 0 and gate.active == 1
        await asyncio.sleep(0.1)  # the holder never releases
        again = await asyncio.wait_for(gate.acquire(), 1)
        assert gate.reclaimed == 1 and gate.active == 1
        held()  # a late release of the reclaimed slot does not free someone else's
        assert gate.active == 1
        again()
        assert gate.active == 0
        assert SlotGate(0).info()["capacity"] == 0 and (await SlotGate(0).acquire())() is None  # disabled: no-op

    asyncio.run(main())


# ---------------------------------------------------------------------- tool-result cap
def test_cap_tool_results_cuts_the_middle_deterministically_and_keeps_other_parts():
    big = "H" * 600 + "M" * 3000 + "T" * 400
    msgs = [{"role": "user", "content": "x" * 5000}, {"role": "tool", "tool_call_id": "1", "content": big},
            {"role": "tool", "tool_call_id": "2", "content": "short"},
            {"role": "tool", "tool_call_id": "3", "content": [{"type": "text", "text": "a" * 2000}, {"type": "image_url", "image_url": {"url": "x"}}]}]
    again = json.loads(json.dumps(msgs))
    assert _cap_tool_results(msgs, 1000) and _cap_tool_results(again, 1000)
    assert msgs == again  # same input, same output: the prompt cache keeps matching
    out = msgs[1]["content"]
    assert out.startswith("H" * 600) and out.endswith("T" * 125) and "[router: 3000 of these 4000 characters were left out" in out
    assert "at most 1000 characters" in out
    assert msgs[0]["content"] == "x" * 5000 and msgs[2]["content"] == "short" and isinstance(msgs[3]["content"], list)
    assert not _cap_tool_results([{"role": "tool", "content": "y" * 1000}], 1000)


# ---------------------------------------------------------------------- metrics
def test_metrics_render_prometheus_text():
    m = Metrics()
    m.inc("requests_total", path="chat")
    m.inc("requests_total", 2, path="agent_llm")
    m.observe("tool_seconds", 1.5)
    m.add_timings({"prompt_n": 10, "cache_n": 90, "predicted_n": 5, "prompt_ms": 20, "predicted_ms": 100}, compaction=True)
    text = m.render({"slots_active": 1})
    assert '# TYPE llama_mcp_router_requests_total counter' in text
    assert 'llama_mcp_router_requests_total{path="agent_llm"} 2' in text and 'llama_mcp_router_requests_total{path="chat"} 1' in text
    assert "llama_mcp_router_tool_seconds_sum 1.5" in text and "llama_mcp_router_tool_seconds_count 1" in text
    assert 'llama_mcp_router_cache_tokens_total{kind="compaction"} 90' in text and "llama_mcp_router_slots_active 1" in text


# ---------------------------------------------------------------------- warm-up
def test_warmup_records_the_first_request_and_fills_only_never_used_slots(tmp_path):
    posted = []

    def handler(request):
        if request.url.path == "/slots":
            return httpx.Response(200, json=[{"id": 0, "is_processing": False, "n_prompt_tokens": 0},
                                             {"id": 1, "is_processing": False, "n_prompt_tokens": 4000},
                                             {"id": 2, "is_processing": True, "n_prompt_tokens": 0}])
        posted.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [], "timings": {"prompt_n": 35000}})

    async def main():
        async with httpx.AsyncClient(base_url="http://backend", transport=httpx.MockTransport(handler)) as client:
            w = Warmup(client, str(tmp_path / "warmup.json"), SlotGate(2), Metrics())
            assert await w.run_once() == []  # nothing recorded yet
            first = {"model": "m", "stream": True, "timings_per_token": True, "tools": [{"t": 1}], "reasoning_effort": "max",
                     "max_completion_tokens": 16384,
                     "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "task"},
                                  {"role": "user", "content": "runtime context"}, {"role": "user", "content": "skills"}]}
            assert w.record(first) and not w.record(dict(first))  # unchanged prefix: no rewrite
            later = dict(first, messages=first["messages"] + [{"role": "assistant", "content": "x"}])
            assert not w.record(later)
            done = await w.run_once()
            # a new Warmup (router restart) knows the recorded prefix
            assert not Warmup(client, str(tmp_path / "warmup.json"), SlotGate(2), Metrics()).record(first)
            return done, w

    done, w = asyncio.run(main())
    assert [d["slot"] for d in done] == [0] and done[0]["prompt_n"] == 35000 and w.last["slots"] == done
    assert posted == [{"model": "m", "tools": [{"t": 1}], "reasoning_effort": "max", "max_tokens": 1, "max_completion_tokens": 1,
                       "stream": False, "id_slot": 0,
                       "cache_prompt": True, "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "ok"}]}]


def test_warmup_skips_a_slot_that_took_a_request_since_the_first_check(tmp_path):
    slots = [{"id": 0, "is_processing": False, "n_prompt_tokens": 0}, {"id": 1, "is_processing": False, "n_prompt_tokens": 0}]
    posted = []

    def handler(request):
        if request.url.path == "/slots":
            return httpx.Response(200, json=slots)
        posted.append(json.loads(request.content)["id_slot"])
        slots[1]["is_processing"] = True  # an agent request arrives on slot 1 while slot 0 is being warmed
        return httpx.Response(200, json={"choices": [], "timings": {"prompt_n": 35000}})

    async def main():
        async with httpx.AsyncClient(base_url="http://backend", transport=httpx.MockTransport(handler)) as client:
            w = Warmup(client, str(tmp_path / "warmup.json"), SlotGate(2), Metrics())
            w.record({"model": "m", "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "task"}]})
            return await w.run_once()

    assert [d["slot"] for d in asyncio.run(main())] == [0] and posted == [0]


# ---------------------------------------------------------------------- router: queue, drain, health
def test_drain_refuses_new_agent_turns_and_full_queue_refuses_new_work(tmp_path):
    c, backend = make(tmp_path)
    with c:
        router = c.app.state.router
        h = c.get("/router/health").json()
        assert h["version"] == __version__ and h["draining"] is False and h["agents"] == {"sessions": 0, "live": 0, "running_turns": 0}
        assert c.post("/router/drain").json()["draining"] is True
        r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "/agent do it"}]})
        assert r.status_code == 503 and r.headers["retry-after"] == "60"
        r = c.post("/v1/chat/completions", json={"model": "local", "messages": [{"role": "user", "content": "plain"}]})
        assert r.status_code == 200  # plain chats are short: still served while draining
        assert c.delete("/router/drain").json()["draining"] is False
        assert c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "/agent do it"}]}).status_code == 200

        router.gate.capacity, router.gate.max_queue, router.gate.active = 1, 1, 1  # one request running, one waiting
        fut = asyncio.new_event_loop().create_future()
        router.gate._heap.append((0, 0, fut))
        r = c.post("/v1/chat/completions", json={"model": "local", "messages": [{"role": "user", "content": "plain"}]})
        assert r.status_code == 429 and int(r.headers["retry-after"]) >= 5
        router.gate._heap.clear()
        router.gate.active = 0
        text = c.get("/router/metrics").text
    assert 'llama_mcp_router_rejected_total{reason="draining"} 1' in text and 'llama_mcp_router_rejected_total{reason="queue_full"} 1' in text
    assert 'llama_mcp_router_agent_turns_total{result="ok"} 1' in text and 'llama_mcp_router_agent_turns_total{result="rejected"} 1' in text
    assert 'llama_mcp_router_tool_calls_total{status="completed"} 1' in text and "llama_mcp_router_agent_sessions 1" in text


def test_plain_chats_go_through_the_slot_gate(tmp_path):
    backend = FakeBackend()
    cfg = RouterConfig(backend="http://backend", selector=NoTools(), slots=2)
    with TestClient(create_app(cfg, transport=httpx.ASGITransport(app=backend.app()))) as c:
        for stream in (False, True):
            r = c.post("/v1/chat/completions", json={"model": "local", "stream": stream, "messages": [{"role": "user", "content": "hi"}]})
            assert r.status_code == 200
        h = c.get("/router/health").json()
    assert h["queue"]["capacity"] == 2 and h["queue"]["active"] == 0 and h["chats"] == 0  # every slot given back


def test_agent_llm_caps_tool_results_counts_stats_and_records_warmup(tmp_path):
    seen = []
    port = free_port()
    extra = [{"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
             {"role": "tool", "tool_call_id": "c1", "content": "A" * 3000}]
    app = live_app(tmp_path, backend=timing_backend(seen), default=True, warmup=True, tool_result_max_chars=1000,
                   env={"FAKE_LLM_CALLS": "2", "FAKE_LLM_SYSTEM": "You are an agent.", "FAKE_LLM_EXTRA": json.dumps(extra),
                        "FAKE_LLM_URL": "http://127.0.0.1:%d/agent/llm/{session}/v1/chat/completions" % port})
    with live(app, port=port) as base:
        r = httpx.post(base + "/v1/chat/completions", json={"messages": [{"role": "user", "content": "go"}]},
                       headers={"X-Conversation-Id": "ops-c::agent"}, timeout=30)
        assert r.status_code == 200
        sid = r.json()["router"]["agent_session"]
        info = httpx.get(base + "/agent/sessions/%s" % sid).json()
        metrics = httpx.get(base + "/router/metrics").text
        health = httpx.get(base + "/router/health").json()
    tool = seen[1]["messages"][-1]["content"]
    assert len(seen) == 2 and tool.startswith("A" * 875) and "[router: 2000 of these 3000 characters" in tool
    warm = json.loads((tmp_path / "sessions" / "warmup.json").read_text())
    assert warm["messages"] == [{"role": "system", "content": "You are an agent."}] and "stream" not in warm
    st = info["stats"]
    assert st["requests"] == 2 and st["prompt_n"] == 200 and st["cache_n"] == 1800 and st["tool_calls"] == 1 and info["turns"] == 1
    assert 'llama_mcp_router_requests_total{path="agent_llm"} 2' in metrics and 'llama_mcp_router_prompt_tokens_total{kind="agent_llm"} 200' in metrics
    assert health["agents"]["sessions"] == 1 and health["warmup"] == {"at": None}
    meta = json.loads((tmp_path / "sessions" / "index.json").read_text())
    assert next(iter(meta.values()))["stats"]["requests"] == 2  # persisted for after a restart


# ---------------------------------------------------------------------- retention, memory
def test_retention_removes_old_idle_sessions_only(tmp_path):
    c, _ = make(tmp_path)
    with c:
        for conv in ("old", "new"):
            c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "/agent " + conv}]}, headers={"X-Conversation-Id": conv + "::m"})
    index_path = tmp_path / "sessions" / "index.json"
    index = json.loads(index_path.read_text())
    old_key = next(k for k in index if ":c:old" in k)
    index[old_key]["last_used"] = time.time() - 10 * 86400
    index_path.write_text(json.dumps(index))
    old_dir = tmp_path / "sessions" / index[old_key]["dir"]
    assert old_dir.is_dir()
    root = str(tmp_path / "sessions")
    assert AgentManager(AgentConfig(command=["x"], root=root)).prune() == []  # off by default
    mgr = AgentManager(AgentConfig(command=["x"], root=root, retention_days=7))
    assert mgr.prune() == [index[old_key]["id"]]
    assert not old_dir.exists() and list(json.loads(index_path.read_text())) == [k for k in index if k != old_key]


def test_memory_dir_is_shared_per_user_and_kept_out_of_git(tmp_path):
    c, _ = make(tmp_path, memory=True)
    with c:
        sids = []
        for conv in ("m1", "m2"):
            r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "/agent hi"}]}, headers={"X-Conversation-Id": conv + "::m"})
            sids.append(r.json()["router"]["agent_session"])
    mem = tmp_path / "sessions" / "anonymous" / "memory"
    assert (mem / "MEMORY.md").read_text().startswith("# Memory")
    for sid in sids:
        ws = tmp_path / "sessions" / "anonymous" / sid / "workspace"
        assert os.path.realpath(ws / ".agents" / "memory") == str(mem)
        # editors replace files by renaming a temporary file over them: works through the shared directory
        tmp = ws / ".agents" / "memory" / "MEMORY.md.tmp"
        tmp.write_text("# Memory\n- %s\n" % sid)
        os.replace(tmp, ws / ".agents" / "memory" / "MEMORY.md")
        assert (mem / "MEMORY.md").read_text() == "# Memory\n- %s\n" % sid
        assert "/.agents/memory\n" in (ws / ".git" / "info" / "exclude").read_text()
        subprocess.run(["git", "add", "-A"], cwd=ws, check=True)
        tracked = subprocess.run(["git", "ls-files", "--cached"], cwd=ws, capture_output=True, text=True).stdout
        assert ".agents/memory" not in tracked
    cfg = AgentConfig(command=["x"], sandbox="bwrap", sandbox_hide=[])
    argv = bwrap_argv(cfg, "/s/d", "/s/d/workspace", [("/s/memory", "/s/d/workspace/.agents/memory")])
    i = argv.index("/s/memory")
    assert argv[i - 1] == "--bind" and argv[i + 1] == "/s/d/workspace/.agents/memory" and argv.index("--bind") < i
    with pytest.raises(ValueError):
        AgentManager(AgentConfig(command=["x"], root=str(tmp_path), memory_dir="../x"))._memory_rel()
