import json
import os
import shutil
import subprocess
import sys

import httpx
import pytest
from starlette.testclient import TestClient

from conftest import FakeBackend
from llama_mcp_router import Selection, Selector
from llama_mcp_router.agent import AgentConfig
from llama_mcp_router.proxy import RouterConfig, create_app

HERE = os.path.dirname(__file__)


class NoTools(Selector):
    name = "none"

    async def select(self, q, t):
        return Selection([])


def agent_cfg(tmp_path, **kw):
    skills = tmp_path / "skills" / "demo-skill"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text("---\nname: demo-skill\ndescription: demo\n---\nbody\n")
    return AgentConfig(command=[sys.executable, os.path.join(HERE, "fake_acp_agent.py")], root=str(tmp_path / "sessions"),
                       skills_dirs=[str(tmp_path / "skills")], files={"agent.patch.yml": "home: {home}\n"}, **kw)


def make(tmp_path, backend=None, **kw):
    b = backend or FakeBackend()
    cfg = RouterConfig(backend="http://backend", selector=NoTools(), agent=agent_cfg(tmp_path, **kw))
    return TestClient(create_app(cfg, transport=httpx.ASGITransport(app=b.app()))), b


def events(text):
    return [json.loads(e[6:]) for e in text.split("\n\n") if e.startswith("data: {")]


def deltas(text, field):
    return "".join(e["choices"][0]["delta"].get(field) or "" for e in events(text))


def test_trigger_routes_to_agent_streams_and_commits(tmp_path):
    c, backend = make(tmp_path)
    with c:
        msgs = [{"role": "user", "content": "/agent make a file"}]
        r = c.post("/v1/chat/completions", json={"model": "local", "stream": True, "messages": msgs})
        assert r.status_code == 200 and backend.requests == []  # never reached the LLM backend directly
        assert "thinking about: make a file" in deltas(r.text, "reasoning_content")
        assert "▶ write out.txt" in deltas(r.text, "reasoning_content") and "permission denied" in deltas(r.text, "reasoning_content")
        content = deltas(r.text, "content")
        assert content.startswith("done turn 1") and "out.txt" in content and "/archive.zip" in content
        sid = content.split("/agent/sessions/")[1].split("/")[0]
        ws = tmp_path / "sessions" / "anonymous" / sid / "workspace"
        assert (ws / "out.txt").read_text() == "turn 1: make a file (permission=no)\n"  # trigger stripped, escalation denied
        assert (ws / ".agents/skills/demo-skill/SKILL.md").exists()
        assert (tmp_path / "sessions" / "anonymous" / sid / "agent.patch.yml").read_text().startswith("home: " + str(tmp_path / "sessions" / "anonymous" / sid / "home"))
        log = subprocess.run(["git", "log", "--format=%s"], cwd=ws, capture_output=True, text=True).stdout.split("\n")
        assert log[0] == "turn 1: make a file" and "session start" in log

        # second turn of the same chat: same session, same process, history not resent
        msgs += [{"role": "assistant", "content": content}, {"role": "user", "content": "add another line"}]
        r2 = c.post("/v1/chat/completions", json={"model": "local", "stream": False, "messages": msgs})
        d = r2.json()
        assert d["choices"][0]["message"]["content"].startswith("done turn 2") and d["router"]["agent_session"] == sid
        assert (ws / "out.txt").read_text().splitlines()[1] == "turn 2: add another line (permission=no)"

        info = c.get("/agent/sessions/%s" % sid).json()
        assert "out.txt" in info["files"] and info["log"][0]["subject"].startswith("turn 2")
        f = c.get("/agent/sessions/%s/files/out.txt" % sid)
        assert f.status_code == 200 and f.text.count("turn") == 2
        z = c.get("/agent/sessions/%s/archive.zip" % sid)
        assert z.status_code == 200 and z.content[:2] == b"PK"
        for bad in ("../index.json", ".git/config", "nope.txt"):
            assert c.get("/agent/sessions/%s/files/%s" % (sid, bad)).status_code == 404
        assert c.get("/agent/sessions/%s/files/out.txt" % ("0" * 32)).status_code == 404


def test_model_name_routes_to_agent_and_allow_policy(tmp_path):
    c, backend = make(tmp_path, permissions="allow", model_id="my-agent")
    with c:
        r = c.post("/v1/chat/completions", json={"model": "my-agent", "messages": [{"role": "user", "content": "hello"}]})
        content = r.json()["choices"][0]["message"]["content"]
        sid = content.split("/agent/sessions/")[1].split("/")[0]
        assert (tmp_path / "sessions" / "anonymous" / sid / "workspace" / "out.txt").read_text() == "turn 1: hello (permission=ok)\n"
        # other chats are untouched by the bridge
        c.post("/v1/chat/completions", json={"model": "local", "messages": [{"role": "user", "content": "plain question"}]})
        assert len(backend.requests) == 1


def test_models_endpoint_lists_agent(tmp_path):
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def models(request):
        return JSONResponse({"object": "list", "data": [{"id": "bonsai", "object": "model"}], "models": [{"name": "bonsai", "model": "bonsai"}]})

    app = Starlette(routes=[Route("/v1/models", models)])
    cfg = RouterConfig(backend="http://backend", selector=NoTools(), agent=agent_cfg(tmp_path))
    with TestClient(create_app(cfg, transport=httpx.ASGITransport(app=app))) as c:
        d = c.get("/v1/models").json()
    assert [m["id"] for m in d["data"]] == ["bonsai", "agent"] and d["models"][-1]["model"] == "agent"


def test_session_survives_router_restart_via_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_RESUME_OK", "1")
    msgs = [{"role": "user", "content": "/agent first"}]
    c, _ = make(tmp_path)
    with c:
        sid = c.post("/v1/chat/completions", json={"messages": msgs}).json()["router"]["agent_session"]
    c2, _ = make_existing(tmp_path)
    with c2:
        d = c2.post("/v1/chat/completions", json={"messages": msgs + [{"role": "assistant", "content": "x"}, {"role": "user", "content": "again"}]}).json()
    assert d["router"]["agent_session"] == sid  # same workspace after a restart
    out = (tmp_path / "sessions" / "anonymous" / sid / "workspace" / "out.txt").read_text().splitlines()
    assert out[0].startswith("turn 1: first") and out[1].startswith("turn 1: again")  # new process, resumed session id


def make_existing(tmp_path):
    cfg = RouterConfig(backend="http://backend", selector=NoTools(),
                       agent=AgentConfig(command=[sys.executable, os.path.join(HERE, "fake_acp_agent.py")], root=str(tmp_path / "sessions")))
    return TestClient(create_app(cfg, transport=httpx.ASGITransport(app=FakeBackend().app()))), None


def test_empty_agent_request_explains_usage(tmp_path):
    c, _ = make(tmp_path)
    with c:
        r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "/agent"}]})
    assert "Send a task" in r.json()["choices"][0]["message"]["content"]


def test_bwrap_argv_hides_home_run_tmp_and_binds_session(tmp_path):
    from llama_mcp_router.agent import bwrap_argv

    cfg = AgentConfig(command=["x"], sandbox="bwrap", sandbox_ro=[str(tmp_path)], sandbox_network=False)
    argv = bwrap_argv(cfg, "/s/dir", "/s/dir/workspace")
    home = os.path.expanduser("~")
    assert argv[:4] == ["bwrap", "--ro-bind", "/", "/"]
    assert ["--tmpfs", home] == argv[argv.index(home) - 1: argv.index(home) + 1]
    assert "--tmpfs" in argv and "/run" in argv and "/tmp" in argv
    assert argv[argv.index("--bind") + 1: argv.index("--bind") + 3] == ["/s/dir", "/s/dir"]
    assert argv.index("--ro-bind", 2) > argv.index(home)  # mounted back after hiding
    assert argv[-1] == "--unshare-net" and "--die-with-parent" in argv


import shutil  # noqa: E402

import pytest  # noqa: E402


@pytest.mark.skipif(not shutil.which("bwrap") or subprocess.run(["bwrap", "--ro-bind", "/", "/", "true"]).returncode != 0, reason="bubblewrap not usable here")
def python_mounts():
    """Directories the test interpreter needs, including every hop of a symlinked (venv / uv-managed) python."""
    dirs, p = [sys.prefix, sys.base_prefix], sys.executable
    for _ in range(10):
        dirs.append(os.path.dirname(os.path.abspath(p)))
        if not os.path.islink(p):
            break
        p = os.path.join(os.path.dirname(p), os.readlink(p))
    dirs.append(os.path.dirname(os.path.realpath(sys.executable)))
    return sorted(set(dirs))


@pytest.mark.skipif(not shutil.which("bwrap"), reason="bubblewrap not installed")
def test_agent_runs_inside_bwrap_and_cannot_see_home(tmp_path):
    secret = os.path.expanduser("~/.llama_mcp_router_test_secret")
    with open(secret, "w") as f:
        f.write("top secret")
    try:
        root = tmp_path / "sessions"
        cfg = AgentConfig(command=[sys.executable, "-c", PROBE], root=str(root), sandbox="bwrap", sandbox_ro=python_mounts() + [HERE])
        rc = RouterConfig(backend="http://backend", selector=NoTools(), agent=cfg)
        with TestClient(create_app(rc, transport=httpx.ASGITransport(app=FakeBackend().app()))) as c:
            d = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "/agent go"}]}).json()
        content = d["choices"][0]["message"]["content"]
        assert "home_visible=False" in content and "write_ok=True" in content, content
    finally:
        os.remove(secret)


PROBE = r'''
import json, os, sys
for line in sys.stdin:
    m = json.loads(line); i = m.get("id"); meth = m.get("method")
    out = lambda r: (sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": i, "result": r}) + "\n"), sys.stdout.flush())
    if meth == "initialize": out({"protocolVersion": 1, "agentCapabilities": {}})
    elif meth == "session/new": out({"sessionId": "x"})
    elif meth == "session/prompt":
        vis = os.path.exists(os.path.expanduser("~/.llama_mcp_router_test_secret")) or os.path.exists("/home/%s/.llama_mcp_router_test_secret" % os.environ.get("USER", ""))
        try:
            open("probe.txt", "w").write("x"); ok = True
        except OSError:
            ok = False
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": "x", "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "home_visible=%s write_ok=%s" % (vis, ok)}}}}) + "\n"); sys.stdout.flush()
        out({"stopReason": "end_turn"})
'''


def test_default_mode_sends_every_chat_to_agent_with_chat_optout(tmp_path):
    c, backend = make(tmp_path, default=True)
    with c:
        r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "no prefix at all"}]})
        assert r.json()["choices"][0]["message"]["content"].startswith("done turn 1") and backend.requests == []
        r2 = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "chat: just a quick question"}]})
        assert r2.json()["choices"][0]["message"]["content"] == "ok"  # plain llama-server path
        assert backend.requests[0]["messages"][0]["content"] == "just a quick question"  # opt-out prefix stripped
        r3 = c.post("/v1/chat/completions", json={"messages": [{"role": "system", "content": "s"}]})  # no user message
        assert r3.json()["choices"][0]["message"]["content"] == "ok"


def test_install_dsh_plugin(tmp_path):
    from llama_mcp_router.cli import main

    assert main(["install-dsh-plugin", str(tmp_path)]) == 1  # not a dsh install
    (tmp_path / "node_modules" / "@deepseek-ai" / "dsh-tools").mkdir(parents=True)
    assert main(["install-dsh-plugin", str(tmp_path)]) == 0
    text = (tmp_path / "plugins" / "llama-mcp-router.mjs").read_text()
    assert "agent/pre-step" in text and "find_tools" in text and "/router/select" in text


def sid_of(r):
    return r.json()["router"]["agent_session"]


def test_conversation_id_header_keeps_chats_with_same_first_message_apart(tmp_path):
    c, _ = make(tmp_path, default=True)
    hi = [{"role": "user", "content": "hi"}]
    with c:
        a = sid_of(c.post("/v1/chat/completions", json={"messages": hi}, headers={"X-Conversation-Id": "conv-a::agent"}))
        b = sid_of(c.post("/v1/chat/completions", json={"messages": hi}, headers={"X-Conversation-Id": "conv-b::agent"}))
        assert a != b  # same first message, different Web UI conversations
        more = hi + [{"role": "assistant", "content": "x"}, {"role": "user", "content": "next"}]
        r = c.post("/v1/chat/completions", json={"messages": more}, headers={"X-Conversation-Id": "conv-a::other-model"})
        assert sid_of(r) == a and r.json()["choices"][0]["message"]["content"].startswith("done turn 2")
        # without the header the first-message hash is the fallback key
        assert sid_of(c.post("/v1/chat/completions", json={"messages": hi})) not in (a, b)
        assert sid_of(c.post("/v1/chat/completions", json={"messages": hi}, headers={"X-Conversation-Id": "bad id/../x"})) not in (a, b)


def test_per_user_api_keys_isolate_sessions(tmp_path):
    users = tmp_path / "users.json"
    users.write_text(json.dumps({"key-bob": "bob"}))
    c, backend = make(tmp_path, default=True, users={"key-alice": "alice"}, users_file=str(users))
    cfgfile = tmp_path / "agent.json"
    cfgfile.write_text(json.dumps({"command": ["x"], "users": {"key-alice": "alice"}, "users_file": str(users)}))
    assert AgentConfig.load(str(cfgfile)).users == {"key-alice": "alice", "key-bob": "bob"}
    hi = {"messages": [{"role": "user", "content": "hi"}]}
    conv = {"X-Conversation-Id": "same-conv"}
    with c:
        r = c.post("/v1/chat/completions", json=hi, headers=conv)
        assert r.status_code == 401 and "API key" in r.json()["error"]["message"]
        assert c.post("/v1/chat/completions", json=hi, headers={**conv, "Authorization": "Bearer nope"}).status_code == 401
        a = sid_of(c.post("/v1/chat/completions", json=hi, headers={**conv, "Authorization": "Bearer key-alice"}))
        b = sid_of(c.post("/v1/chat/completions", json=hi, headers={**conv, "Authorization": "Bearer key-bob"}))
        assert a != b
        assert (tmp_path / "sessions" / "alice" / a / "workspace" / "out.txt").exists()
        assert (tmp_path / "sessions" / "bob" / b / "workspace" / "out.txt").exists()
        mine = c.get("/agent/sessions", headers={"Authorization": "Bearer key-alice"}).json()
        assert mine["user"] == "alice" and [s["session"] for s in mine["sessions"]] == [a] and mine["sessions"][0]["title"] == "hi"
        assert c.get("/agent/sessions").status_code == 401
        assert c.get("/agent/sessions/%s/files/out.txt" % b).status_code == 200  # file links stay capability URLs
        # the plain (chat:) path is not gated by the agent's user keys
        assert c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "chat: q"}]}).status_code == 200
        assert len(backend.requests) == 1


def test_session_listing_needs_users_map(tmp_path):
    c, _ = make(tmp_path)
    with c:
        assert c.get("/agent/sessions").status_code == 404


def test_process_cap_stops_idle_agents_and_resumes_them(tmp_path):
    c, _ = make(tmp_path, default=True, max_processes=1)
    with c:
        mgr = c.app.state.router.agents
        hi = [{"role": "user", "content": "hi"}]
        a = sid_of(c.post("/v1/chat/completions", json={"messages": hi}, headers={"X-Conversation-Id": "a"}))
        b = sid_of(c.post("/v1/chat/completions", json={"messages": hi}, headers={"X-Conversation-Id": "b"}))
        assert sum(1 for s in mgr.sessions.values() if s.conn and s.conn.alive) == 1
        more = hi + [{"role": "assistant", "content": "x"}, {"role": "user", "content": "back"}]
        assert sid_of(c.post("/v1/chat/completions", json={"messages": more}, headers={"X-Conversation-Id": "a"})) == a
        assert sum(1 for s in mgr.sessions.values() if s.conn and s.conn.alive) == 1
        out = (tmp_path / "sessions" / "anonymous" / a / "workspace" / "out.txt").read_text().splitlines()
        assert out[0] == "turn 1: hi (permission=no)" and out[1].endswith("back (permission=no)")  # same workspace, resumed
        assert b != a
