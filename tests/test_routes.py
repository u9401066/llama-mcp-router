import json
import os
import sys

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from conftest import FakeBackend
from llama_mcp_router.agent import AgentConfig
from llama_mcp_router.proxy import RouterConfig, Upstream, create_app, load_routes
from llama_mcp_router.selectors import AllSelector

HERE = os.path.dirname(__file__)


def upstream_app(seen):
    async def chat(request: Request):
        body = await request.json()
        seen.append({"body": body, "conv": request.headers.get("x-conversation-id"), "auth": request.headers.get("authorization")})
        if body.get("stream"):
            async def gen():
                yield 'data: {"choices":[{"index":0,"delta":{"content":"from upstream"}}]}\n\n'
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream", headers={"x-upstream": "1"})
        return JSONResponse({"choices": [{"index": 0, "message": {"role": "assistant", "content": "from upstream"}}]})

    return Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"])])


def make(tmp_path, agent=False):
    seen, backend = [], FakeBackend()
    up = Upstream(url="http://up", model="sentinel", prefix="med:", name="sentinel-svc", transport=httpx.ASGITransport(app=upstream_app(seen)))
    agent_cfg = AgentConfig(command=[sys.executable, os.path.join(HERE, "fake_acp_agent.py")], root=str(tmp_path / "s"), default=True) if agent else None
    cfg = RouterConfig(backend="http://backend", selector=AllSelector(), routes=[up], agent=agent_cfg)
    app = backend.app()
    app.router.routes.append(Route("/v1/models", lambda r: JSONResponse({"object": "list", "data": [{"id": "bonsai", "object": "model"}]})))
    return TestClient(create_app(cfg, transport=httpx.ASGITransport(app=app))), seen, backend


def test_prefix_and_model_go_to_the_upstream_unchanged(tmp_path):
    c, seen, backend = make(tmp_path)
    with c:
        r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "  MED: chest pain"}]},
                   headers={"X-Conversation-Id": "c1::m", "Authorization": "Bearer k"})
        assert r.json()["choices"][0]["message"]["content"] == "from upstream"
        assert seen[0]["body"]["messages"][0]["content"] == "  MED: chest pain"  # passed through as sent
        assert seen[0]["conv"] == "c1::m" and seen[0]["auth"] == "Bearer k"
        r = c.post("/v1/chat/completions", json={"model": "sentinel", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
        assert "from upstream" in r.text and r.headers["x-upstream"] == "1"
        r = c.post("/v1/chat/completions", json={"model": "local", "messages": [{"role": "user", "content": "plain"}]})
        assert r.json()["choices"][0]["message"]["content"] == "ok" and len(backend.requests) == 1  # everything else: llama-server
    assert len(seen) == 2


def test_routes_win_over_the_default_agent_and_are_listed(tmp_path):
    c, seen, backend = make(tmp_path, agent=True)
    with c:
        r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "med: q"}]})
        assert r.json()["choices"][0]["message"]["content"] == "from upstream" and len(seen) == 1
        ids = [m["id"] for m in c.get("/v1/models").json()["data"]]
        assert ids == ["bonsai", "sentinel", "agent"]


def test_unreachable_upstream_is_a_502(tmp_path):
    up = Upstream(url="http://127.0.0.1:9", prefix=["med:"])
    cfg = RouterConfig(backend="http://backend", selector=AllSelector(), routes=[up])
    with TestClient(create_app(cfg, transport=httpx.ASGITransport(app=FakeBackend().app()))) as c:
        r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "med: q"}]})
    assert r.status_code == 502 and "not reachable" in r.json()["error"]["message"]


def test_load_routes(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"routes": [{"url": "http://x", "prefix": "med:", "transport": "ignored"}]}))
    assert load_routes(str(p))[0].prefix == ["med:"] and load_routes(str(p))[0].transport is None
    assert load_routes(os.path.join(HERE, "..", "examples", "routes.json"))[0].model == "sentinel"
    with pytest.raises(ValueError):
        Upstream(url="http://x")
