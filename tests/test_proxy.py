import json

import httpx
import pytest
from starlette.testclient import TestClient

from conftest import TOOLS, FakeBackend, call
from llama_mcp_router import BM25Selector, LayaSelector, Selection, Selector
from llama_mcp_router.proxy import RouterConfig, called_tool_names, create_app, last_user_query


class Pick(Selector):
    name = "pick"

    def __init__(self, *names):
        self.names = list(names)

    async def select(self, query, tools):
        self.query = query
        return Selection(list(self.names))


def make(backend, selector=None, **kw):
    cfg = RouterConfig(backend="http://backend", selector=selector or Pick("pm_export"), **kw)
    return TestClient(create_app(cfg, transport=httpx.ASGITransport(app=backend.app())))


def user(text):
    return {"role": "user", "content": text}


def test_inject_filters_tools_and_sets_headers(backend):
    sel = Pick("pm_export")
    with make(backend, sel) as c:
        r = c.post("/v1/chat/completions", json={"model": "m", "messages": [user("export to bibtex please")]})
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "ok"
    sent = backend.requests[0]["tools"]
    assert [t["function"]["name"] for t in sent] == ["pm_export"]
    assert r.headers["x-router-tools"] == "pm_export" and r.headers["x-router-pool"] == "5"
    assert sel.query == "export to bibtex please"


def test_always_exclude_and_continuation_tools(backend):
    msgs = [user("find papers on propofol"), {"role": "assistant", "content": "", "tool_calls": [call("pm_gene")]}, {"role": "tool", "tool_call_id": "c1", "content": "x"}]
    with make(backend, Pick("pm_export"), always=["pm_search"], exclude=["fs_*"]) as c:
        c.post("/v1/chat/completions", json={"messages": msgs})
    assert [t["function"]["name"] for t in backend.requests[0]["tools"]] == ["pm_search", "pm_export", "pm_gene"]


def test_client_tools_are_routed_and_win_name_collisions(backend):
    mine = {"type": "function", "function": {"name": "pm_export", "description": "MINE", "parameters": {"type": "object"}}}
    with make(backend, Pick("pm_export")) as c:
        c.post("/v1/chat/completions", json={"messages": [user("hello there my friend")], "tools": [mine]})
    sent = backend.requests[0]["tools"]
    assert len(sent) == 1 and sent[0]["function"]["description"] == "MINE"


def test_no_server_tools_and_empty_pool_passthrough(backend):
    with make(backend, use_server_tools=False) as c:
        c.post("/v1/chat/completions", json={"messages": [user("hi")]})
    assert "tools" not in backend.requests[0]


def test_bypass_header_and_body_flag(backend):
    with make(backend) as c:
        c.post("/v1/chat/completions", json={"messages": [user("hi")]}, headers={"x-router-bypass": "1"})
        c.post("/v1/chat/completions", json={"messages": [user("hi")], "router": False})
    assert all("tools" not in b and "router" not in b for b in backend.requests)


def test_tool_choice_forces_inclusion(backend):
    with make(backend, Pick("pm_export")) as c:
        c.post("/v1/chat/completions", json={"messages": [user("hello there my friend")], "tool_choice": {"type": "function", "function": {"name": "pm_icd"}}})
    assert {t["function"]["name"] for t in backend.requests[0]["tools"]} == {"pm_export", "pm_icd"}


def test_selector_failure_falls_back(backend):
    class Boom(Selector):
        name = "boom"

        async def select(self, q, t):
            raise RuntimeError("laya down")

    with make(backend, Boom()) as c:
        r = c.post("/v1/chat/completions", json={"messages": [user("anything at all here")]})
    assert r.status_code == 200 and len(backend.requests[0]["tools"]) == 5
    with make(FakeBackend(), Boom(), fallback="none") as c2:
        c2.post("/v1/chat/completions", json={"messages": [user("anything at all here")]})


def test_max_tools_caps_selector_output(backend):
    with make(backend, Pick("pm_search", "pm_export", "pm_gene"), max_tools=2) as c:
        c.post("/v1/chat/completions", json={"messages": [user("anything at all here")]})
    assert len(backend.requests[0]["tools"]) == 2


def test_select_endpoint_passthrough_and_health(backend):
    with make(backend, BM25Selector(top_k=1)) as c:
        r = c.post("/router/select", json={"query": "look up the BRCA1 gene"})
        assert r.json()["selected"] == ["pm_gene"] and r.json()["pool"] == 5
        assert c.get("/props").json() == {"hello": "props"}
        assert c.get("/tools").status_code == 200
        assert c.get("/router/health").json()["selector"] == "bm25"


def test_agent_mode_runs_server_tools_until_answer():
    script = [{"role": "assistant", "content": None, "tool_calls": [call("pm_export", '{"q": "x"}')]}, {"role": "assistant", "content": "done"}]
    b = FakeBackend(script)
    with make(b, Pick("pm_export"), mode="agent") as c:
        r = c.post("/v1/chat/completions", json={"messages": [user("export my papers")]})
    data = r.json()
    assert data["choices"][0]["message"]["content"] == "done"
    assert data["router"]["iterations"] == 2 and data["router"]["selected"] == ["pm_export"]
    assert b.executed == [{"tool": "pm_export", "params": {"q": "x"}}]
    assert data["usage"]["total_tokens"] == 24
    tool_msg = b.requests[1]["messages"][-1]
    assert tool_msg == {"role": "tool", "tool_call_id": "c1", "content": "result for pm_export"}


def test_agent_mode_returns_client_tool_calls_untouched():
    script = [{"role": "assistant", "content": None, "tool_calls": [call("my_tool")]}]
    b = FakeBackend(script)
    mine = {"type": "function", "function": {"name": "my_tool", "description": "client side", "parameters": {"type": "object"}}}
    with make(b, Pick("my_tool"), mode="agent") as c:
        r = c.post("/v1/chat/completions", json={"messages": [user("do it for me please")], "tools": [mine]})
    assert r.json()["choices"][0]["finish_reason"] == "tool_calls" and b.executed == []


def test_agent_mode_stream_is_emulated():
    with make(FakeBackend(), mode="agent") as c:
        r = c.post("/v1/chat/completions", json={"messages": [user("hello there friend")], "stream": True})
    lines = [l for l in r.text.split("\n\n") if l]
    assert lines[-1] == "data: [DONE]" and json.loads(lines[0][6:])["choices"][0]["delta"]["content"] == "ok"


def test_helpers():
    assert last_user_query([user("a long enough first question"), {"role": "assistant", "content": "x"}, user("ok")]) == "a long enough first question\nok"
    assert last_user_query([{"role": "user", "content": [{"type": "text", "text": "hello world, this is long"}]}]) == "hello world, this is long"
    assert called_tool_names([{"tool_calls": [call("a")]}, {}]) == {"a"}


def test_sticky_keeps_tool_list_stable_across_turns(backend):
    class Seq(Selector):
        name = "seq"

        def __init__(self, picks):
            self.picks = list(picks)

        async def select(self, q, t):
            return Selection(self.picks.pop(0))

    sel = Seq([["pm_export"], ["pm_gene"], ["pm_export"], ["pm_search"]])
    base = [{"role": "system", "content": "s"}, user("first question about export")]
    names = lambda i: [t["function"]["name"] for t in backend.requests[i]["tools"]]
    with make(backend, sel, sticky=True) as c:
        c.post("/v1/chat/completions", json={"messages": base})
        c.post("/v1/chat/completions", json={"messages": base + [user("and now genes")]})
        c.post("/v1/chat/completions", json={"messages": base + [user("export again")]})
        c.post("/v1/chat/completions", json={"messages": [user("a different conversation entirely")]})
    assert names(0) == ["pm_export"]
    assert names(1) == ["pm_export", "pm_gene"]  # grows, order kept
    assert names(2) == ["pm_export", "pm_gene"]  # subset of previous -> identical list (cache hit)
    assert names(3) == ["pm_search"]  # other conversation is independent


def test_default_follows_selector_exactly(backend):
    sel = Pick("pm_export")
    with make(backend, sel) as c:
        c.post("/v1/chat/completions", json={"messages": [user("first question about export")]})
        sel.names = ["pm_gene"]
        c.post("/v1/chat/completions", json={"messages": [user("first question about export")]})
    assert [t["function"]["name"] for t in backend.requests[1]["tools"]] == ["pm_gene"]


def test_abstain_sends_no_tools_and_drops_tool_choice(backend):
    class Quiet(Selector):
        name = "quiet"

        async def select(self, q, t):
            return Selection([], {}, abstain=True)

    with make(backend, Quiet()) as c:
        r = c.post("/v1/chat/completions", json={"messages": [user("thanks, that is all for today")], "tool_choice": "auto"})
    assert r.status_code == 200 and "tools" not in backend.requests[0] and "tool_choice" not in backend.requests[0]
    assert r.headers["x-router-tools"] == ""


class Ranked(Selector):
    name = "ranked"

    async def select(self, query, tools):
        return Selection(["pm_export"], {}, ranking=["pm_gene", "pm_export", "pm_search"], hint="exporting citations (90%)")


def test_reorder_sends_all_tools_most_relevant_first(backend):
    with make(backend, Ranked(), apply="reorder") as c:
        c.post("/v1/chat/completions", json={"messages": [user("export my citations please")]})
    names = [t["function"]["name"] for t in backend.requests[0]["tools"]]
    assert len(names) == 5 and names[:3] == ["pm_gene", "pm_export", "pm_search"]


def test_hint_is_appended_to_last_user_message_only(backend):
    msgs = [{"role": "system", "content": "sys"}, user("first question here"), {"role": "assistant", "content": "a"}, user("export my citations")]
    with make(backend, Ranked(), hint=True) as c:
        c.post("/v1/chat/completions", json={"messages": msgs})
    sent = backend.requests[0]["messages"]
    assert sent[1]["content"] == "first question here" and sent[0]["content"] == "sys"
    assert sent[3]["content"].startswith("export my citations") and "exporting citations (90%)" in sent[3]["content"]
    assert [t["function"]["name"] for t in backend.requests[0]["tools"]] == ["pm_export"]


def test_apply_all_with_hint_keeps_every_tool(backend):
    with make(backend, Ranked(), apply="all", hint=True) as c:
        c.post("/v1/chat/completions", json={"messages": [user("export my citations please")]})
    assert len(backend.requests[0]["tools"]) == 5 and "Routing hint" in backend.requests[0]["messages"][0]["content"]


def test_laya_selection_carries_ranking_and_hint():
    import httpx

    def handler(request):
        return httpx.Response(200, json={"answers": {"tool_group": {"probabilities": {"export": 0.1, "gene": 0.7, "search": 0.2}}}})

    from conftest import GROUPS, TOOLS
    import asyncio

    sel = LayaSelector(groups=GROUPS, views="single", max_groups=1, transport=httpx.MockTransport(handler))
    res = asyncio.run(sel.select("brca1", TOOLS))
    assert res.names == ["pm_gene"] and res.ranking[:3] == ["pm_gene", "pm_search", "pm_export"] and "genes (70%)" in res.hint
    assert set(res.ranking) == {t["function"]["name"] for t in TOOLS}


# ------------------------------------------------------------------ escalation via catalog meta-tool
from llama_mcp_router.proxy import META_TOOL, catalog_tool, expand_tools, requested_tools  # noqa: E402


def names_of(tools):
    return [t["function"]["name"] for t in tools]


def test_catalog_and_expand_helpers():
    cat = catalog_tool(TOOLS, TOOLS[:1])
    fn = cat["function"]
    assert fn["name"] == META_TOOL and "pm_search" not in fn["description"] and "- pm_gene:" in fn["description"]
    assert fn["parameters"]["properties"]["names"]["items"]["enum"] == ["pm_export", "pm_gene", "pm_icd", "fs_read"]
    assert catalog_tool(TOOLS, TOOLS) is None
    assert names_of(expand_tools(TOOLS, TOOLS[:1] + [cat], ["pm_gene"])) == ["pm_search", "pm_gene"]
    assert len(expand_tools(TOOLS, TOOLS[:1], ["nope"])) == 5  # nothing valid -> everything
    assert requested_tools('{"names": ["a", 3]}') == ["a"] and requested_tools("not json") == [] and requested_tools({"names": "x"}) == []


def test_escalation_json_reruns_with_requested_tools():
    script = [{"role": "assistant", "content": None, "tool_calls": [call(META_TOOL, '{"names": ["pm_gene"]}')]},
              {"role": "assistant", "content": None, "tool_calls": [call("pm_gene", '{"q": "BRCA1"}')]}]
    b = FakeBackend(script)
    with make(b, Pick("pm_export"), escalate=True) as c:
        r = c.post("/v1/chat/completions", json={"messages": [user("look up the BRCA1 gene")]})
    assert names_of(b.requests[0]["tools"]) == ["pm_export", META_TOOL]
    assert names_of(b.requests[1]["tools"]) == ["pm_export", "pm_gene"]  # max_escalations=1 -> no catalog in the rerun
    d = r.json()
    assert d["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "pm_gene"
    assert d["usage"]["total_tokens"] == 24 and r.headers["x-router-loaded"] == "pm_gene"
    assert b.requests[1]["messages"] == b.requests[0]["messages"]  # rerun from the original conversation


def test_escalation_not_triggered_by_real_tool_and_absent_when_nothing_left(backend):
    b = FakeBackend([{"role": "assistant", "content": None, "tool_calls": [call("pm_export", "{}")]}])
    with make(b, Pick("pm_export"), escalate=True) as c:
        r = c.post("/v1/chat/completions", json={"messages": [user("export my papers to bibtex")]})
    assert len(b.requests) == 1 and r.json()["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "pm_export"
    with make(backend, Pick("pm_export"), escalate=True, apply="all") as c:
        c.post("/v1/chat/completions", json={"messages": [user("export my papers to bibtex")]})
    assert META_TOOL not in names_of(backend.requests[0]["tools"])


def test_escalation_with_abstain_sends_only_the_catalog(backend):
    class Quiet(Selector):
        name = "quiet"

        async def select(self, q, t):
            return Selection([], {}, abstain=True)

    with make(backend, Quiet(), escalate=True) as c:
        c.post("/v1/chat/completions", json={"messages": [user("hello, how are you today?")]})
    assert names_of(backend.requests[0]["tools"]) == [META_TOOL]


def _events(text):
    return [json.loads(e[6:]) for e in text.split("\n\n") if e.startswith("data: {")]


def test_escalation_stream_swallows_meta_round_and_streams_rerun():
    script = [{"role": "assistant", "reasoning_content": "need gene tools", "content": None, "tool_calls": [call(META_TOOL, '{"names": ["pm_gene"]}')]},
              {"role": "assistant", "reasoning_content": "now call it", "content": None, "tool_calls": [call("pm_gene", '{"q": "BRCA1"}')]}]
    b = FakeBackend(script)
    with make(b, Pick("pm_export"), escalate=True) as c:
        r = c.post("/v1/chat/completions", json={"messages": [user("look up the BRCA1 gene")], "stream": True})
    ev = _events(r.text)
    text = r.text
    assert META_TOOL not in text.replace("loading tools", "")  # the meta call itself never reaches the client
    reasoning = "".join((e["choices"][0]["delta"].get("reasoning_content") or "") for e in ev)
    assert "need gene tools" in reasoning and "router: loading tools pm_gene" in reasoning and "now call it" in reasoning
    names = [tc["function"]["name"] for e in ev for tc in (e["choices"][0]["delta"].get("tool_calls") or []) if (tc.get("function") or {}).get("name")]
    args = "".join(tc["function"].get("arguments", "") for e in ev for tc in (e["choices"][0]["delta"].get("tool_calls") or []))
    assert names == ["pm_gene"] and json.loads(args) == {"q": "BRCA1"}
    assert text.rstrip().endswith("data: [DONE]") and text.count("[DONE]") == 1
    assert [e["choices"][0]["finish_reason"] for e in ev if e["choices"][0]["finish_reason"]] == ["tool_calls"]


def test_escalation_stream_passes_normal_answers_through_unchanged():
    b = FakeBackend([{"role": "assistant", "reasoning_content": "hm", "content": "just text", "tool_calls": None}])
    with make(b, Pick("pm_export"), escalate=True) as c:
        r = c.post("/v1/chat/completions", json={"messages": [user("tell me a joke please")], "stream": True})
    ev = _events(r.text)
    assert "".join(e["choices"][0]["delta"].get("content") or "" for e in ev) == "just text" and len(b.requests) == 1


def test_escalation_in_agent_mode():
    script = [{"role": "assistant", "content": None, "tool_calls": [call(META_TOOL, '{"names": ["pm_gene"]}')]},
              {"role": "assistant", "content": None, "tool_calls": [call("pm_gene", '{"q": "x"}')]},
              {"role": "assistant", "content": "BRCA1 is a gene."}]
    b = FakeBackend(script)
    with make(b, Pick("pm_export"), escalate=True, mode="agent") as c:
        d = c.post("/v1/chat/completions", json={"messages": [user("look up the BRCA1 gene")]}).json()
    assert d["choices"][0]["message"]["content"] == "BRCA1 is a gene." and d["router"]["escalations"] == 1
    assert b.executed == [{"tool": "pm_gene", "params": {"q": "x"}}]
    assert names_of(b.requests[1]["tools"]) == ["pm_export", "pm_gene"]


def test_catalog_mode_with_none_selector(backend):
    from llama_mcp_router import NoneSelector, load_selector

    assert isinstance(load_selector("none"), NoneSelector)
    with make(backend, NoneSelector(), escalate=True) as c:
        c.post("/v1/chat/completions", json={"messages": [user("look up the BRCA1 gene please")]})
    assert names_of(backend.requests[0]["tools"]) == [META_TOOL]
    assert len(backend.requests[0]["tools"][0]["function"]["parameters"]["properties"]["names"]["items"]["enum"]) == 5
    msgs = [user("look up the BRCA1 gene please"), {"role": "assistant", "content": "", "tool_calls": [call("pm_gene")]}, {"role": "tool", "tool_call_id": "c1", "content": "x"}]
    b3 = FakeBackend()
    with make(b3, NoneSelector(), escalate=True) as c:
        c.post("/v1/chat/completions", json={"messages": msgs})
    assert names_of(b3.requests[0]["tools"]) == ["pm_gene", META_TOOL]  # tools already used stay loaded


def test_passthrough_keeps_content_encoding_for_compressed_pages():
    import gzip

    from starlette.applications import Starlette
    from starlette.responses import Response as SResponse
    from starlette.routing import Route

    seen = {}

    async def page(request):
        seen["ae"] = request.headers.get("accept-encoding")
        if "gzip" not in (request.headers.get("accept-encoding") or ""):
            return SResponse("gzip required", status_code=415)
        return SResponse(gzip.compress(b"<html>ui</html>"), media_type="text/html", headers={"Content-Encoding": "gzip"})

    app = Starlette(routes=[Route("/", page)])
    cfg = RouterConfig(backend="http://backend", selector=Pick())
    with TestClient(create_app(cfg, transport=httpx.ASGITransport(app=app))) as c:
        r = c.get("/", headers={"Accept-Encoding": "gzip"})
        assert r.status_code == 200 and r.headers.get("content-encoding") == "gzip" and r.text == "<html>ui</html>"
        assert seen["ae"] == "gzip"
        r = c.get("/", headers={"Accept-Encoding": "identity"})
        assert r.status_code == 415 and seen["ae"] == "identity"  # same behaviour as talking to the backend directly
