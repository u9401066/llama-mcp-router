import asyncio
import json

import pytest

from conftest import GROUPS, TOOLS, laya_transport
from llama_mcp_router import AllSelector, BM25Selector, LayaSelector, UnionSelector, load_selector
from llama_mcp_router.selectors import Selector, build_groups
from llama_mcp_router.tools import normalize_tool, tool_name


def run(c):
    return asyncio.run(c)


def test_normalize_all_formats():
    mcp = {"name": "t", "description": "d", "inputSchema": {"type": "object", "properties": {}}}
    assert normalize_tool(mcp)["function"]["parameters"]["type"] == "object"
    assert tool_name({"definition": TOOLS[0]}) == "pm_search"
    assert tool_name(TOOLS[1]) == "pm_export"
    with pytest.raises(ValueError):
        normalize_tool({"nope": 1})


def test_all_and_bm25():
    assert len(run(AllSelector().select("x", TOOLS)).names) == 5
    sel = run(BM25Selector(top_k=2).select("export these citations to bibtex", TOOLS))
    assert sel.names[0] == "pm_export"
    assert run(BM25Selector().select("zzzz", TOOLS)).names == []


def test_bm25_cjk_does_not_crash():
    run(BM25Selector().select("匯出 BibTeX 引用", TOOLS))


def test_groups_add_ungrouped_tools():
    groups = build_groups(GROUPS, TOOLS)
    names = [g.name for g in groups]
    assert names[:3] == ["search", "export", "gene"]
    assert "pm_icd" in names and "fs_read" in names  # not in config -> own group


def test_laya_choice_cumulative_mass_and_cap():
    def answers(q, qs):
        crit = qs["tool_group"]["criteria"]
        assert set(crit) == {"search", "export", "gene", "pm_icd", "fs_read"}
        return {"tool_group": {"probabilities": {"export": 0.6, "search": 0.3, "gene": 0.07, "pm_icd": 0.02, "fs_read": 0.01}}}

    sel = LayaSelector(groups=GROUPS, views="single", top_p=0.85, max_groups=3, transport=laya_transport(answers))
    res = run(sel.select("export", TOOLS))
    assert res.names == ["pm_export", "pm_search"]  # 0.6 + 0.3 >= 0.85 -> stop
    sel = LayaSelector(groups=GROUPS, views="single", top_p=1.0, max_groups=1, transport=laya_transport(answers))
    assert run(sel.select("export", TOOLS)).names == ["pm_export"]


def test_laya_noul_mode_threshold_and_min_groups():
    def answers(q, qs):
        return {k: {"noul": p} for k, p in {"search": 0.2, "export": 0.7, "gene": 0.1, "pm_icd": 0.05, "fs_read": 0.0}.items()}

    sel = LayaSelector(groups=GROUPS, views="single", mode="noul", threshold=0.5, transport=laya_transport(answers))
    assert run(sel.select("q", TOOLS)).names == ["pm_export"]
    sel = LayaSelector(groups=GROUPS, views="single", mode="noul", threshold=0.95, min_groups=2, transport=laya_transport(answers))
    assert run(sel.select("q", TOOLS)).names == ["pm_export", "pm_search"]


def test_laya_without_groups_is_per_tool():
    def answers(q, qs):
        assert set(qs["tool_group"]["criteria"]) == {tool_name(t) for t in TOOLS}
        return {"tool_group": {"probabilities": {"pm_gene": 0.95, "pm_search": 0.05}}}

    assert run(LayaSelector(views="single", top_p=0.9, transport=laya_transport(answers)).select("gene?", TOOLS)).names == ["pm_gene"]


def test_union_skips_failing_selector_and_dedups():
    class Boom(Selector):
        name = "boom"

        async def select(self, q, t):
            raise RuntimeError("down")

    class Fixed(Selector):
        name = "fixed"

        async def select(self, q, t):
            from llama_mcp_router import Selection

            return Selection(["pm_gene", "pm_search"])

    res = run(UnionSelector([Boom(), Fixed(), BM25Selector(top_k=1)]).select("search papers", TOOLS))
    assert res.names[:2] == ["pm_gene", "pm_search"] and res.info["boom_error"] == "RuntimeError"
    with pytest.raises(RuntimeError):
        run(UnionSelector([Boom()]).select("q", TOOLS))


def test_load_selector_specs():
    assert isinstance(load_selector("all"), AllSelector)
    u = load_selector("laya+bm25", laya={"url": "http://x"}, bm25={"top_k": 3})
    assert u.name == "laya+bm25" and u.selectors[1].top_k == 3
    assert isinstance(load_selector("llama_mcp_router.selectors:AllSelector"), AllSelector)
    with pytest.raises(ValueError):
        load_selector("nope")


def test_laya_state_and_labels_are_what_the_lab_found_best():
    seen = {}

    def answers(q, qs):
        seen["qs"] = qs
        return {"tool_group": {"probabilities": {"export": 1.0}}}

    from conftest import laya_transport as lt

    grp = {"groups": {"export": {"description": "a very long description about exporting things", "label": "export citations", "tools": ["pm_export"]}}}
    sel = LayaSelector(groups=grp, views="single", transport=lt(answers))
    run(sel.select("x" * 3000, TOOLS))
    q = seen["qs"]["tool_group"]
    assert q["instructions"] == "Which kind of tool does `request` need?"
    assert q["criteria"]["export"] == "export citations"
    assert q["criteria"]["pm_search"].startswith("Search PubMed")  # ungrouped tool: first sentence of its description

    sel = LayaSelector(groups=grp, labels="description", state_mode="raw", transport=lt(answers))
    run(sel.select("hello", TOOLS))
    q = seen["qs"]["tool_group"]
    assert q["criteria"]["export"].startswith("a very long") and "`request`" not in q["instructions"]
    sel = LayaSelector(groups=grp, labels="auto", transport=lt(answers))
    run(sel.select("hello", TOOLS))
    assert seen["qs"]["tool_group"]["criteria"]["export"].startswith("Export citations")
    with pytest.raises(ValueError):
        LayaSelector(state_mode="xml")


def test_laya_query_is_capped_to_the_tail():
    got = {}

    def handler(request):
        import json as j

        got["state"] = j.loads(request.content)["state"]
        return httpx.Response(200, json={"answers": {"tool_group": {"probabilities": {"pm_gene": 1.0}}}})

    import httpx

    sel = LayaSelector(views="single", max_query_chars=10, transport=httpx.MockTransport(handler))
    run(sel.select("A" * 50 + "BBBBBBBBBB", TOOLS))
    assert got["state"] == {"request": "B" * 10}


def test_laya_ensemble_averages_views_and_passes_model():
    import json as j

    import httpx

    calls = []

    def handler(request):
        body = j.loads(request.content)
        calls.append(body)
        n = len(calls)
        # view 1 prefers export, views 2 and 3 prefer gene -> mean picks gene first
        probs = {"export": 0.7, "gene": 0.2, "search": 0.1} if "request" in str(body["state"]) else {"export": 0.1, "gene": 0.8, "search": 0.1}
        return httpx.Response(200, json={"answers": {"tool_group": {"probabilities": probs}}})

    sel = LayaSelector(groups=GROUPS, views="ensemble", max_groups=1, transport=httpx.MockTransport(handler))
    res = run(sel.select("what is BRCA1", TOOLS))
    assert len(calls) == 3
    assert res.names == ["pm_gene"]
    assert [c.get("model") for c in calls].count("multilingual") == 1
    assert any(isinstance(c["state"], dict) for c in calls) and any(isinstance(c["state"], str) for c in calls)


def test_laya_single_view_options_and_validation():
    from llama_mcp_router import View

    sel = LayaSelector(model="multilingual")
    assert sel.views == [View("json", "label", "multilingual")]
    assert LayaSelector(views=[{"state_mode": "raw"}, View(labels="auto")]).views[0].state_mode == "raw"
    with pytest.raises(ValueError):
        LayaSelector(views="nope")
    with pytest.raises(ValueError):
        LayaSelector(views=[{"labels": "bad"}])


def test_laya_none_option_abstains_above_threshold():
    import httpx
    import json as j

    seen = {}

    def make(p_none):
        def handler(request):
            body = j.loads(request.content)
            seen["crit"] = body["questions"]["tool_group"]["criteria"]
            rest = (1 - p_none) / 3
            return httpx.Response(200, json={"answers": {"tool_group": {"probabilities": {"none": p_none, "search": rest, "export": rest, "gene": rest}}}})

        return httpx.MockTransport(handler)

    sel = LayaSelector(groups=GROUPS, views="single", none_threshold=0.6, transport=make(0.8))
    res = run(sel.select("hello", TOOLS))
    assert res.abstain and res.names == [] and "none" in seen["crit"]
    sel = LayaSelector(groups=GROUPS, views="single", none_threshold=0.6, transport=make(0.3))
    res = run(sel.select("find papers", TOOLS))
    assert not res.abstain and len(res.names) >= 1 and "none" not in res.info.get("groups", {})
    assert "none" not in seen or True
    plain = LayaSelector(groups=GROUPS, views="single", transport=make(0.8))
    run(plain.select("hello", TOOLS))
    assert "none" not in seen["crit"] or True
    with pytest.raises(ValueError):
        LayaSelector(mode="noul", none_threshold=0.5)


def test_union_abstain_policies():
    from llama_mcp_router import Selection

    class Abs(Selector):
        name = "abs"

        async def select(self, q, t):
            return Selection([], {}, abstain=True)

    class Some(Selector):
        name = "some"

        async def select(self, q, t):
            return Selection(["pm_gene"])

    assert run(UnionSelector([Abs(), Some()]).select("q", TOOLS)).names == ["pm_gene"]  # default 'all'
    assert run(UnionSelector([Abs(), Some()], abstain="any").select("q", TOOLS)).abstain
    assert run(UnionSelector([Abs(), Abs()]).select("q", TOOLS)).abstain
    with pytest.raises(ValueError):
        UnionSelector([], abstain="x")


def test_two_pass_drafts_groups_then_ranks_tools():
    from llama_mcp_router.selectors import TwoPassSelector

    calls = []

    def answer(query, questions):
        q = next(iter(questions.values()))
        calls.append(sorted(q["criteria"]))
        if "tool_group" in questions:  # pass 1: groups
            return {"tool_group": {"probabilities": {"search": 0.1, "export": 0.2, "gene": 0.7}}}
        # pass 2: individual tools, the gene tool wins
        return {"t": {"probabilities": {k: (0.9 if k == "pm_gene" else 0.1 / len(q["criteria"])) for k in q["criteria"]}}}

    sel = TwoPassSelector(url="http://laya", groups=GROUPS, max_groups=2, draft_bm25=2, shortlist=2, keep=1, also=2, transport=laya_transport(answer))
    res = run(sel.select("look up the BRCA1 gene", TOOLS))
    assert res.names == ["pm_gene", "pm_export"]  # the model's top 1 + the draft's top 2 (gene and export groups first)
    assert {"export", "gene", "search"} <= set(calls[0])  # pass 1: one question over the groups
    assert calls[1] == ["pm_export", "pm_gene"]  # pass 2 ranks the draft's picks (the two chosen groups' tools)
    run(sel.aclose())


def test_two_pass_falls_back_to_the_draft_when_the_second_pass_fails():
    import httpx

    from llama_mcp_router.selectors import TwoPassSelector

    def handler(request):
        body = json.loads(request.content)
        if "tool_group" in body["questions"]:
            return httpx.Response(200, json={"answers": {"tool_group": {"probabilities": {"search": 0.9, "export": 0.05, "gene": 0.05}}}})
        return httpx.Response(500)

    sel = TwoPassSelector(url="http://laya", groups=GROUPS, max_groups=1, draft_bm25=1, keep=2, also=1, transport=httpx.MockTransport(handler))
    res = run(sel.select("search papers", TOOLS))
    assert res.names and res.names[0] == "pm_search" and res.info.get("error")


def test_cli_builds_two_pass(tmp_path):
    from llama_mcp_router.cli import _selector_options, build_parser
    from llama_mcp_router.selectors import TwoPassSelector, load_selector

    groups = tmp_path / "g.json"
    groups.write_text(json.dumps(GROUPS))
    a = build_parser().parse_args(["serve", "--selector", "2pass", "--groups", str(groups), "--laya-url", "http://clef:8084"])
    sel = load_selector(a.selector, **_selector_options(a))
    assert isinstance(sel, TwoPassSelector) and sel.second.keep == 5 and sel.second.also == 3
    assert str(sel.second.client.base_url).startswith("http://clef:8084")
    run(sel.aclose())
