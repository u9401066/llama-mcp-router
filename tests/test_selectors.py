import asyncio

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

    sel = LayaSelector(groups=GROUPS, top_p=0.85, max_groups=3, transport=laya_transport(answers))
    res = run(sel.select("export", TOOLS))
    assert res.names == ["pm_export", "pm_search"]  # 0.6 + 0.3 >= 0.85 -> stop
    sel = LayaSelector(groups=GROUPS, top_p=1.0, max_groups=1, transport=laya_transport(answers))
    assert run(sel.select("export", TOOLS)).names == ["pm_export"]


def test_laya_noul_mode_threshold_and_min_groups():
    def answers(q, qs):
        return {k: {"noul": p} for k, p in {"search": 0.2, "export": 0.7, "gene": 0.1, "pm_icd": 0.05, "fs_read": 0.0}.items()}

    sel = LayaSelector(groups=GROUPS, mode="noul", threshold=0.5, transport=laya_transport(answers))
    assert run(sel.select("q", TOOLS)).names == ["pm_export"]
    sel = LayaSelector(groups=GROUPS, mode="noul", threshold=0.95, min_groups=2, transport=laya_transport(answers))
    assert run(sel.select("q", TOOLS)).names == ["pm_export", "pm_search"]


def test_laya_without_groups_is_per_tool():
    def answers(q, qs):
        assert set(qs["tool_group"]["criteria"]) == {tool_name(t) for t in TOOLS}
        return {"tool_group": {"probabilities": {"pm_gene": 0.95, "pm_search": 0.05}}}

    assert run(LayaSelector(top_p=0.9, transport=laya_transport(answers)).select("gene?", TOOLS)).names == ["pm_gene"]


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
    sel = LayaSelector(groups=grp, transport=lt(answers))
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

    sel = LayaSelector(max_query_chars=10, transport=httpx.MockTransport(handler))
    run(sel.select("A" * 50 + "BBBBBBBBBB", TOOLS))
    assert got["state"] == {"request": "B" * 10}
