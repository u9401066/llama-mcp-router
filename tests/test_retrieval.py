import asyncio
import json

import httpx
import pytest

from conftest import TOOLS, mk
from llama_mcp_router import HTTPEmbedder, LayaRerankSelector, Retriever, RetrieverSelector, load_selector
from llama_mcp_router.retrieval import is_cjk, tool_text

VOCAB = ["search", "pubmed", "papers", "export", "bibtex", "gene", "icd", "mesh", "file", "disk", "文獻", "匯出", "基因"]


def fake_vec(text):
    t = text.lower()
    return [float(t.count(w)) for w in VOCAB] + [0.01]


def embed_transport(calls):
    def handler(request):
        body = json.loads(request.content)
        calls.append(len(body["input"]))
        return httpx.Response(200, json={"data": [{"index": i, "embedding": fake_vec(x)} for i, x in enumerate(body["input"])]})

    return httpx.MockTransport(handler)


def run(c):
    return asyncio.run(c)


def test_tool_text_and_cjk():
    assert tool_text(TOOLS[0]).startswith("pm search: Search PubMed")
    assert is_cjk("找基因") and is_cjk("ジーン") and not is_cjk("gene")


def test_bm25_only_retriever():
    r = Retriever()
    assert run(r.rank("export bibtex", TOOLS))[0] == "pm_export"
    assert run(r.rank("x", [])) == []


def test_embedding_retriever_caches_docs_and_uses_embeddings_for_cjk():
    calls = []
    r = Retriever(HTTPEmbedder("http://e", transport=embed_transport(calls)))
    assert run(r.rank("匯出 bibtex", TOOLS))[0] == "pm_export"  # CJK -> embeddings only
    first = list(calls)
    run(r.rank("look up a gene", TOOLS))
    assert calls[len(first):] == [1]  # only the query is embedded the second time (docs cached)
    ranked = run(r.rank("search pubmed papers", TOOLS))
    assert ranked[0] == "pm_search" and set(ranked) == {t["function"]["name"] for t in TOOLS}


def test_retriever_selector_and_registry():
    sel = RetrieverSelector(top_k=2)
    res = run(sel.select("export to bibtex", TOOLS))
    assert res.names[0] == "pm_export" and len(res.names) <= 2 and res.ranking[0] == "pm_export"
    assert isinstance(load_selector("laya-rerank", **{"laya-rerank": {"url": "http://l"}}), LayaRerankSelector)
    assert isinstance(load_selector("retrieve"), RetrieverSelector)


def laya_pick(prefer):
    seen = []

    def handler(request):
        body = json.loads(request.content)
        crit = body["questions"]["t"]["criteria"]
        seen.append(crit)
        n = len(crit)
        probs = {k: (0.9 if k == prefer else 0.1 / max(1, n - 1)) for k in crit}
        return httpx.Response(200, json={"answers": {"t": {"probabilities": probs}}})

    return httpx.MockTransport(handler), seen


def big_pool(n):
    return TOOLS + [mk("srv%d_tool%d" % (i % 7, i), "Generic operation number %d on things" % i) for i in range(n)]


def test_laya_rerank_chunks_keeps_and_unions():
    pool = big_pool(60)
    tr, seen = laya_pick("pm_icd")
    sel = LayaRerankSelector(url="http://l", shortlist=24, keep=2, also=3, chunk=12, transport=tr)
    res = run(sel.select("search pubmed papers about genes", pool))
    assert len(seen) == 3 and all(len(c) <= 12 for c in seen)  # two chunks + final round
    assert res.names[0] == "pm_icd" and len(res.names) <= 5
    assert all(":" in v for v in seen[0].values()) and "pm search:" in "".join(seen[0].values())


def test_laya_rerank_falls_back_to_retriever_when_laya_fails():
    def boom(request):
        return httpx.Response(503, json={"detail": "busy"})

    sel = LayaRerankSelector(url="http://l", keep=2, also=2, transport=httpx.MockTransport(boom))
    res = run(sel.select("export to bibtex", TOOLS))
    assert res.info["error"] == "HTTPStatusError" and res.names[0] == "pm_export" and len(res.names) == 4


def test_laya_rerank_single_chunk_and_tiny_pool():
    tr, seen = laya_pick("pm_gene")
    sel = LayaRerankSelector(url="http://l", keep=1, also=1, transport=tr)
    res = run(sel.select("look up gene", TOOLS))
    assert len(seen) == 1 and res.names[0] == "pm_gene"


def test_sanitize_schema_inlines_nested_defs_and_drops_huge_limits():
    from llama_mcp_router.tools import sanitize_schema, sanitize_tool

    nested = {"type": "object", "properties": {"manifest": {"type": "object", "$defs": {"Status": {"type": "string", "enum": ["a", "b"]}},
              "properties": {"status": {"$ref": "#/$defs/Status", "description": "d"}}}}}
    out = sanitize_schema(nested)
    st = out["properties"]["manifest"]["properties"]["status"]
    assert st == {"type": "string", "enum": ["a", "b"], "description": "d"} and "$defs" not in json.dumps(out)
    rec = {"type": "object", "properties": {"node": {"$ref": "#/$defs/Node"}}, "$defs": {"Node": {"type": "object", "properties": {"child": {"$ref": "#/$defs/Node"}}}}}
    r = sanitize_schema(rec, max_depth=3)
    assert "$ref" not in json.dumps(r)  # recursion terminates, unconstrained at the end
    limits = {"type": "string", "maxLength": 2000, "minLength": 1, "items": {"maxItems": 5}}
    assert sanitize_schema(limits) == {"type": "string", "minLength": 1, "items": {"maxItems": 5}}
    t = mk("x", "y")
    assert sanitize_tool(t) == t  # untouched when nothing to fix
    assert sanitize_schema({"$ref": "#/$defs/Missing", "description": "z"}) == {"description": "z"}
