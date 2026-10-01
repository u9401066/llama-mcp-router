import json

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route


def mk(name, desc):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}


TOOLS = [
    mk("pm_search", "Search PubMed for papers about a topic"),
    mk("pm_export", "Export citations to BibTeX or RIS reference manager formats"),
    mk("pm_gene", "Look up a gene in NCBI Gene"),
    mk("pm_icd", "Convert ICD diagnosis codes to MeSH terms"),
    mk("fs_read", "Read a file from disk"),
]
GROUPS = {
    "always": [],
    "groups": {
        "search": {"description": "searching literature", "tools": ["pm_search"]},
        "export": {"description": "exporting citations", "tools": ["pm_export"]},
        "gene": {"description": "genes", "tools": ["pm_gene"]},
    },
}


class FakeBackend:
    """Stands in for llama-server: /tools, /v1/chat/completions, /tools execution."""

    def __init__(self, script=None):
        self.requests = []
        self.executed = []
        self.script = list(script or [])

    def app(self):
        async def tools(request):
            return JSONResponse([{"type": "mcp", "tool": t["function"]["name"], "definition": t} for t in TOOLS])

        async def run_tool(request):
            body = await request.json()
            self.executed.append(body)
            return JSONResponse({"plain_text_response": "result for %s" % body["tool"]})

        async def chat(request):
            body = await request.json()
            self.requests.append(body)
            msg = self.script.pop(0) if self.script else {"role": "assistant", "content": "ok"}
            finish = "tool_calls" if msg.get("tool_calls") else "stop"
            return JSONResponse({"id": "x", "model": "fake", "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}})

        async def props(request):
            return JSONResponse({"hello": "props"})

        return Starlette(routes=[Route("/tools", tools, methods=["GET"]), Route("/tools", run_tool, methods=["POST"]), Route("/v1/chat/completions", chat, methods=["POST"]), Route("/props", props)])


def call(name, args="{}", id="c1"):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": args}}


@pytest.fixture
def backend():
    return FakeBackend()


def laya_transport(answer_for):
    """MockTransport imitating laya-serve; answer_for(query, questions) -> answers dict."""

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        state = body["state"]
        return httpx.Response(200, json={"answers": answer_for(state["request"] if isinstance(state, dict) else state, body["questions"]), "_body": body})

    return httpx.MockTransport(handler)
