import json

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
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


def sse(msg, finish):
    """Split a message into OpenAI stream chunks the way llama-server does (reasoning, content, tool calls by piece)."""

    def ev(delta, fin=None):
        return "data: " + json.dumps({"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}) + "\n\n"

    out = [ev({"role": "assistant"})]
    if msg.get("reasoning_content"):
        out.append(ev({"reasoning_content": msg["reasoning_content"]}))
    if msg.get("content"):
        out.append(ev({"content": msg["content"]}))
    for i, c in enumerate(msg.get("tool_calls") or []):
        out.append(ev({"tool_calls": [{"index": i, "id": c["id"], "type": "function", "function": {"name": c["function"]["name"], "arguments": ""}}]}))
        a = c["function"]["arguments"]
        for j in range(0, len(a), 5):
            out.append(ev({"tool_calls": [{"index": i, "function": {"arguments": a[j : j + 5]}}]}))
    out.append(ev({}, finish))
    out.append("data: [DONE]\n\n")
    return out


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
            if body.get("stream"):
                return StreamingResponse(sse(msg, finish), media_type="text/event-stream")
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
