import json

import httpx
from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from conftest import FakeBackend
from llama_mcp_router.proxy import RouterConfig, create_app
from llama_mcp_router.selectors import AllSelector
from llama_mcp_router.sentinel import Check, SentinelConfig

RENDERED = "<|im_start|>user\ncase<|im_end|>\n<|im_start|>assistant\n<think>\n"


class Gen:
    """llama-server stand-in: /apply-template and a streamed /completion whose text depends on the prompt."""

    def __init__(self, script):
        self.script, self.prompts = script, []

    def app(self):
        async def apply(request):
            return JSONResponse({"prompt": RENDERED})

        async def completion(request):
            body = await request.json()
            prompt = body["prompt"]
            self.prompts.append(prompt[len(RENDERED):])
            text = self.script(prompt[len(RENDERED):])

            async def gen():
                for i in range(0, len(text), 7):
                    yield "data: %s\n\n" % json.dumps({"content": text[i:i + 7], "stop": False, "timings": {"prompt_n": 10, "prompt_ms": 5.0, "predicted_n": i // 7 + 1, "predicted_ms": 10.0 * (i // 7 + 1)}})
                yield "data: %s\n\n" % json.dumps({"content": "", "stop": True, "timings": {"prompt_n": 10, "prompt_ms": 5.0, "predicted_n": 50, "predicted_ms": 500.0}})

            return StreamingResponse(gen(), media_type="text/event-stream")

        async def chat(request):
            return JSONResponse({"choices": [{"index": 0, "message": {"role": "assistant", "content": "plain"}, "finish_reason": "stop"}]})

        return Starlette(routes=[Route("/apply-template", apply, methods=["POST"]), Route("/completion", completion, methods=["POST"]),
                                 Route("/v1/chat/completions", chat, methods=["POST"]), Route("/v1/models", lambda r: JSONResponse({"data": []}))])


def critic(flag):
    """Fake SystemOne critic: flag(step) -> {check: probability}."""
    seen = []

    def handler(request):
        body = json.loads(request.content)
        step = body["state"]["reasoning_step"]
        seen.append(body["state"])
        probs = flag(step)
        return httpx.Response(200, json={"answers": {q: {"type": "noul", "noul": probs.get(q, 0.01)} for q in body["questions"]}})

    return httpx.MockTransport(handler), seen


CHECKS = [Check(id="red_flag", question="red flag missed?", action="rollback", correction="I must first rule out the dangerous causes."),
          Check(id="closure", question="premature closure?", action="inject", correction="Wait, let me list the alternatives."),
          Check(id="contra", question="contraindication?", action="rollback", correction="I must check contraindications first.")]


def make(script, flag, **kw):
    gen = Gen(script)
    transport, seen = critic(flag)
    cfg = RouterConfig(backend="http://backend", selector=AllSelector(), sentinel=SentinelConfig(checks=CHECKS, min_step_chars=10, **kw))
    app = create_app(cfg, transport=httpx.ASGITransport(app=gen.app()))
    app.state.router.critic.client = httpx.AsyncClient(base_url="http://critic", transport=transport)
    return TestClient(app), gen, seen


def ask(c, text="med: 58M chest pain, sweating", stream=True):
    r = c.post("/v1/chat/completions", json={"model": "local", "stream": stream, "messages": [{"role": "user", "content": text}]})
    assert r.status_code == 200
    if not stream:
        return r.json()
    chunks = [json.loads(e[6:]) for e in r.text.split("\n\n") if e.startswith("data: {")]
    reasoning = "".join(ch["choices"][0]["delta"].get("reasoning_content") or "" for ch in chunks)
    content = "".join(ch["choices"][0]["delta"].get("content") or "" for ch in chunks)
    return reasoning, content, chunks[-1]


GOOD = "Rule out ACS and aortic dissection first.\n\nPlan: ECG and troponin now.\n</think>\n\nMost likely ACS until proven otherwise."


def test_clean_reasoning_passes_untouched_and_the_case_reaches_the_critic():
    c, gen, seen = make(lambda p: GOOD, lambda s: {})
    with c:
        reasoning, content, last = ask(c)
    assert reasoning == "Rule out ACS and aortic dissection first.\n\nPlan: ECG and troponin now.\n"
    assert content == "Most likely ACS until proven otherwise." and "[sentinel" not in reasoning
    assert last["sentinel"]["interventions"] == [] and last["sentinel"]["steps_checked"] == 3  # 2 steps + 1 answer paragraph
    assert last["timings"]["predicted_n"] == 50 and last["choices"][0]["finish_reason"] == "stop"
    assert seen[0]["case"] == "58M chest pain, sweating"  # trigger prefix stripped
    assert "previous_reasoning" not in seen[0] and seen[1]["previous_reasoning"].startswith("Rule out ACS")


def test_red_flag_step_is_withdrawn_and_regenerated_from_the_accepted_prefix():
    def script(prefix):
        return "It is just a muscle strain, give an NSAID.\n\nmore" if "dangerous causes" not in prefix else GOOD

    c, gen, _ = make(script, lambda s: {"red_flag": 0.93} if "muscle strain" in s else {})
    with c:
        reasoning, content, last = ask(c)
    assert "muscle strain" not in reasoning and "muscle strain" not in content  # the bad step never reaches the client
    assert "[sentinel: red_flag (p=0.93) – step withdrawn" in reasoning and "I must first rule out the dangerous causes." in reasoning
    assert gen.prompts == ["", "I must first rule out the dangerous causes.\n\n"]  # rollback: restart without the bad step
    assert content.startswith("Most likely ACS") and last["sentinel"]["interventions"][0]["action"].startswith("step withdrawn")


def test_closure_gets_an_in_context_reflection():
    first = "This is definitely pneumonia, nothing else.\n\n"

    def script(prefix):
        return first + "more" if not prefix else "Alternatives: PE, heart failure.\n</think>\n\nPneumonia or PE; get CTPA."

    c, gen, _ = make(script, lambda s: {"closure": 0.95} if "definitely pneumonia" in s else {})
    with c:
        reasoning, content, last = ask(c)
    assert reasoning.startswith(first) and "Wait, let me list the alternatives." in reasoning  # the step stays, a reflection follows
    assert gen.prompts[1] == first + "\nWait, let me list the alternatives.\n\n"
    assert content == "Pneumonia or PE; get CTPA."


def test_interventions_are_capped_so_disagreement_cannot_loop():
    c, gen, _ = make(lambda p: "Probably reflux.\n\n</think>\n\nGive antacids.", lambda s: {"red_flag": 0.9} if "eflux" in s or "antacid" in s else {})
    with c:
        reasoning, content, last = ask(c)
    assert len(gen.prompts) == 3  # two rollbacks (max_per_check=2), then the step is let through
    assert "still flagged (p=0.90), intervention limit reached" in reasoning
    assert content.startswith("Give antacids.\n\n> ⚠️ Sentinel: this answer may still contain: red_flag (p=0.90).")  # never released silently
    assert [i["check"] for i in last["sentinel"]["interventions"]] == ["red_flag", "red_flag"]
    assert last["sentinel"]["flagged"] == [{"check": "red_flag", "p": 0.9}]


def test_unsafe_answer_is_withdrawn_and_the_model_thinks_again():
    def script(prefix):
        if "contraindications" not in prefix:
            return "Diabetes with eGFR 20.\n</think>\n\nStart metformin 1000 mg twice daily."
        return "Metformin is contraindicated below eGFR 30.\n</think>\n\nUse insulin or a renally dosed DPP-4 inhibitor."

    c, gen, _ = make(script, lambda s: {"contra": 0.94} if s.startswith("Start metformin") else {})
    with c:
        reasoning, content, last = ask(c)
    assert "metformin 1000" not in content and content == "Use insulin or a renally dosed DPP-4 inhibitor."
    assert "answer withdrawn" in reasoning
    assert gen.prompts[1] == "Diabetes with eGFR 20.\n\nI must check contraindications first.\n\n"


def test_unavailable_critic_fails_open():
    gen = Gen(lambda p: GOOD)
    cfg = RouterConfig(backend="http://backend", selector=AllSelector(), sentinel=SentinelConfig(checks=CHECKS, min_step_chars=10))
    app = create_app(cfg, transport=httpx.ASGITransport(app=gen.app()))
    app.state.router.critic.client = httpx.AsyncClient(base_url="http://critic", transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    with TestClient(app) as c:
        reasoning, content, last = ask(c)
    assert content.startswith("Most likely ACS") and last["sentinel"]["interventions"] == []


def test_routing_model_name_non_stream_and_models_list():
    c, gen, _ = make(lambda p: GOOD, lambda s: {})
    with c:
        d = c.post("/v1/chat/completions", json={"model": "sentinel", "messages": [{"role": "user", "content": "chest pain"}]}).json()
        assert d["choices"][0]["message"]["content"].startswith("Most likely ACS") and d["sentinel"]["steps_checked"] == 3
        assert d["usage"]["completion_tokens"] == 50 and d["choices"][0]["message"]["reasoning_content"].startswith("Rule out ACS")
        plain = c.post("/v1/chat/completions", json={"model": "local", "messages": [{"role": "user", "content": "hello"}]}).json()
        assert plain["choices"][0]["message"]["content"] == "plain"  # no trigger: the normal path
        assert "sentinel" in [m["id"] for m in c.get("/v1/models").json()["data"]]


def test_config_validation(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        SentinelConfig(checks=[])
    with pytest.raises(ValueError):
        Check(id="x", question="q", action="explode")
    p = tmp_path / "s.json"
    p.write_text(open(__file__.replace("tests/test_sentinel.py", "examples/sentinel-medical.json")).read())
    cfg = SentinelConfig.load(str(p))
    assert [c.id for c in cfg.checks] == ["red_flag", "contraindication", "closure"] and cfg.triggers == ["med:"]


def test_reasoning_that_never_ends_gets_its_answer_requested():
    def script(prefix):
        return "Final answer: rule out SAH with CT." if prefix.endswith("</think>\n\n") else "Thinking about SAH and CT.\n\nMore thinking"

    c, gen, _ = make(script, lambda s: {})
    with c:
        reasoning, content, last = ask(c)
    assert content == "Final answer: rule out SAH with CT." and gen.prompts[1].endswith("More thinking\n</think>\n\n")
    assert [a["stop"] for a in last["sentinel"]["attempts"]] == [None, None]


def test_warning_names_every_check_still_flagged():
    c, gen, _ = make(lambda p: "Think.\n</think>\n\nGive succinylcholine and send home.", lambda s: {"red_flag": 0.8, "contra": 0.7} if "succinyl" in s else {},
                     max_interventions=0)
    with c:
        reasoning, content, last = ask(c)
    assert "red_flag (p=0.80), contra (p=0.70)" in content and len(gen.prompts) == 1
    assert last["sentinel"]["flagged"] == [{"check": "red_flag", "p": 0.8}, {"check": "contra", "p": 0.7}]
