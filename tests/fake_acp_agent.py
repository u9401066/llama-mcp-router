"""A tiny ACP v1 agent for tests: writes the prompt into a file in cwd, asks for one permission, answers."""
import json
import os
import select
import sys
import time

sessions = {}


def send(m):
    sys.stdout.write(json.dumps(m) + "\n")
    sys.stdout.flush()


def read():
    line = sys.stdin.readline()
    return json.loads(line) if line else None


def update(sid, u):
    send({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": sid, "update": u}})


while True:
    m = read()
    if m is None:
        break
    if "method" not in m:
        continue
    mid, method, p = m.get("id"), m["method"], m.get("params") or {}
    if method == "initialize":
        time.sleep(float(os.environ.get("FAKE_SLOW_START", "0")))
        send({"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": 1, "agentInfo": {"name": "fake"},
                                                       "agentCapabilities": {"sessionCapabilities": {"resume": {}}}}})
    elif method == "session/new":
        sid = "s%d" % (len(sessions) + 1)
        sessions[sid] = {"cwd": p["cwd"], "turns": 0}
        send({"jsonrpc": "2.0", "id": mid, "result": {"sessionId": sid}})
    elif method == "session/resume":
        if p["sessionId"] in sessions or os.environ.get("FAKE_RESUME_OK"):
            sessions.setdefault(p["sessionId"], {"cwd": p["cwd"], "turns": 0})
            send({"jsonrpc": "2.0", "id": mid, "result": {}})
        else:
            send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32000, "message": "unknown session"}})
    elif method == "session/prompt":
        sid = p["sessionId"]
        st = sessions[sid]
        st["turns"] += 1
        text = p["prompt"][0]["text"]
        update(sid, {"sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": "thinking about: " + text}})
        for _ in range(int(os.environ.get("FAKE_LLM_CALLS", "0"))):  # model calls through the router's /agent/llm proxy
            import urllib.request
            req = urllib.request.Request(os.environ["FAKE_LLM_URL"], data=json.dumps({"stream": True, "messages": [{"role": "user", "content": text}]}).encode(),
                                         headers={"content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp.read()
        if "slow" in text:  # a long turn that can be cancelled (session/cancel) while it works
            cancelled, t0 = False, time.time()
            while time.time() - t0 < 1.5 and not cancelled:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    c = read()
                    cancelled = bool(c and c.get("method") == "session/cancel")
            if cancelled:
                with open(os.path.join(st["cwd"], "cancelled.txt"), "a") as f:
                    f.write("turn %d cancelled\n" % st["turns"])
                send({"jsonrpc": "2.0", "id": mid, "result": {"stopReason": "cancelled"}})
                continue
        update(sid, {"sessionUpdate": "tool_call", "toolCallId": "t1", "title": "write out.txt", "kind": "edit", "status": "pending"})
        send({"jsonrpc": "2.0", "id": 900 + st["turns"], "method": "session/request_permission",
              "params": {"sessionId": sid, "toolCall": {"toolCallId": "t1", "title": "escalate sandbox"},
                         "options": [{"optionId": "ok", "name": "Allow", "kind": "allow_once"}, {"optionId": "no", "name": "Reject", "kind": "reject_once"}]}})
        ans = read()
        granted = ans["result"]["outcome"].get("optionId")
        with open(os.path.join(st["cwd"], "out.txt"), "a") as f:
            f.write("turn %d: %s (permission=%s)\n" % (st["turns"], text, granted))
        update(sid, {"sessionUpdate": "tool_call_update", "toolCallId": "t1", "status": "completed"})
        update(sid, {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "done turn %d" % st["turns"]}})
        send({"jsonrpc": "2.0", "id": mid, "result": {"stopReason": "end_turn"}})
    elif method == "session/cancel":
        pass
    else:
        send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "unknown"}})
