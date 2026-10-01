"""Dump an MCP stdio server's tools (tools/list) as OpenAI-format tool definitions.

    python benchmarks/dump_mcp_tools.py out.json uvx zotero-keeper
"""
import json
import subprocess
import sys
import time

out, cmd = sys.argv[1], sys.argv[2:]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open("/tmp/mcp_dump_err.log", "w"), text=True, bufsize=1)


def send(m):
    p.stdin.write(json.dumps(m) + "\n")
    p.stdin.flush()


def recv(id_, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        line = p.stdout.readline()
        if not line:
            break
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if m.get("id") == id_:
            return m
    raise SystemExit("no response")


send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "dump", "version": "0"}}})
recv(1)
send({"jsonrpc": "2.0", "method": "notifications/initialized"})
tools, cursor, i = [], None, 2
while True:
    send({"jsonrpc": "2.0", "id": i, "method": "tools/list", "params": ({"cursor": cursor} if cursor else {})})
    r = recv(i)["result"]
    tools += r["tools"]
    cursor = r.get("nextCursor")
    i += 1
    if not cursor:
        break
p.terminate()
json.dump([{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t.get("inputSchema") or {"type": "object", "properties": {}}}} for t in tools], open(out, "w"), ensure_ascii=False, indent=1)
print(len(tools), "tools ->", out)
