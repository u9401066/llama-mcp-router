"""Assemble the real 600+-tool pool from MCP tools/list dumps.

Tool names get the server key as prefix, exactly like llama-server's --mcp-servers-config does.
Private servers (not published) are added when $PRIVATE_TOOLS_DIR contains servers_private.json + their dumps;
otherwise the public subset is used.
"""
import json
import os
from pathlib import Path

HERE = Path(__file__).parent


def load(include_private: bool = True):
    servers = json.load(open(HERE / "servers.json"))
    priv = os.environ.get("PRIVATE_TOOLS_DIR")
    if priv and include_private and (Path(priv) / "servers_private.json").exists():
        servers.update(json.load(open(Path(priv) / "servers_private.json")))
    tools, server_of, used = [], {}, {}
    for key, s in servers.items():
        src = (Path(priv) if s.get("private") else HERE / "real") / (s["file"] + ".json") if (not s.get("private") or priv) else None
        if src is None or not src.exists() or (s.get("private") and not include_private):
            continue
        for t in json.load(open(src)):
            f = dict(t["function"], name="%s_%s" % (key, t["function"]["name"]))
            tools.append({"type": "function", "function": f})
            server_of[f["name"]] = key
        used[key] = s
    return tools, server_of, used


if __name__ == "__main__":
    t, so, s = load()
    print(len(t), "tools,", len(s), "servers,", sum(len(json.dumps(x)) for x in t) // 1000, "KB of schema")
