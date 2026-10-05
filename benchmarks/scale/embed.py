"""Precompute multilingual-e5-small embeddings for real_lab.py (CPU is fine).
Needs torch + transformers:  PRIVATE_TOOLS_DIR=... python benchmarks/scale/embed.py"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))
from pool import load  # noqa: E402
from llama_mcp_router.tools import first_sentence, tool_description, tool_name  # noqa: E402

M = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-small")
POOL = "cls" if "bge" in M else "mean"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModel.from_pretrained(M).eval()
PFX = ("", "") if "bge" in M else ("query: ", "passage: ")


def embed(texts, prefix):
    out = []
    for i in range(0, len(texts), 64):
        b = tok([prefix + t for t in texts[i:i + 64]], padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            h = model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        v = h[:, 0] if POOL == "cls" else (h * m).sum(1) / m.sum(1)
        out.append(torch.nn.functional.normalize(v, dim=-1).numpy())
    return np.concatenate(out)


def tool_text(t):
    return "%s: %s" % (tool_name(t).replace("_", " "), first_sentence(tool_description(t), 300))


if __name__ == "__main__":
    tools, server_of, servers = load()
    qs = sorted({json.loads(l)["query"] for f in ("queries_real.jsonl",) for l in open(HERE / f, encoding="utf-8") if l.strip()})
    out = HERE / os.environ.get("EMBED_OUT", "emb.npz")
    np.savez_compressed(out, q=embed(qs, PFX[0]), t=embed([tool_text(t) for t in tools], PFX[1]),
                        s=embed(["%s: %s" % (k, v["label"]) for k, v in servers.items()], PFX[1]))
    (HERE / "emb_index.json").write_text(json.dumps({"queries": qs, "tools": [tool_name(t) for t in tools], "servers": list(servers)}, ensure_ascii=False))
    print(len(qs), "queries,", len(tools), "tools,", len(servers), "servers")
