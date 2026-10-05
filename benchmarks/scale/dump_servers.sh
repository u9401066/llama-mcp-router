#!/usr/bin/env bash
# How benchmarks/scale/real/*.json were produced: each MCP server was cloned and installed once in a
# throw-away environment under ~/mcp-servers, its tools/list was dumped (no tool is ever executed), and the
# environments were deleted again. Only the tool definitions are needed to benchmark tool selection.
# Some servers needed `uv pip install "mcp<2"`. Two private servers were dumped the same way but are not published.
OUT=$1; mkdir -p "$OUT"; D=~/llama-mcp-router/benchmarks/dump_mcp_tools.py; PY=~/llama-mcp-router/.venv/bin/python
H=~/mcp-servers
run() { name=$1; dir=$2; shift 2; ( cd "$dir" && NCBI_EMAIL=you@example.com timeout 180 $PY $D "$OUT/$name.json" "$@" ) > /tmp/dump_$name.log 2>&1 && echo "$name: $(tail -1 /tmp/dump_$name.log)" || echo "$name: FAILED ($(tail -1 /tmp/dump_$name.log | cut -c1-120); stderr: $(tail -1 /tmp/mcp_dump_err.log 2>/dev/null | cut -c1-160))"; }
run pubmed-search-mcp       $H/pubmed-search-mcp            .venv/bin/pubmed-search-mcp
run zotero-keeper           $H/zotero-keeper/mcp-server     .venv/bin/zotero-keeper
run research-data-explorer  $H/research-data-explorer       .venv/bin/research-data-explorer
run med-paper-assistant     $H/med-paper-assistant          .venv/bin/med-paper-assistant
run nsforge-mcp             $H/nsforge-mcp                  .venv/bin/nsforge-mcp
run creativity-generation-unit $H/creativity-generation-unit .venv/bin/cgu-server
run rootcause-mcp           $H/rootcause-mcp                .venv/bin/rootcause-mcp
run pharmacy-mcp            $H/pharmacy-mcp                 .venv/bin/pharmacy-mcp
run medical-calc-mcp        $H/medical-calc-mcp             .venv/bin/medical-calc-mcp
run academic-figures-mcp    $H/academic-figures-mcp         .venv/bin/afm-server
run atomic-workflow         $H/atomic-workflow              .venv/bin/atomic-workflow
run mcp-libre               $H/mcp-libre                    .venv/bin/mcp-libre
run sympy-mcp               $H/sympy-mcp                    .venv/bin/python server.py
run reaper-reapy-mcp        $H/reaper-reapy-mcp             .venv/bin/reaper_reapy_mcp
run automl-stat-mcp         $H/automl-stat-mcp/automl-mcp-server/src $H/automl-stat-mcp/.venv/bin/python main.py
run medagent-copilot        $H/medagent-copilot             .venv/bin/python src/mcp_server.py
run openevidence-mcp        $H/openevidence-mcp             node dist/server.js
