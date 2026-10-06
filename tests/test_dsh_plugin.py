"""Runs the DeepSeek Harness plugin (integrations/dsh-plugin.mjs) under Node with a fake DSH context and a fake router."""

import json
import os
import shutil
import subprocess

import pytest

PLUGIN = os.path.join(os.path.dirname(__file__), "..", "src", "llama_mcp_router", "integrations", "dsh-plugin.mjs")


def node_ok():
    node = shutil.which("node")
    if not node:
        return False
    out = subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip().lstrip("v")
    return int(out.split(".")[0] or 0) >= 18


SCRIPT = r"""
import http from 'node:http'
import * as plugin from './plugin.mjs'

const queries = []
const server = http.createServer(async (req, res) => {
  let raw = ''
  for await (const c of req) raw += c
  const body = JSON.parse(raw)
  queries.push(body.query)
  const words = body.query.split(/\s+/).filter(Boolean)
  const selected = body.tools.map((t) => t.function.name).filter((n) => words.some((w) => n.endsWith(w)))
  res.setHeader('content-type', 'application/json')
  res.end(JSON.stringify({ selected }))
})
await new Promise((r) => server.listen(0, '127.0.0.1', r))
const url = `http://127.0.0.1:${server.address().port}`

const GUIDE = 'PubMed Search MCP Server\nUse unified_search first.\n═════════════════════════\nlong part\nmore'
const MCP = ['mcp__pubmed__unified_search', 'mcp__pubmed__get_gene', 'mcp__pubmed__save_pipeline']

function fake(config, { mcpAfterMs = 0 } = {}) {
  const handlers = {}
  const registered = []
  const t0 = Date.now()
  const connected = () => Date.now() - t0 >= mcpAfterMs
  const schemas = () => [
    { name: 'bash' }, { name: 'workflow' },
    ...(connected() ? MCP.map((name) => ({ name, description: name, parameters: {} })) : []),
    ...registered.map((t) => ({ name: t.name })),
  ]
  const ctx = {
    on(name, fn) { (handlers[name] ||= []).push(fn) },
    tools: { schemas, register: (t) => registered.push(t) },
    systemPrompt: { assemble: (context) => assemble(context) },
  }
  async function assemble(context) {
    const assembly = {
      sections: [{ name: 'base', text: 'You are DSH.' }, { name: 'mcp:pubmed', text: connected() ? `### MCP server: pubmed\n\n${GUIDE}` : '' }],
      tools: schemas(), contexts: [], variables: {},
    }
    const hs = handlers['system-prompt/assemble'] || []
    const run = (i) => (i < hs.length ? hs[i](assembly, context, () => run(i + 1)) : Promise.resolve(assembly))
    return run(0)
  }
  const claim = (agent, text) => (handlers['agent/inbox/claimed'] || []).forEach((h) => h({ agent, turn: 1, message: { role: 'user', content: [{ type: 'text', text }] } }))
  plugin.apply(ctx, { routerUrl: url, log: false, ...config })
  const tool = (name) => registered.find((t) => t.name === name)
  return { assemble, claim, tool }
}
const names = (a) => a.tools.map((t) => t.name)
const out = {}

// grow policy, startup wait, hide, instructions as a guide tool
{
  const f = fake({ policy: 'grow', servers: ['pubmed'], startupWaitMs: 3000, hide: ['workflow'], instructions: 'tool', instructionsChars: 60 }, { mcpAfterMs: 300 })
  const agent = {}
  f.claim(agent, 'please unified_search')
  const t = Date.now()
  const a1 = await f.assemble({ agent })
  out.waited = Date.now() - t
  out.step1 = names(a1)
  out.section = a1.sections.find((s) => s.name === 'mcp:pubmed').text
  out.step2 = names(await f.assemble({ agent }))  // same turn, no new message: unchanged
  f.claim(agent, '<system-reminder>ignored get_gene</system-reminder>')
  out.reminder = names(await f.assemble({ agent }))
  f.claim(agent, 'now get_gene')
  out.turn2 = names(await f.assemble({ agent }))
  out.find = await f.tool('find_tools').execute({ query: 'save_pipeline' }, { agent })
  out.afterFind = names(await f.assemble({ agent }))
  out.guide = (await f.tool('mcp_server_guide').execute({ server: 'pubmed' })).guide
  out.other = names(await f.assemble({ agent: {} }))  // another agent: nothing selected yet, all visible
}
// session policy (default): routed once, then only find_tools changes the list
{
  const f = fake({})
  const agent = {}
  const n0 = queries.length
  f.claim(agent, 'unified_search'); out.session1 = names(await f.assemble({ agent }))
  f.claim(agent, 'get_gene'); out.session2 = names(await f.assemble({ agent }))
  await f.tool('find_tools').execute({ query: 'get_gene' }, { agent })
  out.session3 = names(await f.assemble({ agent }))
  out.sessionQueries = queries.slice(n0)
}
// turn policy replaces the selection
{
  const f = fake({ policy: 'turn', always: ['mcp__pubmed__save_pipeline'] })
  const agent = {}
  f.claim(agent, 'unified_search'); await f.assemble({ agent })
  f.claim(agent, 'get_gene')
  out.turnPolicy = names(await f.assemble({ agent }))
}
// router down: fail open
{
  const f = fake({ routerUrl: 'http://127.0.0.1:9', timeoutMs: 1000 })
  const agent = {}
  f.claim(agent, 'unified_search')
  out.down = names(await f.assemble({ agent }))
}
out.queries = queries
console.log(JSON.stringify(out))
server.close()
"""


@pytest.mark.skipif(not node_ok(), reason="needs Node.js >= 18")
def test_dsh_plugin_routes_on_the_step_being_assembled(tmp_path):
    stub = tmp_path / "node_modules" / "@deepseek-ai" / "dsh-tools"
    stub.mkdir(parents=True)
    (stub / "package.json").write_text(json.dumps({"name": "@deepseek-ai/dsh-tools", "type": "module", "exports": "./index.js"}))
    (stub / "index.js").write_text("export const defineTool = (t) => t\n")
    shutil.copy(PLUGIN, tmp_path / "plugin.mjs")
    (tmp_path / "run.mjs").write_text(SCRIPT)
    p = subprocess.run(["node", "run.mjs"], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout.strip().splitlines()[-1])

    for k in ("step1", "step2", "reminder", "turn2", "afterFind", "other", "turnPolicy", "down", "session1", "session2", "session3"):
        out[k] = sorted(out[k])
    base = ["bash", "find_tools", "mcp_server_guide"]
    # first request of the session waited for the MCP server, then carried only the selected MCP tool; 'workflow' hidden
    assert out["waited"] >= 250 and out["step1"] == sorted(base + ["mcp__pubmed__unified_search"])
    assert out["step2"] == out["step1"] and out["reminder"] == out["step1"]  # stable prefix; reminders are not queries
    assert out["turn2"] == sorted(base + ["mcp__pubmed__unified_search", "mcp__pubmed__get_gene"])  # grow keeps earlier tools
    assert out["find"] == {"loaded": ["mcp__pubmed__save_pipeline"]} and "mcp__pubmed__save_pipeline" in out["afterFind"]
    assert out["section"].startswith("### MCP server: pubmed\n\nPubMed Search MCP Server\nUse unified_search first.\n\nTools of this server")
    assert "- get_gene: mcp__pubmed__get_gene" in out["section"] and out["section"].endswith('server "pubmed" before complex or unfamiliar tasks.)')
    assert "long part" not in out["section"] and out["guide"].endswith("long part\nmore")
    assert "mcp__pubmed__get_gene" in out["other"] and "mcp__pubmed__save_pipeline" in out["other"]
    assert out["turnPolicy"] == sorted(["bash", "find_tools", "workflow", "mcp__pubmed__get_gene", "mcp__pubmed__save_pipeline"])
    assert out["down"] == sorted(["bash", "find_tools", "workflow", "mcp__pubmed__unified_search", "mcp__pubmed__get_gene", "mcp__pubmed__save_pipeline"])
    assert out["queries"][:2] == ["please unified_search", "now get_gene"]
    assert out["session1"] == out["session2"] == sorted(["bash", "find_tools", "workflow", "mcp__pubmed__unified_search"])
    assert out["session3"] == sorted(out["session1"] + ["mcp__pubmed__get_gene"])
    assert out["sessionQueries"] == ["unified_search", "get_gene"]  # turn 1 routed, turn 2 not; then find_tools
