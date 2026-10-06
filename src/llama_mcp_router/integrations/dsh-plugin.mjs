// llama-mcp-router as a DeepSeek Harness (DSH) plugin: per-turn MCP tool routing.
//
// Before every new user turn (`agent/pre-step`) the plugin sends the user's message and the agent's MCP tools
// (`mcp__<server>__<tool>`) to a running llama-mcp-router (`POST /router/select`, i.e. Laya + BM25 with your tool
// groups) and hides every MCP tool that was not selected, for that agent only (`agent.ctx.tools.restrict`).
// DSH's own tools (bash, files, skills, todo, ...) are never touched. A `find_tools` tool lets the model load any
// hidden MCP tool by describing it. If the router is unreachable, nothing is hidden (fail open).
//
// Profile patch entry:
//   - insert:
//       - id: llama-mcp-router
//         name: '<dsh install>/plugins/llama-mcp-router.mjs'   # `llama-mcp-router install-dsh-plugin <dsh install>`
//         config: { routerUrl: 'http://127.0.0.1:8001' }
import { defineTool } from '@deepseek-ai/dsh-tools'

export const name = 'llama-mcp-router'
export const inject = ['tools']

const DEFAULTS = { routerUrl: 'http://127.0.0.1:8001', prefix: 'mcp__', timeoutMs: 8000, searchK: 8, startupWaitMs: 6000, log: true }

/** mcp__pubmed__unified_search -> pubmed_unified_search (the names llama-server and the router's groups use). */
export function routerName(name) {
  const parts = name.split('__')
  return parts[0] === 'mcp' && parts.length >= 3 ? `${parts[1]}_${parts.slice(2).join('__')}` : name
}

function textOf(messages) {
  return (messages || [])
    .flatMap((m) => (m.content || []).filter((b) => b && b.type === 'text').map((b) => b.text || ''))
    .join('\n')
    .trim()
}

export function apply(ctx, config) {
  const cfg = { ...DEFAULTS, ...(config || {}) }
  const base = String(cfg.routerUrl).replace(/\/$/, '')
  const states = new WeakMap() // agent -> { dispose, keep: Set, loaded: Set }
  const log = (...a) => cfg.log && console.error('[llama-mcp-router]', ...a)

  const routed = () => ctx.tools.schemas().filter((t) => t.name.startsWith(cfg.prefix))

  async function select(query, tools) {
    const back = new Map(tools.map((t) => [routerName(t.name), t.name]))
    const body = {
      query,
      tools: tools.map((t) => ({ type: 'function', function: { name: routerName(t.name), description: t.description || '', parameters: t.parameters || {} } })),
    }
    const r = await fetch(`${base}/router/select`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(cfg.timeoutMs),
    })
    if (!r.ok) throw new Error(`router ${r.status}`)
    const data = await r.json()
    return (data.selected || []).map((n) => back.get(n)).filter(Boolean)
  }

  function state(agent) {
    let st = states.get(agent)
    if (!st) states.set(agent, (st = { dispose: undefined, keep: new Set(), loaded: new Set() }))
    return st
  }

  function mask(agent) {
    const st = state(agent)
    st.dispose?.()
    st.dispose = undefined
    const deny = routed().map((t) => t.name).filter((n) => !st.keep.has(n) && !st.loaded.has(n))
    if (deny.length) st.dispose = agent.ctx.tools.restrict({ deny })
    return deny.length
  }

  ctx.on('agent/pre-step', async (payload, next) => {
    const text = textOf(payload.messages)
    let tools = text ? routed() : []
    // MCP servers connect asynchronously: on a session's first turn their tools may not be registered yet.
    for (let waited = 0; text && !tools.length && waited < cfg.startupWaitMs && !payload.signal?.aborted; waited += 200) {
      await new Promise((r) => setTimeout(r, 200))
      tools = routed()
    }
    if (text && !tools.length) log(`turn ${payload.turn}: no MCP tools registered after ${cfg.startupWaitMs} ms (MCP server not connected?), nothing to route`)
    if (tools.length) {
      const st = state(payload.agent)
      try {
        st.keep = new Set(await select(text, tools))
        const hidden = mask(payload.agent)
        log(`turn ${payload.turn}: ${st.keep.size + st.loaded.size} of ${tools.length} MCP tools visible, ${hidden} hidden:`, [...st.keep].join(','))
      } catch (e) {
        st.dispose?.()
        st.dispose = undefined
        log('selection failed, all MCP tools visible:', e && e.message)
      }
    }
    return next()
  })

  ctx.tools.register(defineTool({
    name: 'find_tools',
    description:
      'Only the MCP tools most likely needed for the current request are loaded. If none of your tools fits, call this first ' +
      'with a short description of the tool you need (e.g. "convert ICD codes to MeSH"); matching tools are loaded and can be called right after.',
    parameters: { query: { type: 'string', required: true, description: 'what the needed tool should do' } },
    output: {
      schema: { type: 'object', additionalProperties: false, properties: { loaded: { type: 'array', required: true, items: { type: 'string' } } } },
      render: (_args, value) => [{
        type: 'text',
        text: value.loaded.length ? `Loaded tools (callable now): ${value.loaded.join(', ')}` : 'No matching tools found.',
      }],
    },
    async execute(args, exec) {
      const agent = exec.agent
      const tools = routed()
      if (!agent || !tools.length) return { loaded: [] }
      const st = state(agent)
      let found = []
      try {
        found = (await select(String(args.query || ''), tools)).slice(0, cfg.searchK)
      } catch (e) {
        log('find_tools failed, loading every MCP tool:', e && e.message)
        found = tools.map((t) => t.name)
      }
      for (const n of found) st.loaded.add(n)
      mask(agent)
      log(`find_tools(${JSON.stringify(args.query)}) -> ${found.join(',')}`)
      return { loaded: found }
    },
  }))
}
