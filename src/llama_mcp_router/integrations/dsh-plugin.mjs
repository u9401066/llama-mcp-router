// llama-mcp-router as a DeepSeek Harness (DSH) plugin: MCP tool routing inside the agent.
//
// DSH assembles the system prompt and the tool list for every model step (`system-prompt/assemble`). This plugin
// shapes that assembly, so the change applies to the very request being built:
//   * routing: the user's newest message and the agent's MCP tools (`mcp__<server>__<tool>`) go to a running
//     llama-mcp-router (`POST /router/select`: Laya + BM25 with your tool groups); unselected MCP tools are left out
//     of the request. DSH's own tools are not routed. `find_tools` loads more MCP tools when the model needs them.
//   * policy: the tool list is part of the prompt prefix (system prompt + tools, before the history), so every change
//     makes llama-server re-process the whole conversation (measured: 49k tokens, 20 s on a 27B model); an unchanged
//     prefix lets it reuse its KV cache across turns (<1 s).
//       'session' (default): route once, on the session's first message; afterwards only `find_tools` (the model asking
//                 for more) changes the list.
//       'grow':   route every user turn and add the new picks (the list only grows).
//       'turn':   route every user turn and replace the list (smallest prompt, re-processes the conversation each turn).
//   * startup: MCP servers connect asynchronously; a session's first request waits for them (`servers`,
//     `startupWaitMs`) so the first turn has its tools and later turns do not change the prompt prefix.
//   * always: MCP tool names that stay visible whatever the routing picks (e.g. the server's main search tool).
//   * hide: tool names left out of every request (e.g. DSH tools a small model does not need).
//   * instructions: 'keep' | 'tool' (each MCP server's instructions become a short summary, a one-line-per-tool catalog
//     of that server and an `mcp_server_guide` tool returning the full text) | 'drop'.
// If the router is unreachable, nothing is hidden (fail open).
//
// Profile patch entry:
//   - insert:
//       - id: llama-mcp-router
//         name: '<dsh install>/plugins/llama-mcp-router.mjs'   # `llama-mcp-router install-dsh-plugin <dsh install>`
//         config: { routerUrl: 'http://127.0.0.1:8001', servers: [pubmed] }
import { defineTool } from '@deepseek-ai/dsh-tools'

export const name = 'llama-mcp-router'
export const inject = ['tools', 'systemPrompt']

export const DEFAULTS = {
  routerUrl: 'http://127.0.0.1:8001',
  prefix: 'mcp__',
  timeoutMs: 8000,
  searchK: 8,
  policy: 'session',
  servers: [],
  startupWaitMs: 6000,
  always: [],
  hide: [],
  instructions: 'keep',
  instructionsChars: 400,
  catalogChars: 70,
  log: true,
}

/** mcp__pubmed__unified_search -> pubmed_unified_search (the names llama-server and the router's groups use). */
export function routerName(name) {
  const parts = name.split('__')
  return parts[0] === 'mcp' && parts.length >= 3 ? `${parts[1]}_${parts.slice(2).join('__')}` : name
}

/** The user's own words in a message (DSH's injected <system-reminder> blocks are skipped). */
export function userText(message) {
  if (!message || message.role !== 'user') return ''
  const blocks = typeof message.content === 'string' ? [{ type: 'text', text: message.content }] : message.content || []
  return blocks
    .filter((b) => b && b.type === 'text' && typeof b.text === 'string' && !b.text.trimStart().startsWith('<system-reminder>'))
    .map((b) => b.text)
    .join('\n')
    .trim()
}

/** First sentence of a tool description, at most `chars` characters. */
export function firstSentence(text, chars) {
  const t = String(text || '').trim().split('\n')[0]
  const m = t.match(/^(.+?[.。!?！？])(\s|$)/)
  const s = (m ? m[1] : t).trim()
  return s.length > chars ? s.slice(0, chars - 1).trimEnd() + '…' : s
}

/** Short form of an MCP server's instructions: its first lines, up to `chars` characters. */
export function summarize(text, chars) {
  const lines = String(text || '').split('\n')
  let out = ''
  for (const line of lines) {
    if (/^[═─=\-]{8,}\s*$/.test(line.trim())) break
    if (out.length + line.length + 1 > chars) break
    out += (out ? '\n' : '') + line
  }
  return out.trim()
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

export function apply(ctx, config) {
  const cfg = { ...DEFAULTS, ...(config || {}) }
  const base = String(cfg.routerUrl).replace(/\/$/, '')
  const hide = new Set(cfg.hide || [])
  const always = new Set(cfg.always || [])
  const servers = (cfg.servers || []).map(String)
  const states = new WeakMap() // agent -> { started, query, keep: Set | undefined, loaded: Set }
  const guides = new Map() // MCP server -> full instructions ('tool' mode)
  const log = (...a) => cfg.log && console.error('[llama-mcp-router]', ...a)

  const routed = () => ctx.tools.schemas().filter((t) => t.name.startsWith(cfg.prefix) && !hide.has(t.name))
  const mcpReady = () => {
    const names = routed().map((t) => t.name)
    return servers.length ? servers.every((s) => names.some((n) => n.startsWith(`${cfg.prefix}${s}__`))) : names.length > 0
  }

  function state(agent) {
    let st = states.get(agent)
    if (!st) states.set(agent, (st = { started: false, query: undefined, keep: undefined, loaded: new Set() }))
    return st
  }

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

  // The user's newest message, claimed by the agent just before its step is assembled.
  ctx.on('agent/inbox/claimed', (...args) => {
    const p = args.find((a) => a && typeof a === 'object' && 'message' in a)
    const text = p && p.agent ? userText(p.message) : ''
    if (text) state(p.agent).query = text
  })

  function catalog(server) {
    const tools = ctx.tools.schemas().filter((t) => t.name.startsWith(`${cfg.prefix}${server}__`) && !hide.has(t.name))
    if (!tools.length) return ''
    const lines = tools.map((t) => `- ${t.name.slice(cfg.prefix.length + server.length + 2)}: ${firstSentence(t.description, cfg.catalogChars)}`)
    return `Tools of this server (only some are loaded; call find_tools to load others):\n${lines.join('\n')}`
  }

  function shapeSections(sections) {
    if (cfg.instructions === 'keep') return sections
    return sections.map((s) => {
      if (!s || typeof s.name !== 'string' || !s.name.startsWith('mcp:') || typeof s.text !== 'string' || !s.text.trim()) return s
      const server = s.name.slice(4)
      if (cfg.instructions === 'drop') return { ...s, text: '' }
      const body = s.text.replace(/^### MCP server: [^\n]*\n+/, '')
      guides.set(server, body)
      const parts = [`### MCP server: ${server}`, summarize(body, cfg.instructionsChars), catalog(server),
        `(Full usage guide: call mcp_server_guide with server "${server}" before complex or unfamiliar tasks.)`]
      return { ...s, text: parts.filter(Boolean).join('\n\n') }
    })
  }

  async function shape(out, st) {
    let tools = hide.size ? out.tools.filter((t) => !hide.has(t.name)) : out.tools
    const mcp = tools.filter((t) => t.name.startsWith(cfg.prefix))
    if (mcp.length && st.query !== undefined && (cfg.policy !== 'session' || st.keep === undefined)) {
      const query = st.query
      st.query = undefined
      try {
        const picked = await select(query, mcp)
        if (cfg.policy === 'turn') {
          st.keep = new Set(picked)
          st.loaded = new Set()
        } else {
          st.keep = new Set([...(st.keep || []), ...picked])
        }
        log(`selected ${picked.length} of ${mcp.length} MCP tools (${cfg.policy}: ${st.keep.size + st.loaded.size} visible):`, picked.join(','))
      } catch (e) {
        log('selection failed, keeping', st.keep ? 'the previous selection' : 'every MCP tool', ':', e && e.message)
      }
    }
    if (st.keep) tools = tools.filter((t) => !t.name.startsWith(cfg.prefix) || st.keep.has(t.name) || st.loaded.has(t.name) || always.has(t.name))
    return { ...out, tools, sections: shapeSections(out.sections || []) }
  }

  ctx.on('system-prompt/assemble', async (_assembly, context, next) => {
    const agent = context && context.agent
    if (!agent) return next()
    const st = state(agent)
    if (!st.started) {
      st.started = true
      if (cfg.startupWaitMs > 0 && !mcpReady()) {
        const t0 = Date.now()
        while (!mcpReady() && Date.now() - t0 < cfg.startupWaitMs && !context.signal?.aborted) await sleep(100)
        log(mcpReady() ? `MCP servers ready after ${Date.now() - t0} ms` : `MCP servers not ready after ${cfg.startupWaitMs} ms (have: ${routed().length} tools)`)
        // Assemble again so the tools and instructions that just arrived are part of the session's first request.
        if (routed().length) return ctx.systemPrompt.assemble(context)
      }
    }
    return shape(await next(), st)
  })

  ctx.tools.register(defineTool({
    name: 'find_tools',
    description:
      'Only the MCP tools most likely needed are loaded. If none of your tools fits, call this first with a short description ' +
      'of the tool you need (e.g. "convert ICD codes to MeSH"); matching tools are loaded and can be called in your next step.',
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
      log(`find_tools(${JSON.stringify(args.query)}) -> ${found.join(',')}`)
      return { loaded: found }
    },
  }))

  if (cfg.instructions === 'tool') {
    ctx.tools.register(defineTool({
      name: 'mcp_server_guide',
      description: "Full usage guide of an MCP server (how to use its tools well). The system prompt only has each guide's first lines.",
      parameters: { server: { type: 'string', required: true, description: 'MCP server name, e.g. "pubmed"' } },
      output: {
        schema: { type: 'object', additionalProperties: false, properties: { guide: { type: 'string', required: true } } },
        render: (_args, value) => [{ type: 'text', text: value.guide }],
      },
      async execute(args) {
        const server = String(args.server || '')
        const guide = guides.get(server)
        return { guide: guide || `No guide for "${server}". Servers with a guide: ${[...guides.keys()].join(', ') || 'none'}.` }
      },
    }))
  }
}
