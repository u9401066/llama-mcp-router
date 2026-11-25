"""Agent bridge: run an existing coding agent per chat session, behind the OpenAI chat API.

Any agent that speaks the Agent Client Protocol (ACP v1, JSON-RPC over stdio) can be plugged in --
DeepSeek Harness (``dsh --profile acp``), Qwen Code (``qwen --acp``), opencode (``opencode acp``), goose,
codex-acp, ... -- through :class:`AgentConfig`. Each chat conversation gets:

* its own directory with a git workspace (the agent's ``cwd``) and a private agent home/state directory,
* its own agent process (started on demand, stopped after ``idle_ttl``; the workspace stays),
* skills copied into the workspace (``skills_target``), so they are versioned with the work,
* a git commit after every turn, and download links for the files that changed.

Only the newest user message is sent to the agent; the agent keeps (and compacts) its own history, so a
long session is not bounded by the chat client's context window.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger("llama_mcp_router.agent")

ACP_PROTOCOL_VERSION = 1


@dataclass
class AgentConfig:
    command: List[str]  # argv; "{session_dir}", "{workspace}", "{home}" are substituted
    env: Dict[str, str] = field(default_factory=dict)  # extra env, same placeholders ("~" expanded)
    files: Dict[str, str] = field(default_factory=dict)  # per-session files to write (path relative to session_dir -> text)
    root: str = "~/agent-sessions"
    model_id: str = "agent"  # requests with this model go to the agent
    triggers: List[str] = field(default_factory=lambda: ["/agent", "@agent", "agent:"])  # ...or whose first user message starts so
    default: bool = False  # every chat goes to the agent unless its first user message starts with an opt-out prefix
    # Optional per-person access: API key -> user name (or users_file, a JSON file with that mapping). When set, agent chats
    # need "Authorization: Bearer <key>" (the Web UI sends its API-key setting this way), each user's sessions live under
    # <root>/<user>/ and GET /agent/sessions lists only their own.
    users: Dict[str, str] = field(default_factory=dict)
    users_file: Optional[str] = None
    max_processes: int = 4  # live agent processes at once; idle ones are stopped first (their workspaces stay)
    optout: List[str] = field(default_factory=lambda: ["chat:"])
    skills_dirs: List[str] = field(default_factory=list)  # copied into each new workspace
    skills_target: str = ".agents/skills"
    idle_ttl: float = 1800.0
    start_timeout: float = 120.0
    prompt_timeout: float = 3600.0
    permissions: str = "deny"  # answer to session/request_permission: "deny" or "allow"
    commit: bool = True
    name: str = "agent"
    # Outer sandbox around the whole agent process (agent-independent). "bwrap": the host filesystem is
    # read-only, the paths in sandbox_hide are replaced by empty tmpfs (default: your home, /run with the
    # docker socket, /tmp), sandbox_ro paths are mounted back read-only (e.g. the agent's runtime) and only
    # the session directory is writable. Network stays shared unless sandbox_network is false.
    sandbox: Optional[str] = None
    sandbox_ro: List[str] = field(default_factory=list)
    sandbox_hide: List[str] = field(default_factory=lambda: ["~", "/run", "/tmp"])
    sandbox_network: bool = True
    # The agent's model requests (/agent/llm/...) that set no reasoning effort get the chat's own (the Web UI's thinking
    # level: reasoning_effort, or "none" for enable_thinking=false) or else this default (None: llama-server's default).
    reasoning_effort: Optional[str] = None
    # Reasoning effort for the agent's context-compaction requests (DSH: the last user message says it is "acting as a
    # compaction engine"), overriding the turn's: summarising needs little thinking, and at "max" each one took 1-2
    # minutes of decoding. None: same as the other requests.
    compaction_effort: Optional[str] = None
    # Send the agent's tool schemas compacted (tools.compact_tool): no titles, regex patterns, alias/Optional unions or
    # layout -- typically half the tokens, a smaller prompt prefix to process whenever the tool list changes.
    compact_tools: bool = False
    # Move the lines of the agent's system prompt that name its session (DSH: "Your working directory is <session
    # workspace>.") to the start of the first message after it, so all sessions send the same tokens up to their first
    # user message. llama-server can then reuse that shared prefix (tools + system prompt) across sessions: with SWA
    # models (Inkling, Gemma) it restores the context checkpoint it made at the first user message, which it cannot use
    # when the prompts differ a few tokens earlier, so a new session (or a slot switching sessions) re-processed it all.
    stable_prefix: bool = False
    # Fill llama-server's idle, never-used slots (it has just started) with the prompt prefix all agent sessions share
    # (tools + system prompt, recorded from the latest session's first model request in <root>/warmup.json), so the
    # first sessions after a restart do not pay for it. Slots that hold any prompt are never touched.
    warmup: bool = False
    warmup_interval: float = 30.0  # seconds between /slots checks
    # Delete session directories (workspace included) whose last turn is older than this many days; None keeps them.
    # Sessions with a running agent process are kept. Checked hourly.
    retention_days: Optional[float] = None
    # Per-user notes shared by all of that user's sessions: the directory <root>/<user>/memory (MEMORY.md and whatever
    # else the agent keeps there), mounted (bwrap) or linked at memory_dir in each workspace and kept out of the
    # workspace's git. A directory, not a single file: editors save by writing a temporary file and renaming it over the
    # target, which fails on a single-file bind mount (EBUSY). It is not put into the prompt, so the shared prompt prefix
    # stays the same; a skill tells the agent when to read and update it. Without a users map every chat is "anonymous"
    # and shares one directory.
    memory: bool = False
    memory_dir: str = ".agents/memory"
    # Tool results in the agent's model requests longer than this many characters are cut in the middle, with a note
    # telling the model how much is missing and to read in smaller pages. Applied the same way to every request, so the
    # prompt cache is unaffected. None: pass tool results unchanged.
    tool_result_max_chars: Optional[int] = None
    # Show the agent's reasoning and answer while the model generates them. Some agents (DSH's ACP bridge) send each
    # model message only once it is complete, so a chat saw nothing for minutes and then the whole text at once. With
    # this the router takes the text from the agent's model stream (/agent/llm) as it passes, sends it to the chat and
    # drops the agent's later copy of the same text. Only the main agent's requests of the current turn are shown (they
    # carry tools and the user's message); compaction, title and subagent requests are not. Leave it off for agents that
    # already stream over ACP, or the text appears twice.
    live_stream: bool = False

    def __post_init__(self) -> None:
        if self.users_file:
            with open(os.path.expanduser(self.users_file), encoding="utf-8") as f:
                extra = json.load(f)
            self.users = {**{str(k).strip(): str(v) for k, v in extra.items()}, **self.users}
        if any(not k or not v for k, v in self.users.items()):
            raise ValueError("agent users: every API key and user name must be non-empty")

    @classmethod
    def load(cls, path: str) -> "AgentConfig":
        with open(os.path.expanduser(path), encoding="utf-8") as f:
            data = json.load(f)
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def bwrap_argv(cfg: "AgentConfig", session_dir: str, workspace: str, binds: Sequence[Tuple[str, str]] = ()) -> List[str]:
    """bubblewrap prefix confining an agent process to its session (see AgentConfig.sandbox). ``binds``: extra
    (source, destination) read-write mounts, e.g. the user's memory directory into the workspace."""
    argv = ["bwrap", "--ro-bind", "/", "/"]
    for h in cfg.sandbox_hide:
        path = os.path.abspath(os.path.expanduser(h))
        argv += ["--tmpfs", path]
        if path == "/run" and os.path.isdir("/run/systemd/resolve"):
            argv += ["--ro-bind", "/run/systemd/resolve", "/run/systemd/resolve"]  # keep DNS, hide sockets
    for ro in cfg.sandbox_ro:
        path = os.path.abspath(os.path.expanduser(ro))
        if os.path.exists(path):
            argv += ["--ro-bind", path, path]
    argv += ["--bind", session_dir, session_dir]
    for src, dst in binds:
        argv += ["--bind", src, dst]
    argv += ["--dev", "/dev", "--proc", "/proc", "--unshare-pid", "--die-with-parent", "--chdir", workspace]
    if not cfg.sandbox_network:
        argv.append("--unshare-net")
    return argv


class AcpError(RuntimeError):
    pass


_TIMINGS = object()  # queue marker: the session's model timings changed
_STATUS = object()  # queue marker: a status line for the user (e.g. waiting for a model slot)
_LIVE = object()  # queue marker: model text as it is generated (AgentConfig.live_stream)
_STAT_KEYS = ("requests", "compactions", "prompt_n", "cache_n", "predicted_n", "prompt_ms", "predicted_ms", "tool_calls", "tool_s")
MEMORY_HEADER = ("# Memory\n\nDurable notes about this user, shared by all of their agent sessions. Keep entries short and dated;\n"
                 "remove ones that are no longer true.\n")


def safe_owner(owner: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", owner or "anonymous")[:64] or "anonymous"


def _subst(s: str, mapping: Dict[str, str]) -> str:
    for k, v in mapping.items():
        s = s.replace("{" + k + "}", v)
    return os.path.expanduser(s)


class AcpConnection:
    """One agent process; JSON-RPC 2.0, one message per line on stdin/stdout."""

    def __init__(self, argv: List[str], cwd: str, env: Dict[str, str], stderr_path: str, permissions: str = "deny"):
        self.argv, self.cwd, self.env, self.stderr_path, self.permissions = argv, cwd, env, stderr_path, permissions
        self.proc: Optional[asyncio.subprocess.Process] = None
        self._next = 0
        self._pending: Dict[int, asyncio.Future] = {}
        self._reader: Optional[asyncio.Task] = None
        self.listener: Optional[Callable[[Dict[str, Any]], None]] = None
        self.capabilities: Dict[str, Any] = {}
        self._stderr = None

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    async def start(self, timeout: float) -> Dict[str, Any]:
        self._stderr = open(self.stderr_path, "ab")
        self.proc = await asyncio.create_subprocess_exec(
            *self.argv, cwd=self.cwd, env=self.env, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=self._stderr, limit=64 * 1024 * 1024)
        self._reader = asyncio.ensure_future(self._read())
        res = await self.request("initialize", {
            "protocolVersion": ACP_PROTOCOL_VERSION,
            "clientCapabilities": {"fs": {"readTextFile": False, "writeTextFile": False}, "terminal": False},
            "clientInfo": {"name": "llama-mcp-router", "version": "0"},
        }, timeout)
        self.capabilities = res.get("agentCapabilities") or {}
        return res

    async def _send(self, msg: Dict[str, Any]) -> None:
        assert self.proc and self.proc.stdin
        self.proc.stdin.write((json.dumps(msg) + "\n").encode())
        await self.proc.stdin.drain()

    async def request(self, method: str, params: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        if not self.alive:
            raise AcpError("agent process is not running")
        self._next += 1
        i = self._next
        fut = asyncio.get_event_loop().create_future()
        self._pending[i] = fut
        await self._send({"jsonrpc": "2.0", "id": i, "method": method, "params": params})
        try:
            msg = await asyncio.wait_for(fut, timeout)
        finally:
            self._pending.pop(i, None)
        if "error" in msg:
            raise AcpError("%s failed: %s" % (method, (msg["error"] or {}).get("message", msg["error"])))
        return msg.get("result") or {}

    async def notify(self, method: str, params: Dict[str, Any]) -> None:
        if self.alive:
            await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    async def _answer(self, msg: Dict[str, Any]) -> None:
        method, mid = msg.get("method"), msg.get("id")
        if method == "session/request_permission":
            opts = (msg.get("params") or {}).get("options") or []
            want = "allow" if self.permissions == "allow" else "reject"
            pick = next((o for o in opts if str(o.get("kind", "")).startswith(want)), None)
            outcome = {"outcome": "selected", "optionId": pick["optionId"]} if pick else {"outcome": "cancelled"}
            title = ((msg.get("params") or {}).get("toolCall") or {}).get("title")
            log.info("permission request %r -> %s", title, outcome)
            if self.listener:
                self.listener({"method": "router/permission", "params": {"title": title, "granted": want == "allow" and bool(pick)}})
            await self._send({"jsonrpc": "2.0", "id": mid, "result": {"outcome": outcome}})
        else:  # fs/*, terminal/*: not offered in clientCapabilities
            await self._send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "method not supported by client: %s" % method}})

    async def _read(self) -> None:
        assert self.proc and self.proc.stdout
        try:
            while True:
                line = await self.proc.stdout.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if "method" in msg and "id" in msg:
                    await self._answer(msg)
                elif "method" in msg:
                    if self.listener:
                        self.listener(msg)
                elif msg.get("id") in self._pending:
                    fut = self._pending[msg["id"]]
                    if not fut.done():
                        fut.set_result(msg)
        finally:
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(AcpError("agent process exited (code %s)" % (self.proc.returncode if self.proc else "?")))

    async def close(self) -> None:
        if self.proc and self.proc.returncode is None:
            try:
                self.proc.stdin.close()  # ACP agents shut down on stdin EOF
                await asyncio.wait_for(self.proc.wait(), 5)
            except Exception:  # noqa: BLE001
                self.proc.terminate()
                try:
                    await asyncio.wait_for(self.proc.wait(), 5)
                except asyncio.TimeoutError:
                    self.proc.kill()
        if self._reader:
            self._reader.cancel()
        if self._stderr:
            self._stderr.close()


_TIMING_KEYS = ("prompt_n", "prompt_ms", "cache_n", "predicted_n", "predicted_ms", "draft_n", "draft_n_accepted")


def sum_timings(parts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Add up llama-server ``timings`` of several requests into one block the Web UI can show (tokens, time, speed)."""
    if not parts:
        return None
    out: Dict[str, Any] = {k: 0 for k in _TIMING_KEYS}
    for t in parts:
        for k in _TIMING_KEYS:
            v = t.get(k)
            if isinstance(v, (int, float)):
                out[k] += v
    for k in ("prompt_ms", "predicted_ms"):
        out[k] = round(out[k], 3)
    out["prompt_per_token_ms"] = out["prompt_ms"] / out["prompt_n"] if out["prompt_n"] else 0.0
    out["prompt_per_second"] = 1000.0 * out["prompt_n"] / out["prompt_ms"] if out["prompt_ms"] else 0.0
    out["predicted_per_token_ms"] = out["predicted_ms"] / out["predicted_n"] if out["predicted_n"] else 0.0
    out["predicted_per_second"] = 1000.0 * out["predicted_n"] / out["predicted_ms"] if out["predicted_ms"] else 0.0
    out["requests"] = len(parts)
    return out


@dataclass
class AgentSession:
    id: str
    key: str
    owner: str
    dir: str
    workspace: str
    home: str
    acp_session: Optional[str] = None
    conn: Optional[AcpConnection] = None
    turns: int = 0
    last_used: float = field(default_factory=time.time)
    lock: Optional[asyncio.Lock] = None
    # llama-server timings of the agent's model requests in the current turn (seen by the router's /agent/llm proxy)
    done_timings: List[Dict[str, Any]] = field(default_factory=list)
    inflight: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    notify: Optional[Callable[[Optional[Dict[str, Any]]], None]] = None  # (prompt_progress or None) -> event for the turn
    effort: Optional[str] = None  # reasoning effort for the current turn's model requests (AgentConfig.reasoning_effort)
    status: Optional[Callable[[str], None]] = None  # status line -> event for the turn (set while a turn runs)
    stats: Dict[str, float] = field(default_factory=lambda: {k: 0 for k in _STAT_KEYS})  # totals over the session's life
    live: Optional[Callable[[str, str], None]] = None  # ("thought"|"message", text) -> event for the turn (live_stream)
    anchor: str = ""  # start of the current turn's prompt, whitespace collapsed: marks the main agent's model requests
    # Text shown live per model request, without whitespace, until the agent's own copy of it arrives and is dropped
    live_pending: List[Dict[str, Any]] = field(default_factory=list)
    _rid: int = 0
    _notified: float = 0.0
    _progressed: float = 0.0

    def begin_request(self) -> int:
        self._rid += 1
        return self._rid

    def update_request(self, rid: int, timings: Optional[Dict[str, Any]] = None, progress: Optional[Dict[str, Any]] = None) -> None:
        if timings is not None:
            self.inflight[rid] = timings
        now = time.monotonic()
        if progress is not None and self.notify and now - self._progressed >= 0.2:  # prompt processing, as it happens
            self._progressed = now
            self.notify(progress)
        elif timings is not None and self.notify and now - self._notified >= 0.25:  # live speed while the model generates
            self._notified = now
            self.notify(None)

    def live_request(self) -> Dict[str, Any]:
        """Start recording the text of one model request that is shown live (see live_copy); live_end when it is over."""
        e: Dict[str, Any] = {"thought": "", "message": "", "open": True}
        self.live_pending = self.live_pending[-7:] + [e]
        return e

    def live_add(self, e: Dict[str, Any], kind: str, text: str) -> None:
        e[kind] += _squash(text)

    def live_end(self, e: Dict[str, Any]) -> None:
        e["open"] = False
        self._live_prune()

    def _live_prune(self) -> None:
        self.live_pending = [x for x in self.live_pending if x["open"] or x["thought"] or x["message"]]

    def live_copy(self, ev: Dict[str, Any]) -> bool:
        """True if ``ev``, a thought or message from the agent, repeats text already shown live (that part is used up).
        Compared without whitespace, as agents may trim it; text that does not match is passed on (shown twice rather
        than lost)."""
        kind = ev.get("type")
        if kind not in ("thought", "message") or ev.get("live") or not self.live_pending:
            return False
        n = _squash(ev.get("text") or "")
        if not n:
            return False
        for e in self.live_pending:
            if e[kind].startswith(n):
                e[kind] = e[kind][len(n):]
                self._live_prune()
                return True
        return False

    def end_request(self, rid: int, timings: Optional[Dict[str, Any]], compaction: bool = False) -> None:
        self.inflight.pop(rid, None)
        self.stats["requests"] += 1
        self.stats["compactions"] += int(compaction)
        if timings:
            self.done_timings.append(timings)
            for k in ("prompt_n", "cache_n", "predicted_n", "prompt_ms", "predicted_ms"):
                self.stats[k] += float(timings.get(k) or 0)
        if self.notify:
            self._notified = time.monotonic()
            self.notify(None)

    def timings(self) -> Optional[Dict[str, Any]]:
        return sum_timings(self.done_timings + list(self.inflight.values()))


async def _git(cwd: str, *args: str) -> str:
    p = await asyncio.create_subprocess_exec("git", "-c", "user.name=llama-mcp-router", "-c", "user.email=agent@localhost", *args,
                                             cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await p.communicate()
    if p.returncode != 0:
        raise RuntimeError("git %s: %s" % (" ".join(args), err.decode(errors="replace").strip()))
    return out.decode(errors="replace")


class AgentManager:
    """Maps chat conversations to agent sessions (one workspace + one agent process each)."""

    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.root = os.path.abspath(os.path.expanduser(cfg.root))
        os.makedirs(self.root, exist_ok=True)
        self._index_path = os.path.join(self.root, "index.json")
        try:
            with open(self._index_path, encoding="utf-8") as f:
                self._index: Dict[str, Dict[str, Any]] = json.load(f)
        except (OSError, ValueError):
            self._index = {}
        self.sessions: Dict[str, AgentSession] = {}
        self._create_lock: Optional[asyncio.Lock] = None  # created inside the running loop (Python 3.9)
        self._reaper: Optional[asyncio.Task] = None
        self._starting = 0  # agent processes being started (not yet in a session's conn)
        self._pruned_at = 0.0
        self.on_tool: Optional[Callable[[str, float], None]] = None  # (status, seconds) per finished tool call

    def running_turns(self) -> int:
        return sum(1 for s in self.sessions.values() if s.lock and s.lock.locked())

    def live_processes(self) -> int:
        return sum(1 for s in self.sessions.values() if s.conn and s.conn.alive)

    def summary(self) -> Dict[str, int]:
        return {"sessions": len(self._index), "live": self.live_processes(), "running_turns": self.running_turns()}

    # ------------------------------------------------------------------ persistence
    def _save_index(self) -> None:
        tmp = self._index_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=1)
        os.replace(tmp, self._index_path)

    def by_id(self, sid: str) -> Optional[Dict[str, Any]]:
        if not re.fullmatch(r"[0-9a-f]{32}", sid or ""):
            return None
        for v in self._index.values():
            if v.get("id") == sid:
                return v
        return None

    def _dir(self, meta: Dict[str, Any]) -> str:
        return os.path.join(self.root, meta["dir"]) if meta.get("dir") else os.path.join(self.root, meta["id"])

    def list_for(self, owner: str) -> List[Dict[str, Any]]:
        out = [{"session": m["id"], "created": m.get("created"), "turns": m.get("turns", 0), "title": m.get("title", "")}
               for m in self._index.values() if m.get("owner", "anonymous") == owner]
        return sorted(out, key=lambda m: -(m["created"] or 0))

    # ------------------------------------------------------------------ lifecycle
    def _placeholders(self, s: AgentSession) -> Dict[str, str]:
        return {"session_dir": s.dir, "workspace": s.workspace, "home": s.home, "session": s.id}

    async def _prepare_dirs(self, s: AgentSession) -> None:
        new = not os.path.isdir(s.workspace)
        os.makedirs(s.workspace, exist_ok=True)
        os.makedirs(s.home, exist_ok=True)
        for rel, text in self.cfg.files.items():
            path = os.path.join(s.dir, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(_subst(text, self._placeholders(s)))
        if new:
            await _git(s.workspace, "init", "-q")
            for src in self.cfg.skills_dirs:
                src = os.path.expanduser(src)
                if not os.path.isdir(src):
                    continue
                for name in os.listdir(src):
                    if os.path.isdir(os.path.join(src, name)):
                        shutil.copytree(os.path.join(src, name), os.path.join(s.workspace, self.cfg.skills_target, name), dirs_exist_ok=True)
            with open(os.path.join(s.workspace, "README.md"), "w", encoding="utf-8") as f:
                f.write("# Agent workspace\n\nFiles created in this chat session. Every turn is a git commit.\n")
            await _git(s.workspace, "add", "-A")
            await _git(s.workspace, "commit", "-q", "-m", "session start")
        if self.cfg.memory:
            self._link_memory(s)

    def memory_path(self, owner: str) -> str:
        """The owner's memory directory (shared by all of their sessions)."""
        return os.path.join(self.root, safe_owner(owner), "memory")

    def _memory_rel(self) -> str:
        rel = os.path.normpath(self.cfg.memory_dir.strip("/"))
        if rel in ("", ".") or rel.split(os.sep)[0] == "..":
            raise ValueError("memory_dir must be a directory inside the workspace, got %r" % self.cfg.memory_dir)
        return rel

    def _link_memory(self, s: AgentSession) -> None:
        """Make the owner's memory directory appear at memory_dir in the workspace (bwrap: a bind mount over an empty
        directory, set up in _spawn; otherwise a symlink) and keep that path out of the workspace's git."""
        mem = self.memory_path(s.owner)
        os.makedirs(mem, exist_ok=True)
        if not os.path.exists(os.path.join(mem, "MEMORY.md")):
            with open(os.path.join(mem, "MEMORY.md"), "w", encoding="utf-8") as f:
                f.write(MEMORY_HEADER)
        rel = self._memory_rel()
        target = os.path.join(s.workspace, rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.islink(target) and (self.cfg.sandbox == "bwrap" or os.path.realpath(target) != os.path.realpath(mem)):
            os.unlink(target)
        if self.cfg.sandbox == "bwrap":
            os.makedirs(target, exist_ok=True)
        else:
            if os.path.isdir(target) and not os.path.islink(target) and not os.listdir(target):
                os.rmdir(target)  # the empty mount point left by a bwrap configuration
            if not os.path.lexists(target):
                os.symlink(mem, target)
            elif not os.path.islink(target):
                log.warning("session %s: %s exists in the workspace; memory not linked", s.id, rel)
        exclude = os.path.join(s.workspace, ".git", "info", "exclude")
        line = "/" + rel + "\n"  # no trailing slash: also matches the symlink
        try:
            with open(exclude, encoding="utf-8") as f:
                present = line in f.read()
        except OSError:
            present = False
        if not present and os.path.isdir(os.path.dirname(exclude)):
            with open(exclude, "a", encoding="utf-8") as f:
                f.write(line)

    async def _make_room(self) -> None:
        live = [x for x in self.sessions.values() if x.conn and x.conn.alive]
        while len(live) + self._starting >= max(1, self.cfg.max_processes):
            idle = sorted((x for x in live if not (x.lock and x.lock.locked())), key=lambda x: x.last_used)
            if not idle:
                if self._starting and not live:
                    await asyncio.sleep(0.2)  # every slot is an agent that is still starting
                    live = [x for x in self.sessions.values() if x.conn and x.conn.alive]
                    continue
                raise AcpError("all %d agent slots are busy; try again in a moment" % self.cfg.max_processes)
            log.info("agent process limit (%d): stopping idle session %s", self.cfg.max_processes, idle[0].id)
            await idle[0].conn.close()  # type: ignore[union-attr]
            idle[0].conn = None
            live = [x for x in live if x is not idle[0]]

    async def _connect(self, s: AgentSession) -> None:
        await self._make_room()
        self._starting += 1
        try:
            await self._spawn(s)
        finally:
            self._starting -= 1

    async def _spawn(self, s: AgentSession) -> None:
        ph = self._placeholders(s)
        argv = [_subst(a, ph) for a in self.cfg.command]
        env = dict(os.environ)
        env.update({k: _subst(v, ph) for k, v in self.cfg.env.items()})
        if self.cfg.sandbox == "bwrap":
            binds = [(self.memory_path(s.owner), os.path.join(s.workspace, self._memory_rel()))] if self.cfg.memory else []
            argv = bwrap_argv(self.cfg, s.dir, s.workspace, binds) + argv
            if "HOME" not in self.cfg.env:
                env["HOME"] = s.home  # the real home is hidden inside the sandbox
        elif self.cfg.sandbox:
            raise ValueError("unknown agent sandbox %r (supported: bwrap)" % self.cfg.sandbox)
        conn = AcpConnection(argv, s.workspace, env, os.path.join(s.dir, "agent.stderr.log"), self.cfg.permissions)
        try:
            await conn.start(self.cfg.start_timeout)
            resumed = False
            caps = conn.capabilities.get("sessionCapabilities") or {}
            if s.acp_session and ("resume" in caps or conn.capabilities.get("loadSession")):
                method = "session/resume" if "resume" in caps else "session/load"
                try:
                    await conn.request(method, {"sessionId": s.acp_session, "cwd": s.workspace, "mcpServers": []}, self.cfg.start_timeout)
                    resumed = True
                except AcpError as e:
                    log.info("could not %s %s (%s); starting a new agent session", method, s.acp_session, e)
            if not resumed:
                res = await conn.request("session/new", {"cwd": s.workspace, "mcpServers": []}, self.cfg.start_timeout)
                s.acp_session = res["sessionId"]
                self._index[s.key]["acp_session"] = s.acp_session
                self._save_index()
        except BaseException:  # failed or cancelled (e.g. Stop) while starting: do not leave the process behind
            await conn.close()
            raise
        s.conn = conn
        log.info("agent session %s: %s (acp %s)", s.id, "resumed" if resumed else "started", s.acp_session)

    async def get(self, key: str, owner: str = "anonymous", title: str = "") -> AgentSession:
        if self._create_lock is None:
            self._create_lock = asyncio.Lock()
        async with self._create_lock:
            s = self.sessions.get(key)
            if s is None:
                meta = self._index.get(key)
                if meta is None:
                    sid = uuid.uuid4().hex
                    meta = self._index[key] = {"id": sid, "owner": owner, "dir": os.path.join(safe_owner(owner), sid), "created": time.time(),
                                               "acp_session": None, "turns": 0, "title": " ".join(title.split())[:80]}
                    self._save_index()
                d = self._dir(meta)
                s = AgentSession(meta["id"], key, meta.get("owner", "anonymous"), d, os.path.join(d, "workspace"), os.path.join(d, "home"),
                                 meta.get("acp_session"), turns=meta.get("turns", 0), lock=asyncio.Lock())
                s.stats.update({k: v for k, v in (meta.get("stats") or {}).items() if k in s.stats})
                await self._prepare_dirs(s)
                self.sessions[key] = s
            if self._reaper is None:
                self._reaper = asyncio.ensure_future(self._reap())
        return s

    async def _reap(self) -> None:
        while True:
            await asyncio.sleep(min(60.0, max(1.0, self.cfg.idle_ttl / 4)))
            now = time.time()
            for s in list(self.sessions.values()):
                if s.conn and not (s.lock and s.lock.locked()) and now - s.last_used > self.cfg.idle_ttl:
                    log.info("agent session %s idle, stopping its process", s.id)
                    await s.conn.close()
                    s.conn = None
            if self.cfg.retention_days and now - self._pruned_at >= 3600:
                self._pruned_at = now
                self.prune(now)

    def _last_activity(self, meta: Dict[str, Any]) -> float:
        if meta.get("last_used"):
            return float(meta["last_used"])
        try:  # sessions from before v0.15 have no last_used: the workspace's last commit touched .git/index
            return os.path.getmtime(os.path.join(self._dir(meta), "workspace", ".git", "index"))
        except OSError:
            return float(meta.get("created") or 0)

    def prune(self, now: Optional[float] = None) -> List[str]:
        """Delete sessions idle for more than retention_days (directory and index entry); returns their ids."""
        if not self.cfg.retention_days:
            return []
        cutoff = (now or time.time()) - self.cfg.retention_days * 86400
        root = os.path.realpath(self.root) + os.sep
        removed = []
        for key, meta in list(self._index.items()):
            if self._last_activity(meta) >= cutoff:
                continue
            s = self.sessions.get(key)
            if s is not None and ((s.conn and s.conn.alive) or (s.lock and s.lock.locked())):
                continue
            d = os.path.realpath(self._dir(meta))
            if d.startswith(root):
                shutil.rmtree(d, ignore_errors=True)
            self._index.pop(key, None)
            self.sessions.pop(key, None)
            removed.append(meta.get("id"))
        if removed:
            self._save_index()
            log.info("retention (%s days): removed %d sessions", self.cfg.retention_days, len(removed))
        return removed

    async def aclose(self) -> None:
        if self._reaper:
            self._reaper.cancel()
        for s in self.sessions.values():
            if s.conn:
                await s.conn.close()

    # ------------------------------------------------------------------ a turn
    async def turn(self, key: str, text: str, owner: str = "anonymous", effort: Optional[str] = None) -> AsyncIterator[Dict[str, Any]]:
        """Yield events: {"type": "thought"|"message"|"tool"|"plan"|"permission"|"done", ...}. ``effort``: the chat's
        reasoning effort, used for this turn's model requests that set none (default: AgentConfig.reasoning_effort)."""
        s = await self.get(key, owner, text)
        assert s.lock is not None
        async with s.lock:
            s.last_used = time.time()
            s.effort = effort or self.cfg.reasoning_effort
            if s.conn is None or not s.conn.alive:
                yield {"type": "status", "text": "starting agent"}
                await self._connect(s)
            queue: asyncio.Queue = asyncio.Queue()
            s.conn.listener = queue.put_nowait  # type: ignore[union-attr]
            s.done_timings, s.inflight = [], {}
            s.notify = lambda progress=None: queue.put_nowait((_TIMINGS, progress))
            s.status = lambda text: queue.put_nowait((_STATUS, text))
            s.live_pending = []
            if self.cfg.live_stream:
                s.anchor = " ".join(text.split())[:200]
                s.live = lambda kind, t: queue.put_nowait((_LIVE, kind, t))
            prompt = asyncio.ensure_future(s.conn.request(  # type: ignore[union-attr]
                "session/prompt", {"sessionId": s.acp_session, "prompt": [{"type": "text", "text": text}]}, self.cfg.prompt_timeout))
            finished = False
            tool_t0: Dict[str, float] = {}
            try:
                while True:
                    getter = asyncio.ensure_future(queue.get())
                    done, _ = await asyncio.wait({getter, prompt}, return_when=asyncio.FIRST_COMPLETED)
                    if getter in done:
                        ev = self._timings_event(s, getter.result()) or _event(getter.result())
                        if ev and ev["type"] != "none" and not s.live_copy(ev):
                            self._track_tool(s, ev, tool_t0)
                            yield ev
                        continue
                    getter.cancel()
                    while not queue.empty():
                        msg = queue.get_nowait()
                        ev = self._timings_event(s, msg) or _event(msg)
                        if ev and ev["type"] != "none" and not s.live_copy(ev):
                            self._track_tool(s, ev, tool_t0)
                            yield ev
                    res = prompt.result()  # raises AcpError on failure
                    finished = True
                    s.turns += 1
                    s.last_used = time.time()
                    self._index[s.key]["turns"] = s.turns
                    self._save_index()
                    commit, changed = await self._commit(s, text)
                    yield {"type": "done", "stop": res.get("stopReason"), "session": s.id, "commit": commit, "changed": changed,
                           "timings": s.timings()}
                    return
            finally:
                s.conn.listener = None  # type: ignore[union-attr]
                s.notify = None
                s.status = None
                s.live = None
                s.anchor = ""
                if s.key in self._index:
                    self._index[s.key].update(last_used=time.time(), stats={k: round(v, 3) for k, v in s.stats.items()})
                    self._save_index()
                if not finished and not prompt.done():
                    await s.conn.notify("session/cancel", {"sessionId": s.acp_session})  # type: ignore[union-attr]
                    prompt.cancel()

    def _track_tool(self, s: AgentSession, ev: Dict[str, Any], started: Dict[str, float]) -> None:
        """Count finished tool calls and their duration (session stats and AgentManager.on_tool)."""
        if ev.get("type") != "tool" or not ev.get("id"):
            return
        if ev.get("new"):
            started.setdefault(ev["id"], time.monotonic())
        elif ev.get("status") in ("completed", "failed") and ev["id"] in started:
            dur = time.monotonic() - started.pop(ev["id"])
            s.stats["tool_calls"] += 1
            s.stats["tool_s"] += dur
            if self.on_tool:
                self.on_tool(ev["status"], dur)

    @staticmethod
    def _timings_event(s: AgentSession, msg: Any) -> Optional[Dict[str, Any]]:
        if isinstance(msg, tuple) and msg and msg[0] is _STATUS:
            return {"type": "status", "text": msg[1]}
        if isinstance(msg, tuple) and msg and msg[0] is _LIVE:
            return {"type": msg[1], "text": msg[2], "live": True}
        if not (isinstance(msg, tuple) and msg and msg[0] is _TIMINGS):
            return None
        t, p = s.timings(), msg[1]
        return {"type": "timings", "timings": t, "prompt_progress": p} if t or p else {"type": "none"}

    def live_session(self, sid: str) -> Optional[AgentSession]:
        """The running session with this id (its agent process calls the router's /agent/llm/<id>/ proxy)."""
        return next((x for x in self.sessions.values() if x.id == sid), None)

    async def _commit(self, s: AgentSession, text: str):
        if not self.cfg.commit:
            return None, []
        try:
            if not (await _git(s.workspace, "status", "--porcelain")).strip():
                return None, []
            await _git(s.workspace, "add", "-A")
            title = " ".join(text.split())[:60]
            await _git(s.workspace, "commit", "-q", "-m", "turn %d: %s" % (s.turns, title))
            commit = (await _git(s.workspace, "rev-parse", "--short", "HEAD")).strip()
            changed = []
            for line in (await _git(s.workspace, "show", "--name-status", "--format=", "HEAD")).splitlines():
                parts = line.split("\t")
                if len(parts) >= 2:
                    changed.append({"status": parts[0][:1], "path": parts[-1]})
            return commit, changed
        except RuntimeError as e:
            log.warning("commit failed in %s: %s", s.workspace, e)
            return None, []

    # ------------------------------------------------------------------ files
    def workspace_of(self, sid: str) -> Optional[str]:
        meta = self.by_id(sid)
        return os.path.join(self._dir(meta), "workspace") if meta else None

    async def describe(self, sid: str) -> Optional[Dict[str, Any]]:
        ws = self.workspace_of(sid)
        if not ws or not os.path.isdir(ws):
            return None
        files = [f for f in (await _git(ws, "ls-files")).splitlines() if f]
        logs = [dict(zip(("commit", "date", "subject"), l.split("\t", 2))) for l in (await _git(ws, "log", "--format=%h\t%cI\t%s", "-n", "50")).splitlines()]
        meta = self.by_id(sid) or {}
        live = self.live_session(sid)
        stats = dict(live.stats) if live else meta.get("stats")
        return {"session": sid, "files": files, "log": logs, "turns": meta.get("turns", 0), "stats": stats}

    async def archive(self, sid: str) -> Optional[bytes]:
        ws = self.workspace_of(sid)
        if not ws:
            return None
        p = await asyncio.create_subprocess_exec("git", "archive", "--format=zip", "HEAD", cwd=ws, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await p.communicate()
        return out if p.returncode == 0 else None

    def file_path(self, sid: str, rel: str) -> Optional[str]:
        ws = self.workspace_of(sid)
        if not ws:
            return None
        full = os.path.realpath(os.path.join(ws, rel))
        if not full.startswith(os.path.realpath(ws) + os.sep) or "/.git/" in full + "/" or not os.path.isfile(full):
            return None
        return full


def _squash(text: str) -> str:
    return "".join(text.split())


def _text(content: Any) -> str:
    if isinstance(content, dict):
        return content.get("text") or ""
    if isinstance(content, list):
        return "".join(_text(c.get("content", c)) if isinstance(c, dict) else "" for c in content)
    return ""


def _event(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """ACP session/update notification -> a small, client-neutral event."""
    if msg.get("method") == "router/permission":
        return {"type": "permission", **msg["params"]}
    if msg.get("method") != "session/update":
        return None
    u = (msg.get("params") or {}).get("update") or {}
    kind = u.get("sessionUpdate")
    if kind == "agent_message_chunk":
        return {"type": "message", "text": _text(u.get("content"))}
    if kind == "agent_thought_chunk":
        return {"type": "thought", "text": _text(u.get("content"))}
    if kind in ("tool_call", "tool_call_update"):
        return {"type": "tool", "id": u.get("toolCallId"), "title": u.get("title"), "kind": u.get("kind"), "status": u.get("status"), "new": kind == "tool_call"}
    if kind == "plan":
        return {"type": "plan", "entries": [e.get("content") for e in u.get("entries") or []]}
    return None
