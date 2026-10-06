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
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

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


def bwrap_argv(cfg: "AgentConfig", session_dir: str, workspace: str) -> List[str]:
    """bubblewrap prefix confining an agent process to its session (see AgentConfig.sandbox)."""
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
    argv += ["--bind", session_dir, session_dir, "--dev", "/dev", "--proc", "/proc", "--unshare-pid", "--die-with-parent", "--chdir", workspace]
    if not cfg.sandbox_network:
        argv.append("--unshare-net")
    return argv


class AcpError(RuntimeError):
    pass


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
        return {"session_dir": s.dir, "workspace": s.workspace, "home": s.home}

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
            argv = bwrap_argv(self.cfg, s.dir, s.workspace) + argv
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

    async def aclose(self) -> None:
        if self._reaper:
            self._reaper.cancel()
        for s in self.sessions.values():
            if s.conn:
                await s.conn.close()

    # ------------------------------------------------------------------ a turn
    async def turn(self, key: str, text: str, owner: str = "anonymous") -> AsyncIterator[Dict[str, Any]]:
        """Yield events: {"type": "thought"|"message"|"tool"|"plan"|"permission"|"done", ...}."""
        s = await self.get(key, owner, text)
        assert s.lock is not None
        async with s.lock:
            s.last_used = time.time()
            if s.conn is None or not s.conn.alive:
                yield {"type": "status", "text": "starting agent"}
                await self._connect(s)
            queue: asyncio.Queue = asyncio.Queue()
            s.conn.listener = queue.put_nowait  # type: ignore[union-attr]
            prompt = asyncio.ensure_future(s.conn.request(  # type: ignore[union-attr]
                "session/prompt", {"sessionId": s.acp_session, "prompt": [{"type": "text", "text": text}]}, self.cfg.prompt_timeout))
            finished = False
            try:
                while True:
                    getter = asyncio.ensure_future(queue.get())
                    done, _ = await asyncio.wait({getter, prompt}, return_when=asyncio.FIRST_COMPLETED)
                    if getter in done:
                        ev = _event(getter.result())
                        if ev:
                            yield ev
                        continue
                    getter.cancel()
                    while not queue.empty():
                        ev = _event(queue.get_nowait())
                        if ev:
                            yield ev
                    res = prompt.result()  # raises AcpError on failure
                    finished = True
                    s.turns += 1
                    s.last_used = time.time()
                    self._index[s.key]["turns"] = s.turns
                    self._save_index()
                    commit, changed = await self._commit(s, text)
                    yield {"type": "done", "stop": res.get("stopReason"), "session": s.id, "commit": commit, "changed": changed}
                    return
            finally:
                s.conn.listener = None  # type: ignore[union-attr]
                if not finished and not prompt.done():
                    await s.conn.notify("session/cancel", {"sessionId": s.acp_session})  # type: ignore[union-attr]
                    prompt.cancel()

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
        return {"session": sid, "files": files, "log": logs}

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
