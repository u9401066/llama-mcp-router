---
name: memory
description: Use at the start of a task that may depend on the user's preferences or earlier work, and whenever you learn a durable fact about the user or their project worth keeping for future chats.
---
# The user's long-term memory

`.agents/memory/` is shared by all of this user's chat sessions (it is not part of this workspace's git); the notes
are in `.agents/memory/MEMORY.md`.
Your context only covers this chat; this file is how the next chat knows what you learned.

- **Read it** at the start of a task when preferences, earlier results or project conventions could matter.
- **Add to it** when you learn something durable: a stated preference (language, citation style, format), a project
  fact (paths, names, decisions), or a result the user will build on. One dated line per fact, e.g.
  `- 2025-11-25: prefers tables in Traditional Chinese; cite PMIDs.`
- **Fix it** when an entry turns out wrong or outdated: edit or delete the line, do not append a contradiction.
- Never store secrets, passwords, API keys or personal data about other people.
- Keep it under ~100 lines; merge or drop old entries rather than growing it forever.
