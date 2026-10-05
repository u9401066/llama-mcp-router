---
name: work-in-files
description: Use for any task that produces code, data, tables or long text. Write results to files in the workspace instead of pasting them into the chat.
---
# Work in files, not in the chat

The chat window is small; the workspace is not. Every file you write is kept and versioned (each turn is a git commit).

1. Put scripts in `scripts/`, inputs in `data/`, results in `outputs/` (create the folders when needed).
2. Write long content (code, tables, reports, > ~30 lines) to a file with a descriptive name, e.g. `outputs/primes_up_to_100.txt`.
3. Run what you wrote (`python3 scripts/x.py`) and check the result before you answer.
4. In the chat, answer briefly: what you did, the key result in 1–5 lines, and the file paths.
5. Never print a whole large file back into the chat; quote only the lines that matter.
