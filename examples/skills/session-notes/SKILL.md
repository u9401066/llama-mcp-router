---
name: session-notes
description: Use at the start and end of every multi-step or multi-turn task, to keep NOTES.md as the working memory of this session.
---
# Keep NOTES.md as working memory

Your context is limited and older turns may be compacted. `NOTES.md` in the workspace root is what survives.

- **Start of a turn:** if `NOTES.md` exists, read it first.
- **End of a turn:** update it. Keep these sections short:
  - `## Goal` — what the user wants overall
  - `## Done` — what exists now (file paths) and key results
  - `## Decisions` — choices made and why
  - `## Next` — open steps or questions
- Replace outdated lines instead of appending forever; keep it under ~60 lines.
