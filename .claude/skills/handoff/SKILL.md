---
name: handoff
description: Wrap up a session — write a timestamped summary + next-session kickoff prompt to docs/sessions/. Use at a check-in or when ending a work session.
---
Write a session handoff to the repo's living docs. Terse, scannable, bullets over prose.

1. Timestamp via Bash: `date +%Y-%m-%d-%H%M` (sortable → filename) and `date '+%Y-%m-%d %H:%M %Z'` (human).
2. New file `docs/sessions/<sortable>-<kebab-slug>.md` (slug = the session's main topic; never overwrite an existing file).
3. Two sections:

   ## Summary <human-ts>
   Goal; what changed **and why** (decisions, gotchas, surprises — NOT a file dump or restated git log); current
   build/test/git state. Cite files as `path:line`. Include the full `git status --short` so nothing uncommitted is lost.

   ## Kickoff — next session
   A ready-to-paste prompt for a fresh agent with no memory of this session: the next steps in **priority order**, the
   exact `pixi run` / shell commands to use, and any caveats. Restate scope + license guardrails if relevant.

4. If `progress.md` / `CLAUDE.md` "current status" lines are now stale, fix the one or two lines.
5. Print the path written.
