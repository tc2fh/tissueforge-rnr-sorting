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
6. **Commit this session's work** (standing authorization — see CLAUDE.md working agreements; this is
   the deliberate exception to "commit only when asked"). Rules:
   - Run the gate first (`pixi run test`) and **only commit if green**; state the result in the message
     (e.g. `tests: 81 passed`). If you can't run it, say so in the message rather than implying green.
   - **Stage selectively** — `git add` the specific files this session created/modified (incl. the new
     handoff + any `progress.md`/`CLAUDE.md` edits). **Never `git add -A`/`git add .`**: do NOT commit
     the read-only oracle repos (`tvm/`, `3DVertVor/`, `tissue-forge/`, `cellGPU/`, `VertAX/`,
     `gpu_reference_papers/` — they carry their own `.git`) or unrelated prior-session artifacts
     (e.g. stray `rnr/exports/*` blobs) unless they are part of this session's work.
   - If on the default branch (`main`), branch first; otherwise commit on the current feature branch.
   - Use the session's commit-message trailer (the `Co-Authored-By` / `Claude-Session` lines).
   - **Do NOT push** unless the user asks — committing is authorized here, pushing is not.