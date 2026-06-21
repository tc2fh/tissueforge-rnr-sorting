---
name: resume
description: Resume work in a fresh session — read the most recent docs/sessions/ handoff and continue from its kickoff.
---
Pick up from the latest session handoff.

1. Newest handoff: `ls docs/sessions/*.md | sort | tail -1`. Read it fully, plus any files it references.
2. Verify reality before acting — the handoff reflects when it was written, not now: run `git status --short`, and
   re-confirm any build/test state it assumes (e.g. `pixi run test`) only if cheap and relevant.
3. Restate the priority-ordered next steps in 2–3 lines, then start the first one.
4. Honor the scope + license guardrails in CLAUDE.md (no growth / C++-port hardening unless asked; `tvm/` + `3DVertVor/`
   are read-only oracles).
