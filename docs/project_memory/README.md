# Project memory snapshot

These files are a snapshot of Claude Code's **persistent auto-memory** for this project,
copied from `~/.claude/projects/<workspace>/memory/` on the original (macOS) machine. They
capture the hard-won findings of this project — one fact per file, plus `MEMORY.md` as the
index — and are committed here so they travel with the repo (e.g. to the Windows/WSL2 sweep box).

- `MEMORY.md` — the index (one line per memory). Read this first.
- `*.md` — one finding each, with `name` / `description` frontmatter and `[[links]]` between them.

## Re-activating as live memory on another machine

Claude Code derives its memory directory from the workspace's **absolute path**, so the folder
name differs per machine. To make these the live memory on a new machine:

1. Run `claude` once in the cloned workspace so it creates
   `~/.claude/projects/<mangled-new-path>/memory/`.
2. Copy these files into that folder.

If you skip this, nothing is lost: `../../CLAUDE.md` already encapsulates the essential project
state, and `../../rnr/PORTING_NOTES.md` + the other `docs/` notes carry the deep history.

> This is a point-in-time snapshot, not a live mirror. The canonical live memory remains in
> `~/.claude/` on whichever machine you're actively working from; re-snapshot if it drifts.
