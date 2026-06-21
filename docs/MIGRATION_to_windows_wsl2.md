# Moving this project to a Windows PC (WSL2) for long sweeps

This workspace was developed on macOS (osx-arm64). The science is **headless batch**
(no rendering needed for sweeps), so the lowest-friction home on a Windows box is
**WSL2 / Ubuntu** — the bash build script, conda/pixi, and the whole toolchain port
almost directly to `linux-64`. This guide is the runbook; hand it to Claude Code on
the new machine and it can execute most of it.

> **Don't copy the macOS build.** `tissue-forge_build/` (clang-compiled), `.pixi/`, and
> `oracle_run/` are machine-specific and regenerable. They are gitignored on purpose.
> The engine **must be rebuilt** on the new machine.

---

## 0. The repo layout (4 git repos, nested but independent — NOT submodules)

| Path | GitHub | Branch | Role |
|---|---|---|---|
| workspace root | `tc2fh/tissueforge-rnr-sorting` (public) | `main` | our code: `rnr/`, `CLAUDE.md`, `docs/`, `pixi.toml`, `pixi.lock` |
| `tissue-forge/` | `tc2fh/tissue-forge` (fork) | `feat/native-rnr-reconnection` | the engine fork (native I↔H + periodic geom + orientation repair + active motility) |
| `tvm/` | `ZhangTao-SJTU/tvm` | `main` | GPL oracle (read-only reference — license boundary, see CLAUDE.md) |
| `3DVertVor/` | `Manning-Research-Group/3DVertVor` | `3DVertexForce` | MIT/GPL-derived oracle (read-only reference) |

The root repo's `.gitignore` excludes the three nested repos, so they are cloned
**separately into the same folder layout** on the new machine.

---

## 1. One-time WSL2 setup on the Windows PC

In an **Administrator PowerShell**:

```powershell
wsl --install -d Ubuntu     # reboot if prompted; sets up Ubuntu under WSL2
```

Then open **Ubuntu**, and inside it install **pixi** (conda-backed; required because
TissueForge's deps are conda packages):

```bash
curl -fsSL https://pixi.sh/install.sh | bash
exec $SHELL          # reload PATH
```

You do **not** need to apt-install gcc/cmake/etc. — pixi pulls the conda toolchain
(`gcc_linux-64`, cmake, ninja, swig) into the env.

---

## 2. Clone all four repos into the same layout

```bash
mkdir -p ~/Work/SegoLab && cd ~/Work/SegoLab
git clone https://github.com/tc2fh/tissueforge-rnr-sorting.git VertexModeling
cd VertexModeling
git clone -b feat/native-rnr-reconnection https://github.com/tc2fh/tissue-forge.git
git clone -b main                          https://github.com/ZhangTao-SJTU/tvm.git
git clone -b 3DVertexForce                 https://github.com/Manning-Research-Group/3DVertVor.git
```

Result: `tissue-forge/`, `tvm/`, `3DVertVor/` sit inside `VertexModeling/`, exactly as
on the Mac.

---

## 3. Point pixi at linux-64 and install the toolchain

`pixi.toml` currently lists `platforms = ["osx-arm64"]`. Add `linux-64`:

```toml
platforms = ["osx-arm64", "linux-64"]
```

Then:

```bash
pixi install        # re-solves pixi.lock for linux-64 and builds the toolchain env
```

If `pyvoro-mmalahe` (pip sdist, in `[pypi-dependencies]`) fails to build, comment it out
and rely on the scipy Voronoi/Delaunay fallback (CLAUDE.md, Phase 0) — it is only used
for the initial packing.

---

## 4. Build the engine fork on Linux (the one real porting task)

The build is driven by `build_tissue_forge_osx.sh`, which has **macOS-only bits** that must
be dropped for `linux-64` (x86_64):

- **Remove** `-DCMAKE_APPLE_SILICON_PROCESSOR:STRING=arm64` and the `CMAKE_OSX_DEPLOYMENT_TARGET`
  / `CMAKE_OSX_SYSROOT` blocks (macOS-only).
- **Remove** `-DTF_ENABLE_AVX:BOOL=OFF` / `-DTF_ENABLE_SSE4:BOOL=OFF` — those were forced off
  for Apple silicon; on x86_64 leave TF's defaults (ON) for speed, which matters for sweeps.
- **Keep** everything else verbatim: `CMAKE_PREFIX_PATH` / `FIND_ROOT_PATH` / `INSTALL_PREFIX`
  = `$CONDA_PREFIX`, `Python_EXECUTABLE=$CONDA_PREFIX/bin/python`,
  `LIBXML_INCLUDE_DIR=$CONDA_PREFIX/include/libxml2`, Ninja, the `--target install` step.

**Good first task for Claude Code on the new machine:** *"Create `build_tissue_forge_linux.sh`
from `build_tissue_forge_osx.sh`, dropping the macOS-only CMake flags as noted in
docs/MIGRATION_to_windows_wsl2.md §4, and add a `build-tf` task override / new pixi task that
calls it on linux-64."* Then:

```bash
pixi run build-tf        # tens of minutes the first time; incremental relink after
pixi run verify          # gate: engine imports + both solvers init headless
```

Expect to fix a few **gcc-vs-clang** compile nits in the fork's C++ additions (upstream
TissueForge builds on Linux, so the base is fine; the native RNR / motility code is the
risk surface). This is exactly the kind of iterate-on-build-errors loop Claude Code is good
at — point it at the failing compile output.

---

## 5. Confirm the science reproduces, then run sweeps

```bash
pixi run test            # 49-test gate (round-trip reversibility, Condition-4 vetoes,
                         # periodic dynamics, clamp-free active-motility rate, native motility)
pixi run sort-oracle     # one periodic two-type sort (NOISE_MODEL=native default)
pixi run overnight       # full ensemble + figs + video (background-friendly; the long sweep)
```

See `pixi.toml [tasks]` and `CLAUDE.md` "Current status" for the full task list
(`probe-active`, `dpmax`, `fig1e`, `fig1f`, `video`).

> **Open Phase-3 polish (carried over):** regenerate the canonical `fig1e`/`fig1f` with the
> **native** drive — `run_overnight.py` still uses the Python `active` comparison model; add a
> `MODEL=native` fig selector and point it at `native`. Good sweep-box task.

---

## 6. Using Claude Code on the new machine

- **Install inside WSL2**, not native Windows, so it shares the Linux toolchain:
  ```bash
  npm install -g @anthropic-ai/claude-code   # (or the native installer); then `claude`
  ```
- `cd ~/Work/SegoLab/VertexModeling && claude` — it **auto-loads `CLAUDE.md`**, so the full
  project context (the four RNR conditions, the data model, the active-motility finding, the
  phase plan) is available immediately. `rnr/PORTING_NOTES.md` and `docs/` carry the deep
  history.
- **VS Code:** from WSL run `code .` to open the workspace over the WSL remote; the Claude Code
  extension works there too.
- **Curated auto-memory:** a snapshot of the Mac's `~/.claude/projects/.../memory/` (MEMORY.md +
  per-finding notes) is committed in this repo at [`docs/project_memory/`](project_memory/), so it
  travels with the clone. CLAUDE.md already encapsulates the essential state, but to re-activate the
  full curated memory as *live* memory on the new machine, follow
  [`docs/project_memory/README.md`](project_memory/README.md): run `claude` once to create
  `~/.claude/projects/<mangled-new-path>/memory/`, then copy the snapshot files into it. (The folder
  name is derived from the absolute workspace path, so it differs from the Mac's.)

---

## 7. Sanity checklist

- [ ] WSL2 Ubuntu + pixi installed
- [ ] all four repos cloned in the right layout & on the right branches
- [ ] `linux-64` added to `pixi.toml` platforms; `pixi install` clean
- [ ] `build_tissue_forge_linux.sh` authored; `pixi run build-tf` succeeds
- [ ] `pixi run verify` prints "TissueForge + vertex solver OK"
- [ ] `pixi run test` = 49 green
- [ ] `pixi run sort-oracle` produces a CSV in `rnr/exports/`
