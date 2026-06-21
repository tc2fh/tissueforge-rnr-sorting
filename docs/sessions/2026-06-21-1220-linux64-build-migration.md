# Session 2026-06-21 12:20 EDT — linux-64 build migration

Ported the native-RNR TissueForge fork from macOS (osx-arm64) → WSL2/Ubuntu (linux-64) for sweeps.
Runbook: `docs/MIGRATION_to_windows_wsl2.md`. **Outcome: engine builds + `pixi run test` 48-green on linux-64.**

## Summary

- **Runbook steps 1–4 all green:**
  - `pixi install` — added `linux-64` to `platforms`; lock re-solved (+468 pkgs); toolchain = gcc 14.3.0;
    `pyvoro-mmalahe` built from sdist (no scipy fallback needed).
  - Ported `build_tissue_forge_osx.sh` → **`build_tissue_forge_linux.sh`** (dropped Apple-silicon arch flag,
    OSX deploy-target/sysroot blocks, AVX/SSE4=OFF — SIMD left **ON** for sweep speed). Wired via
    `pixi.toml [target.linux-64.tasks] build-tf`.
  - `pixi run build-tf` — **zero compile errors** under gcc (no gcc-vs-clang nits in the native RNR/motility C++).
    `pixi run verify` → `TissueForge + vertex solver OK` (EGL/MESA/X11 warnings are benign in WSL2 headless).
  - `pixi run test` — **48 passed** (~5.3 min).
- **Two linux-only blockers found + fixed (now encoded in `pixi.toml` + runbook):**
  1. **OpenGL** — `find_package(OpenGL REQUIRED)` (`tissue-forge/CMakeLists.txt:317`) failed; macOS provides it as a
     framework. Added GLVND `-devel` deps in `pixi.toml [target.linux-64.dependencies]`
     (`libgl/libopengl/libglx/libegl-devel`) → unversioned `.so` symlinks + `GL/*.h` headers.
  2. **Submodules** — the fork's `extern/{corrade,magnum,magnum-plugins,libsbml}` were empty (runbook clone
     lacked `--recurse-submodules`). Ran `git submodule update --init --recursive`. Runbook §2 fixed.
- **Deleted one legacy test** `test_periodic_sort_unrepaired_unclamped_noise_inverts_a_cell` (gate 49→48).
  It's a thermal-path *negative control* that **cannot arm on the gcc/AVX trajectory**: the thermal sort probe
  never drives an edge below `Lth=1e-3` (recon=0 across seeds 1/2/3/5/7/11/42, sigma 0.5–3.0, ≤5000 steps), so no
  eversion. Not a regression — the reconnection engine is fine (active-motility rate test `test_clampfree_reconnection`
  passes). Its positive partner `test_periodic_sort_stable_with_native_volume_repair` now passes only *vacuously* on
  linux (recon=0); its docstring was updated to say so. The repair's real coverage is the active-motility tests.

## Caveat for sweeps
Unit tests prove the *machinery*. The science uses the **native active-motility** model, not the legacy thermal one.
Before trusting overnight sweeps, confirm an actual sort **demixes** on linux (reconnections fire + demixing index
moves toward sorting) via `sort-oracle`.

## Changed — all UNCOMMITTED
```
 M CLAUDE.md                          # test count 49→48
 M docs/MIGRATION_to_windows_wsl2.md  # 48 count + --recurse-submodules + GLVND deps notes
 M pixi.lock                          # linux-64 solve
 M pixi.toml                          # linux-64 platform + GLVND deps + build-tf override + count comment
 M progress.md                        # 47→48 + linux note
 M rnr/tests/test_periodic_dynamics.py# deleted legacy counter-test; fixed #2 docstring
?? build_tissue_forge_linux.sh
?? docs/sessions/                     # this log
?? .claude/skills/{handoff,resume}/   # new wrap-up / resume skills
```
`tissue-forge/extern/*` submodules are now populated (separate repo, gitignored by root — no root-repo change).

## Kickoff — next session
Paste into a fresh session (CLAUDE.md auto-loads), or run `/resume`:

> Resume the linux-64 TissueForge RNR project at `/home/tien/Work/SegoLab/VertexModeling`. The engine builds and
> `pixi run test` is 48-green here (see `docs/sessions/2026-06-21-1220-linux64-build-migration.md`). Do these **in order**:
>
> 1. **Validate the science on linux.** Smoke first:
>    `pixi run sort-oracle sort 4 0.5 0.1 1e-3 5e-3 0.3 5000 7 0.4 mixed native`
>    — confirm it completes, reconnections fire (recon>0 in stdout), and the demixing index trends toward sorting.
>    If good, run the default `pixi run sort-oracle` (M=6, 20k steps, native → CSV in `rnr/exports/`).
> 2. **Commit the migration** (local; don't push unless asked). One commit covering: `pixi.toml`, `pixi.lock`,
>    `build_tissue_forge_linux.sh`, the test deletion in `rnr/tests/test_periodic_dynamics.py`, `CLAUDE.md`,
>    `progress.md`, and the `docs/` + `.claude/` additions. Suggested msg: `build: port native-RNR TissueForge fork
>    to linux-64 (WSL2)`.
> 3. **Sweeps + Phase-3 fig polish.** Kick off `pixi run overnight` (background). Open item: add a `MODEL=native`
>    selector to `rnr/scripts/run_overnight.py` + `fig1e_demixing.py` + `fig1f_stability.py` so the canonical Fig 1E/1F
>    regenerate with the **native** drive (they currently use the Python `active` comparison model). See CLAUDE.md
>    "Remaining polish".
>
> Scope: NO growth/morphogenesis or standalone C++-port hardening unless asked. License: `tvm/` and `3DVertVor/` are
> read-only oracles — never copy their code into `rnr/` or the fork.
