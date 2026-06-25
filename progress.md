# RNR in TissueForge — Progress

> Companion to `CLAUDE.md`. **Canonical technical record:** `rnr/PORTING_NOTES.md`; reasoning
> index: the auto-memory under `.claude/.../memory/`; per-session logs: `docs/sessions/`.
> Last updated: **2026-06-24** — **NEW phase: GPU port** of the 3D vertex model + RNR (Warp→CUDA,
> forward-sim only). **THE FULL STAGED PLAN (Gates A–E) IS DONE & green.** Gates A–D + the H→I scheduler
> + on-GPU detection + device gather (a whole reconnection round runs with NO `from_warp`), PLUS **Gate E**
> — Stage-1 physics: geometry + the 4 sorting forces + overdamped integrate + director rotational
> diffusion as fp64 Warp kernels, composed into `engine.forward_step`, validated host==TF (float32),
> GPU==host (fp64), a deterministic GPU trajectory == host to 9e-16, and a mixed IC that demixes
> (3DVertVor cell sorting on the GPU). Full gate `pixi run test` → **127 passed** (79 GPU tests on the
> RTX 5090). **2026-06-24 PM — paper-scale stability BUG FOUND then FIXED.** A cell ballooned once
> cumulative I→H crossed ~1000. Root cause (refined): a face wound inconsistent with its `b1/b2`
> breaks cell closure (`Σ sense·snorm≠0`) → wrong, origin-dependent volume → balloon. Two sources:
> near-degenerate INITIAL foam faces (zero area ⇒ arbitrary winding) sitting just above `lth` so
> H→I never collapses them (dominant); + occasional `b1/b2`-inconsistent I↔H surgery output. No
> geometric per-face test detects mis-winding (it's topological). **FIX:** `orient_warp.orient_repair_warp`
> (closure-residual ring-flip) after each sweep in `engine.forward_step`. **PAPER SCALE PASSES:**
> n=10 (2000 cells) 100k steps STABLE, 6010 I→H, vol [0.90,1.05], het 0.497→0.447; `pixi run test`=127.
> (`tvm`/3DVertVor never need this — they re-derive orientation topologically every reconnection via
> `updatePolygonDirections`; our implicit ring+b1/b2 model loses the invariant.) See memory
> `gpu-rnr-scale-corruption` (RESOLVED) + session log below.** Other optional/post-plan:
> batched multi-mesh stepping, a faster periodic-foam builder (O(N²) setup ceiling), hand-CUDA-in-fork.
> Plan + progress §10: `docs/2026-06-24_gpu-3d-vertex-model-exploration.md`; latest session log:
> `docs/sessions/2026-06-24-2345-gpu-rnr-winding-balloon-fixed.md`.

## Status — the Phase-2 goal is REPRODUCED ✅

3DVertVor/Manning **Fig 1E + 1F reproduced** in TissueForge's 3D vertex model, faithfully
(clamp-free), via native 3D I↔H reconnection (the missing 3D T1).

| Phase | State |
|---|---|
| 0 — environment + jammed control (no reconnection) | ✅ done |
| 1 — I↔H reconnection + round-trip reversibility gate | ✅ done |
| 2 — wire into the loop + reproduce sorting | ✅ done — Fig 1E/1F, clamp-free active motility |
| 3 — standalone C++ `MeshQuality` port hardening | later (native I↔H + repair already on the fork; PORTING_NOTES §6) |
| — growth / morphogenesis | later (out of scope) |

## What closed Phase 2 (2026-06-11/12)

The oracle's noise is **active self-propulsion** (`x += dt·motility`, sub-Lth/step), **not** thermal
Brownian noise (√dt, 14–45× Lth — which starves the reconnect trigger and had forced a "noise-clamp"
departure). Switching the harness to the faithful active model removed the last departure and
reproduces the prior thermal+clamp result *with no clamp*:

- **Fig 1E** — area-demixing σ-ordered: `S_area = {0.022, 0.057, 0.116}` for σ = {0.1, 0.2, 0.5}
  (3-seed ensemble, 100k steps). Count-based DP stays ≈ 0 (finite-N limited at N=216).
- **Fig 1F** — the demixed state holds for every σ: `DP/DP_max = {0.89, 0.90, 0.91}` (> 0.8) vs the
  mixed IC ≈ 0 — direct evidence the energetics + native RNR are correct.

Details: `rnr/PORTING_NOTES.md` §6n; memory `active-motility-not-thermal-noise`.

## Pipeline (`pixi run <task>`)

`test` (48-test gate) · `probe-active` (clamp-free rate gate) · `sort-oracle` (one sort → CSV) ·
`overnight` (full 18-sim ensemble + figures + video, failure-tolerant) · `video` · `dpmax` ·
`fig1e` · `fig1f` (`MODEL=active`) · `build-tf` / `verify` (engine build + smoke).

Deliverables in `rnr/exports/`: `fig1e_demixing_active.png`, `fig1f_stability_active.png`,
`sort_active_demixing.gif`, + 18 active CSVs (`sort_oracle_M6_S*_..._active[_demixed].csv`).

## Layout

- `rnr/` — modules (`topology`, `reconnect`, `conditions`, `operator`, `metrics`, `geometry`) +
  `scripts/` (the 9-script faithful pipeline) + `tests/` (48 green) + `PORTING_NOTES.md`.
- `tissue-forge/` — engine fork; `tissue-forge_build/` — its CMake build tree.
- `tvm/`, `3DVertVor/`, `oracle_run/` — GPL reference oracles (read / compare only; never copied).
- `docs/` — findings notes; `reference_pdfs/` — Okuda 2013, Manning 2024, Zhang–Schwarz.
