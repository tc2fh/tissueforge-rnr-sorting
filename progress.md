# RNR in TissueForge — Progress

> Companion to `CLAUDE.md`. **Canonical technical record:** `rnr/PORTING_NOTES.md`; reasoning
> index: the auto-memory under `.claude/.../memory/`; per-session logs: `docs/sessions/`.
> Last updated: **2026-06-24** — **NEW phase: GPU port** of the 3D vertex model + RNR (Warp→CUDA,
> forward-sim only). **Gates A + B + C done & green** (incl. **C2c** — the iterated I→H sweep glued
> end-to-end on the GPU) — the novel result is realized: parallel, conflict-free, element-count-CHANGING
> I→H on a GPU-resident ragged 3D mesh (atomic reservation + simultaneous count-changing surgery on the
> RTX 5090), validated against the host reference. The **H→I reverse detector (C0′)** is also done.
> Full gate `pixi run test` → **81 passed**. Next: finish the H→I scheduler (host footprint/reserve +
> `h_to_i_batch_kernel`), then on-GPU detection, Gate D (compaction), Gate E (force kernels + Fig 1E/1F
> sorting). Plan + progress §10: `docs/2026-06-24_gpu-3d-vertex-model-exploration.md`; latest session
> log: `docs/sessions/2026-06-24-0938-gpu-rnr-gate-c-sweep-and-h-detector.md`. (Prior GPU sessions:
> `docs/sessions/2026-06-24-0859-gpu-rnr-gate-b-and-c.md`, `…-0715-gpu-3d-vertex-port.md`.)

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
