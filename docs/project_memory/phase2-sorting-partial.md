---
name: phase2-sorting-partial
description: "pixi run sort demixes; SCALE-UP (N=5/dt=1e-4) gives demixing index D -0.043->-0.098 (2.3x), het 460->447, clean stability vs frozen control; new metrics.demixing_index + tests (pixi run test green 11); dt must scale DOWN with block density"
metadata: 
  node_type: memory
  type: project
  originSessionId: 1d93d81f-d80c-47e1-8eed-690fbb4f10b8
---

Phase 2 (wire reconnection into the loop + reproduce sorting) is COMPLETE — gate met.
`pixi run sort` (`rnr/scripts/sort_with_reconnect.py`) runs a matched ON-vs-frozen-control
pair in one universe and demixes. Baseline (N=4, 91 cells, dt=2e-4): het pairs 194→189 vs
frozen 194, bounded stability (worst min_vol −6.69).

The hard part was the DYNAMICS, not the reconnection: under heterotypic tension a
reconnection can trigger a TF signed-volume winding SIGN-FLIP that reverses the
VolumeConstraint force and inflates a cell without bound. Tamed with small dt; the faithful
regime needs a native abs(volume) repair (Phase 3). See
[[faithful-instability-is-winding-signflip]]. dt is the MASTER stability lever (not the
energy gate, not a guard): dt=1e-3 → runaway (min_vol→−8e4, metric freezes on corrupted
mesh, made an earlier "194→190" look clean); dt=2e-4 keeps sign-flips transient.

## Scale-up (2026-05-30) — dramatic + quantified, still pure Python

- **`rnr/metrics.demixing_index`**: cell-avg `2·(het_frac − ½)` ∈ [−1,+1], SIGNED so more
  sorted = more negative (departure-note vs 3DVertVor's positive convention; chosen to fall
  with the het/area/energy curves). Topological → flat under shape relaxation / frozen
  control. `contact_summary` returns it; `pixi run sort` plots `D(t)` ON-vs-OFF (headline,
  6-panel). Deterministic gates: hand-enumerated minimal-[I] (`D=0.4` wedges=A/caps=B; `D=−1`
  uniform) + Kelvin (planar split `D<−0.4`, salt-pepper `|D|<0.25`). `pixi run test` green (11).
- **Headline run** `pixi run sort 8000 0.45 1.0 0.0001 5` (N=5 → 189 cells/block, dt=1e-4):
  **D −0.043 → −0.098 (2.3×)** vs control pinned at −0.043; **het 460→447 (−13)** vs frozen
  460; het-area 0.385 vs 0.417; energy 438 vs 459; 285 reconnections; sustained plateau below
  OFF. Clean stability: worst min_vol −1.28, bad≤7 transient. Artifact:
  `rnr/exports/sort_with_reconnect_n5_dt1e4.{png,csv}`.
- **KEY FINDING — dt must scale DOWN with block density** (extends the dt lever): same N=5 run
  at dt=2e-4 corrupted the mesh (15–25 sign-flips, min_vol −7.9); dt=1e-4 reaches the same
  D≈−0.1 plateau cleanly. Rule of thumb: scaling `N_PER_AXIS` up ⇒ scale `dt` down (~÷2 per +1).
- **Demix is PARTIAL** (D≈−0.1, not −1) and TWO Step-3 tuning knobs were TESTED — both fail to
  beat greedy (so greedy D≈−0.10 is near the ACCESSIBLE optimum for the I↔H move-set on a finite
  189-cell block, NOT a tuning miss):
  - stronger het λ: ΔE sign is λ-independent → identical greedy decisions; λ=2 hits the SAME
    D≈−0.099 plateau ~2× faster (by t≈2600) then destabilises (min_vol −4.5) — stronger drive
    wants a smaller dt, like a denser block.
  - Metropolis annealing (new `operator.OperatorParams.temperature`; uphill kept w.p. exp(−dE/T),
    T=0=greedy default so tests unchanged; schedule T0·exp(−step/τ)): T0=0.5/τ=4000 churned MORE
    (recon 406 vs 285) but settled WORSE on D (≈−0.079; uphill moves changed coordination, raised
    per-cell het frac) AND then DESTABILISED catastrophically (after t≈6800: min_vol→−343, 45/189
    bad cells) — uphill churn triggers the sign-flip cascade. Worse on BOTH counts. One hot
    schedule only; kept as a tested default-off knob + C++-port note.
  Path to full demix (D→−0.3+) likely needs a LARGER/PERIODIC system (3DVertVor uses periodic
  1728 cells), not more tuning of the finite block. Artifacts: `_n5_dt1e4` (greedy headline),
  `_n5_anneal` (annealing experiment). Details in `docs/rnr_sorting_notes.md §7`.
  `sort_with_reconnect.py` now takes N_PER_AXIS (5th arg, SPAN=2·N auto-tracks edge ~0.707) +
  TEMP0/ANNEAL_TAU (6th/7th) and checkpoints CSV+PNG every 2000 steps.

## Tasks (two deliverables)
- `pixi run sort` (`sort_with_reconnect.py`) — the PRAGMATIC demo that sorts.
- `pixi run sort-faithful` (`faithful_run.py`) — the reference-faithful DIAGNOSTIC that
  anti-sorts + carries the orientation diagnostic proving the native repair is needed.
3DVertVor oracle comparison still deferred (optional; never copy its GPL code).
