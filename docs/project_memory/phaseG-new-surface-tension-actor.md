---
name: phaseg-new-surface-tension-actor
description: Phase G het tension = a NEW native σ·A surface-tension actor; do NOT modify/delete the existing Adhesion actor
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 2159783a-b26d-4549-812d-8b6845222894
---

For Phase G of the native RNR port (the faithful heterotypic-tension drive), build a **NEW,
separate** native per-type-pair linear surface-tension actor (e.g. `SurfaceTension` /
`HeterotypicTension`): `energy = σ_ij·A_ij` on heterotypic interior surfaces (homotypic σ=0).
The force is the area-gradient — a near-clone of `SurfaceAreaConstraint::force`'s Surface-variant
`ftotal` loop scaled by a constant σ prefactor + a body-type-pair gate, bound like `Adhesion`'s
`typePairs` (+ SWIG `.i` wrap + a `BodyTypeSpec` field so Python sets σ_ij like `adhesion={...}`).

**Hard constraint (user, 2026-05-30):** do **NOT** modify or delete the existing `Adhesion`
actor (`tissue-forge/source/models/vertex/solver/actors/tfAdhesion.{h,cpp}`). Create the new
actor alongside it.

**Why:** TF's Body-`Adhesion` energy is `0.5·λ·Σ|edge|` (perimeter/line tension), but the
paper/oracle (Lawson-Keister/Manning 2024, `reference_pdfs/Manning2024*.pdf`; 3DVertVor) minimize
`σ_ij·A_ij` (area tension). Different geometry, different sorting drive — this is the known oracle
mismatch ([[oracle-comparison-ceiling-physical]]). Reimplement from the Okuda + Manning PDFs, NOT
from GPL `tvm`/`3DVertVor`.

**How to apply:** Phase G is done AFTER the reconnection port (Phases A–F) lands. Do not fold it
into the reconnection port. See `docs/native_rnr_port_plan.md` §Phase G and the approved plan at
`~/.claude/plans/idempotent-riding-wind.md`. Related: [[phase2-sorting-partial]].
