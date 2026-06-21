---
name: adhesion-force-is-already-area-tension
description: "CRITICAL Phase-G finding — TF's Body-Adhesion FORCE is already area tension −∇(λ·A_het); only its energy() is perimeter, and energy() is never used by the integrator"
metadata: 
  node_type: memory
  type: project
  originSessionId: 81faf2b8-7985-490c-9a90-7729d1e724b3
---

**TF's Body-`Adhesion` *force* is ALREADY the area-based heterotypic tension `−∇(λ·A_het)`** —
the σ·A drive Phase G was meant to add. Verified 2026-06-01 against the fork source + numerically.

The chain (all confirmed):
1. `Adhesion_force_Body` (tfAdhesion.cpp:68-104) = `0.25·λ·Σ_{het surfaces of v on b} ftotal_loop`,
   where `ftotal_loop` is the EXACT `SurfaceAreaConstraint::force(Body*,Vertex*)` area-gradient
   cross-product loop.
2. Derived + numerically verified (`Surface` area uses centroid = vertex mean): `ftotal_loop = −2·∇A_s`
   (cos similarity 0.9966–0.9999 vs −2∇A; SAC's own force `ftotal_loop·λ(A−constr)` = `−∇[λ(A−constr)²]`
   exactly, a strong consistency check).
3. The solver sums `force(b1,v)+force(b2,v)` for each shared surface (VertexForce iterates
   `v->getBodies()`), so per het surface: `2·0.25·λ·(−2∇A_s) = −λ·∇A_s`. Total = `−∇(λ·A_het)`. ∎
4. Actor `energy()` is NEVER called by the integrator — only by C-API getters
   (`wraps/C/.../tfCMeshObj.cpp:112,159`). Dynamics is purely force-based (overdamped v=μF). So the
   perimeter form in `Adhesion_energy_Body` (`0.5λΣ|edge|`) is INERT in the dynamics.

**Consequence for Phase G:** a σ·A actor whose force mirrors `Adhesion_force_Body` (the kickoff
plan's design) is FORCE-IDENTICAL to `Adhesion(λ=σ)` and CANNOT "beat the Phase-F baseline" on
sorting (energy-gate-OFF path). Its only real value: a CONSISTENT `energy()=σ·A` (matters only if the
native reconnection ENERGY GATE is used — which the plan keeps OFF) + a decoupled σ_ij API.

**The real ceiling is STABILITY, not the tension type.** Phase-F empirically demixes with Adhesion
(D −0.043→−0.078, het_area 0.48→0.36) but a cell INVERTS ~step 12k (Lth=0.4 INT=10) / ~step 14k
(INT=20) — slow min_vol erosion from accumulated reconnection perturbations (§6e validated only to
8000 steps). Deeper/longer sorting is blocked by this erosion (oracle's abs+flip orientation repair,
§6d#3), not by needing area tension. See [[native-instability-is-displacement-overshoot]],
[[oracle-comparison-ceiling-physical]]. Phase-F baseline script: `rnr/scripts/phase_f_baseline.py`.
