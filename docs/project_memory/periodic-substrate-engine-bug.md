---
name: periodic-substrate-engine-bug
description: "RESOLVED 2026-06-10: periodic substrate blow-up was FlatSurfaceConstraint (a default surface actor) using a raw non-min-image centroid-vertex offset, NOT an MD-engine cutoff/ghost bug. One-line fix; periodic foam now integrates stably."
metadata: 
  node_type: memory
  type: project
  originSessionId: 041eedb3-d9e4-4f2e-ade5-85481ac1b810
---

**RESOLVED 2026-06-10.** The P5 periodic-substrate blow-up (a space-filling periodic Kelvin foam
inverts a cell within ~hundreds of steps even at σ=0, no reconnection, no noise) was **NOT** the
MD-engine cutoff/cell-list/ghost bug diagnosed on 2026-06-09. That earlier diagnosis was wrong:
the "cutoff ≤ 0.8 is stable" finding was the radius>cell **freeze** (mesh not integrating), and the
real spurious force is **invariant to both cutoff (3.0→0.5) and radius (1.0→0.01)**.

**Actual root cause:** every `SurfaceType` auto-binds two default actors in its C++ ctor
(`tfSurface.cpp` `SurfaceType::SurfaceType`, default `flatLam=convexLam=0.1`):
**`FlatSurfaceConstraint`** + `ConvexPolygonConstraint` — present on EVERY surface regardless of the
Python spec. `FlatSurfaceConstraint::force/energy` computed the out-of-plane offset as a **raw**
`getCentroid() - getPosition()`. The centroid is min-image-correct but the vertex position is the raw
global coord, so for a surface **wrapping a periodic box wall** the two are in different images →
offset ≈ box-sized. With the actor's `×mass/_Engine.dt×lam` ("snap to plane in one step") prefactor
that's a spurious ≈1/dt force (~10⁴ at dt=1e-4) along the wall normal on every wall vertex → inversion.
Only manifests in the periodic bulk (finite cluster has no wrap faces) and the static P1–P3/§6g tests
only checked rest-geometry VALUES, never an integrated foam.

**Localized by:** instrumenting the per-step force path — `scripts/diag_force_when.py` (force=0 at rest,
~10⁴ after one step) then temporary C++ `engine_step` stage logging showed the force enters at
`MeshSolver::preStepJoin` (the actor `_forces` buffer), and per-vertex actor dump named the culprit.

**Fix (1 file, LGPL fork, NO mdcore change):** `actors/tfFlatSurfaceConstraint.cpp` — use
`meshRelativePosition(source->getCentroid(), target->getPosition())` (from `tf_mesh_metrics.h`) in
both `force()` and `energy()`. It's the identity when `mesh.periodic_geometry` is OFF, so finite-cluster
behavior is unchanged. Same omission §6g fixed for Volume/SurfaceArea constraints — these two default
surface actors were simply missed then. `ConvexPolygonConstraint` already min-images
(`metrics::relativePosition`), left as-is. Rebuild: `pixi run build-tf` (or incremental ninja+install).

**Verified:** σ=0 periodic foam integrates 4000+ steps at min_vol=max_vol=4.000 (was inverting by
~step 500); σ=1 and σ=1+kT=0.1 noise also STABLE at the LARGE cutoff (3.0) — so engine thermal noise
stays alive AND the foam is stable, dissolving the old "small-cutoff workaround kills the noise"
dilemma (the cutoff was never the cause). New dynamic gate `rnr/tests/test_periodic_dynamics.py`
(subprocess-isolated, since tf.init is one-per-process); `pixi run test` = 42 green. Full writeup
`docs/periodic_substrate_engine_bug.md` (resolution header) + PORTING_NOTES §6i.

**Separate, NOW ALSO RESOLVED (2026-06-10, [[periodic-sort-noise-overshoot-fixed]]):** a *sort* run
(reconnection ON + noise) destabilized on the first reconnection — the thermal noise (DISP_STD=0.0141)
overshooting the post-reconnection Lth=1e-3 gap (~14×) and everting a cell. Fixed by a per-vertex
trust-region clamp on the noise (0.4×min-image nn-dist); periodic sort now STABLE 20000 steps/84
reconnections, `pixi run test` = 44 green. P5 substrate AND reconnection-under-dynamics are unblocked;
remaining work is the SCIENCE (sigma/Lth sweeps to deepen sorting past the D≈−0.10 ceiling; oracle
params V0=1, K_V=10, K_A=1, tension=2(A−A0)+σ, s0=5.4/5.6, Lth=1e-3, dtr=10dt, kT=0.1; harness
`scripts/sort_periodic_oracle.py`). This supersedes the optimistic-P5 framing AND its own prior
(wrong) engine-bug attribution.
