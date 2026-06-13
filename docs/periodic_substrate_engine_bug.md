# P5 blocker — the periodic vertex-mesh substrate blow-up (2026-06-09 diagnosis, 2026-06-10 RESOLVED)

## ⚑ RESOLUTION (2026-06-10): it was NOT an MD-engine bug — it was `FlatSurfaceConstraint`

The cutoff/cell-list/ghost diagnosis below is **WRONG**. Instrumenting the per-step force path
(`scripts/diag_force_when.py`, then C++ stage-by-stage logging in `engine_step`) showed the spurious
~1732/10666 wall force is:
- injected at `MeshSolver::preStepJoin` → it's a **vertex-model actor force** (the `_forces` buffer),
  not an engine force;
- **cutoff- AND radius-invariant** (identical at cutoff 3.0→0.5 and radius 1.0→0.01) — so the whole
  "cutoff threshold ≈ min-image distance" story (point 6 below) was an artifact of the radius>cell
  **freeze**, not a real cell-list/ghost effect;
- present with **zero** user actors, because every `SurfaceType` auto-binds **`FlatSurfaceConstraint`**
  + `ConvexPolygonConstraint` (default `lam=0.1`) in its C++ constructor.

`FlatSurfaceConstraint::force` used a **raw** `centroid - vertex_position`; for a surface wrapping a
periodic wall the centroid (min-image) and the vertex (raw) sit in different images, so the offset is
≈ box-sized, and the actor's `×mass/dt×lam` prefactor turns it into a ≈1/dt force → cell inversion.

**Fix:** min-image that offset via `meshRelativePosition(...)` in `force()`+`energy()` of
`actors/tfFlatSurfaceConstraint.cpp` (no-op when `mesh.periodic_geometry` is off). One file, in the
LGPL fork; **no MD-engine change**. `ConvexPolygonConstraint` already min-images. This is the same
omission §6g fixed for Volume/SurfaceArea constraints — these two default actors were missed then.

**Verified:** σ=0 periodic foam now integrates 4000+ steps with min_vol=max_vol=4.000 (was inverting
by ~step 500); σ=1 and σ=1+kT=0.1 thermal noise also STABLE at the LARGE cutoff (so noise is alive —
the "small cutoff kills noise" dilemma is gone). New dynamic gate `rnr/tests/test_periodic_dynamics.py`;
`pixi run test` = 42 green; finite-cluster behavior unchanged. See PORTING_NOTES.md §6i.

**Still open (separate issue):** a *sort* run (reconnection on) still destabilizes on the first
reconnection — that is the known reconnection displacement-overshoot (§6e/§7), not this substrate bug.

---

_Original (2026-06-09) diagnosis below — retained for the evidence chain, but its root-cause
attribution (MD-engine cutoff/ghost) is superseded by the resolution above._

## TL;DR

P5 (run the sort in a periodic bulk) was failing because the **periodic Kelvin-foam
substrate is dynamically unstable at the engine level** — cells invert within a few hundred
steps **even with zero physics** (σ=0, no reconnection, no noise). This is NOT reconnection
violence and NOT a tuning problem. It is a spurious force the **MD engine** applies to
vertices on/near the periodic box walls, coming from the **cutoff / cell-list / periodic-ghost
machinery**, not from the vertex-model actors. The static P1–P3 tests passed because they only
checked rest-geometry/force *values* on hand-built configs; they never integrated a full
periodic foam under dynamics.

A small cutoff (≤ ~0.8) makes the substrate perfectly stable — but the **same cutoff knob also
kills the engine's thermal noise** (`tf.Force.random`), so the two cannot be satisfied at once
through this knob. And in quick tests, the periodic bulk did **not** sort even when stabilized.

## The evidence chain (each step is a runnable diagnostic)

1. **Substrate blows up with zero physics.** `rnr/scripts/probe_periodic_substrate.py 4000 0.0`
   (σ=0, no reconnection, no noise) → a cell inverts (min_vol < 0) by step ~500.
2. **Not an integration overshoot.** Sweeping dt DOWN (1e-4 → 1e-7) makes it *worse* at fixed
   step count, not better → the blowup is a near-instant huge force at t≈0, not accumulation.
3. **Localized to wrap vertices.** `rnr/scripts/diag_periodic_forces.py`: at t=0 the geometry is
   perfect (every vol=4.0, every face area 0.5–1.3, uniform shape index 5.315), yet a specific
   set of vertices get |F|≈1732 (=1000·√3) while the rest are at equilibrium. Every high-force
   vertex sits **on/adjacent to a box wall**.
4. **Not the actor forces.** `rnr/scripts/diag_swig_force.py`: calling the SWIG actor
   `force(Body,Vertex)` directly on the rest geometry gives a **net-zero, balanced** force on a
   wrap vertex (each of its 4 bodies contributes 0.1438, they cancel). A Python replication of
   the `SurfaceAreaConstraint::force` loop agrees (~0.14/body, wrap faces handled correctly).
   With **all** actor coefficients zeroed (`volume_lam=0, surface_area_lam=0, adhesion=0`) the
   boundary vertices **still** fly. So the force is not the mesh actors.
5. **Not mass/density, not a pairwise potential, not a BC velocity.** Vertex mass is uniform
   (boundary = interior); body `density` defaults to 0 (so particles keep default mass);
   two bare mesh particles 0.7 apart (even across the wall) feel **zero** force; all BCs are
   plain PERIODIC with zero velocity.
6. **It is the cutoff / cell-list / ghost machinery.** Substrate stability vs `tf.init(cutoff=…)`:
   | cutoff | cells/axis (L=8) | substrate |
   |--------|------------------|-----------|
   | 0.5–0.8 | 10–16 | **STABLE** (min_vol=4.000 for 3000+ steps) |
   | 1.0–3.0 | 2–8 | UNSTABLE (cell inverts in ~250 steps) |
   The threshold (~0.8–1.0) ≈ the **minimum-image distance between mesh vertices on opposite
   box walls (~1.0)**: when cutoff reaches that, the engine pairs a wall vertex with the
   periodic ghost of an opposite-wall vertex and the in-step force evaluation corrupts the
   wrap-face geometry → huge force. `position_changed()` itself is idempotent and correct; the
   corruption happens during the engine's in-step cell-list/ghost rebuild + `engine_force`.
7. **Requires touching the walls.** The identical Kelvin foam built in a **big box** (universe
   dim=24, foam in [8,16]³, away from the walls) is **stable at cutoff=3.0** — same as the
   validated finite cluster (dim=60, cutoff=3.0). So the bug needs the foam to reach the
   periodic boundary (where ghosts are created).

## Why the cutoff workaround is not a fix

- `cutoff ≤ 0.8` stabilizes the substrate, BUT the **engine thermal noise dies with cutoff**:
  bare-vertex diffusion after 2000 steps at kT=0.1 is 0.00 (cut 0.7) / 0.21 (1.0) / 0.40 (2.0) /
  0.42 (3.0). The bound `tf.Force.random` is applied through the **same cutoff-dependent
  integration path** (the non-verlet midpoint `engine_force` that also produces the spurious
  force). So: small cutoff ⇒ stable but **no noise**; large cutoff ⇒ noise works but **unstable**.
- The paper's sorting *requires* thermal noise (kT=0.1) to fluidize the tissue, so "small cutoff"
  alone can't reproduce it.

## And even when stabilized, the periodic bulk did not sort (preliminary)

Using the stable small cutoff (0.7) + **manual Python per-vertex noise** (calibrated diffusive
displacement, bypassing the dead engine noise) + native reconnection:
- **Perfect Kelvin lattice:** a symmetric mechanical equilibrium — het-tension forces are
  perfectly balanced, so with no noise it is *frozen* (zero motion, zero reconnections); with
  noise it churns *randomly* (D stays ~0, het-area stays ~0.49) and the volume spread grows
  until a cell inverts.
- **Disordered (random-Voronoi) periodic foam, no noise:** reconnects ~424× in the first 1000
  steps then **jams** (reconnections stop, D≈−0.007, het-area≈0.52); cell volumes become very
  unequal (0.77–9.95 vs V0=4).
- **Smaller Lth (0.1, 0.05):** gentler/fewer reconnections, still **no demixing**.

So in these quick tests the periodic bulk sorts **no better than — actually worse than — the
finite cluster** (D≈0 vs the finite −0.065). This is *preliminary* (short runs, σ=1 vs the
paper's 0.04–0.64, crude manual noise), but it means a periodic-mesh engine fix is **not
guaranteed** to unlock the paper's deep sorting; the sorting deficit needs its own investigation.

## What a real engine fix would look like (if chosen)

The mesh vertex particles have **no pairwise potential**, so they should not participate in the
nonbonded cell-list/ghost force machinery at all. The fix is to make the in-step force/geometry
evaluation use the mesh's **canonical (min-image) vertex positions** rather than the
cell-list-reassigned ghost images — e.g. exclude mesh particles from ghost generation / the
nonbonded loop while still letting them receive bound forces (noise) and be integrated. That
would let a **large cutoff** be used (keeping engine noise alive) without the spurious wrap force.
This is a non-trivial change in `tf_engine_advance.cpp` / the space/cell-list + `tfMeshSolver`
interaction, and should be gated by a new **dynamic** test: integrate a full periodic foam at
σ=0 for N steps and assert min_vol>0 / max_vol≈V0 (the static P1–P3 tests miss this).

## Reproducers (all under `rnr/scripts/`, pure-Python, no engine rebuild)

- `probe_periodic_substrate.py` — substrate blows up at σ=0 (the headline).
- `diag_periodic_forces.py` — t=0 geometry is perfect but wrap vertices get |F|≈1732.
- `diag_swig_force.py` — direct actor `force()` is balanced (≈0); the step still moves the vertex.
- `diag_force_loop.py` — Python replication of the area-gradient loop (sane ~0.14/body).
- the cutoff scan + big-box control in this session's transcript (one-off scripts).

## UPDATE (2026-06-09, same session) — deeper findings; the derisk path is itself blocked

Following the user's choice to "derisk sorting first" (run the periodic bulk with the stable
small-cutoff workaround + manual noise before paying for the engine fix), I found the workaround
does **not** give a working testbed, and the situation is worse than "a ghost-force bug":

- **The small-cutoff "stability" was the mesh being FROZEN, not stabilized.** Mesh particles have
  `radius=1.0`; the engine cell-list will not integrate a particle bigger than its cell
  (cell≈cutoff), so at `cutoff ≤ ~0.8` the actor forces are *computed* (direct SWIG force = 10.1,
  `p.force` after a step = 471) but **never applied** — an off-equilibrium random foam (cells
  0.18–2.44, V0=1) stays frozen at its initial volumes forever. So my earlier `cutoff=0.7`
  "sort" tests had **dead het tension** — those negative sorting results were meaningless.
- **Whenever the integrator actually moves the mesh particles, the wall-filling foam blows up** —
  at *any* cutoff (lower radius to unfreeze → blows up at cutoff 0.3 too), and at *any* boundary
  condition: periodic, free-slip, AND none all invert a cell by ~step 500. So it is **not** the
  particle-level periodic ghosts. The clean perfect Kelvin lattice at exact equilibrium
  (radius 0.3 < cutoff 0.5 < edge 0.707, zero perturbation, σ=0) blows up too.
- **The discriminator is "foam touches the box walls."** The identical Kelvin foam in a big box
  away from the walls (universe dim=24, foam in [8,16]³) is stable at cutoff=3 — same as the
  validated finite cluster. So the bug is the in-step mesh-geometry handling of vertices on the
  box walls under `mesh.periodic_geometry`, not the actors (rest force is balanced/correct) and
  not the BC ghosts.
- **No Python knob (cutoff, radius, dt, BC) yields stable working periodic-foam dynamics.**

**Consequence:** the periodic-bulk sorting derisk **cannot be done in Python** — there is no
working periodic testbed. And the **finite cluster** (the only stable regime) cannot reach the
paper's faithful regime either: at the oracle's gentle σ (σ/K_A·V0^(2/3) ≈ 0.04–0.64) the free
boundary jams so het faces never collapse to the reconnection trigger (athermal → frozen; with
noise → random reconnections, no sorting), while the strong σ that does drive the finite cluster
(σ/K_A·V0^(2/3) ≈ 4, what the old harness used) is ~6–100× the paper and is not faithful.

**So the engine fix is now a hard PREREQUISITE, not an optional faithfulness improvement.** Scope:
make the in-step mesh geometry read consistent (min-image canonical) particle positions for
box-wall vertices so a space-filling periodic foam integrates stably — likely in the engine's
per-step position path (`Vertex::positionChanged` caches `global_position()`; `Surface/Body`
geometry + the cell-list/space rebuild), NOT in the actor force methods. Gate it with a NEW
**dynamic** test: integrate a periodic foam at σ=0 for N steps, assert min_vol>0 / max_vol≈V0.
The exact offending line needs in-engine instrumentation (the behavioral box is fully closed:
wall-touching + integrating + periodic_geometry → blowup; everything else stable).

### Oracle parameters (read from 3DVertVor, for the faithful run once the engine is fixed)
`Energy/Volume.cpp`: K_V (`kv_`) = 10, pressure = −2·kv·(V−V0). `Energy/Interface.cpp`:
polygon tension = `2·(A_cell − A0) + σ_ij`, so K_A = 1, `s0_` = 5.40 (paper 5.6), A0 = s0·V0^(2/3).
`Reconnection/Reconnection.cpp`: `Lth_ = 1e-3` (gentle; placement at ±0.5·Lth / Lth·v̂). `main.py`:
N = Lx·Ly·Lz random points in [0,L]³ ⇒ **V0 = 1**, N = 1728 in 12³, `dtr = 10·dt`. Paper noise:
white thermal, kT=0.1, μ=1 (the GPL checkout uses active `motility_` instead; thermal line
`Run.cpp:1344` is commented). Harness: `rnr/scripts/sort_periodic_oracle.py` (params wired;
blocked by the engine).

## Bottom line for the project

- The P5 "blows up even at Lth=0.01" mystery (from the prior session's
  `probe_periodic_substrate.py`) is **explained**: it is this engine cutoff/ghost bug, present
  regardless of reconnection.
- The periodic vertex mesh (P0–P4) is **not** dynamically sound, despite the green static tests.
- Decision needed (see the session report): (A) fix the engine periodic-ghost/mesh-particle
  interaction (deep C++; the "correct" path, but the sorting deficit means uncertain payoff);
  (B) investigate the sorting deficit itself (noise/σ/volume-stiffness/reconnection placement),
  possibly via direct numerical comparison to the 3DVertVor oracle's exact setup; or (C) revisit
  whether the finite-cluster path can be pushed deeper. Recommend (B)-then-(A): confirm the
  periodic regime *can* sort (against the oracle) before paying for the engine fix.
