---
name: periodic-sort-noise-overshoot-fixed
description: "RESOLVED 2026-06-10 — the periodic SORT (reconnection ON + thermal noise) blow-up was the thermal noise overshooting the post-reconnection Lth gap, NOT a placement/min-image bug. A clean recon×noise factorial showed it needs BOTH; DISP_STD=0.0141 is ~14x Lth=1e-3 so one noise kick everts a freshly-reconnected cell (transient, self-recovering). FIXED by a per-vertex trust-region clamp on the NOISE (cap at 0.4*min-image nn-dist). 20000 steps/84 reconnections STABLE; pixi run test=44; gate test_periodic_dynamics.py + PORTING_NOTES §6j"
metadata:
  node_type: memory
  type: project
  originSessionId: native-rnr-reconnection
---

After the substrate fix ([[periodic-substrate-engine-bug]], §6i), a periodic *sort*
(`sort_periodic_oracle.py sort 4 0.5 0.1 1e-3 1e-3 1.9`) still blew up on the FIRST
reconnection (`min_vol −1.78` by step 500, recon~1) while substrate-only was stable.

**Diagnosis (scripts `diag_recon_overshoot.py` + `diag_read_side_effect.py`):**
- The negative volume is **TRANSIENT and self-recovering** — a cell everts then un-everts
  (worst −3.1 at the dip, back to +0.65 a few hundred steps later). Reading `Body.volume`
  every step vs only at the end gives the IDENTICAL final state → the volume getter has no
  side effect; the "−1.78 at step 500" was just the checkpoint catching the transient dip.
- **Factorial (recon × noise), 600 steps:** recon OFF + noise ON → STABLE (+0.29); recon ON +
  noise OFF → STABLE (+0.28); recon ON + noise ON → **UNSTABLE (−3.1)**. Needs BOTH.
- **Mechanism = a magnitude mismatch:** native I→H places two vertices `Lth=1e-3` apart
  (placement is ALREADY min-image — `rnr_placeIToH` uses `meshPositionNear`/`meshWrapPosition`;
  the kickoff's "suspect #1 placement-not-min-image" was checked and is FALSE). One
  Euler–Maruyama kick is `DISP_STD = sqrt(2·μ·kT·dt) = sqrt(2·0.1·1e-3) = 0.0141` ≈ **14×** the
  Lth gap → throws a fresh vertex past its neighbour → eversion. (DISP_STD≫Lth holds for the
  oracle's params too; the oracle survives via orientation repair [§6d#3] — it TOLERATES the
  transient eversion; we PREVENT it.)

**Fix (Python harness, NO engine rebuild):** a per-vertex TRUST-REGION on the noise — cap each
vertex's noise displacement at `NOISE_CLAMP=0.4 × (min-image nearest-neighbour distance)`. Normal
vertices (nn≈0.5, cap≈0.2 ≫ kick 0.014) never bind; only Lth-scale fresh-reconnection vertices do.
Position-level analogue of [[winding-clamp-stabilizes-sort]] but applied to the NOISE only (force
dynamics untouched — justified since recon-ON+noise-OFF is already stable). Implicit-edge topology
is cached, rebuilt only on `num_vertices` change (reconnection) → O(V) reads + O(E) vectorised
(`np.minimum.at`) per step. Implemented in `sort_periodic_oracle.py` (arg 10 NOISE_CLAMP, default
0.4; 0=old unstable) + `probe_periodic_sort.py`.

**Result:** `sort 4 0.5 0.1 1e-3 1e-3 1.9 20000` STABLE for 20000 steps / 84 reconnections
(`worst_min=0.270` = initial Voronoi min, never lower; `worst_max=2.618`, never inflates).

**Science — σ/Lth sweep (M=4, 64 cells, 20000 steps) answers the kickoff's open question "does a
STABLE periodic bulk break the D≈−0.10 finite-cluster ceiling?": YES, once the T1 RATE is raised.**
σ=0.5,Lth=1e-3 → 84 recon, D=−0.089 (STABLE); σ=1.0,Lth=1e-3 → 88 recon, D=−0.099 (STABLE);
σ=0.5,**Lth=0.05 → 6973 recon, D=−0.115 (STABLE**, worst_min 0.077) — **CEILING BROKEN**;
σ=1.0,Lth=0.05 → D=−0.173 by step 6.5k then **EVERTS @ ~11.5k (UNSTABLE)**. So the −0.10 ceiling was
the slow reconnection rate at oracle-gentle Lth, NOT finiteness (Lth=0.05 is a DEPARTURE — Okuda
wants small Lth). Force-overshoot boundary: the doubly-aggressive regime (high σ AND high Lth) sorts
deepest but everts because the NOISE clamp doesn't limit deterministic FORCE overshoot (a tighter
noise clamp doesn't help → confirmed force-driven). A Python TOTAL-displacement clamp (after
`tf.step`) was tried and FAILS here (everts faster) because at Lth=0.05 a reconnection fires almost
every step → the Python post-step clamp must SKIP reconnection steps (handles invalidated) and its
per-vertex pull-back distorts faces under strong tension. → **The force-overshoot trust-region must
be NATIVE (in the vertex integrator, every step, consistent with in-doQuality reconnection)**, not a
Python pass. Reverted; harness keeps the clean noise-only clamp. See
[[oracle-comparison-ceiling-physical]], [[native-instability-is-displacement-overshoot]].

**Gate:** `test_periodic_dynamics.py` +2 subprocess tests over `probe_periodic_sort.py`:
`…_stable_with_reconnection_and_noise` (clamp 0.4 → STABLE) and
`…_unclamped_noise_inverts_a_cell` (clamp 0 → UNSTABLE within ~500 steps, the load-bearing
non-no-op check). **`pixi run test` = 44 green.** Detail: PORTING_NOTES §6j.

**Port note (Phase 3):** prototype clamps Python-applied noise only. The C++ port should put the
per-vertex displacement trust-region in the vertex INTEGRATOR (cap total per-step motion at a
fraction of nn-dist) so it also covers ENGINE noise (`tf.Force.random`, applied inside `tf.step()`)
+ post-reconnection force overshoot — one mechanism for all three displacement sources.
See [[native-instability-is-displacement-overshoot]].

**SUPERSEDED for the faithful regime (2026-06-11):** the noise clamp here was a flagged DEPARTURE.
It is now replaced by the paper's own faithful mechanism — a native orientation repair (oracle
stabilizer #3) — which makes the faithful σ=0.5/Lth=1e-3/kT=0.1 sort STABLE 20000 steps with NO
clamp. The clamp code remains but is not needed for faithful runs. See
[[native-orientation-repair-faithful]]. (The trust-region note above still applies ONLY to the
aggressive-Lth departure regime, which the repair does not cover — it fails by inflation, not
eversion, there.)
