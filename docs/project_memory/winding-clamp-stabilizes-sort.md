---
name: winding-clamp-stabilizes-sort
description: Python-side PER-VERTEX DISPLACEMENT CLAMP (operator.stable_step) prevents the winding sign-flip from Python (no native abs-volume needed for the prototype); makes the sort STABLE (min_vol>0 throughout) AND DEEPER (D -0.10 vs old -0.096 with min_vol -218)
metadata: 
  node_type: memory
  type: project
  originSessionId: 87d0df5f-99f0-4f5a-85bd-0192878140c6
---

Resolved the [[faithful-instability-is-winding-signflip]] blow-up FROM PYTHON (that memory
said the abs-volume repair was native/C++-only — true for the force-level fix, but a
preventive position-level clamp works in the prototype).

**What:** `rnr/operator.py::stable_step(bodies, rel_frac=0.4)` wraps `tf.step()` with a
per-vertex displacement limiter: snapshot positions, step, then pull any vertex back so it
moves at most `rel_frac` × (distance to its nearest connected neighbour). `rel_frac<0.5`
means a vertex can't cross a neighbour in one step → no face can evert → no winding flip.
TF exposes `Vertex.position` setter (updateChildren=True refreshes surfaces) but NO
Surface winding-reversal, so PREVENT (clamp) rather than repair. `b.volume` is a cache →
must call `b.position_changed()` on touched bodies after clamping (tfBody.cpp).

**Why it's not just a hack:** it's a trust-region limiter on overdamped gradient descent —
under normal relaxation per-step motion is tiny (cap never binds); it binds only on the
pathological overshoot, holding the collapsing edge short until the energy-gated
reconnection resolves it. A flagged DEPARTURE from the paper's force-level abs(volume); the
native C++ port should still use abs(volume) in VolumeConstraint.

**Measured (`pixi run check-clamp`, 2026-05-30, N=189, dt=1e-4, validated config):**
clamp OFF → min_vol goes negative @ step ~2100, blows to −218 @ ~4260 (cells shoot off).
clamp ON (frac=0.4) → worst min_vol = **+2.089**, never non-positive over 4500 steps, AND
D = **−0.102** @ step 4500 still descending (vs old plateau −0.096 reached only at ~5940
WITH min_vol −0.75). So the instability was NEVER doing the sorting — removing it makes the
sort cleaner AND deeper. Clamp rate self-decays (≈50/step early → ≈5/step as it settles).

Tests: 5 pure `_clamp_to` unit tests in `rnr/tests/test_stability.py` (pixi run test → 19
green). Behavioural gate = `pixi run check-clamp` (render-free) + the re-rendered
`pixi run sort-video` (clamp on by default, arg 7 = CLAMP_FRAC). Confirms
[[phase2-sorting-partial]] energy-gate-sorts; supersedes the "stability needs native abs"
caveat for the PROTOTYPE.
