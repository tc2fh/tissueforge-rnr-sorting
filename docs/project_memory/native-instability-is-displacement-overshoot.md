---
name: native-instability-is-displacement-overshoot
description: "RESOLVED 2026-05-31 — the native check-clamp-native instability was TWO per-vertex displacement overshoots (A: periodic-image wrap of the corner cluster; B: post-reconnection overshoot storm), NOT a volume/winding problem. FIXED by centring the cluster (Fix A) + the oracle reconnection regime: a new native reconnect_interval throttle knob (oracle dtr=10*dt) at Lth=0.4 (Fix B); a parallel-implement scheduler race also fixed (serial reconnection executor). Hardened gate (min_vol>0 AND max_vol<3*V0) now PASSES; no volume code touched"
metadata:
  node_type: memory
  type: project
  originSessionId: 45b1bb77-f074-4dcc-96fb-e158554c2408
---

A 2026-05-30 trigger investigation (scripts `rnr/scripts/diag_inflation.py` / `diag_fling.py` /
`diag_centered.py`; PORTING_NOTES §6c/§6d) RE-DIAGNOSED the `pixi run check-clamp-native 4500 0.0001`
failure. It is **NOT a volume/winding problem** — it is **two independent per-vertex DISPLACEMENT
OVERSHOOTS**; the negative/huge `Body.volume` is a downstream symptom. This CORRECTS
[[faithful-instability-is-winding-signflip]] (the winding flip is the symptom, not the cause).

Evidence (patched-engine runs; cluster corner `[0,10]³` vs centred `[25,35]³` in periodic `[0,60]³`):
- **A — periodic-image WRAP** (reconnection-INDEPENDENT): corner cluster, reconnect OFF → a boundary
  vertex on x≈0 drifts across 0 and teleports **exactly +60** (the box dim) to x≈60 at step ~2090
  (caught by `diag_fling.py`, single-step jump 60.000, dnv=0) → cell becomes a box-spanning sheet.
  Cause: TF's vertex MESH has no minimum-image awareness (PORTING_NOTES §5.1). **Centring the cluster
  / non-periodic BC removes it** (centred+reconnect-OFF is fully stable, max step-disp ≈2e-4).
- **B — post-reconnection OVERSHOOT STORM** (survives centring): reconnecting EVERY step at
  `Lth=0.45` leaves the mesh out of equilibrium → 5–7-cell-width single-step overshoots → new short
  edges → cascade → inflation (max_vol 30972). New vertices are placed fine (offset <2.3, NOT an
  Appendix-1 bug). Pre-existing in the Python prototype (conftest.py "post-reconnection overshoot").

**Both volume fixes were tried and FAIL (fork then reverted to clean Phase-D `6d67617`):** Approach B
(force FLOOR/ABS in tfVolumeConstraint.cpp) UNSTABLE (FLOOR −536, ABS −118338); Approach A
(orientation-consistent positive `Body` volume + geometric-sign force) gives a MISLEADING
`STABLE +0.302` — the gate's min-only metric is fooled while every cell inflates (1779/30972).

**Oracle (3DVertVor) comparison** (read-only, GPL, file:line in PORTING_NOTES §6d): same overdamped
Euler integrator as TF (`Run/Run.cpp:1252,1345`) — the integrator is NOT the difference. It stays
stable via 4 stabilizers TF lacks: (1) **minimum-image periodicity** in all geometry/forces
(Volume/Interface/Polygon .cpp) ⇒ a wrap is a non-event [fixes A]; (2) **throttled+tiny-Lth
reconnection** — `Lth=1e-3` (vs our 0.45), checked every `dtr=10·dt` (vs our every step), placement
±Lth≈0.001 [fixes B]; (3) **orientation repair** `if(volume_<0){volume_=fabs; flip polygonDirections}`
(Cell.cpp:216) [the eversion mode; what Approach A approximated — legit but insufficient alone];
(4) COM drift removal. Energy = `Σk_v(V−V0)² + Σ tension·area` (area-based — confirms Phase-G plan).

**Why:** the whole Phase-E "robust volume backstop" premise (docs/native_volume_fix_plan.md) is
wrong; volume robustness can't fix a displacement overshoot. **How to apply:** for A, centre the
cluster in the gate harness (no engine change); for B, adopt the oracle reconnection regime (small
`reconnect_length` + an every-N-steps throttle knob) — trade-off: smaller Lth / larger interval ⇒
stable but slower sorting. See [[phaseD-native-doquality-wired-blocked-by-E]],
[[winding-clamp-stabilizes-sort]] (the Python clamp "worked" only because it caps per-step
displacement, incidentally catching both overshoots), [[oracle-comparison-ceiling-physical]],
[[phaseG-new-surface-tension-actor]].

**RESOLVED 2026-05-31 (PORTING_NOTES §6e; fork commit on feat/native-rnr-reconnection — NO volume
code touched).** Both fixes implemented; `pixi run check-clamp-native` (no args) now PASSES with
volumes near target, and `pixi run test` = 32 green (31 + a reconnect_interval throttle test).
1. **Gate HARDENED first** (the old min-only gate was foolable by uniform inflation — exactly how
   Phase-E Approach A printed a misleading "STABLE +0.302" while inflating to 30972): now FAILS if
   `max_vol > 3*V0` or a vertex leaves the box, not just `min_vol<=0`. (Healthy baseline max_vol =
   7.438 = boundary-clipped Voronoi cells, not V0=4.)
2. **Fix A** — centred both gate harnesses to box=[25,35]³ in [0,60]³ (no vertex wraps).
3. **Fix B** — config sweep alone is INSUFFICIENT (centred, dt=1e-4, 4500 steps: edges only reach
   ~0.3–0.45, NOT ~1e-3, so Lth≤0.2 ⇒ zero reconnections / D frozen, Lth=0.45-every-step ⇒ storm).
   Added a native **`reconnect_interval`** knob on MeshQuality (the oracle's dtr; a reconnectCounter
   gates the pass by `counter % interval == 0`, default 1; getter/setter/IO/SWIG). The I↔H
   surgery/placement was NOT changed. **Chosen default = Lth=0.4, INTERVAL=10** (oracle dtr=10*dt):
   robustly stable (worst_min ~2.2 across runs, durable to 8000 steps), reconnects continuously
   (~34 events, D drifts −0.043→−0.06), max_vol stays at the 7.438 baseline. Throttling tames the
   Lth=0.45 storm (−2.7 → stable at INT≥5); Lth=0.45 INT=10 is marginal (a run dipped to 0.37).
4. **Scheduler RACE** (latent Phase-C/D bug surfaced under the now-stable long runs): the pass ran
   `implement()` in PARALLEL but a ReconnectionOperation's `targets` cover only its 9 surfaces, not
   the shared outer vertices/bodies the I↔H also mutates ⇒ concurrent mutation race ⇒ intermittent
   segfault + nondeterministic eversions. Fixed with a SERIAL reconnection executor
   (`MeshQuality_doOperationsReconnectionSerial`); throttled+sparse so serial cost is nil. Energy
   gate stays OFF (D≈−0.06 is the expected faithful depth; the Phase-G σ·A actor is what demixes).
