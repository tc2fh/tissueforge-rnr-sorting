---
name: faithful-instability-is-winding-signflip
description: "PARTLY CORRECTED by [[native-instability-is-displacement-overshoot]]: the winding sign-flip is the SYMPTOM, not the root cause; the cause is a per-vertex displacement overshoot (periodic wrap or post-reconnection), and native abs-volume repair does NOT stabilize the gate. Observations below (negatives are flips, V_geo>0, energy gate sorts) still hold."
metadata: 
  node_type: memory
  type: project
  originSessionId: 95ddd839-821c-4df2-a7ea-6f511d9c761e
---

> **⚠️ CORRECTED (2026-05-30) — see [[native-instability-is-displacement-overshoot]].** A native
> trigger investigation proved the root cause is a per-vertex DISPLACEMENT OVERSHOOT — either a
> periodic-image wrap of the corner-placed cluster (+60 teleport) or a post-reconnection overshoot
> storm — and the negative/inflated volume is the *symptom*. The recommendation below to "adopt a
> 3DVertVor abs-flip volume repair as THE key port fix" is WRONG as stated: implemented natively
> (Approach A), abs/orientation-consistent volume only converts the eversion runaway into an
> *inflation* runaway and gives a misleading gate pass. The real fix is config (centre/non-periodic
> BC) + the oracle reconnection regime (tiny `Lth` + every-N-steps throttle). Point 2 below
> ("instability is dt overshoot, not reconnection") was directionally right; points 4 & the abs-flip
> recommendation were the misattribution. The observations (negatives ARE winding flips with
> positive V_geo; energy gate is what sorts) remain valid.

The reference-faithful experiment (2026-05-30, `rnr/scripts/faithful_probe.py` +
`faithful_run.py`) settled the Phase-2 open fork. Stiff energetics: vol_lam=10,
surface_area_lam=1.0 (quadratic), homo tension 0, het tension λ=1, salt-and-pepper
Kelvin block (91 cells). Findings (each is a measured result, don't re-derive):

1. **Faithful tiny Δl_th is FEASIBLE.** Under linear het tension, interior het edges
   collapse to ~1e-4 (the reference Lth scale) even in our finite block. My pre-compaction
   worry ("faces may not collapse far in a finite block") is resolved — they do.
2. **The instability is dt OVERSHOOT, not reconnection and not threshold-size.** The
   FROZEN substrate (reconnection OFF entirely) inverts cells at dt=1e-3 but is STABLE at
   dt=2e-4 over the same physical time. So the inversions are the integrator overshooting
   a collapsing face, independent of any reconnection.
3. **The negatives are TF WINDING SIGN-FLIPS, not geometric collapse.** Orientation
   diagnostic (independent orientation-free `V_geo` via outward-normal divergence theorem +
   convex hull): 21/25 (dt=1e-3) and 8/8 (dt=2e-4) negative-volume cells have POSITIVE
   V_geo, several at near-EXACT magnitude (TF −10.10 / V_geo +10.61; TF −4.347 / +4.347),
   with hugely INFLATED hulls (up to 73× target). Geometry is intact; only TF's signed-
   volume sign is wrong. This is exactly 3DVertVor's abs(volume)+flip case. NB this
   REVERSES the pre-compaction claim "my negatives are genuine collapse so abs-flip would
   mask them" — that was true only for the OLD negative-tension energetics (V→−6e5); under
   FAITHFUL energetics it is a sign-flip.
4. **Mechanism = runaway-inflation feedback.** winding flip → TF signed volume negative →
   VolumeConstraint force `∝(V−V₀)` reverses sign → cell inflates without bound → cascade.
   3DVertVor's abs-flip is ESSENTIAL here because it restores the correct force direction
   and breaks the runaway — it is not cosmetic.
5. **Volume guard is a proven no-op AND the wrong tool.** At faithful Lth `cum.reverted=0`
   (the mutate-half never leaves an immediately-negative neighbourhood → reconnect.py
   winding is sound, matches Phase-1 round-trip). The flips appear DYNAMICALLY several
   steps after a reconnection, so reversing the reconnection cannot un-invert the runaway
   cell. Confirms [[phase2-sorting-partial]]'s "reversing doesn't un-invert".
6. **Bare geometric trigger does NOT sort.** het pairs go UP (194→209/217 at dt=1e-3;
   194→196 at dt=2e-4) — anti-sorting, as predicted. The **energy gate** (my flagged
   departure) is the only thing that demixes in finite time here.

**Recommendations (these change operator DEFAULTS — get a nod before editing):**
- DROP the volume guard from default (no-op + wrong tool); keep as optional knob.
- Primary stability lever = ADEQUATE dt (proven 1e-3→2e-4 stabilises frozen) + SMALLER
  reconnection features (place≈Δl_th≈1e-3, not my 5e-3) to shrink the post-reconnection
  kick that still tips marginal cells at 2e-4. NOT a guard.
- ADOPT a 3DVertVor-style orientation/volume repair (abs-flip, or a correct-sign volume
  force) — now warranted because the faithful failure mode IS the sign-flip and the repair
  breaks the runaway. This is NATIVE/C++ (TF computes volume internally; can't inject from
  Python) → it is THE key port finding for `MeshQuality`/VolumeConstraint.
- KEEP the energy gate as an explicit optional departure — it is what sorts in finite time.

**RESOLVED (2026-05-30) from Python:** see [[winding-clamp-stabilizes-sort]]. A per-vertex
displacement clamp (`operator.stable_step`) PREVENTS the overshoot (can't cross a neighbour
in one step) — keeps min_vol > 0 throughout and sorts DEEPER (D −0.10 vs −0.096). So the
native abs-volume repair is still the right C++ port fix, but the prototype no longer needs
it. NB this also means point 4's runaway never starts if the overshoot is clamped.
