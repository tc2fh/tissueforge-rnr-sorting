---
name: native-orientation-repair-faithful
description: "RESOLVED 2026-06-11 — implemented 3DVertVor's FAITHFUL orientation repair (oracle stabilizer #3) natively in TF, superseding the §6j Python noise clamp for the paper's regime. A per-Body orientSign (abs the signed volume + flip the parity when V<0, applied consistently in volume AND the VolumeConstraint force) is the C++ analogue of flipping all polygonDirections_ together. Faithful σ=0.5/Lth=1e-3/kT=0.1 sort STABLE 20000 steps with NO clamp; counter-test (repair off via env) reproduces the exact −1.78 eversion. pixi run test=45; gate test_periodic_dynamics.py; PORTING_NOTES §6k."
metadata:
  node_type: memory
  type: project
  originSessionId: native-rnr-reconnection
---

The user chose to replace the §6j noise-clamp DEPARTURE ([[periodic-sort-noise-overshoot-fixed]])
with the paper's own faithful mechanism: 3DVertVor's **orientation repair** (oracle stabilizer #3,
`Cell.cpp:216-221`). Now done natively on fork `feat/native-rnr-reconnection`.

**Why it was thought "not the fix" but actually is.** §6c rejected a *volume-robustness* attempt
(Approach A) that took a **per-FACE abs** (`Σ|face_contrib|`) → overcounts a half-everted cell →
inflation runaway. The oracle does a **per-CELL** repair: compute the cell's TOTAL signed volume;
if `V<0`, `fabs(V)` AND flip EVERY face direction together, used consistently in volume and force.
Flipping all faces == negating a single per-cell sign — NOT a per-face abs. So the faithful repair
was never actually tried; §6c's verdict was about the broken per-face version (and about
instabilities A/B — periodic wrap + reconnection storm — already fixed by §6g + §6e). Corrects
[[faithful-instability-is-winding-signflip]] / [[native-instability-is-displacement-overshoot]] for
the transient-eversion mode specifically.

**Implementation (the seams).** TF has no per-(cell,face) direction store; `volumeSense(body)` is ±1
from b1/b2 identity. One per-Body sign captures the oracle's flip-all exactly:
- `tfBody.h`: private `FloatP_t orientSign` (+`getVolumeOrientSign()`).
- `tfBody.cpp`: after the volume sum in BOTH `updateInternals()` and `positionChanged()` —
  `orientSign=1; if(repairEnabled && volume<0){orientSign=-1; volume=-volume;}`. Memoryless (sign of
  the raw signed volume each step) — equivalent to the oracle's persistent flip and strictly safer
  (oracle re-checks only at init + per reconnection pass; TF's force breaks on even one negative
  step, §6j, so flip every step in the hot path).
- `tfVolumeConstraint.cpp:60`: `force *= source->getVolumeOrientSign()` (gradient stays restoring
  when getVolume() is abs'd). `tfNormalStress.cpp:50`: same (completeness; not in faithful model).
- Healthy cells (V>0) → orientSign=+1 → STOCK behaviour, so the 44 prior tests are unaffected.

**Why:** the noise clamp was a flagged departure (locally-adaptive timestep); the orientation repair
is what the paper's code actually does, so faithful runs need no clamp.

**How to apply:** env `TF_VERTEX_NO_VOLUME_REPAIR=1` disables it (stock signed volume) — used only by
the load-bearing gate counter-test; default ON. `orientSign` is the C++ port seam (= polygonDirections_).
The repair is ORTHOGONAL to the displacement trust-region (§6j port note): it fixes the volume-SIGN
mode only. The deep DEPARTURE regime (σ=1, Lth=0.05) confirms this — with the repair the eversion is
CURED (min stays +0.15) but the cell still INFLATES (max 5.2) by displacement-MAGNITUDE overshoot,
which is the trust-region's job, not the repair's (and Lth=0.05 is a departure anyway — Okuda wants
Lth small). So: repair = faithful fix for the paper's regime; trust-region = optional, only for the
aggressive-Lth departure. See [[oracle-comparison-ceiling-physical]].

**Result.** Faithful σ=0.5, Lth=1e-3, kT=0.1, μ=1, s0≈5.6, clamp OFF: **STABLE 20000 steps / no
eversion (worst_min 0.047>0) / no inflation (worst_max 2.618)** via existing actors + native repair.
Counter-test (repair OFF via env, clamp OFF): min_vol −1.78 by step 500 (reproduces §6j EXACTLY ⇒
repair is load-bearing). Gate: `test_periodic_dynamics.py` (+1 net test); **pixi run test = 45**.
