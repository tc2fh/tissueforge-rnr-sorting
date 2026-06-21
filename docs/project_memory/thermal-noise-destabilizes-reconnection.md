---
name: thermal-noise-destabilizes-reconnection
description: "PARTLY SUPERSEDED 2026-06-11 by [[active-motility-not-thermal-noise]]: the observation (thermal tf.Force.random noise WORSENS stability+sorting) holds, but the root cause is NOT 'reconnection robustness vs kinetic trapping' — it is that thermal Brownian (√dt) is the WRONG NOISE MODEL. The oracle uses ACTIVE motility (dt-scaled, sub-Lth/step); thermal √dt noise is 14–45× Lth/step and sabotages the reconnect trigger. The line 34 claim 'tf.Force.random IS faithful for the noise' is WRONG for reconnection (faithful only for long-time diffusion). Use active motility instead."
metadata: 
  node_type: memory
  type: project
  originSessionId: 81faf2b8-7985-490c-9a90-7729d1e724b3
---

**Faithful 3DVertVor reproduction (existing actors + thermal noise) shows thermal noise HURTS, not
helps, at the sigma=1 regime.** Harness: `rnr/scripts/sort_faithful_3dvertvor.py` (`pixi run
sort-faithful-3dv`). Model = VolumeConstraint + SurfaceAreaConstraint(s0=5.6) + body-Adhesion (the
area-based sigma*A_het force, [[adhesion-force-is-already-area-tension]]) + tf.Force.random white
noise (mu=1, kT=1.63e-5*std^2). Native geometric reconnection (gate OFF), centred, LTH=0.4 INT=10.

kT sweep (sigma=1, 10k steps): the athermal control is the BEST and only stable run.
| kT   | stable? | D plateau | het-area | reconnections |
| 0    | ✅      | **−0.065** | 0.375   | 2844 |
| 0.02 | ❌ blows up | −0.064 | 0.388 | 3054 |
| 0.05 | ❌ blows up | −0.057 | 0.396 | 3403 |
| 0.1  | ❌ blows up | −0.031 | 0.429 | 4185 |

MONOTONIC: more noise → more reconnection churn → faster cell inversion → SHALLOWER demixing. The
noise jitters edges below LTH → more reconnections → more post-reconnection overshoots → volume
erosion → inversion (the §6e instability B, now noise-driven and un-tamed by the INT=10 throttle).

**Implication:** the binding constraint is RECONNECTION ROBUSTNESS, not kinetic trapping — until a
noise-perturbed near-inversion can RECOVER, noise can't fluidize toward sorting. This points back to
the oracle's orientation/volume repair (abs+flip, PORTING_NOTES §6d#3) as the likely prerequisite for
viable noise — distinct from the reverted §6c volume work (that was for the athermal overshoot;
noise is a new perturbation source). Open: does the paper's WEAKER-sigma regime (sigma/K_A~1 vs our
10) + more throttle tolerate noise? (being tested). Athermal D≈−0.065 is itself shallow — consistent
with [[oracle-comparison-ceiling-physical]] (even the periodic oracle plateaus ~0.06–0.10 at matched
sigma). tf.Force.random IS faithful/usable for the noise (calibration in PORTING_NOTES §6f).
