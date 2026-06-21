---
name: phased-native-doquality-wired-blocked-by-e
description: native RNR Phase D done (committed 6d67617) — reconnection wired into live doQuality; faithful gate fails — but Phase E (volume robustness) is NOT the fix: instability RE-DIAGNOSED as displacement overshoot, see [[native-instability-is-displacement-overshoot]]
metadata: 
  node_type: memory
  type: project
  originSessionId: 359ed126-144d-46f9-8057-3cf2f65de2ac
---

> **⚠️ UPDATE (2026-05-30): Phase E was ATTEMPTED and the premise was DISPROVEN.** "Phase E =
> robust/abs volume" does NOT fix the gate — the failure is NOT a winding sign-flip but a per-vertex
> DISPLACEMENT OVERSHOOT (periodic wrap of the corner cluster + post-reconnection overshoot storm).
> See [[native-instability-is-displacement-overshoot]]. The fork was reverted to clean Phase-D
> (`6d67617`); no engine change shipped; `pixi run test` = 31 green. The Phase-D WIRING below is
> still done & correct — only the "what fixes the gate" conclusion changed. Real fix (next session):
> config (centre/non-periodic BC) + oracle reconnection regime (tiny Lth + every-N-steps throttle).

Native RNR **Phase D DONE for wiring**, committed `6d67617` on `feat/native-rnr-reconnection`
(after Phase C `f47e62b`). The native I↔H reconnection now runs as the active geometric pass
INSIDE the live quality loop: `MeshSolver::postStepStart` → `Mesh::getQuality().doQuality()`
(tfMeshSolver.cpp:541) — so `tf.step()` fires it whenever `mesh.quality` is set, not only via
`forceReconnect*`. New seams: `MeshQuality.stockQualityOps` (Python `stock_quality_operations`,
default True; **set False to isolate native RNR from the crashy stock collapses**) and
`ReconnectionOperation.enforceTrigger` (live ops re-check Okuda Condition 2; `forceReconnect*`
bypass; `reconnect_length=0` disables). Energy gate stays OFF for the faithful path. Chains still
built SERIALLY; check/mutate split intact. See [[phaseB-native-walk-conditions-done]].

**Phase E is now the next required step** (do NOT start unasked). Gate
`pixi run check-clamp-native 4500 0.0001` (new `rnr/scripts/check_clamp_native.py`; native
equivalent of `check-clamp 4500 0 0.0001`, all of: python-op/clamp/stock-ops/energy-gate OFF):
**worst min_vol = −29.512, first non-positive @ step ~2100 → UNSTABLE**. min_vol holds (4.0→3.62)
to step 2000 then a cell inverts — the dynamic winding sign-flip ([[faithful-instability-is-winding-signflip]]),
milder than the Python-op control (−218) but still inverts. D stayed ≈−0.04…−0.056 (geometric-alone
sorts weakly; energy gate / Phase-G σ·A is the demix drive — see [[phaseG-new-surface-tension-actor]]).
Phase E = robust/abs signed volume in `VolumeConstraint`/orientation-consistent `Body` volume.

**Why:** proves the reconnection port (A–D) is complete + correct, and pins exactly what remains.
**How to apply:** for a faithful in-step sorting run, do Phase E first; until then the native pass
runs but transients can invert a cell. `pixi run test` = 31 green (28 + 3 live doQuality).

**Reference gotcha (now fixed in `tfMesh.i`):** attaching a Python `tfv.Quality()` via
`mesh.quality = q` double-freed in `~MeshQuality` at teardown (abort
`POINTER_BEING_FREED_WAS_NOT_ALLOCATED`) because `Mesh::setQuality` takes ownership but the SWIG
proxy kept `thisown=1`. The setter now does `_quality.thisown = 0` on transfer. Phase C avoided
this by using a *detached* Quality with `forceReconnect`. Debug tip: tfv.init() segfaults inside
the command sandbox — run `pixi run` build/test/gate UNSANDBOXED; lldb (`--batch -o run -k "bt all"`)
gives the native backtrace.
