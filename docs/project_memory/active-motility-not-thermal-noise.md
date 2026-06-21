---
name: active-motility-not-thermal-noise
description: "RESOLUTION 2026-06-11 — the 3DVertVor/Manning oracle drives vertices with ACTIVE SELF-PROPULSION (motility), NOT thermal Brownian noise. Its thermal line Run.cpp:1344 (cR·ndist, cR=sqrt(2μ·kB·T·dt), √dt) is COMMENTED OUT; the live line :1345 is x+=dt·motility, motility=temperature·⟨cell director⟩, scaling as dt NOT √dt. Per-step displacement ≤ dt·v0 ≈ 0.1×Lth (sub-Lth), so the plain instantaneous-edge reconnect trigger (Reconnection.cpp:34 edge<Lth=1e-3) catches collapses with NO clamp. This is the faithful fix that removes the §6j/§6l noise clamp and the §6f tf.Force.random thermal substitution (which matched long-time diffusion D=μ·kT but NOT per-step displacement = what the trigger needs). Base tvm DOES use thermal cR·ndist; the Manning fork switched to active motility."
metadata:
  node_type: memory
  type: project
  originSessionId: native-rnr-reconnection
---

**UPDATE 2026-06-13 (Phase 3): this active model is now NATIVE in the C++ engine** — see
[[native-active-motility-drive]]. The Python `add_noise_active` injection described below is retired
to a comparison fallback; production uses `NOISE_MODEL=native` (engine-side per-cell director +
active force). The physics/finding below is unchanged.

The clamp-free-reconnection kickoff (`docs/clampfree_reconnection_kickoff_prompt.md`) assumed the
oracle uses thermal noise and asked how it catches sub-Lth edges when per-step noise ≈ 45× Lth.
**Wrong premise.** Reading the oracle settles it:

- `3DVertVor/Run/Run.cpp:1345`: `x += velocity·dt + dt·motility`. The thermal line `:1344`
  (`+ cR·ndist`, `cR=sqrt(2·mu·kB·T·dt)`) is **commented out**. (Base `tvm/Run/Run.cpp:164` keeps it
  — the Manning fork switched models.)
- `motility` = ACTIVE self-propulsion: each CELL has a director `n_c∈S²` that rotates with
  active-Brownian rotational diffusion (`Run.cpp:1287` Dr=1, std=√(2·Dr·dt), the ONLY √dt term —
  on ORIENTATION); per-vertex `motility=temperature·⟨n_c⟩` over incident cells (`Vertex.cpp:78–86`).
  So per-step displacement scales as **dt** (ballistic), `≤ dt·v0` (v0≡temperature). With v0=0.1,
  dt=1e-3 → ≤ 1e-4 = 0.1×Lth.

**Why this is the whole story.** The reconnect trigger is the plain instantaneous edge length
(`Reconnection.cpp:34`, Lth=1e-3, checked every dtr=10·dt). It works because per-step motion is
sub-Lth: a collapsing edge stays below Lth and is caught — no clamp, no time-averaged trigger
needed. Our harness's §6f `tf.Force.random` substitution calibrated **long-time diffusion** (D=μ·kT)
but the trigger depends on **per-step displacement**; thermal √dt noise is 14–45× Lth/step, blows
fresh edges past the trigger, STARVES reconnection — hence the §6j clamp band-aid.

**Confirmed (probe_active_motility.py, M=4 σ=0.5 3000 steps):** active no-clamp 35 recon (= thermal
+clamp 35, vs starved thermal-clamp0 1); M=6 → 141 (vs thermal+clamp 94); STABLE. v0=0 control → 38:
reconnections are DETERMINISTIC-relaxation-driven — noise's job is to NOT SABOTAGE the trigger
(thermal does, active doesn't) while providing persistent active stirring.

**Science (gate 4 — FULL ENSEMBLE DONE 2026-06-12):** regenerated fig1e/fig1f clamp-free active, M=6,
σ∈{0.1,0.2,0.5}×seed∈{7,8,9}×IC∈{mixed,demixed}, 100k steps (18 sims, all STABLE), via
run_overnight.py. Fig1E: area-demix S_area={0.022,0.057,0.116} for σ={0.1,0.2,0.5} ORDERED (count-DP
≈0, finite-N limited as before). Fig1F: demixed HELD DP/DPmax={0.89,0.90,0.91} (>0.8) vs mixed ≈0 —
demixed is a stable minimum. Artifacts rnr/exports/{fig1e_demixing_active.png,fig1f_stability_active.png,
sort_active_demixing.gif (51-frame video)}. Reproduces the prior thermal+clamp result with NO clamp.
(Op note: survived a mid-run power-loss HIBERNATION — Mac safe-slept on dead battery, processes
resumed intact on power-up; boot time unchanged ⇒ no reboot. Failure-tolerant orchestrator made it a
non-event.) New scripts: run_overnight.py (pool orchestrator), video_periodic_active.py; fig1e/fig1f
gained MODEL=active selector (picks _active CSVs, won't mix with legacy thermal).

**Shipped:** `rnr/scripts/probe_active_motility.py`; `sort_periodic_oracle.py NOISE_MODEL` arg
(default `active`; `thermal` = legacy clamp path + legacy CSV name; active CSVs tagged `…_active`).
Active model rebuilds the vertex-handle/incidence cache EVERY step (a doQuality pass can net-zero the
vertex count via paired I→H/H→I, so num_vertices is an unsafe staleness signal — a cached deleted
handle segfaults; silently crashed M=6 until fixed). Gate `rnr/tests/test_clampfree_reconnection.py`
(active no-clamp rate ≥10 + STABLE; unclamped-thermal starves ≤3); `pixi run test`=47. PORTING_NOTES
§6n. C++ port: the native integrator must use active motility (sub-Lth/step), making the planned
§6j in-integrator clamp unnecessary. Supersedes [[clamp-enables-reconnection-not-just-stability]];
corrects [[thermal-noise-destabilizes-reconnection]] (the destabilization was the wrong noise model);
[[adhesion-force-is-already-area-tension]] energetics unchanged. Repair [[native-orientation-repair-faithful]]
stays ON (orthogonal: transient eversion).
