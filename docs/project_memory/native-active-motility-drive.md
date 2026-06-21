---
name: native-active-motility-drive
description: "DONE 2026-06-13 (Phase 3) — the active self-propulsion drive is now NATIVE in the C++ engine fork feat/native-rnr-reconnection; the Python add_noise_active harness injection is retired to a comparison fallback. Per-cell Body::director (unit vector) + active-Brownian rotational diffusion evolved once/step at the top of MeshSolver::preStepStart + a per-vertex active force v0·⟨incident-cell directors⟩ folded into VertexForce. The seam: the vertex integrator is OVERDAMPED (tf_engine_advance.cpp dx=dt·f·imass) with vertex mass=1 (density 0 ⇒ default MeshParticleType mass ⇒ imass=1, μ=1), so a FORCE v0·⟨n⟩ gives displacement dt·v0·⟨n⟩ = the oracle's dt·motility exactly. API: MeshSolver.set_motility(v0,Dr,seed)/get_motility_v0/get_motility_dr + body.director; v0=0 default (off). Completes the project deliverable: faithful RNR + active cell-sorting runs entirely in TissueForge."
metadata:
  node_type: memory
  type: project
  originSessionId: native-rnr-reconnection
---

Phase 3 port of the §6n active model (memory [[active-motility-not-thermal-noise]]) from the Python
harness into the C++ engine. PORTING_NOTES §6o. The model is unchanged — per-cell director n_c∈S²,
rotational diffusion `n←normalize(n+√(2·Dr·dt)(ξ−n))` (only √dt term, on orientation), per-vertex
active velocity `v0·⟨n_c⟩` over incident cells — only its IMPLEMENTATION moved native.

**The seam (the crux).** TissueForge's vertex integrator is OVERDAMPED
(`mdcore/src/tf_engine_advance.cpp`, `dynamics != PARTICLE_NEWTONIAN`: `dx = dt·f·imass`) and vertex
particles have **mass=1** (`MeshParticleType` PARTICLE_OVERDAMPED; density 0 ⇒ `Vertex::getMass()=0`
⇒ default unit mass ⇒ imass=1, **μ=1**, per §6f + the new calibration). So a per-vertex **force**
`f=v0·⟨n_c⟩` produces displacement `dt·v0·⟨n_c⟩` = the oracle's `dt·motility` (Run.cpp:1345) exactly,
applied with the deterministic mesh forces in one update. Engine wraps PBC during advance ⇒ harness
`% L` dropped; live vertex↔cell adjacency ⇒ harness `_rebuild_incidence` + stale-handle workaround
unneeded. Native is simpler/more robust than the harness.

**C++ (fork feat/native-rnr-reconnection):**
- `Body::director` (FVector3) + getDirector/setDirector(normalizes); {0,0,0}=unset. Mirrors the
  per-cell `orientSign` precedent ([[native-orientation-repair-faithful]]). NOT serialized (like
  orientSign — known limitation, no mid-run save/restore in gates).
- `MeshSolver::setMotility(v0,Dr=1,seed=-1)` + getMotilityV0/Dr; a DEDICATED reproducible
  std::mt19937 (does not perturb tf.Force.random's stream). v0=0 default ⇒ OFF ⇒ existing runs
  unchanged. Seeds all bodies' directors random-on-S².
- Director EVOLUTION: serial loop atop `MeshSolver::preStepStart` (before the parallel force pass ⇒
  no race). Active FORCE: folded into `VertexForce`'s incident-body loop (`force += v0·⟨n⟩`).
- SWIG: set_motility/get_motility_v0/get_motility_dr + body.director property. Forced regen via
  `rm tissue_forgePYTHON_wrap.cxx` ([[tf-swig-subi-needs-forced-regen]]).

**Harness:** `sort_periodic_oracle.py NOISE_MODEL=native` + `probe_active_motility.py MODEL=native`
call set_motility once, do NOTHING per step. `active`=Python injection (comparison fallback),
`thermal`=legacy. native ≈ active statistically (different RNG → not bit-identical).

**Gates (all green, 2026-06-13):** (1) build clean + verify + API reachable. (2) calibration
`probe_native_calibration.py`: displacement = dt·v0·⟨n⟩ (median ratio 1.000, cos 1.000 ⇒ μ=1 +
scaling exact); rot-diff rate ∝ Dr (ratio 2.08, Dr_eff 1.10/2.28 vs leading-order rate=(2/3)Dr).
(3) rate `probe_active_motility.py … native`: M=4 recon=37 (active 35), M=6 146 (active 141), STABLE.
(4) `pixi run test`=49 green (+ test_native_motility.py). (5) science `sort_periodic_oracle.py …
native` M=6 seed7 40k: σ-ordered S_area={0.031,0.035,0.101} for σ={0.1,0.2,0.5} (strong σ resolves;
low-σ needs 3-seed ensemble), demixed-IC HELD (σ=0.5 DP/DP_max≈0.98) vs mixed≈0, and native≈active
(S_area 0.101 vs 0.105, recon 302 vs 299) — direct faithfulness proof. (6) `NOISE_MODEL=native` is
now the sort_periodic_oracle.py DEFAULT; `active`=Python-injection fallback. Full 100k×3-seed
canonical fig1e/1f regen via `pixi run overnight` is the remaining polish. Supersedes the harness
injection as the deliverable.
