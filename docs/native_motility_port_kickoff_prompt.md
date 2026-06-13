# Kickoff — native active-motility drive (Phase 3: make the faithful sort fully native)

**Read first:** `CLAUDE.md` (status banner + **license boundary**); `rnr/PORTING_NOTES.md` **§6n**
(the active-vs-thermal finding — WHY active, the whole premise), **§6g** (the per-step MeshSolver
anatomy — *critical* for where to hook), **§6k** (the native orientation repair = the precedent for
adding per-`Body` state + a per-step native update), **§6f** (noise calibration: mobility μ≈1, and
the `MeshSolver` force→`p->f` path); memories `active-motility-not-thermal-noise`,
`native-orientation-repair-faithful`, `tf-swig-subi-needs-forced-regen`, `py311-source-build-migration`.
**The Python reference to port (OURS, re-derived — port this, don't re-derive):**
`rnr/scripts/sort_periodic_oracle.py` (`add_noise_active` + the `_dirs` director state +
`_rebuild_incidence`) and `rnr/scripts/probe_active_motility.py`.

## Where we are (2026-06-13)

Phases 0–2 are **done**: 3DVertVor/Manning **Fig 1E + 1F reproduced** faithfully, clamp-free
(PORTING_NOTES §6n; `pixi run test` = 47 green). The engine fork `feat/native-rnr-reconnection`
(HEAD `cffa102`, github.com/tc2fh/tissue-forge) already has, **native in C++**: the I↔H reconnection
(inside `doQuality`), periodic min-image geometry (`mesh.periodic_geometry`), and the per-cell
orientation repair (`Body::orientSign`).

**The ONE remaining non-native piece is the active self-propulsion (motility) noise drive.** It lives
in a *Python harness* (`sort_periodic_oracle.py::add_noise_active`), injected via `v.set_position`
before each `tf.step()`. That is the last thing between the validated prototype and a fully-native
"RNR + active cell-sorting" capability in TissueForge — the project's actual deliverable.

## Goal (one sentence)

Move the active drive into the C++ engine — a **per-cell director** that undergoes rotational
diffusion + a **per-vertex active force `F_v = v0·⟨incident-cell directors⟩`** — so the faithful sort
runs entirely in the engine, then retire the Python noise injection.

## The model (re-derived already; this is what to port — NOT GPL `Run.cpp`)

Per the §6n finding, the oracle's "noise" is active self-propulsion, **dt-scaled, not √dt thermal**:
- **Per cell** `c`: a director `n_c ∈ S²`. Rotational diffusion each step (the ONLY √dt term, on
  *orientation*): `n_c ← normalize(n_c + sqrt(2·Dr·dt)·(ξ − n_c))`, `ξ` ~ uniform on S², `Dr = 1`.
- **Per vertex** `v`: motility velocity `u_v = v0 · mean_{c ∋ v} n_c` (`v0` ≡ the oracle's
  `temperature`; our runs use `v0 = 0.1`).
- **Displacement**: the oracle does `x_v += dt·u_v` *alongside* the deterministic `velocity·dt`
  (`Run.cpp:1345`). Per-step magnitude `≤ dt·v0 ≈ 0.1·Lth` — sub-Lth, which is *why* reconnection is
  caught with no clamp.

**The clean native mapping (recommended):** TissueForge's vertex integrator is **overdamped**
(`x += μ·F·dt`, §6f measured μ≈1). So adding a per-vertex force `F_v = v0·⟨n_c⟩` yields displacement
`μ·v0·⟨n_c⟩·dt = v0·⟨n_c⟩·dt` — *exactly* the oracle's `dt·u_v`, applied simultaneously with the
deterministic forces (which is the oracle's single combined update, cleaner than the harness's
inject-then-step). The engine's PBC integrator already wraps `p->x`, so **no manual `% L` is needed**
(the harness's `(P+dx) % L` was a Python-side hack). And the engine maintains the vertex↔cell
adjacency live, so **the per-step `_rebuild_incidence` + the net-zero-count stale-handle workaround
are not needed natively** — the native version is simpler and more robust than the harness.

## ⚡ DERISK FIRST (decides FORCE vs POSITION-increment)

Confirm the mesh integrator is overdamped and **how a per-vertex force maps to displacement**, before
committing to the force approach. Read the integration path: `MeshSolver::preStepStart` →
`preStepJoin` (`tfMeshSolver.cpp:~446–504`, where `p->f[...] += buff[...]`) → the engine advance →
`positionChanged`. Re-confirm §6f's μ≈1 with a one-vertex measurement (apply a known `F`, read
`Δx/dt`). If it's a clean `x += μF·dt`, implement the drive as a **force** (idiomatic, PBC-handled).
If the mesh advance is *not* a clean overdamped force step, fall back to a **direct position
increment** `x += dt·u_v` in a post-force solver hook (the literal-oracle form). Either way the
invariant is: per-step displacement `= dt·v0·⟨n_c⟩`.

## Task 1 — read the seams (determine the design)

- `tfMeshSolver.{h,cpp}` — the per-step hooks. `preStepStart()` (~446) begins force computation;
  `preStepJoin()` (~485–504) sums mesh force into `p->f`. Decide: (a) evolve all cell directors once
  per step in `preStepStart` (before forces), and (b) add the active per-vertex force where the mesh
  force is accumulated.
- `tfBody.{h,cpp}` — `orientSign` (`tfBody.h:92`) is the precedent: add a `director_[3]` unit-vector
  member + getter/setter exactly the same way (init random-on-sphere; serialize if §6k's orientSign
  is serialized).
- `tfVertex.{h,cpp}` — the live vertex→incident-cells adjacency (`getBodies()` / equivalent) for the
  per-vertex director average.
- `actors/tfAdhesion.cpp`, `actors/tfSurfaceAreaConstraint.cpp` — templates for how a force is
  computed per element and bound to types (if you implement the drive as a `MotilityForce` actor).
- The engine RNG used by `tf.Force.random` (grep `source/` for the random-force RNG; it's the
  template for a *seeded, reproducible* per-cell `ξ`). Reproducibility matters for the gate tests.

**Open design decision (pick after reading):** store director on `Body` + evolve in a solver pre-step
hook + add the force in the mesh-force pass (reading `Body.director`) — recommended, mirrors §6k. A
fully stateless per-type actor does NOT fit (the director carries stochastic state). Params `v0`
(active speed) and `Dr` (rotational diffusion) live on a new actor bound to body types (idiomatic,
allows per-type v0) or as a solver/mesh setting (simplest faithful match; the oracle uses one global
`temperature`). Default to uniform v0; expose per-type if cheap.

## Task 2 — implement (C++ on the fork)

1. **Director state**: `Body::director_` (unit vector) + `getDirector/setDirector`; initialize
   random-on-S² (use the engine RNG, seeded).
2. **Rotational diffusion**: once per step (pre-force), evolve every body's director by
   `n ← normalize(n + sqrt(2·Dr·dt)·(ξ − n))`. Re-derive from the physics — do **not** copy
   `3DVertVor/Run.cpp` (`updateVerticesPosition`/`updateSP`); that's GPL. Our Python
   `add_noise_active` is the (MIT-clean) reference.
3. **Active force/displacement**: per vertex, `F_v = v0·(1/N_v)·Σ_{c∋v} n_c`; add to `p->f` in the
   mesh-force pass (force approach) — or the equivalent position increment (fallback).
4. **Params + API**: `v0`, `Dr` settable from Python; default `v0=0`, `Dr=1` (off by default, so
   existing runs are unaffected).
5. **SWIG**: expose the new API. **Gotcha (memory `tf-swig-subi-needs-forced-regen`):** editing a
   sub-`.i` or a `%included` header does NOT re-run SWIG — `rm` the generated
   `tissue_forgePYTHON_wrap.cxx` in `tissue-forge_build/` to force regen, else the new methods never
   reach Python. (`FloatP_t` is float32.)
6. **Build**: `pixi run build-tf` (tens of min first time; incremental relink after a C++ edit),
   then `pixi run verify`.

## Task 3 — wire the harness to the native drive + validate

- Add `NOISE_MODEL=native` to `sort_periodic_oracle.py` (and a variant of `probe_active_motility.py`):
  set `v0`/`Dr` on the solver/actor, and **do nothing in Python per step** (no `add_noise_active`, no
  `_rebuild_incidence`). Keep `NOISE_MODEL=active` (the Python injection) as the side-by-side
  comparison/fallback, and `thermal` as legacy.
- The native and Python-`active` runs must agree (same seed → statistically identical demixing/rate).

## Validation gates (definition of done)

1. **Build + smoke**: `pixi run build-tf` clean; `pixi run verify` OK; new API reachable from Python.
2. **Calibration** (new probe, mirrors §6f): native per-step vertex displacement ≈ `dt·v0·⟨director⟩`
   (confirms μ and the force scaling); per-cell director autocorrelation decays at the expected
   rotational-diffusion rate (confirms `Dr`).
3. **Rate, clamp-free, NATIVE drive** (no Python noise): `probe_active_motility`-equivalent at
   `v0=0.1, Dr=1` gives **~35 reconnections/3000 (M=4, σ=0.5)** and **~141 (M=6)**, STABLE
   (no eversion/inflation) — matching the Python-`active` numbers (§6n).
4. **`pixi run test` ≥ 48 green**: all current tests (round-trip reversibility, Condition-4 vetoes,
   periodic dynamics) stay green, **plus** a new native-active-drive test (rate restored + stable +
   the calibration check). Don't trade reversibility for the drive.
5. **Science reproduces with the NATIVE drive**: fig1e/fig1f (`MODEL=active` CSVs regenerated with
   `NOISE_MODEL=native`) still show σ-ordered area demixing (`S_area ≈ 0.022/0.057/0.116`) and the
   demixed state holding (`DP/DP_max ≈ 0.89/0.90/0.91`). A focused 3-σ × seed-7 run is enough to gate;
   the full ensemble (`pixi run overnight`) is the polish.
6. **The Python `add_noise_active` injection is retired** as the production path (kept only as a
   comparison fallback); `NOISE_MODEL=native` becomes the default.

## Departures / discipline to keep

- **Don't re-open the science.** The port must REPRODUCE the validated §6n results bit-for-similar,
  not change them. If the native drive disagrees with Python-`active`, the port has a bug — debug it,
  don't "retune."
- **License boundary:** re-derive the active update from the physics / our Python; never paste
  `3DVertVor`/`tvm` GPL C++ into the fork.
- **Don't touch** the working native I↔H, periodic geometry, or orientation repair.
- **Faithful = the per-cell director averaged to vertices.** Do NOT shortcut with `tf.Force.random`
  colored noise (`mean>0, duration≫dt`, §6f): that's per-*particle* and step-discretized, not the
  smoothly-rotating per-*cell* self-propulsion — it loses cell coherence. (It's at most a sanity
  foil.)
- Keep the **check/implement split** clean (director-update vs force-add) so the logic is
  inspectable, per the C++-port discipline in PORTING_NOTES §6.

## Pointers

- **Engine seams:** `tissue-forge/source/models/vertex/solver/tfMeshSolver.cpp` (`preStepStart`
  ~446, `preStepJoin` ~502 `p->f += buff`), `tfBody.h:92` (`orientSign` precedent), `tfVertex.{h,cpp}`
  (vertex→bodies), `actors/tfAdhesion.cpp` (force-actor template), `tf_mesh_metrics.{h,cpp}` (periodic
  helpers if the force needs min-image — it shouldn't, it's local). Build tree: `tissue-forge_build/`.
- **Python reference (port this; it's ours):** `sort_periodic_oracle.py::add_noise_active` +
  `_dirs`/`_rebuild_incidence`; `probe_active_motility.py`.
- **Gates today:** `rnr/tests/test_clampfree_reconnection.py` (rate), `test_periodic_dynamics.py`
  (stability), `test_native_roundtrip.py` / `test_roundtrip.py` (reversibility).
- **Papers:** Okuda 2013 (`reference_pdfs/`), Manning 2024 (Eq. 5 dynamics: μ=1, dt=0.01,
  Dr/temperature; our runs dt=1e-3, v0=0.1). Note the paper's "kT_vert" is the active speed v0, not a
  thermal kT — see §6n.
- **Record when done:** new `PORTING_NOTES.md` §6o (native active drive: the API, the seam, the
  calibration); update memory `active-motility-not-thermal-noise` (now native); update `CLAUDE.md`
  status (Phase 3 active drive done). This kickoff lives at `docs/native_motility_port_kickoff_prompt.md`.

## Why this is the right next step

The science is already convincing (§6n); scaling N or sweeping σ only polishes an already-solid
result. The native drive **completes the actual deliverable** — a real RNR + active-sorting capability
*in* TissueForge, not a Python harness around it — and unblocks everything downstream (the eventual
growth/morphogenesis work runs on the native engine). It's also cleanest to do now, while the
noise-model finding (§6n) is fresh and the seams are documented.
