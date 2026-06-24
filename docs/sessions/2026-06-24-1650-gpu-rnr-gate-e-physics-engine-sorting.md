# Session — GPU 3D RNR: Gate E (Stage-1 physics kernels + composed step + sorting)

## Summary 2026-06-24 16:50 EDT

**Goal:** finish the GPU port — **Gate E** (the last item in the staged A–E plan): port the Stage-1
physics (force / geometry / integration) to Warp, compose a full forward step, and validate
end-to-end Fig 1E/1F sorting vs the CPU oracle. **DONE — the full A–E plan is complete; the GPU 3D
vertex engine now runs a whole step on-device and SORTS.** Design study + progress log:
`docs/2026-06-24_gpu-3d-vertex-model-exploration.md` §10.

**What changed & why (decisions / gotchas / findings):**

- **Built host-reference-first** (the `reconnect_csr`→`reconnect_warp` methodology), three new modules:
  - `rnr/gpu/physics_csr.py` — host numpy oracle: surface/body geometry (centroid/area/unnormalized
    normal; volume/area/centroid/orientSign; periodic min-image, first-vertex/first-surface floating
    origin — matches TF exactly) + the **four sorting-physics forces re-derived from the LGPL TF actors
    (read, not copied; each cites its actor)**: VolumeConstraint (`tfVolumeConstraint.cpp:40-63`),
    SurfaceAreaConstraint body variant (`tfSurfaceAreaConstraint.cpp:39-65`), Adhesion body variant
    (`tfAdhesion.cpp:69-105`, heterotypic σ = `0.25·λ·dA_s/dx` over het faces), active drive
    `v0·⟨directors⟩` (`tfMeshSolver.cpp:88-102`) + overdamped integrator.
  - `rnr/gpu/physics_warp.py` — the same as **fp64 Warp kernels** (surface_geom / body_geom / force /
    integrate / director_update).
  - `rnr/gpu/engine.py` — `forward_step` = director→geometry→force→integrate→`reconnect_sweep_*_warp_device`
    (throttled, both directions)→`compact_warp`; + `het_contact_fraction` demixing metric.
- **DECISION — flat/convex regularizers OMITTED.** TF auto-binds `FlatSurfaceConstraint` +
  `ConvexPolygonConstraint` (λ=0.1) on every SurfaceType (`tfSurface.cpp:2348-2349`); they are
  mesh-hygiene, NOT the sorting physics, so the port leaves them out. The force gate validates against
  a TF oracle of just the 4 physics actors (called directly via `tfv.VolumeConstraint(...).force(body,vertex)`
  etc.). **This is the main "faithful" caveat for long runs** (see kickoff #2).
- **GOTCHA — the force gate needs a jittered POLYDISPERSE foam + v0/a0 at the cell mean.** A
  monodisperse Kelvin foam has ~0 NET volume/area force at every vertex (Σ_cell dV/dx=0 by
  space-filling + Kelvin symmetry), so TF's float32 per-cell forces (~1e6) cancel to pure round-off
  noise (~0.1) while the fp64 host gives ~1e-9 — a meaningless comparison. Jitter (10% of spacing) +
  v0=⟨V⟩, a0=⟨A⟩ makes the net forces genuine and well-conditioned.
- **WARP GOTCHA (fp64).** EVERY float literal that meets a `wp.float64` value must itself be
  `wp.float64(...)` ("Input types must be the same", which then makes the referenced `@wp.func` read
  "undefined" at compile). Bit me on `d_minimg`'s `> 0.0`, every `/3.0`, `0.25*sigma`, etc.
- **PERF/CORRECTNESS — per-body force sum restructured as per-(surface,body).** A body `src` defines
  surface `s` ⟺ `src ∈ s2b[s]`, so iterating the vertex's surfaces and, for each, its ≤2 bodies
  enumerates exactly the host's `for src: for s` pairs — NO body dedup for the conservative forces.
  Only the active mean (distinct incident bodies) needs the small O(valence²) first-occurrence dedup.
- **THE "matches the CPU oracle" gate** = with directors frozen (Dr=0) and reconnection off the
  composed GPU step is deterministic and **reproduces the (TF-validated) host reference to fp64 (9e-16)
  over 12 steps**. Sorting itself is validated STATISTICALLY (from a mixed IC the het-contact order
  parameter demixes 0.48→0.42 under heterotypic tension) — determinism stops mattering once the
  director RNG + atomic reconnection ordering kick in, exactly as the plan called for.

- **Benchmarking (discussion only — NO benchmark/batch code was committed; throwaway scripts in
  scratchpad).** GPU `forward_step` vs TF `tf.step()` on the same production foam:
  - **~7×/step** and roughly flat 64→1000 cells — because BOTH are overhead-bound (TF on its ~128-thread
    pool ≈11 ms floor; GPU on kernel-launch + the per-step `n_used` readback — pure physics is 0.78 ms at
    216 cells, mostly launches). So 7× is a floor, not a ceiling.
  - **Large-N via K-fold replication** (disjoint-union SoA, unchanged kernels): 1,728 cells → 3.5 ms/step
    (~6 min/100k); 13,824 → 11.4 ms (~19 min); 27,000 cells/184k verts → 13.1 ms (~22 min). A single
    100k-step run stays in tens-of-minutes out to mesh sizes the CPU can't reach (and `build_periodic_voronoi`
    is O(N²) → can't even construct them on CPU).
  - **Batching is LOW difficulty:** that replication benchmark IS a batch of up to 125 meshes through the
    UNCHANGED kernels (incl. reconnection + compaction) — disjoint topologies never conflict in the
    independent-set reservation. Real work = a concatenating loader + per-batch box/σ as arrays (not
    scalars) + per-batch metrics + a correctness test (~a few days). Payoff: the 18-sim Fig 1E/1F
    ensemble as ONE union run ≈ **~7–8 min** vs ~50 min sequential-GPU / ~40 min CPU-16-core.

**Build / test / git state:**
- **`pixi run test` = 127 passed** earlier this session (was 108). Re-run at handoff after a cosmetic
  test-only cleanup (removed an unused param in `test_gpu_engine.py::_host_step`) — RESULT STATED IN THE
  COMMIT MESSAGE. GPU subset `pytest rnr/tests/test_gpu_*.py -q` → 79 passed (was 60). RTX 5090 / Warp
  1.14 / sm_120 / fp64.
- Branch `migrate/linux64-wsl2`. **This session's files:** `rnr/gpu/{physics_csr,physics_warp,engine}.py`
  (new), `rnr/tests/{test_gpu_physics_csr,test_gpu_physics_warp,test_gpu_engine}.py` (new),
  `docs/2026-06-24_gpu-3d-vertex-model-exploration.md` (§10, modified), `progress.md` (status line), +
  this handoff. Memory `gpu-3d-vertex-direction` updated (outside the repo).
- **Do NOT commit** `rnr/exports/*` (prior-session figs/videos/CSVs) or the read-only oracle repos
  (`tvm/ 3DVertVor/ tissue-forge/ cellGPU/ VertAX/ gpu_reference_papers/` — own `.git`).

Full `git status --short` (the long `rnr/exports/*` list is ALL prior-session output — leave it):
```
 M docs/2026-06-24_gpu-3d-vertex-model-exploration.md   # THIS session (§10)
 M progress.md                                          # THIS session (GPU status line)
 M rnr/exports/{dpmax.json, fig1e_*, fig1f_*}           # prior sessions — leave
?? rnr/gpu/engine.py                                    # THIS (composed forward_step + metric)
?? rnr/gpu/physics_csr.py                               # THIS (host reference: geometry + 4 forces)
?? rnr/gpu/physics_warp.py                              # THIS (fp64 kernels)
?? rnr/tests/test_gpu_engine.py                         # THIS (3: stability, trajectory==host, demix)
?? rnr/tests/test_gpu_physics_csr.py                    # THIS (7: geom/forces == TF)
?? rnr/tests/test_gpu_physics_warp.py                   # THIS (9: geom/forces/integrate/director == host)
?? rnr/exports/{native_*_mixed*.mp4, sort_oracle_M8_*_native[_demixed].csv, vertex_motion_native.gif}  # prior — leave
?? cellGPU/ VertAX/ gpu_reference_papers/               # read-only oracles — DO NOT COMMIT (own .git)
```

## Kickoff — next session

You are continuing a GPU port of TissueForge's 3D vertex model + RNR (Okuda I↔H). Read
`docs/2026-06-24_gpu-3d-vertex-model-exploration.md` (plan + progress §10) first. **THE FULL STAGED
PLAN (Gates A–E) IS DONE & green:** parallel count-changing I↔H both directions, on-GPU detection,
Gate-D compaction, device gather (a whole reconnection round with no `from_warp`), AND **Gate E** —
Stage-1 physics kernels (geometry + the 4 sorting forces + overdamped integrate + director rotational
diffusion) composed into `engine.forward_step`, validated host==TF (float32), GPU==host (fp64), and a
deterministic GPU trajectory == host to 9e-16; from a mixed IC the engine demixes. **79 GPU tests; full
`pixi run test` = 127.**

**Everything below is OPTIONAL / post-plan. Priority order:**

1. **Batched multi-mesh stepping** (HIGH leverage, LOW difficulty — the standout next step). The
   physics/RNR/compaction kernels already run on a disjoint-union SoA (a K-fold-replicated mesh was
   benchmarked through the unchanged kernels). Work: a loader that concatenates K different CSR meshes
   (offset indices, + a batch-id per element — `replicate()` from the scratch bench is ~80%); thread the
   **periodic box and the per-sim params (σ etc.) through the kernels as arrays indexed by batch-id**
   instead of scalars; per-batch `het_contact_fraction` (group by batch-id); a test that disjoint-batch
   reconnection stays isolated. Payoff: the 18-sim Fig 1E/1F ensemble (σ∈{0.1,0.2,0.5} × seed∈{7,8,9} ×
   IC∈{mixed,demixed}, 100k steps, M=6) as ONE union run ≈ **~7–8 min**.
2. **100k-step stability validation (required to call a sort "faithful").** Longest test today = 600
   steps (demixing); 60 steps (full-step stability). Run a long sort and check the mesh stays valid; if
   it degrades, **re-add the flat/convex regularizers** the port omitted — `FlatSurfaceConstraint`
   (`tfFlatSurfaceConstraint.cpp:44-52`, force = `mass/dt·λ·(d·n̂)n̂`, i.e. displacement `λ(d·n̂)n̂`,
   dt/mass-independent) + `ConvexPolygonConstraint`, both λ=0.1, as two more per-surface force kernels.
3. **Faster periodic-foam builder.** `rnr/geometry.build_periodic_voronoi` is O(N²) ghost-tiled brute
   Voronoi → infeasible beyond a few thousand cells. The large-N GPU regime (10k–100k cells, ~20 min/100k)
   can't be exploited until the builder scales (ghost only near-boundary seeds, or a block Voronoi).
4. **Device prefix-sum for the trigger-scan compaction** (currently an O(cands) readback, not O(mesh)).
5. **Hand-CUDA-in-fork** — the two-vehicle plan's second leg (port the validated Warp algorithm to CUDA
   inside the TF fork, reusing `engine_flag_cuda`).

**Caveats / guardrails:**
- **License:** reimplement from Okuda 2013 / our own `rnr/` code. **Never copy GPL `tvm/`.** `cellGPU/`,
  `VertAX/`, `gpu_reference_papers/` are read-only oracles — study, don't paste, **don't commit** (own `.git`).
- **Precision:** RNR placement stays **fp64** (bit-reversible); Gate-E force kernels are fp64 too.
- **Validation:** RNR path = round-trip / body-anchored fingerprint; physics = host==TF (float32) /
  GPU==host (fp64) / deterministic trajectory; sorting = STATISTICAL (het-contact trend), not bit-equality.
- **Warp:** kernels live in real `.py`; every fp64 literal meeting a `wp.float64` must be `wp.float64(...)`;
  a cross-module `@wp.func` must be IMPORTED into the calling module.
- `tf.init()` is one-per-process; tests share the session-scoped `vsolver` fixture (`rnr/tests/conftest.py`).
  The physics tests reuse the periodic foam builder via `from .test_gpu_physics_csr import _build_two_type_foam, _periodic`.
- Scope: GPU-port phase only — don't scope-creep into growth or the in-engine C++/CUDA fork unless asked.
```
pixi run test                                            # full gate (~5.5 min, expect 127)
pixi run python -m pytest rnr/tests/test_gpu_*.py -q     # GPU subset (~37s, kernels cached → 79)
pixi run python -c "import warp as wp; wp.init(); print([d for d in wp.get_devices() if d.is_cuda])"
```
