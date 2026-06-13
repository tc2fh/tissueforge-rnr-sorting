# RNR Reconnection in TissueForge — Project Memory

## What this project is

Goal (this phase): reproduce **3DVertVor-style cell sorting** — multiple cell
types with **heterotypic interfacial tension** — inside **TissueForge's 3D vertex
model**, by adding the one capability TissueForge lacks: a **3D T1 / reversible
network reconnection (RNR)** operation so cells can swap neighbors and actually sort.

Prototype now in **Python** via TissueForge's event system. Harden into a native
C++ `MeshQuality` operation **later** (separate project). Growing-sheet
morphogenesis (Okuda-style volume growth, monolayer→multilayer) is also **later**.

Do NOT scope-creep into growth or the C++ port unless explicitly asked.

## Current status (2026-06-13) — Phase-2 REPRODUCED ✅ + the active drive is now fully NATIVE ✅

3DVertVor/Manning **Fig 1E + 1F are reproduced** in TissueForge's 3D vertex model, faithfully
(clamp-free), with the native 3D I↔H reconnection supplying the missing 3D T1. Phases 0–2 (below)
are done. **Phase 3 (2026-06-13): the active self-propulsion drive — the last non-native piece — is
now NATIVE in the engine** (per-cell `Body::director` + rotational diffusion in
`MeshSolver::preStepStart` + a per-vertex active force `v0·⟨incident directors⟩` in `VertexForce`,
set via `MeshSolver.set_motility`). The faithful RNR + active cell-sorting capability now runs
entirely inside TissueForge; the Python `add_noise_active` injection is retired to a comparison
fallback. Broader standalone C++ port hardening and growth remain **later**.

- **The finding that closed it:** the 3DVertVor/Manning oracle drives vertices by **active
  self-propulsion** (`x += dt·motility`, per-step ≈ 0.1·Lth), **not** thermal Brownian noise
  (√dt ≈ 14–45× Lth, which starves the reconnect trigger). Our harness had substituted thermal
  `tf.Force.random`; the faithful active model removed the last "noise-clamp" departure. Full
  reasoning: `rnr/PORTING_NOTES.md` §6n + memory `active-motility-not-thermal-noise`.
- **Native drive (§6o):** `MeshSolver.set_motility(v0, Dr, seed)` enables it; `body.director` reads
  the per-cell director. The seam: the vertex integrator is overdamped with unit mass (μ=1), so a
  per-vertex force `v0·⟨n⟩` gives displacement `dt·v0·⟨n⟩` = the oracle's `dt·motility` exactly.
  Validate: `pixi run python rnr/scripts/probe_native_calibration.py` (displacement/μ + Dr).
- **Run it:** `pixi run test` (49-test gate) · `probe-active … native` (clamp-free native rate) ·
  `sort-oracle` (one sort, `NOISE_MODEL=native` default; `active`=Python comparison) · `overnight`
  (full ensemble + figs + video) · `dpmax`/`fig1e`/`fig1f` (`MODEL=active`). Deliverables in
  `rnr/exports/`: `fig1e_demixing_active.png`, `fig1f_stability_active.png`, `sort_active_demixing.gif`.
  (Remaining polish: regenerate canonical fig1e/1f with the native drive — point `run_overnight.py`
  at `native` + a `MODEL=native` fig selector; `overnight` as written still uses `active`.)
- **Engine:** fork `feat/native-rnr-reconnection` (native I↔H + periodic min-image geometry +
  per-cell orientation repair + **native active-motility drive**). Build tree `tissue-forge_build/`.
- History lives in `PORTING_NOTES.md` (§6 = native port + noise finding) and the auto-memory;
  `docs/` keeps the findings notes. (Stale kickoff/planning prompts + the earlier-phase
  diagnostic scripts/tasks were pruned 2026-06-12.)

## Repos in this workspace (workspace root = the directory holding this file)

- `tissue-forge/` — the target engine. **LGPL.** Reference its source freely;
  the eventual C++ port will modify a fork of it.
- `tvm/` — Zhang & Schwarz 3D vertex model, the original implementation of the
  Okuda RNR algorithm. **GPL v3.**
- `3DVertVor/` — Manning-lab fork of `tvm` adding multiple cell types + heterotypic
  tension + `.vtu` output. **MIT, but derives from GPL `tvm`.**
- `rnr/` — **our** new prototype code (Python). All new work goes here this phase.

### License boundary — important
`tvm` is **GPL v3**. Do **not** copy/paste code from `tvm/` (or the GPL-derived
parts of `3DVertVor/`) into `rnr/` or into any TissueForge fork — GPL terms would
attach to LGPL TissueForge. Instead **reimplement the algorithm from the Okuda 2013
equations** (see below) against TissueForge's API, and use `tvm/` + `3DVertVor/`
only as a **correctness oracle** (run them, compare numbers/behavior). When in
doubt, re-derive rather than transcribe. Keep a note in commit messages when an
approach was checked against the reference repos.

## The key insight (don't re-derive this each session)

TissueForge **already has the cell-sorting energetics** (the actors exist). The only
missing piece is the 3D reconnection. Specifically:

- **Volume elasticity** → `VolumeConstraint` actor (on bodies). ✅ exists
- **Heterotypic interfacial tension** → carried by a COMBINATION of actors, not one.
  TissueForge's bundled *vertex* cell-sorting example (Osborne et al. 2017; see
  `wraps/python/models/vertex/solver/examples/cell_sorting.py`) uses **`EdgeTension`
  (per cell type) + `PerimeterConstraint` + `SurfaceAreaConstraint` + `Adhesion`**,
  and derives the per-type-pair values from edge-tension params
  (e.g. `adh_ab = 2*adh_hetr - (lam_a + lam_b)`). So σ_ij ≈ EdgeTension(by type)
  combined with Adhesion(by type-pair) — NOT Adhesion alone. ✅ actors exist
  - CAVEAT for 3D: that example is **2D** (`frozen_z=True`, `create_hex2d_mesh`).
    In 2D the interfacial term lives on **edges** (EdgeTension). In 3D it lives on
    **surfaces** (a surface = a cell–cell boundary, `Surface.b1/b2`). So the exact
    decomposition of "heterotypic tension" must be re-derived for 3D against the
    surface-based actors (`SurfaceAreaConstraint` + `Adhesion` on surfaces); do not
    assume the 2D EdgeTension recipe ports verbatim.
- **Surface area / interface energy** → `SurfaceAreaConstraint`. ✅ exists
- **3D T1 / I↔H reconnection** → **does not exist.** See the next paragraph — this is
  the whole project.

**Why the 2D example sorts but 3D won't (the crux):** the bundled example reaches
neighbor exchange via `vertex_merge` + `edge_split` (it sets
`quality.vertex_merge_distance` / `edge_split_distance`). That merge/split pair **is
the 2D T1 transition.** TissueForge has no 3D analogue — its 3D quality ops are only
degenerate collapses (`BodyDemote`, `SurfaceDemote`, `EdgeDemote`). **The RNR I↔H
reconnection IS the 3D T1.** Supplying it is the entire point of this project.
- **3D T1 / I↔H reconnection** → **does not exist.** TissueForge's `MeshQuality`
  has only 2D ops + degenerate-3D collapses (`BodyDemote`, `SurfaceDemote`,
  `EdgeDemote`). No face↔edge swap. **This is the whole project.**

So: most of the work is the reconnection operation + a correct initial packing +
a sorting metric — not porting energies.

## TissueForge data model (verified against source — trust this)

The mesh maps almost 1:1 onto the RNR papers:

- **`Surface`** (= a "polygonal face" / cell–cell boundary) has exactly two body
  pointers, **`b1` and `b2`**. A surface between two bodies *is* the polygon between
  two cells in Okuda/tvm. Members: `vertices` (ordered, winding defines normal),
  `b1`, `b2`, `actors`.
- **`Vertex`** stores `surfaces` (the surfaces it defines) and an underlying
  TissueForge **particle** (integrated with overdamped mechanics). Helpers:
  `connectedVertices()`, `sharedSurfaces(other)`, `getBodies()`, `findSurface(dir)`.
- **`Body`** (= a "cell" / polyhedron) stores `surfaces`. Helpers:
  `getVertices()`, `connectedBodies()`, `findSurface(dir)`.
- Edges are **implicit** (consecutive vertices in a surface's ordered list).
  TissueForge has no explicit Edge object — this differs from `tvm`, which has an
  `Edge` class. Our reconnection must work with implicit edges (vertex pairs).
- Topology-mutating primitives available from Python:
  `Surface.merge`, `Surface.sew`, `Vertex.insert`/`Vertex.insert_c`,
  `Surface.split`, plus `find_vertex(dir=...)`, `find_surface(dir=...)`.

Source files to read when reasoning about topology:
- `tissue-forge/source/models/vertex/solver/tfSurface.{h,cpp}`
- `tissue-forge/source/models/vertex/solver/tfBody.{h,cpp}`
- `tissue-forge/source/models/vertex/solver/tfVertex.{h,cpp}`
- `tissue-forge/source/models/vertex/solver/tfMeshQuality.cpp` (how existing ops are
  structured: a *check/predicate* half + an *implement/mutate* half; mirror this)
- `tissue-forge/source/models/vertex/solver/actors/tfAdhesion.cpp` (the σ_ij machinery)

Reference implementation to compare against (read, don't copy):
- `tvm/Reconnection/Reconnection.cpp` — `I_H()` and `H_I()`. The I→H direction is
  ~400 lines of pointer surgery. Note it relies on an explicit `Edge` and on each
  vertex having exactly 4 neighboring cells.

## The four RNR conditions (Okuda et al. 2013, Biomech Model Mechanobiol 12:627–644)

The PDF is the authority. Implement from these equations, not from `tvm` code.

1. **Condition 1 — polygon face shape / center.** A face with ≥4 edges is split into
   radial triangles from a center point; the center is the **edge-length-weighted
   average of edge midpoints** (Okuda Eq. 3), *not* the plain vertex centroid. This
   keeps center displacement O(Δl_th) under vertex add/remove. In TissueForge,
   `Surface` area/centroid uses a centroid triangulation already — verify whether its
   centroid convention matches Eq. 3 closely enough; if not, this is a place the
   prototype may diverge from the paper and must be noted.
2. **Condition 2 — reconnection trigger.** Reconnect when the max relevant edge
   length < threshold **Δl_th** (NOT the min — `tvm`'s "condition H" uses min and is
   the wrong one). Gaps are then O(Δl_th) and reversible.
3. **Condition 3 — energy gap.** ΔU_{I→H} + ΔU_{H→I} = O(Δl_th) (Eq. 4). Satisfied by
   construction if vertices are placed per Appendix 1 and Δl_th is small. The
   TissueForge energies (Adhesion/area/volume) are smooth functions of geometry, so
   this should hold; verify empirically.
4. **Condition 4 — topological constraints (prevents irreversible patterns).**
   (i) two edges never share two vertices simultaneously;
   (ii) two faces never share two or more edges simultaneously.
   `tvm` adds a third "EXTRA RULE": two cells never share two or more faces. Port all
   three as **guards that veto an illegal reconnection** before mutating.

**Vertex placement (Okuda Appendix 1)** — the formulas that make I↔H reversible:
- H→I (triangle 7-8-9 → edge 10-11): place 10/11 at the triangle's center of mass
  ± `0.5·Δl_th·u_T`, where `u_T` is the unit normal of triangle 7-8-9.
- I→H (edge 10-11 → triangle 7-8-9): place 7/8/9 at the edge midpoint plus
  `(Δl_th / L_max)·v_k`, where the `v` vectors are projections (onto the plane normal
  to the edge) of averaged directions to the six outer neighbors, and `L_max`
  normalizes the largest.

## Environment (pixi — TissueForge is a conda package)

Use **pixi** (not uv): TissueForge's binary is conda-only, so a conda-backed env is
required. `uv`/pip can't pull it.

- TissueForge install: `conda install -c conda-forge -c tissue-forge tissue-forge`
  → in pixi, add channels `conda-forge` and `tissue-forge` and dependency `tissue-forge`.
- On macOS, vertex-model rendering needs the windowing libs that ship with the conda
  package; headless/batch runs are fine for the science.
- **pyvoro** (initial Voronoi geometry) is **pip-only** and its C extension can fail
  to build on modern Python. The original `joe-jordan/pyvoro` is stale; `tvm`'s README
  uses the maintained fork **`pyvoro-mmalahe`**. Try `pyvoro-mmalahe` first via pixi's
  `[pypi-dependencies]`. If it won't build, fall back to scipy's
  `Voronoi`/`Delaunay` to generate the initial packing.
- numpy, scipy, matplotlib (analysis/plots), pytest (tests) → conda-forge.

When setting up: create `pixi.toml` at the workspace root, put env work behind
`pixi run` tasks (e.g. `pixi run test`, `pixi run baseline`, `pixi run sort`).
Verify the env with a one-liner: `import tissue_forge as tf; tf.init()` then
`from tissue_forge.models.vertex import solver as tfv; tfv.init()`.

## Phase plan (current phase = up to 3DVertVor cell sorting)

**Phase 0 — environment + baseline (do before any physics).**
- Stand up the pixi env; confirm `tf.init()` / `tfv.init()` work.
- Run TissueForge's own **vertex-model** example
  `wraps/python/models/vertex/solver/examples/cell_sorting.py` (NOT
  `examples/py/cell_sorting.py`, which is the particle/center-based model) to confirm
  the install and study the real vertex API: `SurfaceTypeSpec` types, `adhesion`
  dicts, `bind_adhesion`, `become()`, `create_hex2d_mesh`, `mesh.quality`. Sibling
  vertex examples worth a look: `cell_migration.py`, `cell_splitting.py`,
  `capillary_loop.py`. Remember it's 2D and sorts via merge/split (the 2D T1).
- Build the 3D control: bodies (cells) in a periodic box with `VolumeConstraint` +
  `SurfaceAreaConstraint` + per-type-pair tension (Adhesion on surfaces; see the
  energetics caveat above), **no reconnection**. Expect it to
  jam and NOT sort. This control proves reconnection is the missing piece.

**Phase 1 — topology toolkit + I↔H, with a round-trip test as the gate.**
- `rnr/topology.py`: given a short (implicit) edge, identify the full I-neighborhood
  (two end-vertices, surrounding bodies, the surfaces and outer vertices involved),
  using TissueForge adjacency helpers. Mirror the *structure* of `tvm`'s `I_H` walk,
  reimplemented against the TF API.
- `rnr/reconnect.py`: forward `i_to_h` and reverse `h_to_i` using TF mesh-editing
  primitives + Appendix-1 vertex placement.
- `rnr/conditions.py`: Condition-4 guards (i), (ii), + the extra rule.
- **GATE:** `rnr/tests/test_roundtrip.py` — build a tiny hand-made configuration,
  apply I→H then H→I, assert topology + geometry return to original within tolerance.
  Reversibility is the entire point of RNR; this test is the definition of "Phase 1
  done." Do not move on without it green. Also test that Condition-4 guards correctly
  veto the known irreversible patterns (Okuda Fig. 6/9: double edges, double trigonal
  faces).

**Phase 2 — wire into the loop + reproduce sorting.**
- `rnr/operator.py`: a per-step TissueForge event that scans for edges < Δl_th and
  runs the validated reconnection on legal ones. (Note: native ops run *inside*
  `MeshQuality::doQuality()` in the C++ step; from Python we run between steps. This
  is a valid prototype, not the final integration.)
- Re-run the two-type setup **with** reconnection. Heterotypic tension should now
  drive sorting.
- `rnr/metrics.py`: a quantitative sorting readout (heterotypic-contact fraction
  over time, and/or a sorting index). "Did it sort?" must be a number.
- Compare qualitatively to a `3DVertVor` run (oracle) if feasible.

**Phase 3 — prep for C++ port (light, ongoing).**
- Keep the Python operator split into check-half + mutate-half so the later C++
  `MeshQualityOperation` subclass is a translation, not a redesign.
- Maintain `rnr/PORTING_NOTES.md` listing every TF API call the prototype depends on
  (these are the seams to re-express in C++).

## Working agreements for Claude Code

- **Tests first for Phase 1.** Write `test_roundtrip.py` and the Condition-4 veto
  tests before/with the reconnection code; treat them as the spec. Plausible-looking
  motion can silently corrupt the mesh — only the round-trip proves correctness.
- **Use the reference repos as an oracle, never as a paste source** (license + the
  data models differ: TF has no explicit Edge, `tvm` does).
- After any topology mutation, assume handles/pointers may be invalidated; re-fetch
  from the mesh rather than reusing stale references.
- Prefer small, runnable increments. Each phase should end with something that runs
  via a `pixi run` task and a passing test or a saved plot/metric.
- When a modeling choice departs from the paper (e.g. centroid convention in
  Condition 1), say so explicitly in code comments and in PORTING_NOTES.md.
- Cite the Okuda equation number in comments where a formula is implemented.
- Don't claim a phase is done until its gate (test or metric) actually passes.

## Citations to keep handy
- TissueForge: Sego et al. (2023), *Sci. Rep.* 13:17886, “General, open-source vertex
  modeling in biological applications using Tissue Forge.”
- RNR algorithm: Okuda et al. (2013), *Biomech. Model. Mechanobiol.* 12:627–644.
- 3DVertVor base: Zhang & Schwarz, *Phys. Rev. Research* 4:043148 (arXiv:2204.07081).
