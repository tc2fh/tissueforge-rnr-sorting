# TissueForge bugs & robustness gaps found during the RNR project

Running list of **TissueForge engine** issues we have hit, for later investigation / possible
upstream reporting. Scope: genuine engine/binding/build bugs — NOT our own modelling departures
(those are in `rnr/PORTING_NOTES.md` §5) and NOT dependency bugs (pyvoro etc., noted at the bottom).

Status legend: **OPEN** (no fix) · **WORKAROUND** (mitigated in our harness, root unfixed) ·
**FIXED-IN-FORK** (patched on `feat/native-rnr-reconnection`, upstream candidate).
Paths are relative to `tissue-forge/`. Last updated 2026-06-23.

| # | Title | Area | Severity | Status |
|---|-------|------|----------|--------|
| 1 | `tf.init(threads=N)` does not bound the vertex/mesh thread pool | engine threading | High (sweeps) | WORKAROUND |
| 2 | Stock 3D `MeshQuality` collapse passes segfault on finite blocks | vertex/MeshQuality | High | WORKAROUND |
| 3 | `MeshQuality` dependency-graph build races on dense graphs | vertex/MeshQuality | Med | WORKAROUND |
| 4 | Vertex renderer: orphaned surfaces drawn; `become()` colour stale | vertex/renderer | Low | WORKAROUND |
| 5 | Default surface actors use raw (non-min-image) coords (periodic) | vertex/actors | High (periodic) | FIXED-IN-FORK |
| 6 | `Mesh::setQuality` SWIG double-free | bindings | Med | FIXED-IN-FORK |
| 7 | SWIG sub-`.i`/header edits skip Python-wrapper regen | build | Low | WORKAROUND |
| 8 | Headless/singleton `tf.init()` foot-guns | init | Low | KNOWN |
| 9 | Headless `screenshot()` UB: by-value `imgCnv_t` calls by-ref converters | rendering | High (headless) | FIXED-IN-FORK |

---

## 1. `tf.init(threads=N)` does not bound the vertex/mesh thread pool  *(OPEN/WORKAROUND)*

**Symptom.** A process launched with `tf.init(windowless=True, threads=1)` still spawns ~32 worker
threads (≈ host logical-core count) and uses *every* core it is allowed. `threads=N` has no effect on
vertex-model parallelism. Running 18 such jobs unpinned on a 32-thread host → loadavg ~188, **zero
step progress** (total thrash). Only OS affinity (`taskset -c`) actually confines a job. (Separately,
OpenBLAS adds ~32 more threads/process until `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1` — standard numpy
behaviour, *not* a TF bug, but it compounded the symptom.)

**Root cause** (traced 2026-06-23, read-only):
- `threads=N` flows Python→`conf.universeConfig.threads` (`wraps/python/tfSimulatorPy.cpp:180,318`)
  → `nr_runners` (`source/tfSimulator.cpp:637`) → `engine_start(&_Engine, nr_runners, nr_runners)`
  (`tfSimulator.cpp:691`). This correctly creates **N MD "runner" pthreads** for the *nonbond/pair*
  force loop (`source/mdcore/src/tfEngine.cpp:1300`, `tfRunner.cpp:397`). So `threads=1` ⇒ 1 runner. ✓
- BUT the **vertex/mesh** force + quality loops use a SEPARATE `ThreadPool` singleton, hardcoded to
  `std::thread::hardware_concurrency() - 1` workers (`source/types/tfThreadPool.h`, ctor ≈ L60), which
  **ignores `conf.threads`**. The mesh solver dispatches `parallel_for(ThreadPool::size(), …)` —
  `source/models/vertex/solver/tfMeshSolver.cpp:427,437,448` (vertices/surfaces force passes) and
  similarly in `tfMeshQuality.cpp`. For a vertex-model run (our entire use case) this pool is the
  dominant CPU consumer and is uncapped.

**Why it matters.** TF *does* scale sublinearly with cores here (per-step 1c≈0.041s, 8c≈0.018s at
M=8), so the right sweep config is 1-core/job × many jobs — but the only way to get it is OS affinity,
because the API knob is a no-op for mesh work.

**Workaround (in `rnr/scripts/run_sweep.py`).** `taskset -c <core>` each job to a distinct core (free-
core pool) + export single-thread BLAS env. On SLURM, `--cpus-per-task=1` gives the same confinement.

**Fix sketch.** Make `ThreadPool` configurable (ctor/`reconfigure(int n)` arg instead of hardcoded
`hardware_concurrency()-1`); initialise it from `conf.universeConfig.threads` in `universe_init`
(`tfSimulator.cpp:629–691`) before the mesh solver starts; ensure `parallel_for`/`tfTaskScheduler`
honour the configured size. Files: `source/types/tfThreadPool.h`, `source/tfSimulator.cpp`,
`source/tfUniverse.{h,cpp}`, `source/types/tfTaskScheduler.h`. See memory `tf-threading-for-sweeps`.

## 2. Stock 3D `MeshQuality` collapse passes segfault on finite Kelvin blocks  *(WORKAROUND)*

Running the *stock* `MeshQuality` 3D ops (degenerate collapses: `SurfaceDemote`/`BodyDemote`/
`EdgeDemote`) on a **finite** Kelvin block segfaults after enough repeated `doQuality()` calls **even
with `reconnectLength=0`** (our reconnection pass off). Our isolated reconnection pass is stable; the
stock collapses are the culprit. Workaround: `mesh.quality = None`, or our fork's
`stock_quality_operations = False` to run only the reconnection pass. Investigate the demote ops'
handling of boundary/degenerate faces on open meshes. Ref: PORTING_NOTES §6b ("STOCK-PASS HAZARD").

## 3. `MeshQuality` dependency-graph construction races on dense graphs  *(WORKAROUND)*

`MeshQuality_constructChains` builds operation chains with `parallel_for` + `appendNext`, whose
loop-check reads/writes the shared `prev`/`next` graph without full locking — fine for sparse stock
passes, but a *dense* graph (our reconnection pass: each candidate touches ~9 surfaces, heavy overlap)
can fabricate a cycle → unbounded `MeshQuality_upstreams` recursion → stack-overflow segfault. A
related race exists in the parallel `doOperations` executor when two ops share an outer vertex/body not
listed in their `targets`. Workaround (fork): build chains and run the reconnection pass **serially**.
A general fix would lock the graph mutation or widen `targets`. Ref: PORTING_NOTES §6b/§6e.

## 4. Vertex `MeshRenderer`: orphaned surfaces still drawn; `become()` colour stale  *(WORKAROUND)*

(a) `Body::destroy()` only detaches the body pointer (`Mesh::remove(Body*)` → `s->remove(b)`); it does
NOT destroy the body's surfaces, and the renderer draws every live surface — so a "destroyed" cell
stays visible. (b) After `surface.become(otherType)` the C++ type/colour are correct but the rendered
colour keeps the creation-time value across redraws. Workarounds: also destroy orphaned surfaces; set
surface colour at creation. Ref: PORTING_NOTES §5b, memories `vertex-destroy-orphans-surfaces`,
`vertex-render-color-gotchas`.

## 5. Default surface actors use raw (non-min-image) coordinates  *(FIXED-IN-FORK)*

Every `SurfaceType` auto-binds `FlatSurfaceConstraint` + `ConvexPolygonConstraint` (default lam 0.1).
`FlatSurfaceConstraint::force/energy` computed the out-of-plane offset as a RAW
`centroid - vertex.position`; for a surface wrapping a periodic boundary the two are in different
images → a spurious ≈1/dt force (~10⁴ at dt=1e-4) → cell inversion. Only manifests in a periodic bulk
(no wrap faces in a finite cluster), so static rest-geometry tests missed it. Fixed in fork
(`source/models/vertex/solver/actors/tfFlatSurfaceConstraint.cpp`) to use `meshRelativePosition`
(min-image, identity when `periodic_geometry` is off). Same omission class as the §6g periodic-geometry
pass; worth auditing all actors. Ref: PORTING_NOTES §6i, memory `periodic-substrate-engine-bug`.

## 6. `Mesh::setQuality` double-free via SWIG ownership  *(FIXED-IN-FORK)*

`Mesh::setQuality` takes ownership and `delete`s the `MeshQuality*` on the next set / `= None` / mesh
dtor (`tfMesh.cpp:116`), but the SWIG proxy for a Python-created `tfv.Quality()` kept `thisown=1`, so
teardown freed it twice → `abort()` in `~MeshQuality`. Latent in stock TF; surfaces whenever a
Python-created Quality is attached. Fixed in fork by setting `_quality.thisown = 0` on transfer
(`wraps/python/.../tfMesh.i`). Ref: PORTING_NOTES §6d (Phase-D).

## 7. SWIG sub-`.i` / header edits skip Python-wrapper regen  *(WORKAROUND/build)*

CMake/Ninja's SWIG step tracks only the top-level `wraps/python/tissue_forge.i`, not transitively
`%include`d sub-`.i` files or the C++ headers SWIG reads. Editing a sub-`.i` (new SWIG property) or a
header (new getter/setter) builds + relinks but **silently skips regenerating the Python wrapper** — the
new symbols never reach Python though the build "succeeds." Workaround: `rm` the generated
`…/tissue_forgePYTHON_wrap.cxx` before building, verify with `grep`/`hasattr`. Fix: add the proper
dependencies to the SWIG custom command. Ref: PORTING_NOTES §6, memory `tf-swig-subi-needs-forced-regen`.

## 8. Headless / singleton `tf.init()` foot-guns  *(KNOWN)*

(a) Headless `tf.init(...)` hangs (≈0 CPU) without `windowless=True` (blocks creating a GL context).
(b) `tf.init()` is a non-reentrant singleton — a 2nd call in one process hangs (forces subprocess-per-
test). Arguably documented behaviour, but both are easy foot-guns worth a clear error instead of a hang.
(c) **Camera getters read stale until the first render** (NOT a bug — a probe artifact worth recording).
`camera_view_front/top/right/...` set the ArcBall's *target* transform (`ab->viewFront`), but
`camera_rotation()`/`camera_center()` return the *current* transform (`ab->crotation()`), which only
LERPs toward the target when the renderer updates the arcball during a draw. So reading the camera
*before any* `screenshot()`/render returns the init default for every preset (looked like "presets do
nothing"); reading *after* a screenshot returns correct, distinct values, and the presets DO render
distinct images (verified: 6 axis-aligned presets → 6 different windowless screenshots, pairwise pixel
diff 28–68). Practical rule: take/await one render before trusting camera getters. (`camera_disable_lagging`
controls the LERP.) So `VIEW=top|right|...` in `video_native_gl.py` is fine.
Ref: PORTING_NOTES §0.

---

## 9. Headless `screenshot()` is UB for every format except JPEG  *(FIXED-IN-FORK)*

**Symptom.** `tf.system.screenshot("x.png")` (and `.bmp`/`.hdr`) aborts the process (exit 134) with
`Trade::AbstractImageConverter::convertToData(): can't convert image with a zero size: Vector(0, <garbage>)`;
`.tga` silently writes 0 bytes. Only `.jpg` and `tf.system.image_data()` (JPEG) work. Long mis-blamed on
the WSL2 Mesa/Zink driver — but the windowless EGL context is fine (auto-`llvmpipe` GL 4.5), and JPEG
proves render+readback work. It is a **TissueForge bug**, driver-independent.

**Root cause.** `source/rendering/tfApplication.cpp` declared the dispatch typedef
`typedef Array<char> (*imgCnv_t)(ImageView2D);` — argument **by value** — but every converter in
`tfImageConverters.{h,cpp}` takes `const ImageView2D&` (**by reference**). `PNGImageData()` etc. did
`getImageData((imgCnv_t)convertImageDataToPNG, …)`, C-casting a `(*)(const ImageView2D&)` to the
incompatible by-value type and calling through it. By-value (a >16-byte struct) vs by-reference are
different SysV AMD64 calling conventions ⇒ the callee reads a garbage `ImageView2D` (width 0) ⇒ zero-size
image ⇒ `convertToData` assert ⇒ abort. JPEG escaped only because `JpegImageData()` passed a *by-value
lambda* matching the wrong typedef — and the widely-used path (Jupyter widget = `image_data()`) is JPEG,
so it went unnoticed upstream. (jpg vs bmp is the tell: same `RGB8Unorm`, same `StbImageConverter` —
only by-value-lambda vs by-ref-cast differs.)

**Fix.** Make the typedef match the real converters: `typedef Array<char> (*imgCnv_t)(const ImageView2D&);`
and change the JPEG lambda param to `const ImageView2D&`. Every `(imgCnv_t)convertImageDataToXxx` cast is
then an exact identity — no UB. After `pixi run build-tf`, all five formats write valid 800×600 files
(png/bmp/jpg/tga/hdr), exit 0; the vertex mesh renders headlessly (surfaces need
`SurfaceTypeSpec.style = {"color":…, "visible":True}`). Engine-source only; no Python API change.
Full write-up: PORTING_NOTES §8; memory `tf-gl-render-broken-wsl2`. (The interactive `tf.run()` GLFW
window is a separate path, not covered.)

---

## Robustness gap (not a discrete bug, tracked elsewhere)

**Post-reconnection / coarse-dt displacement overshoot → cell inflation.** At a larger timestep the
overdamped relaxation after a reconnection can overshoot a vertex by several cell widths, inflating a
cell past the volume guard (max_vol ≫ V0) over a long run — even with the native orientation repair
(which only cures eversion, min_vol<0). Observed 2026-06-23: a dt=5e-3 M=8 sweep inflated ~5/18 runs
between t≈100–1600 (dt=1e-3 is robust). The documented fix is a per-vertex **displacement trust-region
inside the vertex integrator** (so it also covers engine noise) — a planned native item, see
PORTING_NOTES §6j/§7 and memories `native-instability-is-displacement-overshoot`,
`winding-clamp-stabilizes-sort`, `dt-lever-for-faithful-sort`.

## Non-TissueForge (dependencies)

- **pyvoro-mmalahe `dispersion`**: returns overlapping full-box cells when `dispersion` < seed spacing,
  and `periodic=True` returns garbage on a regular lattice. Workarounds in `rnr/geometry.py`
  (full-box dispersion; 3×3×3 ghost-tiling instead of `periodic=`). Memory `pyvoro-dispersion-gotcha`.
