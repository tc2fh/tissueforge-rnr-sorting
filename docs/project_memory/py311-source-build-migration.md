---
name: py311-source-build-migration
description: Plan to move off the py3.9 conda binary to a from-source TissueForge build at Python 3.11 (enables renderer patching)
metadata: 
  node_type: memory
  type: project
  originSessionId: 50be984f-bc37-4828-bb5f-f47b6cae3a57
---

Decided (2026-05-29) to migrate off the prebuilt conda `tissue-forge 0.2.1` (py3.9,
osx-arm64) to a **from-source build at Python 3.11**, which also unblocks the renderer
patches in `patch_visualization_plan.md`.

**Why 3.11:** the py3.9 limit was only the prebuilt arm64 *artifact*. TF source supports
3.9–3.13 (`tissue-forge/CMakeLists.txt:263` caps at `<3.14`; conda recipe matrix builds
3.9–3.13). The real ceiling is **pyvoro-mmalahe 1.3.4** (Aug 2023, classifiers stop at
3.11, no arm64 wheel → always builds from sdist). 3.11 is the safe newest; 3.12/3.13 need
patching pyvoro's Cython build or switching to the scipy Voronoi fallback.

**How to apply:** the `default` pixi env is REPLACED (user doesn't want the py3.9 env
kept) — `pixi.toml` now pins py3.11, drops the conda `tissue-forge` dep + channel, and
carries the build toolchain; a `build-tf` task compiles TF from `./tissue-forge` INTO the
env prefix via `build_tissue_forge_osx.sh` (`CMakeLists.txt:302-312` installs the python
pkg relative to `CMAKE_INSTALL_PREFIX`, so one prefix = no cross-env dylib bridging). Then
apply the renderer patches and rebuild. The `tissue-forge/` clone (currently upstream
`tissue-forge/tissue-forge` on `main`) gets forked first: `origin` → `tc2fh/tissue-forge`,
`upstream` → original, patches on branch `feat/mesh-renderer-clip-visibility`. Full
single-path plan (fork → env reinstall → build → patches → validate) + troubleshooting in
`tissue-forge_build_runbook.md`. Run sequence: fork/branch (§2) → `pixi install` →
`pixi run build-tf` → `pixi run verify`.

DONE (2026-05-29): build + verify GREEN on py3.11. Gotchas hit & solved, all in the
runbook troubleshooting table: (1) `git submodule update --init --recursive` for empty
`extern/`; (2) conda-forge split libxml2 → added `libxml2-devel` for headers; (3) clang 19
+ `tf_types.i:124` `using namespace std;` made TF's `struct queue` collide with
`std::queue` — fixed by a forward declaration `struct queue;` in `tfEngine.h`'s
`namespace TissueForge` (NOT by qualifying the use, NOT by pinning libcxx 15 which is
unsolvable vs modern libxml2-devel/assimp/icu). ~900 build warnings (mostly Magnum
`Generic3D` deprecations in upstream source) are benign.

DONE (2026-05-29): Renderer Patch A (clip planes) + Patch B (STYLE_VISIBLE, Option A)
implemented in `tfMeshRenderer.{h,cpp}`, validated headless, committed (6f092d1, 41aefed)
and pushed to `origin` (tc2fh fork) branch `feat/mesh-renderer-clip-visibility`. Patch A
mirrors AngleRenderer's clip virtuals, forwarding to BOTH face+edge shaders. Patch B
collapses hidden surfaces to their centroid (zero-area tris / zero-len lines). Validation
script: `rnr/scripts/validate_renderer_patch.py` (modes plain/clip/hide; tf.init is
one-per-process so run once per mode). KEY GOTCHA: visibility/color must be set at
SurfaceType CREATION (`style={'visible': False}`); `surface.become()` does NOT propagate
to the renderer in the no-show single-`screenshot` path (a pre-existing become/render-cache
limitation affecting color too, not the patch) — per-instance control needs the optional
Patch C SWIG ownership fix. Patch C not done (type-level visibility suffices for the
interior-Kelvin use case). git author fixed to tien.comlekoglu@gmail.com (was hostname email).

Related: [[phase0-baseline-control]], [[pyvoro-dispersion-gotcha]].
