# TissueForge vertex MeshRenderer — patches & rendering gotchas

> Durable copy of the learnings from the 2026-05-29 session (renderer clip/visibility
> patches + the cell-sorting starting-point view). Kept in-repo so it survives even if
> the assistant's project memory is cleared. Companion to `patch_visualization_plan.md`
> and `tissue-forge_build_runbook.md`.

## 1. Renderer patches (DONE — on the fork)

Two patches to `tissue-forge/source/models/vertex/solver/tfMeshRenderer.{h,cpp}`, committed
and pushed to fork `tc2fh/tissue-forge`, branch `feat/mesh-renderer-clip-visibility`:

- **Patch A — clip planes** (commit `6f092d1`). The vertex `MeshRenderer` built
  clip-capable `Flat3D` shaders but never overrode the `SubRenderer` clip virtuals, so it
  silently ignored `tf.init(clip_planes=...)` / `tf.ClipPlanes`. Fix: implement
  `addClipPlaneEquation` / `removeClipPlaneEquation` / `setClipPlaneEquation`, mirroring
  `AngleRenderer`/`BondRenderer`, **forwarding each op to BOTH `_shaderFaces` and
  `_shaderEdges`** (mesh has two shaders; the references have one). Do NOT seed equations in
  `start()` — `UniverseRenderer` replays them on registration (would double-apply).
  `UniverseRenderer` owns the global GL `ClipDistanceN` enable; don't touch it.

- **Patch B — visibility** (commit `41aefed`). `render_meshFacesEdges` read the resolved
  surface style for colour only, never visibility. Fix: honour `STYLE_VISIBLE` by collapsing
  a hidden surface's geometry to its centroid (zero-area triangles / zero-length lines →
  rasterises to nothing). Fixed-stride buffer layout untouched → still `parallel_for`-safe,
  no solver/buffer changes. `null` style ⇒ visible (keeps colour fallback). `STYLE_VISIBLE`
  is default-on ⇒ backward compatible.

Validation harness: `rnr/scripts/validate_renderer_patch.py` (modes `plain` / `clip` /
`hide`). `tf.init` is one-per-process, so run once per mode. Visibility is set at SurfaceType
CREATION (`style={'visible': False}`), NOT via `become()` (see gotcha #2 below).

Build (incremental, fast — one TU): `pixi run build-tf`. Validate headless with
`tf.system.screenshot`, `windowless=True`.

## 2. Rendering colour/visibility gotchas (headless screenshot path)

> **Re-investigated 2026-05-29** with controlled, separate-process, **pixel-counted** A/B
> tests + source tracing. Of the four "all-one-colour traps" originally logged here, only ONE
> was a real renderer limitation; two were FALSE, and the rest were symptoms of a single
> bug the first pass never isolated — `Body.destroy()` orphaning surfaces. Corrected list
> below; the original (wrong) wording is preserved in §2.1 as a cautionary tale.

How colour actually works: the vertex `MeshRenderer`
(`source/models/vertex/solver/tfMeshRenderer.cpp::render_meshFacesEdges`) computes each
surface's colour FRESH every draw as `s->style ? s->style : s->type()->style`. A surface's
per-instance `style` is `NULL` after creation (ctor inits `style{NULL}`), so colour comes
from the **SurfaceType's** style, re-read every frame.

**Always verify by PIXEL-COUNTING the PNG** (`PIL.Image.open` → count blue vs orange). The
screenshot read channel is flaky and a single eyeballed shot — or a stale `/tmp` file —
fooled the original debugging repeatedly.

### Confirmed

1. **`Body.destroy()` does NOT hide a cell — it orphans the surfaces, which keep rendering.**
   The core gotcha. `Body::destroy → Mesh::remove(Body*)` only does
   `for(auto &s : b->getSurfaces()) s->remove(b);` — it **detaches** the body pointer from
   each surface, then frees the body slot; it never destroys surfaces. The renderer draws
   **every** live surface in the mesh (`parallel_for(mesh->surfaces->size())`, any with
   `objectId() >= 0`) regardless of body membership. Verified: built 189 cells
   (890 quad + 216 hex surfaces), destroyed 188 bodies → bodies=1 but the surface tally was
   **unchanged** (890 + 216). So "destroy boundary cells, keep N interior" renders the WHOLE
   block (its all-square exterior hull reads as one colour from outside — which is what the
   first pass misread as several separate colour bugs).
   **Fix:** after destroying the unwanted bodies, also destroy the orphaned surfaces:
   ```python
   for st in (stA, stB):
       for s in list(st.instances):
           if len(s.getBodies()) == 0:   # no surviving body -> orphan -> still renders
               s.destroy()
   ```
   Verified: reduces a lone kept cell to exactly its 14 faces. Cleanest for a single cell —
   just BUILD only the surfaces you want; see `rnr/scripts/kelvin_single_cell.py`.

2. **Render colour = the SURFACE type's style, fixed at surface CREATION.** Pass the coloured
   `SurfaceType` to the surface constructor. `surface.become(otherType)` does NOT re-render
   the colour in the windowless single-`screenshot` path — VERIFIED real: after `become`, the
   C++ `typeId`/`type()`/`type().style.color` are all correct (e.g. 478 surfaces now type B,
   B.style = orange) and `s.style is None`, yet the render keeps the creation-time colour
   across repeated screenshots AND a `position_changed()`. The data is right; only the
   rendered colour is stale. So set colour at creation. (`view_cluster.py` escapes this only
   because `tf.show()` runs a live loop.) Same caveat applies to visibility (set at creation,
   per §1's validation note).

### Debunked — do NOT cargo-cult (shown FALSE by pixel-identical A/B renders)

3. **❌ "A second BodyType clobbers surface colours."** One BodyType vs two (built by
   construction) → **pixel-identical** multi-colour renders. BodyType count has no effect on
   colour. One BodyType is still the simplest setup but is NOT required; cell identity for the
   sort can safely ride on a 2nd BodyType. (Removes the "open follow-up" the old note raised.)

4. **❌ "Set the camera before the build; moving it after resets colour."** Camera-before vs
   camera-after build → **pixel-identical**, both multi-colour. And a
   `camera_rotate_by_euler_angle` BETWEEN two screenshots *does* change the second image — the
   renderer re-renders fresh each shot. Camera ordering has zero colour effect. The original
   "all-blue after tilt" was trap #1 (the orphaned full block). Useful headless camera API:
   `camera_zoom` takes NO args (it's the scroll handler) — use `camera_zoom_by(delta)`
   (positive = zoom in; `camera_zoom_to` overshoots after a rotate); `decorate_scene(False)`
   hides the box wireframe + grid for a clean cell shot.

5. **⚠️ "Index-based per-face colouring renders all-blue; use nearest-seed."** Not re-tested
   in isolation; almost certainly another symptom of trap #1 (the whole ~189-cell block
   rendered, dominated by hull faces). `nearest_label` (nearest kept seed to face centroid)
   still works and is what the demo uses; revisit only if it recurs after the orphan fix.

### Possible native-port renderer patches (investigated, NOT applied)

- **Orphan/destroy (trap #1):** fixable cleanly — either have `MeshRenderer::draw` skip
  surfaces with no live body (~2 lines), or have `Body::destroy` optionally cascade-destroy
  its exclusively-owned surfaces. Localized, no API change. For the prototype we handle it in
  Python (the snippet above).
- **`become` stale colour (trap #2):** the data is already correct and `draw()` re-reads
  colour every frame, yet the output is stale even across forced re-renders (camera nudge AND
  `tf.step()` both fail to propagate it) — a deeper render-path/object-identity issue needing
  an instrumented build, NOT the "add a per-surface colour override" a scoping subagent
  speculated. Left unpatched; set colour at creation. **Full investigation + fix plan with a
  minimal repro, ruled-out causes, ranked hypotheses (H1 object-identity is the lead), and an
  instrumentation protocol: `docs/become_color_bug_plan.md`.**

### 2.1 Previous (incorrect) wording — kept as a cautionary tale

The first pass logged four separate "all-one-colour" traps — (1) become doesn't render,
(2) a 2nd BodyType clobbers colour, (3) index-vs-geometry colouring, (4) camera-move resets
colour — and prescribed "one BodyType + camera-before-build." Only #1 held up. #2 and #4 were
never controlled-tested (single eyeballed screenshots of a scene that was *also* hitting the
orphan bug) and propagated into three scripts + project memory before being caught. Lesson:
pixel-count, change ONE variable per run, and suspect a single upstream cause before logging
several independent ones.

## 3. Cell-sorting starting-point view

`rnr/scripts/sorting_demo_start.py` — `pixi run python rnr/scripts/sorting_demo_start.py
[out.png] [clip]`. Builds a Kelvin-cell cluster from a pyvoro Voronoi of a 5×5×5 BCC lattice
(interior 14-faced cells kept; the box-clipped boundary layer's bodies AND their orphaned
surfaces destroyed — see §2 trap #1), randomly split into type A (blue) / type B (orange) —
the salt-and-pepper initial condition for the 3D vertex-model sort once RNR reconnection is
wired in. The `clip` arg adds a centre clip plane (Patch A) to reveal the interior mix. Last
verified: 35 cells (A=18, B=17), 746 orphaned boundary surfaces destroyed, both colours
present by pixel count; outputs in `rnr/exports/sorting_demo_{block,clip}.png`.

A companion single-cell proof — `rnr/scripts/kelvin_single_cell.py` → 
`rnr/exports/kelvin_single_cell.png` — builds ONE interior Kelvin cell directly (no destroy
path) and renders it as a truncated octahedron: 6 square (blue) + 8 hexagon (orange) faces.

## 3b. Clip planes — VERIFIED WORKING in the headless screenshot path (2026-06-24)

Re-debugged because the videos "couldn't" show half the tissue. Conclusion: **the native
clip plane is NOT broken** — Patch A (§1) made the vertex `MeshRenderer` honour it, and it
clips correctly windowless via BOTH entry points. The earlier "it does nothing" reading was a
**measurement artifact** (see the pixel-count caveat below), not a renderer bug.

- **Init path:** `tf.init(clip_planes=[(point, normal), ...])`. The C++ parser
  (`tfSimulatorPy.cpp::parse_kwargs`) is **strict and silent**: each entry must be a
  `(point, normal)` **tuple** whose point and normal are **lists** of 3 floats. A tuple-of-tuples,
  a list-of-lists, or numpy arrays for point/normal are **silently dropped** (it logs `LOG_ERROR`,
  off by default) → `tf.rendering.ClipPlanes.len() == 0` and no clipping. `rnr.clip.parse_clip_env`
  returns exactly the right `(list, list)` shape; the videos wrap it as `[(point, normal)]`.
- **Runtime path:** `cp = tf.rendering.ClipPlanes.create(tf.FVector3(point), tf.FVector3(normal))`,
  then `ClipPlanes.len()`, `cp.setEquation(point, normal)` (re-aim), `cp.destroy()`. All work
  headless too (each makes the GL context current via the next `screenshot`).
- **Kept half = the side the normal points toward** (`tfFlat3D.vert`:
  `gl_ClipDistance = dot(instancePosition, planeEq) >= 0`, planeEq = `(normal, -dot(normal,point))`).
  The shader's `instancePosition` is the raw world-space vertex (no `INSTANCED_TRANSFORMATION` on
  the mesh shader), so the plane is defined directly in simulation coordinates `[0, L]`.
- **Camera turntable** (to orbit and see INTO the cut, not just one side):
  `tf.system.camera_rotate_to_euler_angle(FVector3(pitch, yaw, roll))` — angles in **RADIANS**;
  `pitch < 0` looks DOWN (pair with a top cut, i.e. `CLIP=-z` keeping the lower half). Sweep `yaw`
  per frame for the orbit. NOTE `tf.system.camera_rotation()` returns a stale/identical quaternion
  across `camera_view_*` presets — do not use it to read back orientation.
- **PIXEL-COUNT CAVEAT (this is what faked the bug):** TF's headless background is **mid-gray
  (~128,128,128)**, not black. A naive "foreground = non-dark, saturated-ish" pixel count matches
  ~99.9% of an 800×600 frame whether or not geometry is clipped, so it looked like nothing changed.
  **Always eyeball the PNG** (or count against the actual gray bg / the box-wireframe-only frame).
  This is the §2 "pixel-count the PNG" rule again — but the threshold itself must exclude the gray.

Wiring: `video_native_gl.py` (real TF clip plane, init path) and `video_native_cells.py`
(matplotlib per-face polygon clip, `rnr.clip.clip_polygon_halfspace`) both take `CLIP=<sign><axis>`
(`z`,`-z`,`x`,...) + `CLIP_AT=<frac>`; the turntable auto-enables when `CLIP` is set. Clipped
renders write to their OWN `native_*_frames_clip<spec>/` scratch dir + `_clip<spec>` mp4 so they
never clobber a concurrent un-clipped run.

## 4. Git author note

The fork's local commits initially used the hostname email; set to
`tien.comlekoglu@gmail.com` / "Tien Comlekoglu" so commits attribute correctly. Commit
messages end with the `Co-Authored-By` trailer.
