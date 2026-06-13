# PORTING_NOTES — TissueForge API seams the RNR prototype depends on

Purpose: list every TissueForge (TF) Python API the prototype relies on, so the
eventual native C++ `MeshQualityOperation` port (separate project) is a translation,
not a re-discovery. Also records environment gotchas and where the prototype departs
from `CLAUDE.md`. Updated as of Phase 0 (control complete).

All API below was verified against the installed conda package
(`tissue-forge 0.2.1`, py39, osx-arm64) and the source tree in `tissue-forge/`.

---

## 0. Environment gotchas (each one cost real debugging time)

| Gotcha | Symptom | Fix |
|---|---|---|
| **Headless needs `windowless=True`** | `tf.init(...)` hangs forever (≈0 CPU, blocked) when launched from a non-GUI shell on macOS — it blocks creating a GL/window context. | `tf.init(windowless=True, ...)`. Bundled `tissue_forge/examples/windowless.py` is the reference. |
| **`tf.init()` is a singleton** | Calling `tf.init()` a *second time* in one process hangs. | One `tf.init()` per process. For pytest (Phase 1) this means a **session-scoped fixture or a subprocess per test file** — you cannot re-init between tests. |
| **`pyvoro` `dispersion`** | pyvoro-mmalahe returns overlapping **full-box** cells (Σ vol ≫ box vol) when `dispersion` < seed spacing. | Pass `dispersion ≥` max box edge. `rnr.geometry` defaults to the largest box edge. |
| **`MeshQuality` is auto-enabled** | Topology silently changes during stepping even with no reconnection (small contacts collapse; we saw 91→81 contacts). | TF gives every mesh a default `MeshQuality`. Disable with `solver.get_mesh().quality = None` (calls `setQuality(nullptr)`). |
| **Handle identity** | `id(handle)` is unstable — `connected_bodies` etc. return fresh Python wrappers each call, so `id()` double-counts. | Use the mesh id: `BodyHandle.id` / `SurfaceHandle.id` / `VertexHandle.id`. |
| **Low-level topology ops are ONE-DIRECTIONAL** (Phase 1) | `Surface.replace/insert/remove(vertex)` edit the surface's vertex ring but do **not** update the vertex→surface back-pointers (`vh.getSurfaces()` stays stale). | After each, mirror it: `new_v.add(surface)` / `old_v.remove(surface)`. Wrapped in `reconnect._replace_v/_insert_between/_drop_v`. |
| **Body↔surface attach needs BOTH sides** (Phase 1) | `Body.add(surface)` alone leaves `surface.getBodies()` empty; `Surface.add(body)` alone is also insufficient. | Call **both** `surface.add(body)` AND `body.add(surface)` (`reconnect._attach_body`). `surface.refresh_bodies()` is a safe no-op afterward. |
| **`Surface.neighbor_vertices(v)` not iterable** (Phase 1) | Returns a SWIG `std::tuple<VertexHandle,VertexHandle>*` that raises `TypeError: not iterable` and leaks. | Walk the ordered ring yourself: `topology.ring_neighbors(surface, v)` returns `[prev, next]` from `surface.vertices`. |
| **Destroying via handle, not the static** (Phase 1) | `tfv.Vertex.destroy([handle, ...])` / `tfv.Surface.destroy([...])` raise "wrong type" — they want `Vertex*`/`Surface*` vectors, not handles. | Use the handle's no-arg method: `vh.destroy()` / `sh.destroy()`. (`Surface.destroy()` also frees any now-orphaned vertices.) |

---

## 1. Init / solver / mesh

| Python (prototype uses) | C++ symbol | Notes |
|---|---|---|
| `tf.init(windowless=True, dim=[...], cutoff=, dt=)` | `Simulator` init | `bc` defaults to periodic *for MD particles only* — irrelevant to the vertex mesh. |
| `tfv.init()` | `MeshSolver::init` | exported as `solver.init`. |
| `tfv.MeshSolver.get()` | `MeshSolver::get()` | singleton accessor. |
| `solver.get_mesh()` | `MeshSolver::getMesh()` | returns the `Mesh`. |
| `mesh.quality = None` | `Mesh::setQuality(MeshQuality*)` | `None`→`nullptr` disables auto-quality; `mesh.has_quality` then False. |
| `tf.step()` | one integrator step of `dt` | drive headless loops with this (no `tf.show()`). |

## 2. Type specs + energetics (all via `models.vertex.solver.mesh_types`)

| Python | Effect | C++ seam |
|---|---|---|
| `class X(BodyTypeSpec): volume_lam, volume_val, surface_area_lam, surface_area_val, adhesion={name:λ}` | declarative cell type | `BodyType` + bound actors |
| `X.get()` | registers type, **binds** `VolumeConstraint` + `SurfaceAreaConstraint`, returns `BodyType` | `bind_body_type` |
| `BodyTypeSpec.bind_adhesion([A, B])` | binds `Adhesion` for every type **pair** | `bind_types` |
| `SurfaceTypeSpec` (subclass, `.get()`) | surface type for faces (no per-surface energy needed in this control) | `SurfaceType` |

Energy actors used (3D Hamiltonians):
- `VolumeConstraint` = `λ_V (V − V₀)²` per body.
- `SurfaceAreaConstraint` = `λ_A (A − A₀)²` per body.
- `Adhesion` = `λ_ij · A_ij` on each shared surface, by **body-type pair**.
  TF sign convention: **higher λ ⇒ weaker adhesion ⇒ higher interfacial tension.**
  So homotypic λ negative (sticky), heterotypic λ positive (repulsive) ⇒ differential
  adhesion. This is the **3D, surface-based** decomposition of σ_ij — NOT the 2D
  `EdgeTension` recipe from the bundled (2D) `cell_sorting.py`.

## 3. Mesh construction (Voronoi → TF) — `rnr/geometry.py`

| Python | C++ symbol | Notes |
|---|---|---|
| `tfv.Vertex.create(tf.FVector3(x,y,z))` | `Vertex::create(const FVector3&)` | standalone vertex from a position; also a `list[FVector3]` batch overload. |
| `stype(vertices=[vh, ...])` | `SurfaceType::operator()(const std::vector<VertexHandle>&)` | surface from **existing shared** vertex handles (the key to shared faces). Also a `positions=` overload that creates fresh vertices. |
| `btype([surf, ...])` | `BodyType::operator()(surfaces)` | body from surfaces. Passing one `SurfaceHandle` to **two** bodies sets `b1` then `b2` ⇒ shared internal face. |

Construction recipe (validated: body volumes match pyvoro to ~1e-5, space-filling):
global-dedup vertices by rounded position; dedup interior faces by the cell pair so
each becomes **one** surface added to **both** bodies; box-wall faces
(`adjacent_cell < 0`) become single-body surfaces = the cluster's free outer boundary.

## 4. Body / surface query API (used by metrics + future reconnection neighborhood walk)

| Python | C++ | Used for |
|---|---|---|
| `body.volume`, `body.area`, `body.centroid` | `getVolume/getArea/getCentroid` | stability checks, type split |
| `body.connected_bodies` (**property**) | `connectedBodies()` | neighbor pairs (share a surface) |
| `body.contact_area(other)` | `contactArea` | area-weighted metric |
| `body.find_interface(other)` | `findInterface` | the surface(s) between two bodies |
| `body.become(btype)` | `become` | reassign cell type (returns S_OK) |
| `body.type()` → `.name` | `type()` | type-name for het classification |
| `body.id` | mesh id | **stable** identity for dedup |
| `BodyType.instances` | live `BodyHandle`s | re-fetch each call (handles invalidate after mutation) |
| `surface.bodies` | `getBodies()` (b1,b2) | 2 ⇒ interior, 1 ⇒ free boundary |
| `SurfaceType.find_from_name(name).instances` | — | all surfaces of a type |

---

## 4b. Reconnection (I↔H / 3D T1) seams — Phase 1, VERIFIED & in use

The reconnection is built with **Strategy A — manual surface-list surgery** (mirrors
tvm's `I_H`/`H_I` pointer walk, reimplemented on the TF API; NOT a paste). Code:
`rnr/topology.py` (read-only neighbourhood walk), `rnr/conditions.py` (Condition-4
vetoes), `rnr/reconnect.py` (the mutation). Gated green by `rnr/tests/test_roundtrip.py`.

**Why not the high-level primitives:** `Vertex.split` makes only a 2-vertex edge,
`Vertex.merge` collapses two→one, `Vertex.replace_c(surface=)` collapses a face→vertex —
none expresses the edge↔triangle (point→triangular-face) step, so they can't do a clean
reversible I↔H. Manual surgery it is (user decision, 2026-05-30).

### Neighbourhood walk (read-only) — `topology.py`
| Python | Purpose |
|---|---|
| `Vertex.getBodies()` (→ ids) | the 4 cells at an edge endpoint; intersection = 3 side cells, set-difference = the 2 caps |
| `Vertex.shared_surfaces(other)` | the (≤3) surfaces bearing the short edge (filtered to those where the pair is **consecutive**) |
| `Vertex.connected_vertices` | outer vertices (ring-neighbours minus the edge partner) |
| `Surface.vertices` (ordered) + `topology.ring_neighbors` | adjacency within a face ring (since `neighbor_vertices` is unusable, see §0) |
| `Surface.getBodies()` | 2 ⇒ interior face; classify side vs cap faces |
| `Body.find_interface(other)` | the side↔cap faces (must be exactly 1 — also the 4(iii) check) |

The walk yields the canonical Okuda **[I]** = 5 cells / 9 faces / 6 outer vertices (or
**[H]** = same 5 cells + the new triangle). Validated on a real Kelvin interior edge
(`scripts/experiment_neighbourhood.py`, `scripts/check_topology.py`). Non-canonical
neighbourhoods (boundary edges, higher-valence vertices) return `None` → safely skipped,
never mutated.

### Condition-4 guards — `conditions.py`
| Python | Okuda condition |
|---|---|
| `Body.find_interface(b)` length ≥ 2 | 4(iii) extra rule: two cells share ≥2 faces (double trigonal face [β]) |
| `Surface.num_shared_contiguous_vertex_sets(other)` ≥ 2 | 4(ii): two faces share ≥2 separate edges (double edge [α]) |
| (caps `find_interface` must be empty before I→H; side-face/side-cell pairs before H→I) | the directional vetoes |

### Mutation primitives — `reconnect.py`
| Python | Used for | Maintenance (see §0) |
|---|---|---|
| `tfv.Vertex.create(FVector3)` | the 3 new triangle verts (I→H) / 2 new edge verts (H→I) | — |
| `Surface.replace(new_v, old_v)` | swap a vertex in a face ring | + `new_v.add(s)`, `old_v.remove(s)` |
| `Surface.insert(new_v, v1, v2)` | grow a face by inserting between two verts | + `new_v.add(s)` |
| `Surface.remove(v)` | shrink a face | + `v.remove(s)` |
| `stype(vertices=[...])` | create the new triangular face | attach with `s.add(b)` **and** `b.add(s)` for both caps |
| `Surface.add(body)` / `Body.add(surface)` | attach the triangle to the 2 caps | call BOTH directions |
| `VertexHandle.destroy()` / `SurfaceHandle.destroy()` | drop the orphaned edge verts / triangle | no-arg handle method, not the static |
| `Surface.position_changed()` / `Body.position_changed()` | recompute centroid/area/volume after surgery | (`update_internals` is `Body`-only, not on the handle) |

### Vertex placement (Okuda 2013 Appendix 1) — `reconnect.place_i_to_h` / `place_h_to_i`
- I→H Eqs. 46–56; H→I Eqs. 42–45. Computed in numpy from `Vertex.position`.
- H→I normal sign chosen from the outer-vertex clusters (robust to triangle winding)
  rather than a stored orientation flag.

### Reversibility result (the gate)
Topology restores **exactly** (vertex/surface counts; recovered edge is a valid [I] config
between the same 5 cells; caps separated again). Geometry restores within O(Δl_th): all
non-central vertices are byte-identical (never moved); recovered edge endpoints land
within Δl_th (**exactly** when original edge length = Δl_th — minimal config drift < 1e-6;
Kelvin drift ~0.013 with Δl_th = 0.707).

---

## 4c. Phase-2 operator seams (`rnr/operator.py`) — the per-step driver

The operator runs BETWEEN `tf.step()` calls (native port: inside `MeshQuality::doQuality()`).
No NEW TF API beyond §4/§4b — it composes the reconnection with these existing calls:

| Python | Used for |
|---|---|
| `Body.volume` | volume guard (reverse a reconnection that inverts a neighbourhood cell) + `mesh_health` |
| `Body.getSurfaces()` / `Surface.getBodies()` / `Surface.area` + `Body.type().name` | the energy gate's local Σ(λ_ij·A_ij) |
| `Body.id` (stable) | the 5-cell "claim" set for body-disjoint batching + the cooldown key |
| `tf.step()` | drive the loop |
| `Vertex.position` setter (`set_position`, `updateChildren=True`) | the **winding clamp** `stable_step` — pull an over-moved vertex back (refreshes its surfaces) |
| `Vertex.getBodies()` + `Body.position_changed()` | after clamping, re-fetch the cached `Body.volume` (tfBody.cpp recomputes it from surfaces; the vertex setter refreshes surfaces but NOT the body cache) |

**Winding clamp (`stable_step`, the stability guard):** wraps `tf.step()` with a per-vertex
displacement limiter — cap each vertex's per-step move at `rel_frac` (default 0.4) of its
nearest-neighbour distance so it can't cross a neighbour and flip a face winding (see §7
RESOLVED). Native port should NOT need this — use the force-level `abs(volume)` instead — so
it is the one operator seam the C++ port *drops* rather than translates.

Operator design choices (all in the module docstring; these are the C++-port knobs):
- **Trigger** = Condition 2 (`find_short_edges`/`find_small_triangles`, H→I uses the **max**
  triangle edge). **Δl_th must be < equilibrium edge** (Kelvin ≈0.707) or the whole pristine
  foam reconnects at once.
- **Handle re-fetch** = **body-disjoint batching**: disjoint 5-cell neighbourhoods share no
  vertex/surface handles, so a whole disjoint batch applies safely from one scan; rescan up
  to `max_passes`. (Cell count is invariant under I↔H, so the `bodies` list never goes stale —
  only local vertex/surface handles do.)
- **Anti-thrash** = placement hysteresis (`place_scale = Δl_th·(1+hysteresis)`) **+** per-site
  cooldown (lock a reconnected 5-cell set for N steps). Optional `p_transition` (Okuda Fig. 7
  ≈0.01).
- **Volume guard** = tentatively apply, and if any neighbourhood cell volume ≤ `vol_floor`,
  REVERSE via the inverse op (I↔H is exactly reversible). Prevents most blow-ups.
- **Energy gate** (`energy_gate=True`, DEPARTURE — see §5.5) = also reverse if the
  reconnection RAISED local adhesion energy. This is what actually drives sorting.

## 5. Departures from CLAUDE.md (flagged per working agreement)

1. **Boundary = finite cluster, NOT periodic.** The vertex solver has **no** periodic
   support (grep of `source/models/vertex`: zero PBC/wrap/boundary-condition code; the
   mesh generators lay down absolute coords and clamp neighbor links at the edge). The
   universe-level periodic BC acts on MD particles only and would tear cells straddling
   the boundary. Phase 0 uses a free-standing Voronoi cluster (box-clipped outer faces
   = free boundary), per the user's decision.
2. **Packing = pyvoro Voronoi** (user's choice), via `pyvoro-mmalahe`. Builds correctly
   given the `dispersion` fix above.
3. **MeshQuality disabled for the control.** Its only 3D ops are irreversible degenerate
   collapses, which remove small contacts and *falsely mimic* sorting. A pure jamming
   control must freeze topology. Reconnection (Okuda I↔H) will be the **only**
   topology-change operator in later phases.
4. **Condition-1 centroid convention (Phase 1: NOT exercised, no impact yet):** Okuda
   Eq. 3 specifies an edge-length-weighted average of edge midpoints; TF's `Surface`
   centroid uses a centroid triangulation. The reconnection computes Appendix-1 vertex
   placement itself (numpy, from raw `Vertex.position`), so it never relies on TF's
   surface centroid — the divergence does not affect the round-trip. Revisit if a future
   step (e.g. a Condition-1 face-center vertex) starts depending on `Surface.centroid`.
5. **(Phase 2) Energetics convention changed for stability.** Phase 0's control used
   homotypic λ=−1 / heterotypic λ=+1. Under the dense salt-and-pepper sorting load that
   **blows up** (negative tension grows faces unbounded; see §7). Phase 2 uses
   **non-negative tensions** (homo λ=0, het λ>0) + **dt≤0.001**. Still the σ_ij = Adhesion
   (λ·A) by body-type-pair decomposition; only the sign/magnitude + dt changed.
6. **(Phase 2) ENERGY GATE on the reconnection (`operator.energy_gate`).** Okuda's trigger
   is purely geometric; sorting is meant to emerge from long-time self-organisation. In a
   finite block over feasible runtimes that is too slow, and the geometric trigger fires at
   shrinking heterotypic faces whose new cap–cap contact is often heterotypic — so ungated
   reconnection can *raise* het area (~¾ of triggered reconnections are energetically
   uphill). The energy gate reverses uphill reconnections (greedy/Metropolis-at-T=0). It is
   what produces the het-pair drop; a flagged modelling departure (the C++ port can keep it
   optional).

## 5b. Rendering / visualization seams (headless screenshot path)

The vertex `MeshRenderer` (`source/models/vertex/solver/tfMeshRenderer.cpp`) is the only
renderer for the mesh. `render_meshFacesEdges()` reads each surface's colour FRESH every
draw as `s->style ? s->style : s->type()->style` (per-surface `style` is `NULL` after
creation — ctor inits `style{NULL}` — so colour comes from the **SurfaceType's** style).
Working recipe lives in `rnr/scripts/sorting_demo_start.py` (block + `clip` modes) and
`rnr/scripts/kelvin_single_cell.py` (one isolated Kelvin cell, hexagons visible).

Re-investigated 2026-05-29 with controlled, separate-process, **pixel-counted** A/B tests
(the screenshot read channel is flaky — always verify by counting PNG pixels, never by eye):

| Claim | Verdict | Detail |
|---|---|---|
| **`Body.destroy()` hides a cell from the render** | ❌ **FALSE — the core gotcha** | `Body::destroy → Mesh::remove(Body*)` only does `s->remove(b)` (detaches the body pointer); it does **not** destroy surfaces. The renderer draws **every** live surface in the mesh, so orphaned surfaces still render. Destroyed 188/189 bodies → surface count UNCHANGED (890 quad + 216 hex). **Fix:** also destroy orphaned surfaces (`for s in stype.instances: if len(s.getBodies())==0: s.destroy()`), or build only the surfaces you want. |
| `surface.become(otherType)` updates the rendered colour | ❌ FALSE (real limitation) | After `become`, C++ `typeId`/`type()`/`type().style.color` are all correct and `s.style is None`, yet the render keeps the creation-time colour across repeated screenshots + `position_changed()`. The data is right; the rendered colour is stale. **Set colour at surface creation.** |
| "A 2nd BodyType clobbers surface colours" | ❌ DEBUNKED | One vs two BodyTypes → pixel-identical multi-colour renders. No effect. |
| "Camera move after build resets colours" | ❌ DEBUNKED | camera-before vs camera-after build → pixel-identical; a rotate between two screenshots DOES change the image (renderer re-renders each shot). Camera ordering has no colour effect. The original "all-blue after tilt" was the destroy-orphan bug rendering the full block (all-square hull). |

Useful headless rendering API: `tf.system.decorate_scene(False)` (hide box wireframe + grid),
`camera_view_front()`, `camera_rotate_by_euler_angle(FVector3)`, `camera_zoom_by(delta)`
(positive = zoom in; `camera_zoom` takes NO args — it's the scroll handler; `camera_zoom_to`
overshoots after a rotate), `clip_planes=[(point, normal)]` passed to `tf.init` (renderer Patch A).

**Possible native-port renderer patches (investigated, NOT applied — see memory
[[vertex-destroy-orphans-surfaces]] / [[vertex-render-color-gotchas]]):** (a) have
`MeshRenderer::draw` skip surfaces with no live body so `Body::destroy` visually hides cells;
(b) the `become`-stale-colour is a render-refresh issue (data is already correct) and would need
deeper renderer/SWIG debugging than this pass covered — the subagent's "add per-surface colour
override" suggestion is unverified speculation. For the prototype, handle both in Python.

## 6. C++ port seams (Phase 3, light/ongoing)

**BUILD GOTCHA (verified Phase A, 2026-05-30):** CMake/Ninja's SWIG step only tracks the
top-level `wraps/python/tissue_forge.i` as a dependency, NOT the transitively-`%include`d
sub-`.i` files (`tfMeshQuality.i`, ...) or the `%include`d C++ headers. So editing a sub-`.i`
(new SWIG properties) or a header SWIG reads (new getters/setters) builds + relinks but
**silently skips regenerating the Python wrapper** — the new symbols never reach Python though
the build "succeeds." Fix: before `pixi run build-tf`, force a regen with
`rm -f tissue-forge_build/wraps/python/CMakeFiles/TissueForge_py.dir/tissue_forgePYTHON_wrap.cxx`,
then build; verify with `grep -c <NewSymbol> <that .cxx>` (>0) and a Python `hasattr`. Pure C++
changes don't need this. (`FloatP_t` is float32: a Python-set 0.1 reads back 0.10000000149.)

The native op should subclass `MeshQualityOperation` and run **inside**
`MeshQuality::doQuality()` (the prototype runs between `tf.step()` calls instead — a
valid prototype, not the final integration). Keep the prototype split into a
**check/predicate half** and an **implement/mutate half**, mirroring the existing
`tfMeshQuality.cpp` ops, so the port is a translation. Every Python call in §1–§4 maps
to the C++ symbol in its row.

**Highest-value native task (Phase-2 carryover, see §7): a volume-sign / winding repair.**
The Python prototype's one unfixable failure mode is a TF signed-volume winding sign-flip
that reverses the VolumeConstraint force into a runaway. Mitigated in Python only by a
small dt; the proper fix is native — make `Body::getVolume()` / the VolumeConstraint robust
to a transient face-winding flip (clamp to `abs`, or a one-sided `V>0` force), and ensure
the reconnection surgery writes consistent surface winding. This unblocks the Okuda-pure
infinitesimal-feature trigger (drops the energy-gate departure of §5).

## 6b. Native port progress — Phase B (DONE, 2026-05-30)

The C++ port lives in the fork `tissue-forge/source/models/vertex/solver/tfMeshQuality.{h,cpp}`
(+ the SWIG wrap `wraps/python/.../tfMeshQuality.i`), branch `feat/native-rnr-reconnection`.

- **Phase A (scaffolding):** `ReconnectionOperation` stub + `MeshQuality` knobs
  `reconnectLength`/`reconnectHysteresis`/`reconnectEnergyGate` (getters/setters/IO/SWIG), a no-op
  reconnection pass wired into `doQuality`. Gate: build clean, 19 tests green, knobs settable.
- **Phase B (this section):** ported `topology.py` + `conditions.py` into an anonymous-namespace
  block in `tfMeshQuality.cpp` — the read-only neighborhood walk (`rnr_iNeighbourhood` /
  `rnr_hNeighbourhood`), Condition-4 vetoes (`rnr_iToHVeto` / `rnr_hToIVeto` + the
  `rnr_cellsShareMultipleFaces` / `rnr_facesShareMultipleEdges` primitives), and scanners
  (`rnr_findShortEdges` / `rnr_findSmallTriangles`). `ReconnectionOperation` now stores the
  candidate by id with a real `check()` (re-walk + re-veto) / `prep()` (re-fetch + gather
  affected bodies) / `targets` (every touched surface, for the dependency graph). `implement()`
  is still a no-op — the Appendix-1 placement + surface surgery is Phase C.

  **The C++ helper ↔ Python oracle map** (1:1; the C++ uses TF pointer handles directly):
  | Python (`rnr/`) | C++ (`tfMeshQuality.cpp`, anon ns) |
  |---|---|
  | `topology.i_neighbourhood` | `rnr_iNeighbourhood` |
  | `topology.h_neighbourhood` | `rnr_hNeighbourhood` |
  | `topology.is_consecutive` / `other_neighbor` | `rnr_isConsecutive` / `rnr_otherNeighbor` (via `Surface::neighborVertices`) |
  | `topology.find_short_edges` / `find_small_triangles` | `rnr_findShortEdges` / `rnr_findSmallTriangles` |
  | `conditions.i_to_h_veto` / `h_to_i_veto` | `rnr_iToHVeto` / `rnr_hToIVeto` |
  | `conditions.cells_share_multiple_faces` | `rnr_cellsShareMultipleFaces` (`Body::findInterface().size()>=2`) |
  | `conditions.faces_share_multiple_edges` | `rnr_facesShareMultipleEdges` (`Surface::numSharedContiguousVertexSets()>=2`) |

  **Test seam (the Phase-B gate):** three read-only diagnostic methods on `MeshQuality`, JSON in /
  Python `dict`/`list` out, wrapped in `.i` as `analyze_i_reconnection(v10,v11)` /
  `analyze_h_reconnection(tri)` / `find_reconnection_candidates()`. They read the global
  `Mesh::get()`, so a detached `tfv.Quality()` instance suffices (the test mesh keeps its own
  `quality=None`). `find_reconnection_candidates` uses the SAME scanners `doQuality` does.
  Gate test `rnr/tests/test_native_reconnection.py` (6 tests) cross-checks the native walk +
  vetoes edge-for-edge against the Python oracle on the hand-built minimal config and a real
  Kelvin block; `pixi run test` = 25 green.

- **CONCURRENCY (verified): build the reconnection dependency chains SERIALLY.** The stock
  `MeshQuality_constructChains` builds chains with `parallel_for` + `appendNext`, whose loop-check
  reads/writes the shared `prev`/`next` graph without fully locking — fine for the sparse stock
  passes, but the reconnection pass produces a *dense* graph (each candidate touches ~9 surfaces,
  hundreds of candidates with heavy overlap on a Kelvin block) that races and can fabricate a
  cycle → unbounded `MeshQuality_upstreams` recursion → stack-overflow segfault. The reconnection
  pass therefore calls `MeshQualityOperation_checkChain` in a serial loop (valid DAG, no race);
  the subsequent parallel `MeshQuality_doOperations` walk is safe on a valid DAG.

- **STOCK-PASS HAZARD for Phase D (verified, NOT our bug):** running the *stock* `MeshQuality`
  (its degenerate 3D collapses — Surface/Body/EdgeDemote) on a **finite Kelvin block** segfaults
  after enough repeated `doQuality` calls **even with `reconnectLength=0`** (reconnection pass
  off) — this is exactly why `conftest`/CLAUDE.md keep `mesh.quality=None`. Our reconnection pass
  in *isolation* (stock thresholds zeroed, `collision_2d=False`, no forces → no vertex split) is
  stable + inert over 40 calls × 5 processes (verts/surfs unchanged). **Phase D must run the
  reconnection pass WITHOUT the stock collapses** (e.g. an exclude/skip path, or only fire stock
  ops on demand) — or first fix the stock collapses on finite blocks — before a live sorting loop.

- **Phase C (DONE, 2026-05-30):** ported the mutation half into
  `ReconnectionOperation::implement()` for both directions. The placement helpers are native
  `FVector3` translations of `reconnect.place_i_to_h` (Okuda Appendix-1 Eqs. 46-56) and
  `place_h_to_i` (Eqs. 42-45, normal oriented by the cap-side outer-vertex cluster). The surgery
  uses Strategy A manual surface-list edits (`Surface::replace/insert/remove` + mirrored
  `Vertex::add/remove`, `Surface::add/remove(Body)` + mirrored `Body::add/remove`) under
  `MeshSolver::engineLock()`. `numNewVertices/numNewSurfaces` are exact: I->H = 3/+1, H->I =
  2/+0. Source check: member `Surface::destroy()` does **not** cascade-delete the now-orphaned
  triangle vertices after H->I, so the implementation destroys those three vertices explicitly
  after deleting the triangle. The new triangle reuses the trigger side surface's `SurfaceType`.

  **Phase-C test seam:** added `MeshQuality::forceReconnectIToH(v10Id, v11Id)` and
  `forceReconnectHToI(triId)` (SWIG-wrapped as Python dict helpers). They build one
  `ReconnectionOperation`, run `prep()+check()`, reserve the exact new object counts, and call
  the real native `implement()` while bypassing the scan and all stock quality passes.

  **Gate:** `rnr/tests/test_native_roundtrip.py` mirrors the Python round-trip gate on the
  hand-built minimal [I] config and on a Kelvin block, plus the caps-touch veto/no-mutation path.
  `pixi run build-tf` clean; `pixi run test` = **28 passed** (25 existing + 3 native round-trip).
  Behavior was checked against the Python oracle and the Okuda equations; no GPL reference code was
  copied.

- **Phase D (DONE for wiring; faithful run BLOCKED BY Phase E, 2026-05-30):** the native
  reconnection pass now runs as the active geometric pass **inside the live quality loop**
  (`MeshSolver::postStepStart` → `Mesh::getQuality().doQuality()`, verified tfMeshSolver.cpp:541),
  i.e. every `tf.step()` fires it when `mesh.quality` is set — not only via the `forceReconnect*`
  debug entry points. Changes (branch `feat/native-rnr-reconnection`):
  - **`stockQualityOps` knob** on `MeshQuality` (Python `mesh.quality.stock_quality_operations`,
    getter/setter/IO/SWIG; default **True** preserves stock behavior). When **False**, `doQuality`
    skips the legacy vertex/surface/body/collision passes and runs ONLY the reconnection pass. This
    is the isolation the STOCK-PASS HAZARD note (above) demanded: the stock degenerate-collapse
    passes segfault on finite Kelvin blocks, so the Phase-D live harness sets it False to exercise
    native RNR alone. (Smallest clean API; the alternative of zeroing every stock threshold is
    leakier.)
  - **`enforceTrigger`** on `ReconnectionOperation` (default true; the scanner-built live ops use
    it, `forceReconnect*` pass false): `check()` re-validates Okuda **Condition 2** at mutate time —
    I→H only if the interior edge `cfg.length < reconnectLength`, H→I only if the triangle
    `cfg.maxEdge < reconnectLength`. `reconnectLength == 0` ⇒ the scanner returns no candidates ⇒
    pass disabled. `reconnectEnergyGate` stays **False** for the faithful path (geometric trigger
    only — NOT the greedy gate that demixed the Python prototype).
  - Concurrency unchanged: chains still built **serially** (the dense-graph parallel race is NOT
    reintroduced). Check/mutate split intact: scanners + `check()` decide, `implement()` only mutates.
  - **Ownership double-free fix (`wraps/.../tfMesh.i`).** The Phase-D doQuality tests are the first
    to *attach* a Python-created `tfv.Quality()` to the mesh (`mesh.quality = q`). `Mesh::setQuality`
    **takes ownership** (it `delete`s the pointer on the next set / on `= None` / on mesh dtor;
    tfMesh.cpp:116), but the SWIG proxy kept `thisown=1`, so teardown freed the same `MeshQuality`
    twice → `abort()` in `~MeshQuality` (`POINTER_BEING_FREED_WAS_NOT_ALLOCATED`; caught via lldb).
    The `quality` setter now sets `_quality.thisown = 0` on transfer so the C++ Mesh is the sole
    owner. (Latent in stock TF; Phase C dodged it by using a *detached* Quality with `forceReconnect`.)

  **Test seam (the Phase-D gate, `rnr/tests/test_native_doquality.py`, 3 tests):** drive the LIVE
  `do_quality()` scheduler path (scan → `check`/`prep` → `implement`), stock collapses disabled:
  (1) one I→H on the minimal [I] config (+1 vertex / +1 surface, new cap-cap triangle is a valid
  [H]); (2) `reconnect_length = 0` ⇒ no-op; (3) Kelvin-block smoke with stock ops off ⇒ no segfault,
  one mutation, all body volumes stay positive immediately after. `pixi run build-tf` clean;
  `pixi run test` = **31 passed** (28 + 3 native doQuality).

  **Stability gate (plan Phase D): `pixi run check-clamp-native 4500 0.0001`** — the native
  equivalent of `check-clamp 4500 0 0.0001` (new `rnr/scripts/check_clamp_native.py`: same block /
  energetics / dt, but the topology op is the in-step native MeshQuality reconnection — Python
  operator OFF, winding clamp OFF, stock ops OFF, energy gate OFF). **RESULT: worst min_vol =
  −29.512, first non-positive @ step ~2100 → UNSTABLE.** min_vol holds (4.0 → 3.62) through step
  2000, then a cell inverts (the **dynamic winding sign-flip**, memory
  `faithful-instability-is-winding-signflip`). This is *milder* than the Python-operator control
  (−218) — there is no runaway Python-op overshoot — but it still inverts. **Conclusion: the native
  reconnection wiring is complete and correct (all build/unit gates green), but the faithful
  in-step sorting run is BLOCKED BY Phase E** (robust/abs signed volume in `VolumeConstraint` /
  orientation-consistent `Body` volume). Phase E was NOT started (explicit scope guard). D stayed
  ≈ −0.04…−0.056 over the run (geometric-trigger-alone sorts weakly, as expected; the energy gate /
  Phase-G σ·A drive is what demixes — do not flip the energy gate on to "make it sort/stable").

## 6c. Phase E (ATTEMPTED 2026-05-30) — volume robustness is NOT the fix; instability RE-DIAGNOSED
**→ RESOLVED 2026-05-31, see §6e: the §6c re-diagnosis was right; the §6d oracle recipe was implemented and the native gate is now STABLE (no volume code touched).**

Phase E was scoped (`docs/native_volume_fix_plan.md`) as a *robust-volume backstop* for a
presumed "winding sign-flip": make the signed cell volume robust so the `VolumeConstraint` force
can't run away, native-analogue of the Python `stable_step` clamp. **That premise is wrong.** A
careful trigger investigation (the diagnostic scripts `rnr/scripts/diag_inflation.py`,
`diag_fling.py`, `diag_centered.py`) showed the instability is **NOT a volume/winding problem at
all** — it is **two independent per-vertex DISPLACEMENT OVERSHOOTS**; the negative/huge volume is a
downstream *symptom*. The fork engine was reverted to the clean Phase-D state (commit `6d67617`);
**no engine change shipped.** Details:

**What was tried (and why each is not the fix):**
- **Approach B — force-level FLOOR/ABS** in `tfVolumeConstraint.cpp` (`Veff = max(V,1e-3·constr)` /
  `Veff = |V|`). MEASURED on `check-clamp-native 4500 0.0001` (worst signed min_vol): un-patched
  **−29.5**; FLOOR **−536**; ABS **−118338** — *all UNSTABLE, ABS far worse*. Two reasons it can't
  work: (1) the gate reads the **signed `Body.volume`** (`mesh_health` → `b.volume`), which a
  force-only patch never changes; (2) ABS flips only the prefactor `(constr−|V|)` while `ftotal`
  keeps the stored sign → an inconsistent force → a *new* runaway.
- **Approach A — orientation-consistent volume** (`Body::positionChanged`/`updateInternals`:
  `volume = (1/6)Σ|(c_s−c_b)·N|`, positive-definite) **+** the force gradient oriented by the same
  geometric `sign((c_s−c_b)·N)` instead of `volumeSense`. This makes the gate *literally* print
  `STABLE, worst min_vol = +0.302` and a single-eversion micro-test recover — **but it's a
  MISLEADING pass**: the gate's min-only metric is fooled when *every* cell inflates together. The
  run actually blows up to `min_vol = 1779` (corner) / `max_vol = 30972` (centred): the eversion
  runaway was merely converted into an **inflation** runaway. 33 tests passed but the mesh is not
  stable. (The robust-positive volume + orientation repair *is* a legitimate stabilizer — the
  oracle has it, see §6d #3 — but it is **insufficient** for the two real instabilities below.)

**The REAL diagnosis — two displacement overshoots (evidence, patched-engine runs; reconnection
toggled; cluster corner `[0,10]³` vs centred `[25,35]³` in the `[0,60]³` periodic universe):**

| config | reconnect | outcome |
|---|---|---|
| corner | OFF | **fling @ step ~2090** — a boundary vertex on x≈0 drifts across 0 and teleports **exactly +60** (the box dim) to x≈60 → cell becomes a box-spanning sheet (bbox `[59.6,59.5,2.1]`) |
| centred | OFF | **fully stable** 4500 steps (max single-step disp ≈ 2e-4) |
| centred | ON | **reconnection-storm runaway** from ~step 2200: a reconnection nearly every step, each causing **5–7-unit** single-step overshoots → new short edges → cascade → inflate to 30972 |

- **Instability A — periodic-image WRAP.** The gate builds the cluster in the **corner** of a
  **periodic** universe; a vertex sitting on x=0 wraps to x≈60. `diag_fling.py` caught the exact
  +60.000 single-step jump with reconnection OFF (dnv=0). Reconnection-independent; **centring (or
  non-periodic BC) removes it entirely.** Root reason it's fatal for us but not the oracle:
  TF's vertex *mesh* has **no minimum-image awareness** (PORTING_NOTES §5.1), so a wrapped vertex
  tears the cell; the oracle computes all geometry/forces with minimum-image, so a wrap is a
  non-event (§6d #1).
- **Instability B — post-reconnection OVERSHOOT STORM.** Independent of A (survives centring). The
  faithful (energy-gate-OFF) geometric trigger fires *many* reconnections; each leaves the local
  geometry out of equilibrium and at dt=1e-4 the relaxation force overshoots a vertex 5–7 cell-
  widths, creating new short edges → more reconnections → self-sustaining storm. The newly-placed
  vertices are themselves fine (offset < 2.3, **not** an Appendix-1 placement bug). This is exactly
  the "post-reconnection relaxation overshoots at coarse dt" that `conftest.py` already documents in
  the Python prototype — pre-existing, not introduced here.

**Why the prior memory was a MISDIAGNOSIS.** `faithful-instability-is-winding-signflip` /
`docs/native_volume_fix_plan.md` called this a "dt-overshoot winding sign-flip" curable by
abs-volume. The winding flip / negative volume is the **symptom** of a vertex overshooting or
wrapping; the **cause** is the displacement overshoot. The Python `stable_step` clamp "worked" only
because it caps per-step displacement — incidentally catching *both* the 60-unit wrap and the
5–7-unit post-reconnection overshoots.

**Recommended fix (NOT YET IMPLEMENTED — for the next session, see §6d for the oracle recipe):**
- **A:** test-config fix — centre the cluster and/or use non-periodic BC so no vertex ever wraps
  (no engine change; centring alone made centred+reconnect-OFF fully stable above).
- **B:** adopt the 3DVertVor reconnection regime — a **much smaller `reconnect_length`** (oracle
  uses `Lth=1e-3`; ours is `0.45` ≈ 0.64× the equilibrium edge ⇒ ~600× too aggressive) **and
  throttle reconnection to every-N-steps** (oracle `dtr=10·dt`; ours runs every `doQuality`/step).
  Trade-off: smaller `Lth` ⇒ stable but slower sorting (closer to the paper's quasi-static regime).
  The every-N-steps throttle likely wants a small native `MeshQuality` interval knob.

## 6d. Oracle comparison — how 3DVertVor stays stable where we don't (verified against the GPL code; read-only, nothing copied)

`3DVertVor` (Lawson-Keister/Zhang/Fagotto/Manning 2024, the paper `reference_pdfs/Manning2024PLOS…pdf`)
runs stable periodic-bulk dynamics with T1s. Its main loop is **explicit overdamped Euler**
(`Run/Run.cpp:1252` `v=μ·F`, `:1345` `x+=v·dt`) — the **same integrator as TF's `tf.step()`**, so
the scheme is not the difference. Four stabilizers it has and TF lacks (file:line in `3DVertVor/`):

1. **Minimum-image periodicity in the mesh.** Distances/forces/centroids are all wrapped to the
   nearest image (`Energy/Volume.cpp:99-116`, `Energy/Interface.cpp:68-85`, `Polygon/Polygon.cpp:87-104`),
   and vertices are wrapped into `[0,L)` each step (`Run/Run.cpp:1516`). ⇒ a vertex crossing a
   periodic face is computed correctly *across* the boundary; no box-spanning sheet. **TF's vertex
   mesh has none of this** ⇒ our instability A.
2. **Throttled + tiny-threshold reconnection.** Reconnection checked only every `dtr_=10·dt`
   (`tvm.cpp:302`, gated at `Run/Run.cpp:1043`), threshold `Lth_=1.0e-3` (`Reconnection/Reconnection.cpp:23`),
   new vertices placed `±Lth≈0.001` from the collapse (`Reconnection.cpp:318,626`). ⇒ reconnections
   are rare and each is a *negligible* perturbation. **We reconnect every step at Lth=0.45 with
   0.45-scale placement** ⇒ our instability B (the storm).
3. **Orientation repair.** `Cell/Cell.cpp:216-221`: `if(volume_<0){ volume_=fabs(volume_); flip all
   polygonDirections_; }`, the flipped orientation used consistently in volume *and* force. This is
   the abs+flip our Approach A approximated — it cures the eversion mode but, alone, not A or B.
4. **Centre-of-mass drift removal** each step (`Run/Run.cpp:1264-1278`) — minor.

Energy functional (confirms the Phase-G plan): `E = Σ k_v(V−V0)² + Σ tension·area` — interface
tension is **area-based** (`Energy/Interface.cpp`), `dt≈0.01`, `μ=1`, periodic bulk.

## 6e. Phase F (DONE 2026-05-31) — the two REAL fixes; native RNR gate is STABLE

The §6c re-diagnosis was correct (two displacement overshoots, NOT a volume problem) and §6d named
the oracle recipe. Both fixes are now implemented and `pixi run check-clamp-native` PASSES with
volumes near target. **No volume/winding code was touched** — the reverted Phase-E approaches stay
reverted (do NOT re-add them). Fork commit on `feat/native-rnr-reconnection`. Behaviour was checked
against the §6d oracle recipe and the Python prototype; no GPL code copied.

**Gate hardening (done first — the old gate was foolable).** `check_clamp_native.py` checked only
`min_vol > 0`, which a UNIFORM-INFLATION runaway passes (all cells grow together so the minimum
stays positive — exactly how the reverted Phase-E Approach A printed a misleading "STABLE +0.302"
while inflating to 30972). The gate now ALSO tracks `max_vol` + the cluster bounding box and FAILS
if `max_vol > 3·V0` or any vertex leaves the box. Genuine PASS = min_vol > 0 AND max_vol near
baseline AND no fling. (Healthy baseline `max_vol = 7.438` is the boundary-clipped Voronoi cells,
NOT V0=4 — interior Kelvin cells are 4; 3·V0=12 sits above the baseline and far below any runaway.)

**Fix A — centre the cluster (test-config, no engine change).** Both gate harnesses
(`check_clamp_native.py`, `check_clamp_stability.py`) build the cluster at `box=[25,35]³`
(`OFFSET=25`, both expose it as an arg) inside the periodic `[0,60]³` universe, far from every
boundary, so no vertex wraps — removing instability A (the +60 periodic-image teleport) entirely.
(The Python twin `check-clamp` runs with `mesh.quality=None` — its Python greedy-energy-gate
operator + `stable_step` clamp, NOT the native pass — so the native work cannot affect it. That
path is stochastically MARGINAL in BOTH boxes: worst min_vol fluctuates ~1–2 and one centred run
dipped to −0.404, a pre-existing property of the greedy gate amplifying integrator FP jitter, not a
centring regression. The robust deliverable is the NATIVE gate below; the Python twin is the
legacy/oracle path being retired by the port.)

**Fix B — the oracle reconnection regime (config sweep + one native knob).**
- *Config sweep alone is insufficient.* Centred, dt=1e-4, 4500 steps: edges that collapse below the
  trigger only reach ~0.3–0.45 (NOT ~1e-3 — that figure was a longer/quasi-static regime), so
  `Lth≤0.2` ⇒ ZERO reconnections (D frozen at the no-RNR −0.0431); `Lth=0.45` every-step ⇒ the
  overshoot STORM (min_vol −2.7). No single `reconnect_length` both stabilises AND reconnects.
- *New native knob `reconnect_interval`* on `MeshQuality` (= the oracle's `dtr`). The reconnection
  pass runs only every Nth `doQuality()` call — a `reconnectCounter` member gates it
  (`counter % interval == 0`), default 1 = every step. Getter/setter/IO/SWIG;
  `mesh.quality.reconnect_interval`. The I↔H surgery/placement was NOT changed. Throttling to N=10
  (oracle `dtr=10·dt`) lets the mesh relax between reconnections and TAMES the storm: `Lth=0.45`
  goes from min_vol −2.7 (INT=1) to STABLE (INT≥5).

**Stability-vs-sorting curve** (centred, 4500 steps, energy gate OFF, hardened gate; D drifts from
the no-RNR −0.0431 as reconnections fire):

| Lth  | INT | worst_min_vol            | reconnects? (final D, ~events)   | gate |
|------|-----|--------------------------|----------------------------------|------|
| 0.45 | 1   | **−2.7** (storm)         | yes, then blows up               | FAIL |
| 0.2  | 1   | ~3.5                     | **NO** (D frozen −0.043, 0 ev)   | pass* |
| 0.4  | 1   | 1.59                     | yes (−0.076, 35 ev)              | PASS |
| 0.45 | 5/10/20 | 1.60 / 1.74 / 2.08   | yes (~−0.065, ~44 ev)            | PASS (INT=10 MARGINAL: a run dipped 0.37) |
| 0.4  | 10  | **~2.2** (8000-step: 1.36) | yes (−0.056…−0.065, ~34 ev)    | **PASS ← DEFAULT** |
| 0.4  | 20  | ~2.85                    | yes (~−0.06, ~31 ev)             | PASS |

  `*` `Lth≤0.2` "passes" only by doing nothing. **DEFAULT = `Lth=0.4, INTERVAL=10`**: the largest
  *robustly*-stable `Lth` at the oracle `dtr=10` (worst_min ~2.2 across 3 runs = >50% V0;
  `max_vol`=7.438 baseline, zero inflation; durable to 8000 steps). Confirmed trade-off: smaller Lth
  / larger INTERVAL ⇒ more margin, slower sorting. Energy gate stays OFF ⇒ D ≈ −0.06 (the faithful
  geometric-trigger depth, expected — do NOT flip the energy gate on to deepen it).

**Scheduler RACE fix (uncovered during verification — a latent Phase-C/D bug, not from Fix A/B).**
The native reconnection pass ran its operations' `implement()` in PARALLEL
(`MeshQuality_doOperations`' `parallel_for` over `op_heads`). A `ReconnectionOperation`'s `targets`
(hence the dependency graph) cover only its **9 incident surfaces**, NOT the 6 outer vertices / 5
bodies the I↔H surgery also mutates — so two reconnections with disjoint *surface* targets but a
shared outer vertex/body could `implement()` concurrently and race on that shared object →
intermittent heap-corruption **segfault** (~1 per dozen full gate runs) AND nondeterministic small
eversions (a deterministic sim varying run-to-run is the race tell). Fix: a serial executor
`MeshQuality_doOperationsReconnectionSerial` runs THIS pass single-threaded (it is throttled +
sparse, so serial cost is nil). Extends the same serialization rationale chain construction already
uses (§6b CONCURRENCY). After the fix: no segfault in 15+ runs, and the run-to-run min_vol spread
tightened. The C++-port note: widening `targets` to include vertices/bodies would let the parallel
scheduler handle it, but that changes the validated walk; serial is the conservative choice.

## 6f. Faithful 3DVertVor/Manning reproduction — the model maps onto EXISTING actors; the gap is NOISE (2026-06-01)

Triggered by "do we need to rework heterotypic adhesion to match the PDFs/3DVertVor?". Read both
`reference_pdfs/` PDFs + `3DVertVor/Energy/{Interface,Volume}.cpp` + `Run/Run.cpp`. Conclusion: the
het-adhesion FORCE does NOT need reworking — TF already applies it. See memory
[[adhesion-force-is-already-area-tension]].

**The paper/oracle energy** (Manning Eq. 3, p.5; 3DVertVor `Interface.cpp:updateEnergy/updateTension`
+ `Volume.cpp`):

    E = Σ_cells [ K_A (A_i − A0)² + K_V (V_i − V0)² ]  +  Σ_{i≠j type} σ_ij · A_ij

σ_ij·A_ij is **area-based** (the shared interface polygon's AREA, NOT edge length — Manning p.5
explicitly), homotypic σ=0. Dynamics: overdamped `dr/dt = μ∇E + η`, μ=1, dt=0.01, white noise
kT=0.1, shape index s0=A0/V0^(2/3)=5.6 (fluid). `Interface.cpp` builds each polygon tension as
`2(s_cell−s0)+σ_ij` (the SurfaceAreaConstraint part + the σ part) and applies `force = tension·∇A`.

**Maps 1:1 onto EXISTING TF actors** (all verified against fork source):
| paper/oracle term | TF actor | faithful? |
|---|---|---|
| `K_V(V−V0)²` | `VolumeConstraint` | ✅ energy AND force |
| `K_A(A−A0)²` (cell surface elasticity) | `SurfaceAreaConstraint` | ✅ energy AND force (`ftotal_loop=−2∇A`, so force `=−∇[λ(A−constr)²]` exactly) |
| `σ_ij·A_ij` (het, area-based) | **body-`Adhesion`** | ✅ **FORCE** `=−∇(λ·A_het)`; ❌ `energy()` is perimeter but the integrator never calls it |

So `[VolumeConstraint + SurfaceAreaConstraint + body-Adhesion]` forces sum to EXACTLY
`−∇[K_V(V−V0)² + K_A(A−A0)² + σ·A_het]` = the 3DVertVor force. A new σ·A actor (kickoff Phase G)
is **force-redundant** with `Adhesion(λ=σ)`; worth building ONLY for a consistent `energy()` (needed
just for energy reporting / an energy-gated reconnection variant — and the paper's reconnection is
purely geometric, so not for faithfulness). **Deferred.**

**The real missing physics = the STOCHASTIC drive.** Our athermal runs sort only weakly (D plateaus
≈−0.07 then a cell inverts ~step 12k). The paper is Brownian (kT=0.1); the 3DVertVor checkout is an
active-motility fork (the thermal line `Run.cpp:1344` is commented out for a `motility_` line :1345).
Either way our runs had ZERO noise.

**Noise via `tf.Force.random` (verified appropriate).** `Force.random(std, mean, duration)` =
`Gaussian` force: isotropic random direction, magnitude ~N(mean,std), held `ceil(duration/dt)` steps
then resampled (`tfForce.cpp:211-243`, resample gated by `INTEGRATOR_UPDATE_PERSISTENTFORCE`, set
each force-eval). It REACHES vertices: `MeshSolver::preStepJoin` does `p->f += meshforce` into the
same particle buffer the engine integrates, so a force bound to `MeshParticleType`
(`_low._vertex_solver__MeshParticleType_get()`) sums with the mesh forces. Config for faithful WHITE
noise: **`duration = dt`** (resample every step ⇒ delta-correlated) + **`mean = 0`**. CALIBRATION
(measured on free vertices, this build): mobility **μ = 1.000** (= the paper's μ exactly), and
**D = μ·kT, D ≈ 1.63e-5·std²** ⇒ kT_eff = 1.63e-5·std². Caveat: TF draws a Gaussian *magnitude* ×
isotropic *direction*, not 3 per-axis Gaussians — same effective temperature / long-time diffusion,
slightly different per-kick shape (exact per-axis needs a `CustomForce`). `duration≫dt` ⇒ colored /
active-motility noise (the fork's regime), available via `mean>0`.

**Faithful harness:** `rnr/scripts/sort_faithful_3dvertvor.py` (`pixi run sort-faithful-3dv`): existing
actors at s0=5.6, σ=Adhesion-λ_het, native geometric reconnection (gate OFF), centred, hardened
stability, + `tf.Force.random` noise (KT arg → std). `KT=0` = athermal control. Headline test: does
noise drive deeper demixing? (Departures from paper: finite cluster not periodic; dt=1e-4 not 0.01;
σ/K_A regime — ours σ=1,K_A=0.1 gives σ/K_A=10, vs the paper's σ/K_A∈[0.04,1].)

**RESULT (2026-06-01) — noise does NOT unlock sorting; it destabilizes/randomizes. The athermal
strong-σ run is the ceiling.** See memory [[thermal-noise-destabilizes-reconnection]]. Full sweep:
| regime | stable? | D plateau | het-area | reconnects |
| σ=1   kT=0           | ✅ | **−0.065** | **0.375** | 2844 |  ← best
| σ=1   kT=0.1         | ❌ | −0.031 | 0.429 | 4185 |  noise → more churn → cell inverts, shallower D
| σ=0.1 kT=0          | ✅ | −0.043 (FROZEN) | 0.462 | **0** |  weak σ never triggers reconnection
| σ=0.1 kT=0.05 INT=10 | ❌ | −0.034 | 0.468 | 3376 |  noise unfreezes but reconnections are RANDOM (het↑)
| σ=0.1 kT=0.05 INT=30 | ✅ | −0.057 | 0.479 | 1616 |  throttle stabilises, but het flat = marginal

Diagnosis: the binding constraint is **reconnection violence/robustness**, not tension or noise. Our
reconnections displace vertices ~Lth·(1+hyst) ≈ 0.48 (≈70% of the 0.707 edge) PER event; the oracle
uses Lth=1e-3 (negligible). Root chain: finite cluster + dt=1e-4 ⇒ edges only collapse to ~0.3–0.45
(§6e) ⇒ must use big Lth ⇒ violent reconnections ⇒ can't absorb noise's extra churn ⇒ cells invert.
The oracle avoids this via (1) periodic minimum-image mesh (§6d#1), (2) quasi-static near-collapse
+ tiny-Lth gentle reconnection (§6d#2), (3) orientation/volume repair abs+flip (§6d#3). Reproducing
the paper's sorting needs the reconnection regime fixed — NOT a new energetics actor.

## 6g. Periodic vertex-mesh geometry (oracle stabilizer #1, §6d#1) — DONE & VERIFIED (2026-06-01)

The first oracle stabilizer (§6d#1, "minimum-image periodicity in the mesh") is now implemented in
the fork (`feat/native-rnr-reconnection`). A new mesh flag **`mesh.periodic_geometry`** (default OFF,
preserving the historical finite-cluster behaviour) routes EVERY vertex-mesh geometry computation
through the minimum-image convention over the `Universe::dim()` box, so a cell straddling a periodic
face is measured by its SHORT image, not the long box-spanning coordinates. New files
`source/models/vertex/solver/tf_mesh_metrics.{h,cpp}` add the shared helpers
`minimumImage / meshUsesPeriodicGeometry / meshPeriodicBox / meshRelativePosition / meshPositionNear
/ meshWrapPosition`; the flag lives on `Mesh` (`getPeriodicGeometry/setPeriodicGeometry`, IO-serialized,
SWIG property `tfMesh.i`). Routed call sites: `tfSurface.cpp` (centroid/area/perimeter/normal/contains/
merge/sew/split), `tfBody.cpp` (centroid/volume/findVertex/findSurface/isOutside/split), the actor
gradients `tfAdhesion.cpp` / `tfSurfaceAreaConstraint.cpp` / `tfVolumeConstraint.cpp`, and
`tfMeshQuality.cpp` (RNR edge-length `rnr_edgeLength` + the I↔H Appendix-1 placement, which unwraps
locally then wraps new vertices back).

**Design: periodicity is achieved by minimum-image GEOMETRY, not coordinate wrapping.** Two facts make
this correct and sufficient (both verified, see the Explore trace in the 2026-06-01 session):
- The default boundary conditions are already **PERIODIC_FULL**, so the *particle integrator* wraps
  `p->x` into `[0,L)` at boundary crossings on its own. The gap was never the particle positions — it
  was that the vertex-mesh geometry used PLAIN coordinate differences (box-spanning a straddling cell).
- The vertex forces are **topological+geometric** (actors over a vertex's own surfaces/bodies), NOT
  spatial-neighbour-list forces, so the engine's cell list / neighbour periodicity is irrelevant to
  them. Only the geometry needs min-image. The volume recomputation in `Body::positionChanged` is the
  one subtlety: each surface's `_volumeContr` ≡ `(1/6)·unnormalizedNormal·centroid_s`, and the body
  re-dots that same (translation-invariant) `normal` with the surface centroid shifted into the BODY's
  image (`meshPositionNear(s->getCentroid(), bodyCentroid)`) — i.e. it re-expresses every face in one
  common unwrapped frame. Correct provided each surface AND each body is smaller than half the box
  (the standard PBC requirement).

**CAVEAT — `meshWrapPosition` is NOT on the per-step hot path.** A step is: `MeshSolver::preStepStart`
→ forces into `p->f`; engine `engine_advance` moves `p->x` directly; `MeshSolver::postStepStart` →
`Vertex::positionChanged()` re-reads `p->global_position()` into the cache. `Vertex::setPosition`
(where `meshWrapPosition` lives) is hit ONLY by `Vertex::create` and explicit user/reconnection sets —
never by normal integration. This is fine: because all geometry is min-image, wrapping is unnecessary
for correctness; the engine's own PBC keeps `p->x` bounded anyway. Do NOT try to "fix" this by wrapping
in `positionChanged` — that would desync the cache from the integrator's `p->x` and `p->p0`.

**Behavioural note: `metrics::relativePosition` → `meshRelativePosition` swap.** Sites that previously
used `metrics::relativePosition` (engine-BC min-image, per-axis `space_periodic_{x,y,z}`) now use the
mesh flag instead. The mesh path keys off `Mesh::getPeriodicGeometry()` and uses `Universe::dim()` for
the box on ALL THREE axes (no per-axis control; `minimumImage` only skips an axis whose box ≤ 0, which
never happens for a real `dim`). For our cubic sorting box this is equivalent; if a future use needs a
slab geometry (periodic in x,y only), generalize `meshPeriodicBox` to carry per-axis flags.

**Verified correct — gates `rnr/tests/test_periodic_geometry.py` (3 tests) +
`test_native_roundtrip.py::test_native_periodic_roundtrip_across_boundary` (1 test), in
`pixi run test`, 37 green:**
1. *area & volume use the short image* — a unit cube split across x=0 reads area=6, volume=1, matching
   the interior cube; with the flag OFF the same coordinates give the box-spanning ~238 / ~59 (the flag
   is load-bearing).
2. *forces are periodic-correct, not just energies* — `VolumeConstraint` and `SurfaceAreaConstraint`
   gradients on the straddling cube equal the interior cube's to **float32** (`worst |Δ|`: volume
   1.65e-6, area 0.0). The 1.65e-6 is pure float32 roundoff from the `L − x` subtraction.
3. *RNR edge-length is min-image across a boundary* — a minimal Okuda [I] config centred on z=0 (its
   short edge's endpoints in different z-images) is still `valid`+`legal` and the native
   `analyze_i_reconnection` trigger length is the short 0.5, not the ~box-0.5 long image.
4. *(P3 gate) the native I↔H round-trip is periodic-correct across a boundary* — with
   `periodic_geometry=True` on a straddling [I] config (short edge over z=0, raw |dz|≈box−0.5), the
   native `force_reconnect_i_to_h` then `force_reconnect_h_to_i` mutate path (placement + body volumes
   measured min-image, topology walk image-independent) returns counts + adjacency to the original [I]
   and recovers the short edge to ~float32 (min-image edge drift < 1e-4). Bodies stay positive AND
   `< 100` (a box-spanning straddle would read huge). This is the P3 deliverable: the placement code
   (`rnr_placeIToH`/`rnr_placeHToI`, which unwrap locally then `meshWrapPosition` new vertices) is now
   exercised across the wall, not just the edge-length trigger.

**Float32 amplification caveat (for the periodic sorting runs).** Because a straddling cell carries an
extra `L − x` subtraction, its forces differ from an identical interior cell by ~1.6e-6 per eval. In a
*regularized* run (volume + area + adhesion all active) that is negligible. But a chaotically *unstable*
configuration amplifies it: a throwaway A/B probe of a lone cube under PURE volume constraint (shape-
unstable) diverged interior-vs-straddle within ~80 steps — confirmed to be chaos seeded by the 1.6e-6,
NOT a periodic logic bug (the forces match at step 0). Lesson: judge periodic correctness by force/energy
parity (done above), not by long trajectories of an ill-posed config.

**Still TODO for a full periodic sorting run** (this slice is geometry + reconnection-correctness only):
~~a periodic Voronoi initial packing (P4)~~ **DONE — see §6h**, min-image metrics in
`rnr/metrics.py` (Python side, P5), and combining with the §6e reconnection regime. Stabilizers §6d#2
(tiny-Lth quasi-static reconnection) and §6d#3 (orientation/volume repair) remain the other two gaps.
This implements §6d#1 only. **P3 (periodic reconnection round-trip) is now DONE & VERIFIED** (gate #4
above); the periodic *mutate* path — not just the geometry/trigger — is proven correct across a boundary.

## 6h. Periodic mesh GENERATOR (P4) — DONE & VERIFIED (2026-06-01)

The periodic initial packing for the bulk sorting run. Pure-Python, in `rnr/geometry.py`
(`periodic_bcc_seeds` + `build_periodic_voronoi`); on disk only (`rnr/` is not under git). Builds a
SPACE-FILLING Kelvin foam in the periodic box with **no free surface** — every face interior (b1/b2
both set), body adjacency wrapping across the box faces — for use with `mesh.periodic_geometry=True`.
Gate: `rnr/tests/test_periodic_geometry.py::test_periodic_voronoi_pack_is_space_filling_closed_and_wraps`
(+ `…_rejects_sub_box`), `pixi run test` = **39 green**.

**Route: GHOST-TILING (route B), not pyvoro `periodic=`.** pyvoro-mmalahe DOES accept
`periodic=[True,True,True]`, but on this build it returns garbage for a regular lattice (box-spanning
overlapping cells + spurious wall faces) at essentially every `dispersion` — too fragile. Instead we
replicate seeds across a 3×3×3 supercell, run the existing NON-periodic Voronoi, keep only the
central-image cells (then fully surrounded by ghosts → zero wall faces), and remap each central face's
`adjacent_cell` back to its central index (mod n). Reuses the validated finite path; robust.

**Three decisions that make it correct (each was a real failure mode hit + fixed):**
1. **`dispersion` aliasing.** voro++ block-size aliasing makes pyvoro return overlapping cells when
   `dispersion` divides the (highly regular) lattice spacing — e.g. for an n=3 BCC in L=6, disp ∈
   {2,3,4,6} all gave box-spanning overlaps + walls, while disp ∈ {2.5, full-box} gave the exact foam.
   Fix: force a SINGLE brute-force block — default `dispersion` = full ENLARGED-box edge (the finite
   path already does this for its box). O(N²) in the 27·N ghost seeds; fine at our sizes (N≈O(10²)).
2. **Min-image vertex dedup.** A wrap face is reported by its two cells with vertex coords a box vector
   apart (one sees x≈lo⁺, the other the ghost x≈lo⁻); the finite path's raw-coordinate `vkey` would NOT
   merge them, so the wrap face would never become one shared Surface. Fix: canonicalize each vertex
   into `[lo,hi)` (fold + round + refold a value rounded up to exactly hi back to lo) before keying, and
   create the TF vertex at the wrapped position. Faces are deduped by their VERTEX SET (`frozenset` of
   global vertex ids), NOT a `(min,max)` cell-pair key — the pair key breaks if a cell touches the same
   neighbour twice (directly AND through the wrap), possible in a small box. Self-adjacent faces (a cell
   meeting its own image, n<3 per axis) are rejected with a clear error.
3. **The foam box MUST equal the universe box `[[0,dim]]³`.** THE footgun, and the one that cost the
   most time. The engine min-images at `Universe::dim()` (`tf_mesh_metrics.cpp meshPeriodicBox`), so a
   cell straddling a foam-box wall is only measured by its short image when that wall IS a universe
   wall. Building the foam in a SUB-box (e.g. `[24,36]³` inside a `[0,60]³` universe) SILENTLY yields
   box-spanning volumes/centroids for the straddling cells — every straddling Kelvin cell read volume
   ~2000–2700 (vs 32) and its centroid landed mid-box, even though its areas/surfaces were perfect.
   Symptom signature: geometry (areas, surface list, interior cells) all correct, but straddling-cell
   volumes box-spanning and Σvol≈0 (signed contributions cancel). `build_periodic_voronoi` now asserts
   `box == [[0,dim_x],[0,dim_y],[0,dim_z]]` and raises otherwise. (This also surfaced that the engine's
   min-image body **centroid** for a straddling SHARED-surface cell — `Body::updateInternals` line ~185,
   `refreshBodies` reordering b1/b2 from both bodies' centroids — was never exercised before P4; it is
   correct, once the box matches the universe.)

**Verified (gate, `periodic_geometry=True`, BCC n=3 = 54 Kelvin cells in `[0,60]³`):** Σvol = 215999.9995
= boxvol (1e-4 tol); every cell volume = 4000.0 (= boxvol/54), all positive, none box-spanning; all 378
surfaces are 2-body (zero free faces); 145 wrap faces (shared-face body pairs whose raw seeds are >½ box
apart) and 38 straddling bodies, each with a positive min-image-small volume. The negative `…_rejects_
sub_box` gate proves the universe-box guard fires.

**New TF API surface this depends on (P5/port seams):** `tf.Universe.dim` (read box size),
`tfv.Vertex.create`, `stype(vertices=…)` (Surface ctor), `btype(surface_list)` (Body ctor),
`Surface.getBodies()`, `Body.getSurfaces()/getVertices()/volume/area/centroid`, `Vertex.position/id`,
`Body.id`, `tfv.MeshSolver.get().position_changed()`, `mesh.periodic_geometry`.

## 6i. Periodic substrate blow-up = `FlatSurfaceConstraint` not min-image (FIXED & GATED, 2026-06-10)

**The P5 blocker turned out NOT to be the MD engine.** The 2026-06-09 diagnosis
(`docs/periodic_substrate_engine_bug.md`, memory `periodic-substrate-engine-bug`) blamed an
engine cutoff/cell-list/ghost bug: a space-filling periodic Kelvin foam inverted a cell within
~hundreds of steps even at σ=0 with no reconnection and no noise, and the only "stable" cutoff
froze the mesh. **That root-cause was wrong** (the cutoff "stability" was the radius>cell freeze;
the real force is cutoff- AND radius-invariant). The actual cause, found by instrumenting the
per-step force path:

- Every `SurfaceType` auto-binds **two default actors** in its C++ ctor (`tfSurface.cpp`
  `SurfaceType::SurfaceType`, default `flatLam = convexLam = 0.1`): **`FlatSurfaceConstraint`** and
  `ConvexPolygonConstraint`. These are present on EVERY surface regardless of the Python spec.
- `FlatSurfaceConstraint::force/energy` (`actors/tfFlatSurfaceConstraint.cpp`) computed the
  out-of-plane offset as a **RAW** `source->getCentroid() - target->getPosition()`. The centroid
  is min-image-correct (computed via `meshPositionNear`), but the vertex position is the raw
  global coordinate, so for a surface that **wraps a periodic box wall** the two live in different
  periodic images and the difference is ≈ box-sized. With the actor's `×mass/_Engine.dt×lam`
  ("snap to plane in one step") prefactor this is a spurious ≈1/dt force (~10⁴ at dt=1e-4) along
  the wall normal on every wall vertex → cell inversion.
- It only manifested in the periodic bulk because the finite cluster has no wrap faces, and the
  static P1–P3/§6g tests only checked **rest-geometry values**, never an integrated foam.

**Fix (1 file, `actors/tfFlatSurfaceConstraint.cpp`):** use the minimum-image displacement
`meshRelativePosition(source->getCentroid(), target->getPosition())` (from `tf_mesh_metrics.h`) in
both `force()` and `energy()`. `meshRelativePosition` is the identity when `mesh.periodic_geometry`
is OFF, so **finite-cluster behavior is unchanged** (all 39 prior tests still pass).
`ConvexPolygonConstraint` already min-images (via `metrics::relativePosition(..., scent)`), so it
was left as-is. This is the same omission §6g fixed for VolumeConstraint/SurfaceAreaConstraint —
those two default surface actors were simply missed in that periodic-geometry pass.

**Gate (NEW dynamic test — the static tests missed this):** `rnr/tests/test_periodic_dynamics.py`
runs `scripts/probe_periodic_substrate.py` in a subprocess (tf.init is one-per-process) and asserts
a space-filling periodic foam INTEGRATES without inverting/inflating: σ=0 (pure substrate), σ=1
(het tension), and σ=1 + kT=0.1 thermal noise — **all STABLE** (σ=0: min_vol=max_vol=4.000 for
3000 steps; was inverting by ~step 500). `pixi run test` = 42 green. The noise test is the
payoff: a LARGE cutoff (3.0) now gives both engine thermal noise AND a stable foam, dissolving the
old "small-cutoff workaround kills the noise" dilemma (the cutoff was never the cause).

**Port note:** this is a vertex-model **actor** fix, fully in the LGPL fork; nothing in the MD
engine (`mdcore`) changed. Reproducers: `scripts/diag_force_when.py` (force=0 at rest, ~10⁴ after
one step, pre-fix). Substrate now relaxes a random periodic Voronoi toward equilibrium
(`scripts/sort_periodic_oracle.py substrate`: vol 0.18–2.44 → 0.84–1.09 over 2000 steps).
**Still separate:** the reconnection-under-dynamics overshoot (§6e / §7) — a sort run still
destabilizes on the first reconnection, which is the known displacement-overshoot issue, NOT this
substrate bug. **→ NOW RESOLVED for the periodic sort, see §6j.**

## 6j. Periodic sort reconnection blow-up = noise overshooting the Lth gap (FIXED & GATED, 2026-06-10)

**With the substrate fixed (§6i), a periodic *sort* (reconnection ON + thermal noise) still blew
up on the first reconnection** — `sort_periodic_oracle.py sort 4 0.5 0.1 1e-3 1e-3 1.9` went
`min_vol −1.78` by step 500 with only 1 reconnection, while substrate-only (no reconnection) was
stable. Localized with `scripts/diag_recon_overshoot.py` (separates surgery from relaxation) and
`scripts/diag_read_side_effect.py` (a clean recon×noise factorial):

- The negative volume is **transient and self-recovering** (a cell everts then un-everts; worst
  −3.1 at the dip, but `min_vol` is back to +0.65 a few hundred steps later — identical final
  state whether or not you sample volume every step, so reading `Body.volume` has NO side effect).
  The original "−1.78 at step 500" was just the checkpoint catching the transient dip.
- **Factorial (recon × noise), 600 steps, same config:** recon OFF+noise ON → STABLE (+0.29);
  recon ON+noise OFF → STABLE (+0.28); recon ON+noise ON → **UNSTABLE (−3.1)**. The blow-up needs
  BOTH. Each alone is fine.
- **Mechanism (a magnitude mismatch):** a native I→H places two new vertices `Lth = 1e-3` apart
  (`rnr_placeIToH`, Okuda Eqs. 46–48; placement is already min-image — verified, NOT suspect #1).
  But one Euler–Maruyama thermal kick is `DISP_STD = sqrt(2·μ·kT·dt) = sqrt(2·1·0.1·1e-3) = 0.0141`
  — **~14× the post-reconnection gap.** So a single noise step throws a freshly-reconnected vertex
  clean past its neighbour → the cell everts (TF signed volume goes negative). Normal vertices
  (nn ≈ 0.5) are unaffected; only the near-degenerate fresh-reconnection vertices are at risk.
  Note `DISP_STD ≫ Lth` holds for the oracle's own params too — the oracle survives it via
  orientation repair (§6d#3: `if(volume<0){volume=fabs; flip polygonDirections}`), i.e. it
  TOLERATES the transient eversion; we instead PREVENT it.

**Fix (Python harness, no engine rebuild): a per-vertex TRUST-REGION on the noise.** Cap each
vertex's noise displacement at `NOISE_CLAMP · (min-image nearest-neighbour distance)`
(`NOISE_CLAMP = 0.4`). For a normal vertex the cap (~0.2) ≫ the kick (0.014) so it never binds; it
binds only on the Lth-scale fresh-reconnection vertices, holding them next to their partner until
relaxation pulls the gap open. This is the position-level analogue of the proven
`operator.stable_step` clamp (memory `winding-clamp-stabilizes-sort`) but applied to the NOISE
specifically (the deterministic force dynamics are untouched — justified because recon ON+noise OFF
is already stable). Implemented in `sort_periodic_oracle.py` / `probe_periodic_sort.py`:
the implicit-edge topology is **cached and rebuilt only when `num_vertices` changes** (i.e. after a
reconnection), so per-step cost is O(V) reads + O(E) vectorised min-image distances (`np.minimum.at`
over the edge list).

**DEPARTURE (flagged):** this is a locally-adaptive timestep safeguard, not in naive
Euler–Maruyama — it shrinks the effective dt only where the local length scale (a fresh Lth edge)
is far below the global one. It converges to the same SDE (the cap rarely binds). The cleaner,
faithful alternative is the oracle's orientation repair; but memory `native-instability-is-
displacement-overshoot` records that the force-level abs(volume) approximation of it gave runaway
inflation, so the trust-region (prevent, don't repair) is preferred for the prototype.

**Result:** `sort 4 0.5 0.1 1e-3 1e-3 1.9 20000` is **STABLE for 20000 steps / 84 reconnections**
(`worst_min = 0.270` = the initial Voronoi minimum, never lower; `worst_max = 2.618`, never
inflates). D drifts −0.079 → −0.089 (modest demixing; the D ≈ −0.10 ceiling is not broken at
σ=0.5 — reconnection rate is the limiter at gentle Lth, see `oracle-comparison-ceiling-physical`).

**Science — σ/Lth sweep (M=4, 64 cells, 20000 steps), the open kickoff question "does a STABLE
periodic bulk break the D≈−0.10 finite-cluster ceiling?":**
| σ   | Lth  | reconnections | D (end) | hetA  | verdict |
|-----|------|---------------|---------|-------|---------|
| 0.5 | 1e-3 | 84            | −0.089  | 0.405 | STABLE  |
| 1.0 | 1e-3 | 88            | −0.099  | 0.376 | STABLE  |
| 0.5 | 0.05 | 6973          | **−0.115** | 0.345 | STABLE (worst_min 0.077) |
| 1.0 | 0.05 | 3894          | −0.173 @ step 11k then **EVERTS** | 0.289 | UNSTABLE @ ~11.5k |
- The **D≈−0.10 ceiling IS broken** once the T1 rate is raised (Lth 1e-3→0.05 gives ~80× more
  reconnections → D −0.115): the ceiling was the slow reconnection rate at oracle-gentle Lth, NOT
  finiteness. Lth=0.05 is a DEPARTURE (Okuda wants Lth small for O(Δl_th) reversible gaps), so this
  is "the bulk CAN sort deeply," not an oracle-faithful number.
- **Force-overshoot boundary:** the doubly-aggressive regime (σ=1.0 AND Lth=0.05) sorts the deepest
  (D=−0.17 by step 6.5k) but eventually everts a cell — because at high σ the deterministic edge
  collapse is strong and the *noise* clamp does not limit *force* overshoot. A tighter noise clamp
  (0.2) does NOT help (still everts, sooner) → confirms it is force- not noise-driven.
- **A Python total-displacement clamp (noise+force, applied after `tf.step`) was tried and FAILS in
  this regime** (everts at step ~300, worse than no clamp): at Lth=0.05 a reconnection fires almost
  every step, so the Python post-step clamp must SKIP reconnection steps (handles invalidated, nv
  mismatch) and its independent per-vertex pull-back distorts faces under strong tension. **This is
  positive evidence that the force-overshoot trust-region must be NATIVE** — inside the vertex
  integrator, applied every step consistently with the in-`doQuality` reconnection — not a Python
  post-step pass. Reverted; the harness keeps the clean noise-only clamp.

**Gate (NEW):** `test_periodic_dynamics.py` adds two subprocess tests over `probe_periodic_sort.py`:
`test_periodic_sort_stable_with_reconnection_and_noise` (clamp 0.4 → STABLE, min_vol>0 through 2000
steps) and `test_periodic_sort_unclamped_noise_inverts_a_cell` (clamp 0 → UNSTABLE within ~500
steps; the load-bearing check that the clamp isn't a no-op). `pixi run test` = **44 green**.

**Port note (Phase 3):** the prototype clamps Python-applied noise only. The eventual C++ port
should put a per-vertex displacement trust-region in the **vertex integrator** (cap the total
per-step motion at a fraction of nn-distance) so it also covers ENGINE noise (`tf.Force.random`,
applied inside `tf.step()` where Python can't pre-clamp it) and any residual post-reconnection
force overshoot — one mechanism, covering all three displacement sources.
*(UPDATE §6k: for the transient-eversion source the FAITHFUL fix is the oracle's orientation
repair, now implemented natively — the noise clamp above is superseded for the paper's regime.
The displacement trust-region is now wanted only for the aggressive-Lth DEPARTURE regime.)*

## 6k. Native orientation repair (oracle stabilizer #3) — the FAITHFUL replacement for the noise clamp (DONE & GATED, 2026-06-11)

The §6j noise clamp was a flagged DEPARTURE (a locally-adaptive timestep safeguard, not in the
paper). The paper survives the same transient eversion FAITHFULLY via **orientation repair**
(oracle stabilizer #3, §6d#3). This session implements that repair natively, so the periodic
noisy reconnecting sort is stable at the paper's true parameters **without** any clamp.

**Why §6c's "volume robustness is NOT the fix" did not apply.** §6c rejected a *volume robustness*
attempt (Approach A) that took a **per-FACE abs**, `volume = (1/6)Σ|(c_s−c_b)·N|` — summing the
absolute value of each face's contribution. That overcounts a half-everted cell (its faces don't
cancel) → a spuriously large positive volume → the inflation runaway §6c measured. The oracle does
something different: a **per-CELL** repair (`3DVertVor/Cell/Cell.cpp:216-221`). It computes the
cell's TOTAL signed volume `V = Σ_f dir_f·(dP_f/6)`; if `V<0` it takes `fabs(V)` **and flips every
face direction together** (`polygonDirections_[id] = !polygonDirections_[id]` for all faces), then
uses those same directions consistently in the volume AND the force (`Energy/Volume.cpp:87,126`).
Flipping ALL faces together is exactly negating the whole sum — i.e. a **single per-cell sign**, not
a per-face abs. So the faithful repair was never actually tried; §6c's verdict was about the broken
per-face version (and about instabilities A/B — periodic wrap + reconnection storm — which are
independently fixed by §6g periodic min-image + §6e `reconnect_interval`). The remaining §6j
transient-eversion mode is exactly what stabilizer #3 cures.

**TF mapping — a single per-body `orientSign` ≡ the oracle's flip-all-`polygonDirections_`.** TF has
no persistent per-(cell,face) direction store; `Surface::volumeSense(body)` is ±1 from b1/b2
identity (`tfSurface.cpp:954`) and the cell volume is `Σ_s N_s·c_s·volumeSense(this)/6`
(`tfBody.cpp` `updateInternals`/`positionChanged`). Because the oracle's repair only ever flips
ALL faces together, its effect is captured exactly by one per-body sign multiplying `volumeSense`.
Seams (fork `feat/native-rnr-reconnection`):
- **`tfBody.h`** — new private `FloatP_t orientSign` (default +1) + public `getVolumeOrientSign()`.
- **`tfBody.cpp`** — after the volume sum in BOTH `updateInternals()` and `positionChanged()`:
  `orientSign = 1.f; if(repairEnabled && volume < 0.f){ orientSign = -1.f; volume = -volume; }`.
  This is *memoryless* — `orientSign` = sign of the raw signed volume recomputed every step —
  which is equivalent to the oracle's persistent flip (it only flips all-together) and **strictly
  safer**: the oracle re-checks only at init + inside each reconnection pass
  (`Reconnection.cpp:94` → `updatePolygonDirections`) and trusts the sign between, but TF's force is
  sensitive to even ONE negative-volume step (§6j), so TF applies the abs+flip every step in the
  per-step hot path.
- **`tfVolumeConstraint.cpp:60`** — `force ×= source->getVolumeOrientSign()`, so the area-gradient
  stays restoring when `getVolume()` has been abs'd (faithful to the oracle applying the flipped
  directions in its volume force, `Volume.cpp:126`).
- **`tfNormalStress.cpp:50`** — `×= bodies[0]->getVolumeOrientSign()` too (completeness; NormalStress
  is not in the faithful model, but it's the other `volumeSense` consumer).
- **Toggle:** env var `TF_VERTEX_NO_VOLUME_REPAIR=1` falls back to stock signed volume (cached once
  via a file-static lambda in `tfBody.cpp`). Used only by the load-bearing gate counter-test.
  Healthy cells (`V>0`) get `orientSign=+1` ⇒ **stock behaviour**, so nothing else changes (the
  44 prior tests stay green).

**Results (`probe_periodic_sort.py`, M=4 / 64 cells, kT=0.1, μ=1, s0≈5.6 via the existing actors):**
| regime | clamp | repair | outcome |
|---|---|---|---|
| **faithful** σ=0.5, Lth=1e-3 | 0 (OFF) | ON | **STABLE 20000 steps** — min settles ~0.78–0.83 (worst_min 0.047>0), max ~1.15 (worst_max 2.618); no clamp needed |
| faithful σ=0.5, Lth=1e-3 | 0 (OFF) | **OFF** (env) | **UNSTABLE** — min_vol −1.78 @ step 500 (reproduces §6j's exact number ⇒ repair is load-bearing) |
| DEPARTURE σ=1, Lth=0.05 | 0 (OFF) | ON | UNSTABLE, but **evert→inflate**: min stays +0.15 (eversion CURED) while max → 5.2 (inflates) |

- The faithful regime — the paper's regime — is now stable with the paper's own mechanism and **no
  clamp**. This supersedes the §6j noise-clamp departure for faithful runs.
- The deep DEPARTURE regime (aggressive Lth=0.05) confirms the mechanism split: the repair removes
  the winding-SIGN failure (min never goes negative) but the cell still fails by displacement-
  MAGNITUDE overshoot (inflation). That is the **trust-region's** job (the §6j port note), NOT the
  repair's — and it is a departure (Okuda wants Lth small), separate from faithfulness. The repair
  *isolating* evert→inflate is positive evidence the two instabilities are distinct.

**Gate:** `test_periodic_dynamics.py` — `test_periodic_sort_stable_with_native_volume_repair`
(repair ON, clamp 0 → STABLE, no inflation) and `test_periodic_sort_unrepaired_unclamped_noise_
inverts_a_cell` (env repair OFF, clamp 0 → UNSTABLE, the non-no-op check). **`pixi run test` = 45.**

**Port note (Phase 3):** `orientSign` IS the C++ analogue of `polygonDirections_` (one per-body sign
suffices). The env toggle should become a proper `MeshQuality`/solver flag. The repair is orthogonal
to the displacement trust-region (§6j): repair fixes the volume-sign bookkeeping; the trust-region
(if the aggressive-Lth regime is ever wanted) caps per-step displacement. Faithful runs need only
the repair.

## 6l. The clamp ENABLES reconnection (not just stability) — Fig 1E setup (2026-06-11)

Starting the Manning2024 **Fig 1E** reproduction (`docs/fig1e_reproduction_kickoff_prompt.md`,
which said "set clamp=0, the §6k repair provides stability now"), a reconnection-RATE diagnostic
showed that premise is **incomplete**. The §6j noise trust-region clamp is **load-bearing for
SORTING**, not only for stability:

| dt | clamp | recon / 3000 steps (σ=0.5, M=4) |
|---|---|---|
| 0.001 | **0** (kickoff "faithful") | **~1**  (starved) |
| 0.001 | 0.4 (§6j) | ~24 |
| 0.005 | 0.4 | ~11 |
| 0.01  | 0.4 | ~3 (plateaus) |
| 0.01  | 0 | ~0 (through t=600) |
| 0.001 | 0, **interval=1** | ~3 (every-step checking doesn't rescue it) |

**Why.** A native I↔H places two vertices Lth=1e-3 apart; one Euler–Maruyama kick is
DISP_STD=√(2μ·kT·dt) = 0.0141 (dt=1e-3) … 0.0447 (dt=1e-2) = **14–45× Lth**. With clamp=0 the
kick blows a collapsing edge back above the Lth trigger before `doQuality` (every
`reconnect_interval`=10 steps) catches it below threshold ⇒ reconnection starves ⇒ no neighbour
exchange ⇒ **no sorting** (the paper's DP is neighbour-count based). The §6j clamp (cap noise at
0.4×nn-dist) binds ONLY on near-degenerate short edges, letting them persist below Lth so the T1
fires. So the clamp does **two** jobs: (1) prevent the post-reconnection eversion (§6j) AND (2)
**enable reconnection at all**. The §6k orientation repair only addresses (1). Therefore
**clamp=0 + repair ON is stable but FROZEN** — the §6k "faithful stable 20000 steps" result is
genuinely stable, but the tissue is the jammed control, not sorting. (This refines, not refutes,
§6k: the repair IS the faithful eversion fix; it just isn't a reconnection ENABLER.)

A fully clamp-free faithful sort isn't reachable with the current **Python-pre-step-noise**
harness: interval=1 barely helps (~3) because the edge is above Lth at essentially every check.
Catching T1s without the clamp would need the Lth trigger to read a **noise-free / time-averaged**
edge length, or noise applied so the deterministic tension dominates near collapsing edges — an
engine change (Phase-3 port note: put the trust-region in the vertex integrator so it covers
engine noise too, and/or trigger on the mechanical edge length).

**dt.** Larger dt starves reconnection (noise ∝ √dt) and runs `doQuality` fewer times per unit
physical time, so the paper's dt=0.01 is unusable here despite being volume-stable; **dt=0.001**
gives the best reconnection rate. Reaching the paper's t≈2000–4000 at dt=1e-3 needs 2–4e6 steps,
so compare **trend over a feasible window / fraction-of-run** (validated approach, §compare_oracle).

**Fig 1E metric (DONE, re-derived not copied).** DP (Sahu/Schwarz/Manning ref[4], arXiv 2102.05397,
Eq.2) = ⟨2(N_s/N_t − 1/2)⟩, N_s = homotypic nbrs, N_t = total nbrs ⇒ **DP = −`demixing_index`**
exactly (the documented sign flip). DP_max = segregated-config value, <1 at finite N (Sahu SI:
≈ 1 − O(N^−1/3)); `compute_dpmax.py` gives **DP_max(M=6)=0.56**. Paper plots DP/DP_max.

**Fig 1E setup used.** M=6 (N=216, larger than M=4 to shrink the finite-size DP offset),
**clamp=0.4 (flagged DEPARTURE) + repair ON + dt=0.001 + interval=10**, kT=0.1, Lth=1e-3, cut=1.9,
σ∈{0.1,0.2,0.5}, ensemble over seeds (paper ensemble-averages; count-based DP is noisy at our N —
AREA-based het fraction σ-orders within a few k steps, count-DP lags because areas shrink before
neighbours are lost). New: `compute_dpmax.py`, `fig1e_demixing.py`, probe `interval` arg, CSV tag
now carries the seed. `pixi run dpmax` / `pixi run fig1e`.

**RESULT** (3×3 ensemble: σ∈{0.1,0.2,0.5} × seeds 7/8/9, M=6, clamp=0.4 + repair, dt=1e-3,
interval=10, 100k steps each; `rnr/exports/fig1e_demixing.{png,csv}`):
- **Stability: all 9 STABLE** — ~530–560 reconnections each, no eversion (worst_min>0) or
  inflation (worst_max ≈ initial spread). The committed native repair holds the noisy
  reconnecting periodic sort together at the paper's params over long runs and many T1s. (Caveat:
  9 simultaneous 216-cell TF procs OOM-kill ~3 of them — run ≤3–4 parallel.)
- **AREA demixing reproduces the σ-ordered Fig 1E trend** — normalized S_area = 1−⟨hetA(t)⟩/⟨hetA(0)⟩
  ends at **0.022 / 0.037 / 0.090** for σ = 0.1 / 0.2 / 0.5: monotonic and ORDERED (≈4× span). The
  het tension shrinks heterotypic interface area, faster/more for larger σ — the physics of Fig 1E.
- **Count-based DP/DP_max (the paper's EXACT metric) does NOT resolve** — with equal seeds (DP_0 =
  −0.0037 all three) the final DP/DP_max are +0.001 / −0.007 / +0.009, i.e. all within ±0.01 of 0
  and NOT σ-ordered: pure noise. Cause: with the faithful energy-gate-OFF geometric RNR, total
  neighbour count GROWS over the run (e.g. 1687→2219 contacts) as the foam relaxes to s0=5.6 and
  churns under reconnection, and the new contacts are ~50/50, so the het-neighbour FRACTION stays
  ≈0.5. Neighbour-count demixing (the paper's DP) requires DOMAIN formation, which needs much
  larger N (paper N≥512; DP_max→1) and longer t (paper t=10000 vs our t=100 ≈ 1% — dt=1e-3 forced
  by the reconnection-rate constraint above). This is the finite-size + reconnection-rate ceiling
  (`oracle-comparison-ceiling-physical`), now quantified for the count metric.

**Verdict:** TREND match (area, monotonic + σ-ordered) + engine stable, NOT an absolute count-DP
match — exactly the honesty the kickoff asked for. Reaching the paper's count-DP→1 is a system-
size/run-length problem (and possibly wants domain-seeded ICs / the energy gate), not a stability
or correctness gap in the native RNR.

## 6m. Fig 1F — the demixed state is energetically preferred (DONE, the clean count-DP result)

Fig 1F is the counterpart to 1E: initialize the tissue ALREADY DEMIXED and show it STAYS demixed.
This **sidesteps the §6l count-DP limit** — DP starts at DP_max, so we test whether the het tension
HOLDS it there (no domain formation needed). `sort_periodic_oracle.py` gained an 11th arg `demixed`
that seeds a segregated z-slab (== the `compute_dpmax.py` config); plot with `fig1f_stability.py`
(`pixi run fig1f`). `rnr/exports/fig1f_stability.{png,csv}`.

**RESULT (3 seeds × σ∈{0.1,0.2,0.5}, demixed IC, 100k steps; vs the Fig 1E mixed-IC at σ=0.5):**
- **The demixed state HOLDS for every σ** — DP/DP_max stays high over 100k steps / ~550–640
  reconnections: σ=0.1 0.93→0.86, σ=0.2 0.93→0.86, σ=0.5 0.97→0.92 (all >0.8; a slow ~5–7% relax,
  tightest hold at σ=0.5). The mixed-IC at σ=0.5 stays ≈0 (the §6l rate limit). The contrast
  demixed≈0.9 vs mixed≈0 is **direct evidence the energetics + native RNR are correct: the demixed
  state is a stable minimum the het tension maintains** — even at weak σ=0.1. All STABLE (repair).
- This is the **clean COUNT-BASED DP result** Fig 1E couldn't give (1E needs domain nucleation at
  large N; 1F starts past that barrier). So: 1E trend reproduced in AREA + 1F count-DP stability
  reproduced. The only thing still out of reach at N=216/t=100 is mixed→demixed count-DP convergence.
- Op note: a sporadic kill still claimed 1 of 3 demixed σ=0.5 runs near step 92.5k (re-ran solo);
  the deaths are OOM/transient, not deterministic crashes (same seed completes solo).

## 6n. The clamp was a wrong-noise-model artifact — the oracle uses ACTIVE MOTILITY, not thermal noise (RESOLVED & GATED, 2026-06-11)

**The §6j/§6l noise clamp is not faithful and is not needed. It was a band-aid over a wrong noise
MODEL.** The kickoff (`docs/clampfree_reconnection_kickoff_prompt.md`) framed the clamp as load-
bearing for sorting and asked how the oracle catches sub-Lth edges when "the paper's per-step noise
sqrt(2·μ·kT·dt)=0.045 ≈ 45× Lth." That premise is the bug: it assumes the oracle uses THERMAL
Brownian noise (Euler–Maruyama, scaling as **√dt**). It does not.

**What the oracle actually does (read-only, GPL — re-derived, nothing copied):**
- `3DVertVor/Run/Run.cpp:1283 updateVerticesPosition()` advances each vertex by
  `x += velocity·dt + dt·motility` (line **1345**). The THERMAL line right above it
  (`x += … + cR·ndist`, `cR=sqrt(2·mu·kB·T·dt)`, **:1344**) is **commented out.** (The base `tvm`
  at `tvm/Run/Run.cpp:161,164` DOES use that thermal `cR·ndist` — the Manning fork switched models.)
- `motility` is ACTIVE self-propulsion, not noise: each CELL has a director `n_c ∈ S²` that rotates
  by active-Brownian rotational diffusion (`Run.cpp:1287` `Dr=1`, `NoiseStdDev=sqrt(2·Dr·dt)`, the
  ONLY √dt term — on ORIENTATION); each vertex's `motility = temperature·⟨n_c⟩` over its incident
  cells (`Vertex.cpp:78–86`). So the per-step displacement is **`dt·v0` (v0≡temperature), scaling
  as dt, NOT √dt.** With v0=0.1, dt=1e-3 that is **≤ 1e-4 = 0.1×Lth** — *below* the trigger.
- The reconnect trigger is the plain instantaneous edge length (`Reconnection.cpp:34`
  `edge->length_ < Lth_`, `Lth_=1e-3`). It works because per-step motion is sub-Lth: a collapsing
  edge stays below Lth across the dtr=10·dt window and is caught. There is **no √dt jitter to dwarf
  Lth in the first place** — so no clamp, no time-averaged trigger, no special ordering is needed.

**Why our harness needed the clamp.** §6f *noticed* the fork is active-motility but chose
`tf.Force.random` thermal white noise anyway, calibrating to **long-time diffusion** (`D=μ·kT`).
Long-time-diffusion equivalence is **not** per-step-displacement equivalence: the thermal kick is
√(2μ·kT·dt)=0.0141 (dt=1e-3)…0.0447 (dt=1e-2) = **14–45× Lth per step**, which blows freshly-
collapsing edges back over the trigger and STARVES reconnection. The §6j clamp (cap noise at
0.4·nn-dist) crudely re-imposed the active model's intrinsically small per-step motion. The
reconnection trigger cares about **per-step displacement**, the one quantity §6f did not match.

**Measured (this env; `probe_active_motility.py` vs `probe_periodic_sort.py`, M=4 σ=0.5 3000 steps):**
| noise | clamp | recon/3000 | note |
|---|---|---|---|
| thermal √dt | 0.4 | **35** | clamp props it up |
| thermal √dt | 0   | **1**  | STARVED (the §6l bug) |
| **active dt** | **none** | **35** | faithful — rate restored, no clamp |
| active dt (M=6) | none | **141** | scales (vs thermal+clamp 94) |
| active, v0=0 (NO noise) | — | **38** | the reconnections are DETERMINISTIC-relaxation-driven |

The v0=0 control is the key refinement: in this foam the reconnections are driven by the
deterministic relaxation of the Voronoi IC under volume/area/adhesion; the noise's job is **not to
DRIVE reconnection but to NOT SABOTAGE it** (thermal √dt sabotages; sub-Lth active doesn't) while
providing the persistent active stirring the oracle uses to escape local minima. All STABLE
(worst min_vol ≈ initial; no eversion/inflation) at clamp=0 with the §6k repair ON.

**Implementation.** `rnr/scripts/probe_active_motility.py` (new) + `sort_periodic_oracle.py` gained
`NOISE_MODEL` (arg 12, **default `active`**; `thermal` keeps the legacy clamp path + legacy CSV name).
Active CSVs are tagged `…_active.csv`. The active model: per-CELL director array, rotational
diffusion `n ← normalize(n + sqrt(2·Dr·dt)(ξ−n))`, per-vertex `u = v0·⟨n_incident⟩`, `Δx = dt·u`,
periodic-wrapped. **Stale-handle fix:** rebuild the vertex-handle + incidence cache from the LIVE
mesh EVERY step — a single doQuality pass can do several I→H(+1 vert)/H→I(−1) that NET to zero count
change, so `num_vertices` is an unsafe staleness signal (a cached deleted handle segfaults on
`.position`; this silently crashed M=6 until fixed). Gate: `rnr/tests/test_clampfree_reconnection.py`
(active no-clamp rate ≥10 + STABLE; unclamped-thermal starves ≤3). `pixi run test` = **47**.

**Supersedes / corrects:** the §6j noise-clamp DEPARTURE (no longer needed for faithful runs); the
§6f "`tf.Force.random` is faithful" claim (faithful for long-time diffusion, NOT for the
reconnection trigger — use active motility); the §6l framing "the clamp ENABLES reconnection"
(precisely: unclamped THERMAL noise DISABLES it; the faithful active model never disables it). The
§6k orientation repair stays ON and orthogonal (it handles transient eversion; with the active model
no eversion is even provoked in these runs, but the repair remains the faithful safeguard).

**Science reproduces at clamp=0 (gate 4) — FULL ENSEMBLE DONE (2026-06-12).** Regenerated the whole
fig1e/fig1f ensemble with the active model, clamp-free: M=6, σ∈{0.1,0.2,0.5} × seed∈{7,8,9} ×
IC∈{mixed,demixed}, **100k steps** (18 sims, all STABLE — no eversion/inflation, ~340 reconnections
each), orchestrated by `rnr/scripts/run_overnight.py` (bounded-concurrency, failure-tolerant pool).
- **Fig 1E** (`fig1e_demixing_active.png`): σ-ordered AREA-demixing reproduces cleanly — S_area =
  1−hetA(t)/hetA(0) = **{0.022, 0.057, 0.116}** for σ={0.1,0.2,0.5} (monotonic ~5× spread, ORDERED).
  The paper's exact count-based DP/DP_max stays ≈0 for all σ (finite-N limited at N=216, as §6l/§6m;
  its tiny residual is at least σ-ordered too with the 3-seed average). Area is the resolving metric.
- **Fig 1F** (`fig1f_stability_active.png`): the demixed IC **HELD for every σ** — DP/DP_max =
  {0.89, 0.90, 0.91} (all >0.8) vs the mixed IC ≈0 (−0.001). The demixed≈0.9-vs-mixed≈0 contrast is
  direct evidence the energetics + native RNR are correct and the demixed state is a stable minimum.
- **Video** (`sort_active_demixing.gif`, `rnr/scripts/video_periodic_active.py`): 51-frame render of
  the σ=0.5 active sort, hetA 0.526→0.463 over 40k steps. Matches the prior thermal+clamp ensemble
  (§6l/§6m) qualitatively but **with no clamp**. (Op note: a mid-run power-loss hibernation paused the
  batch overnight; processes resumed intact on power-up — the failure-tolerant orchestrator + the
  `fig1{e,f}` MODEL=active selector were the robustness that made that a non-event.)

**C++ port implication (Phase 3).** The eventual native `MeshQuality`/integrator must drive vertices
with the **active self-propulsion** model (per-cell director + `dt·motility`), NOT a √dt thermal
force. Then per-step displacement is sub-Lth by construction and the §6j "in-integrator
trust-region/clamp" port note is **unnecessary** — there is no overshoot to clamp. This both closes
the science gap and removes a planned C++ departure.

## 7. (Phase 2) Stability of reconnection-under-dynamics — DIAGNOSED
**(SUPERSEDED in part — see §6c: the "winding sign-flip" below is the SYMPTOM; the CAUSE is a
per-vertex displacement overshoot, either a periodic-image wrap (corner cluster) or a
post-reconnection overshoot. The force-level abs(volume) "port fix" suggested here does NOT
stabilize the native gate; the real fixes are config (centre/non-periodic) + the oracle's
reconnection regime, §6c/§6d.)**

Established empirically (`rnr/scripts/faithful_probe.py`, `faithful_run.py`,
`sort_with_reconnect.py`). The root mechanism is now understood:

- **The negative volumes are TF winding SIGN-FLIPS, not geometric collapse.** Under linear
  het tension interior edges collapse to ~1e-4 (faithful threshold is feasible). At too-large
  dt the integrator overshoots a near-collapsed face — a vertex crosses through it — so that
  face's winding flips and TF's signed cell volume `V` goes negative *while the cell is
  geometrically intact*. `faithful_run.py`'s orientation diagnostic confirms this:
  independent orientation-free `V_geo > 0` (often exact-magnitude vs the negative `Body.volume`),
  convex hull healthy. This is exactly 3DVertVor's `abs(volume)`+flip case.
- **The runaway is a force-sign feedback loop.** A negative `V` makes the VolumeConstraint
  force `∝(V−V₀)` point the wrong way → the cell inflates without bound (hulls seen ≤73×
  target, min_vol → −8e4) → cascade. This is why a Python "reverse the reconnection" guard
  CANNOT cure it (the flip appears dynamically, several steps *after* the mutate; at mutate
  time the neighbourhood is still positive, so `cum.reverted=0` at faithful features).
- **dt is the master stability lever.** The frozen substrate (reconnection OFF) inverts at
  dt=1e-3 but is STABLE at dt=2e-4 — small enough that per-step displacement stays below the
  face-crossing threshold. `pixi run sort` runs at dt=2e-4 and stays bounded (worst min_vol
  ≈ −7, transient) while demixing (het 194→189). *(The older "negative homotypic λ + dt=0.005
  blows up with reconnection OFF" finding still holds for that substrate; non-negative
  tensions fixed it. The sign-flip runaway is the NEXT layer, fixed by dt.)*
- **The volume guard is feature-size-dependent.** At BIG features a reconnection can directly
  invert a neighbour at mutate time → the guard reverses it (834 fires in the sort demo). At
  faithful-SMALL features it never fires (`cum.reverted=0`) → default OFF in `operator.py`.

**Port fix (the key §6 finding):** repair the volume sign at the source in C++ —
(a) guarantee the reconnection surgery leaves consistent surface winding, and
(b) make the cell volume / VolumeConstraint robust to a transient winding flip (3DVertVor:
`abs(volume)` + flip `polygonDirections_`; better: a one-sided `V>0` force). TF computes
volume internally so the FORCE-LEVEL fix CANNOT be done from Python — it is precisely what
unblocks the Okuda-pure infinitesimal-feature trigger and lets us drop the energy-gate departure.

**RESOLVED for the prototype (2026-05-30) — `operator.stable_step` winding clamp.** The
force-level abs(volume) is native-only, but the overshoot can be PREVENTED from Python: cap
each vertex's per-step displacement at `rel_frac < 0.5` of its nearest-neighbour distance, so
no vertex crosses a neighbour in one step and no winding flips. It's a trust-region limiter on
overdamped gradient descent (binds only on the pathological overshoot; ~50 clamps/step early,
self-decaying to ~5 as the mesh settles). `pixi run check-clamp`: clamp OFF → min_vol −218 by
step ~4260 (cells eject); clamp ON (0.4) → worst min_vol **+2.09**, never non-positive, and the
sort goes **DEEPER** (D −0.10 vs the unstable run's −0.096) — proving the instability was a
parasitic artifact, not the sorting mechanism. The re-rendered `pixi run sort-video` now stays
a clean space-filling tissue throughout (final min_vol +2.97, D −0.096). Flagged DEPARTURE
(§5): the native port still uses abs(volume) and DROPS this clamp. Unit tests:
`rnr/tests/test_stability.py` (the pure `_clamp_to` math).
