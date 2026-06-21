# Building 3D Vertex Models in TissueForge (Python)

A practical guide to constructing and running **3D vertex-model** tissue simulations in
TissueForge from Python, with a focus on **multiple cell types**, **heterotypic interfacial
tension** ("adhesion"), and **neighbour-exchange reconnection** (the 3D T1 / Okuda I↔H operation)
so cells can rearrange and sort.

> **Audience:** new users who want to assemble elaborate multicellular 3D vertex models.
> Everything here is verified against the TissueForge source in this workspace.

---

## 0. Stock TissueForge vs. this fork — read this first

TissueForge ships an open, general vertex-model solver, but **upstream TissueForge cannot perform a
3D neighbour exchange**: its 3D mesh-quality operations are only *degenerate collapses*
(`BodyDemote`, `SurfaceDemote`, `EdgeDemote`). The reversible 3D T1 (a face↔edge swap) — the
operation that actually lets cells swap neighbours and sort in 3D — **does not exist upstream**.

This workspace uses a fork (`feat/native-rnr-reconnection`, github.com/tc2fh/tissue-forge) that adds,
natively in C++:

| Capability | Where | Stock TF? |
|---|---|---|
| 3D **I↔H reconnection** (the 3D T1), inside `doQuality` | `Quality.reconnect_length` etc. | ❌ fork only |
| **Periodic minimum-image** vertex geometry | `mesh.periodic_geometry` | ❌ fork only |
| Per-cell **orientation repair** (winding-flip robustness) | automatic | ❌ fork only |
| **Active self-propulsion** drive (per-cell director) | `MeshSolver.set_motility`, `body.director` | ❌ fork only |
| Volume/area/adhesion energetics, types, 2D T1 (merge/split) | `BodyTypeSpec`, actors, `Quality` | ✅ stock |

Features marked **fork only** are flagged ⭐ throughout. If you are on upstream TissueForge, the
type/energetics/mesh-construction parts of this guide apply, but 3D sorting will *jam* without the
reconnection operation.

Build the fork with `pixi run build-tf` (see the workspace `README`/`pixi.toml`); verify with
`pixi run verify`.

---

## 1. The data model

A vertex-model mesh maps almost 1:1 onto the cell-tissue picture:

- **`Vertex`** — a point in space. Backed by a TissueForge *particle* integrated with **overdamped**
  dynamics (see §3). Knows its incident `surfaces`, `bodies`, `connected_vertices`, `position`.
- **`Surface`** — a polygonal face. A surface between two cells *is* a cell–cell interface. Has
  exactly **two body pointers** (`b1`, `b2`; 0–2 in general), an ordered `vertices` ring (winding
  defines the normal), `area`, `normal`, `centroid`.
- **`Body`** — a cell (a closed polyhedron of surfaces). Has `surfaces`, `vertices`, `volume`,
  `area`, `centroid`, `connected_bodies`.
- **`Mesh`** — the container of all vertices/surfaces/bodies, plus the `quality` operator and the
  `periodic_geometry` flag.
- **Edges are implicit** — consecutive vertices in a surface's ordered ring. There is no explicit
  `Edge` object.

Each kind has a lightweight **handle** variant (`BodyHandle`, `SurfaceHandle`, `VertexHandle`) — the
objects returned by most Python calls. Handles expose the same properties (`.volume`, `.centroid`,
`.become(...)`, …) and stay valid across topology changes as long as the underlying object lives.

Two layers of API:
- **Engine objects** (`Body`, `Surface`, `Vertex`, `BodyType`, `SurfaceType`) and **actors**
  (forces/energies) come from the C++ core via SWIG, re-exported in
  `tissue_forge.models.vertex.solver`.
- **Pythonic helpers**: `BodyTypeSpec`/`SurfaceTypeSpec` (declarative type definitions),
  the `bind` module, and `create_*_mesh` builders.

---

## 2. Install / import

```python
import tissue_forge as tf
from tissue_forge.models.vertex import solver as tfv
from tissue_forge.models.vertex.solver.mesh_types import BodyTypeSpec, SurfaceTypeSpec
```

> **One `tf.init()` per process.** `tf.init()` is a singleton — calling it twice in the same Python
> process hangs. Under pytest, initialise once in a session-scoped fixture and let all test meshes
> share the single universe.

---

## 3. Initialising the engine and solver

```python
L = 6.0                                   # box edge
tf.init(
    dim=[L, L, L],        # universe size; for a PERIODIC bulk the mesh box MUST equal this
    cutoff=1.0,           # MD neighbor cutoff (particle–particle); mesh forces ignore it
    dt=1e-3,              # timestep (see stability notes, §11)
    windowless=True,      # headless/batch; omit (or False) for the interactive renderer
    # bc='periodic',      # boundary conditions; periodic is the default
    # cells=[5, 5, 5],    # optional space-cell decomposition for the MD engine
)
tfv.init()                                # initialise the vertex-model solver (== MeshSolver.init)

mesh = tfv.MeshSolver.get_mesh()          # the live Mesh (static accessor)
```

**Dynamics are overdamped.** Vertex particles use `dynamics = OVERDAMPED`: each step the integrator
does `x += dt · F · (1/m)`. With the default vertex mass `m = 1` (which holds whenever cell
`density = 0`, the default), this is `x += dt · F` — i.e. **gradient descent on the energy**, with
mobility μ = 1. There is no inertia. This is why `dt` is the master stability knob (§11) and why the
active drive (§10) is implemented as a force.

⭐ **Periodic geometry.** For a space-filling periodic bulk, set:

```python
mesh.periodic_geometry = True
```

This makes the vertex-model geometry (volumes, areas, centroids, and the corresponding forces) use
**minimum-image** distances, so a cell straddling the box boundary is measured correctly instead of
spanning the whole box. The engine min-images at `Universe.dim`, so **your foam box must equal the
universe `dim`** (a sub-box is silently treated as box-spanning).

---

## 4. Quickstart — a minimal 3D two-type model

```python
import numpy as np
import tissue_forge as tf
from tissue_forge.models.vertex import solver as tfv
from tissue_forge.models.vertex.solver.mesh_types import BodyTypeSpec, SurfaceTypeSpec

tf.init(dim=[20, 20, 20], cutoff=3.0, dt=1e-3, windowless=True)
tfv.init()
mesh = tfv.MeshSolver.get_mesh()

# --- interface (surface) type: cell–cell faces ---
class Iface(SurfaceTypeSpec):
    pass

# --- two cell (body) types with target volume/area and heterotypic tension ---
class A(BodyTypeSpec):
    volume_lam = 10.0;  volume_val = 1.0          # K_V (V - V0)^2,  V0 = 1
    surface_area_lam = 1.0;  surface_area_val = 5.6   # K_A (A - A0)^2
    adhesion = {"A": 0.0, "B": 0.5}               # interfacial tension TO each type (see §8)

class B(BodyTypeSpec):
    volume_lam = 10.0;  volume_val = 1.0
    surface_area_lam = 1.0;  surface_area_val = 5.6
    adhesion = {"A": 0.5, "B": 0.0}

stype = Iface.get()
btA, btB = A.get(), B.get()
BodyTypeSpec.bind_adhesion([A, B])                # bind the pairwise Adhesion actors

# --- build a block of cells of type A, then flip ~half to B ---
bodies = tfv.create_hex3d_mesh(btA, stype,
                               tf.FVector3(10, 10, 10),   # center
                               4, 4, 4,                   # nx, ny, nz cells
                               1.0, 1.0)                  # in-plane spacing dr, layer spacing dz
bodies = [b for row in bodies for col in row for b in col]   # flatten the 3D array
tfv.MeshSolver.get().position_changed()
rng = np.random.default_rng(0)
for b in bodies:
    if rng.random() < 0.5:
        b.become(btB)

# --- enable the 3D T1 (reconnection) so cells can swap neighbours --- (fork only)
q = tfv.Quality()
q.stock_quality_operations = False     # don't run the degenerate 3D collapses
q.reconnect_length = 1e-3              # Δl_th: reconnect edges shorter than this
q.reconnect_hysteresis = 0.2          # anti-thrash gap
q.reconnect_interval = 10             # run the reconnection pass every 10 steps
q.reconnect_energy_gate = False
mesh.quality = q
mesh.periodic_geometry = True

# --- run ---
for step in range(20000):
    tf.step()
    if step % 1000 == 0:
        vols = [b.volume for b in bodies]
        print(step, "min/max vol", round(min(vols), 3), round(max(vols), 3))
```

> The `create_hex3d_mesh` block is **not periodic** (it has free surfaces). For a true periodic bulk
> (the setup used to reproduce 3DVertVor/Manning cell sorting), use a space-filling Voronoi packing —
> see the project helper `rnr.geometry.build_periodic_voronoi` and `rnr/scripts/sort_periodic_oracle.py`.

---

## 5. Defining cell types — `BodyTypeSpec` / `SurfaceTypeSpec`

You declare types as **subclasses** of the spec base classes; class attributes become the type's
energetics. Calling `.get()` registers the engine type (idempotent) and returns it.

```python
class A(BodyTypeSpec):
    volume_lam = 10.0
    volume_val = 1.0
    ...

btA = A.get()        # registers & returns the BodyType; safe to call repeatedly
```

### 3D cells — `BodyTypeSpec` fields

| Attribute | Meaning | Actor created |
|---|---|---|
| `name` | type name (defaults to the class name) | — |
| `density` | mass density; **leave 0** to keep unit vertex mass / μ = 1 | — |
| `volume_lam`, `volume_val` | `λ(V − V₀)²` volume elasticity | `VolumeConstraint` (3D) |
| `surface_area_lam`, `surface_area_val` | `λ(A − A₀)²` surface elasticity | `SurfaceAreaConstraint` |
| `body_force_comps` | constant force `F` on every vertex of the cell | `BodyForce` |
| `adhesion` | `{other_type_name: λ}` heterotypic interfacial tension (§8) | `Adhesion` (via `bind_adhesion`) |

A constraint is only created if **both** of its fields are set (e.g. `volume_lam` *and*
`volume_val`).

### 2D cells / faces — `SurfaceTypeSpec` fields

| Attribute | Meaning | Actor created |
|---|---|---|
| `name`, `style` | name; `{'color': 'CornflowerBlue', 'visible': True}` | — |
| `density` | mass density (2D only) | — |
| `edge_tension_lam`, `edge_tension_order` | `λLⁿ` edge tension (2D interfacial term) | `EdgeTension` (2D) |
| `perimeter_lam`, `perimeter_val` | `λ(L − L₀)²` perimeter elasticity (2D) | `PerimeterConstraint` (2D) |
| `surface_area_lam`, `surface_area_val` | `λ(A − A₀)²` area elasticity | `SurfaceAreaConstraint` |
| `normal_stress_mag` | force along the surface normal | `NormalStress` |
| `surface_traction_comps` | constant force on the surface | `SurfaceTraction` |
| `adhesion` | `{other_type_name: λ}` (2D: edge-based) | `Adhesion` (via `bind_adhesion`) |

**Which spec do I use?** In a **3D** model, *cells are bodies* → use `BodyTypeSpec`, and surfaces are
the cell–cell interfaces (often a single shared `SurfaceTypeSpec` with no extra energetics). In a
**2D** model, *cells are surfaces* → use `SurfaceTypeSpec` (this is what the bundled
`examples/cell_sorting.py` does, with `frozen_z`).

Change a cell's type at runtime with `body.become(other_type)` (or `surface.become(...)` in 2D).

---

## 6. Energetics — the actors

Every force/energy term is an **actor** bound to a type or to an individual object. The spec classes
above create and bind the common ones for you; you can also construct and bind actors manually.

| Actor | Constructor | Physical term | Dim | Bind to |
|---|---|---|---|---|
| `VolumeConstraint` | `(lam, constr)` | `λ(V − V₀)²` | 3D | body / body type |
| `SurfaceAreaConstraint` | `(lam, constr)` | `λ(A − A₀)²` | 2D/3D | body or surface (type or inst) |
| `Adhesion` | `(lam)` | interfacial tension `λ·A_shared` (3D) / `λ·L_shared` (2D) | 2D/3D | **type pair** |
| `EdgeTension` | `(lam, order=1)` | `λLⁿ` | 2D | surface / surface type |
| `PerimeterConstraint` | `(lam, constr)` | `λ(L − L₀)²` | 2D | surface / surface type |
| `BodyForce` | `(force: FVector3)` | constant `F·x` | 3D | body / body type |
| `SurfaceTraction` | `(force: FVector3)` | constant surface force | 2D/3D | surface / surface type |
| `NormalStress` | `(mag)` | `mag · n̂` | 2D/3D | surface / surface type |
| `ConvexPolygonConstraint` | `(lam=0.1)` | restore face toward convex | 2D | surface / surface type |
| `FlatSurfaceConstraint` | `(lam=0.1)` | restore vertices toward the face plane | 3D | surface / surface type |

> ⭐ Note: every `SurfaceType` auto-binds a `FlatSurfaceConstraint` (default `lam=0.1`). On the fork it
> is minimum-image-aware, so it is safe under `periodic_geometry`.

### Manual binding — the `bind` module

```python
from tissue_forge.models.vertex import solver as tfv

vc = tfv.VolumeConstraint(10.0, 1.0)
tfv.bind.body(vc, btA)             # bind to a BodyType (all bodies of that type) ...
tfv.bind.body(vc, some_body)       # ... or to a single Body / BodyHandle

et = tfv.EdgeTension(5.0, 1)
tfv.bind.surface(et, stype)        # bind to a SurfaceType or a single Surface/SurfaceHandle

adh = tfv.Adhesion(0.5)
tfv.bind.types(adh, btA, btB)      # type-PAIR actor (adhesion between two types)
```

`bind.body` / `bind.surface` accept either a *type* (applies to all members) or an *instance*.
`bind.types` is for pair actors (`Adhesion`).

---

## 7. The overdamped energy your model minimises

For a 3D two-type tissue the total energy is

```
E = Σ_cells [ K_V (V_i − V₀)²  +  K_A (A_i − A₀)² ]   +   Σ_{i≠j interfaces} σ_ij · A_ij
      └ VolumeConstraint ┘   └ SurfaceAreaConstraint ┘        └ Adhesion (heterotypic) ┘
```

and the dynamics is the overdamped descent `dx/dt = −∇E` (μ = 1). The shape index
`s₀ = A₀ / V₀^(2/3)` controls whether the tissue is solid-like or fluid-like (≈ 5.4–5.6 is fluid).
These three actors reproduce the 3DVertVor/Manning energy exactly; what they cannot do alone is let
cells change neighbours — that needs reconnection (§9).

---

## 8. Heterotypic adhesion (interfacial tension) — the sorting drive

This is the term that makes different cell types sort. **Two things commonly trip up new users:**

**(1) In TissueForge's vertex model, the `Adhesion` actor between two types behaves as an interfacial
*tension*, not a glue.** Its force is `−∇(λ · A_shared)` — the gradient of the *shared interface area*
(3D) or shared edge length (2D). A **positive** `λ` *penalises* contact area between the two types, so
they minimise their shared interface and **demix**. So the `adhesion` value is really the
heterotypic surface tension `σ_ij` of the 3DVertVor/Manning model:

- **homotypic** `λ = 0` (a cell pays nothing to touch its own type),
- **heterotypic** `λ = σ > 0` (an A–B interface costs energy → A and B separate),
- larger `σ` ⇒ sharper, faster sorting.

**(2) How to specify it.** Each type lists its interfacial λ *to every other type by name* in its
`adhesion` dict, then you call `bind_adhesion` with all the spec classes:

```python
class A(BodyTypeSpec):
    ...
    adhesion = {"A": 0.0, "B": 0.5}   # A–A tension 0, A–B tension 0.5

class B(BodyTypeSpec):
    ...
    adhesion = {"A": 0.5, "B": 0.0}   # symmetric

A.get(); B.get()                       # register the types FIRST
BodyTypeSpec.bind_adhesion([A, B])     # then bind the pairwise Adhesion actors
```

`bind_adhesion(specs)` walks every unordered type pair, reads the λ from the `adhesion` dicts, builds
one `Adhesion(λ)` per pair, and binds it with `bind.types`. List the value in *both* types' dicts for
clarity (the binder only needs it once). Call it **after** `.get()` on all types.

> The 2D bundled example (`examples/cell_sorting.py`) derives its adhesion from a *combination* of
> `EdgeTension` (per type) and `Adhesion` (per pair) — e.g. `adh_ab = 2·adh_hetr − (λ_a + λ_b)` — and
> those values can be negative. That EdgeTension-based 2D recipe does **not** port verbatim to 3D; in
> 3D the interfacial term lives on *surfaces*, so use a positive body-`Adhesion` λ as the tension σ_ij.

---

## 9. Reconnection — the 3D T1 (neighbour exchange) ⭐ fork only

Without a topology operator, an overdamped vertex model just relaxes to a jammed local minimum; cells
keep their neighbours forever and cannot sort. The **reconnection** operation (Okuda's reversible
network reconnection, the 3D analogue of the 2D T1) lets a short cell–cell edge swap into a small
triangular face (I→H) and back (H→I), exchanging which cells are neighbours.

It runs automatically inside the solver's per-step quality pass once you configure the `Quality`
operator:

```python
q = tfv.Quality()
q.stock_quality_operations = False   # turn OFF the degenerate 3D collapses (recommended)
q.reconnect_length     = 1e-3        # Δl_th: trigger on edges/faces shorter than this
q.reconnect_hysteresis = 0.2         # reconnect re-separates to Δl·(1+hyst); avoids immediate re-trigger
q.reconnect_interval   = 10          # run the reconnection pass every Nth step (oracle uses dtr = 10·dt)
q.reconnect_energy_gate = False      # if True, veto reconnections that raise interfacial energy
q.collision_2d         = False
mesh.quality = q                     # hand the operator to the mesh (replaces the default)
mesh.periodic_geometry = True
# ... then just tf.step(); reconnection happens inside each step.
```

**Choosing `reconnect_length` (Δl_th).** Keep it **small** relative to a typical edge so reconnections
are gentle and reversible (the reference model uses `1e-3`). The trigger is the *instantaneous* edge
length, so per-step vertex motion must stay below Δl_th (it does for overdamped relaxation and for the
active drive in §10; large thermal kicks can blow past it — see §11).

**`stock_quality_operations`.** Leave **off** for 3D sorting: the stock collapses (`BodyDemote` etc.)
are irreversible and can crash on a finite packing. Turn them on only if you specifically want
degenerate cleanup.

**2D T1.** In a 2D model you don't use `reconnect_length`; the 2D neighbour exchange is
`vertex_merge` + `edge_split`, configured by `quality.vertex_merge_distance` and
`quality.edge_split_distance` (see the bundled 2D example).

**Counting reconnections.** A reconnection changes the vertex count, so a cheap monitor is to watch
`mesh.num_vertices` between steps.

### Diagnostics (optional, for testing/inspection)

The fork exposes read-only and forced variants used by the test suite:

- `q.find_reconnection_candidates()` → list of dicts (each with `legal` + `veto_reason`) at the
  current `reconnect_length`.
- `q.analyze_i_reconnection(v10_id, v11_id)` / `q.analyze_h_reconnection(triangle_id)` → the
  neighbourhood walk + Okuda Condition-4 veto for a specific edge/face.
- `q.force_reconnect_i_to_h(v10_id, v11_id)` / `q.force_reconnect_h_to_i(triangle_id)` → force a
  single reconnection (for round-trip / unit tests).

---

## 10. Active self-propulsion (motility) ⭐ fork only

To keep the tissue stirring (so it escapes local minima and actually sorts) the recommended drive is
**active self-propulsion**, not thermal noise. Each **cell** carries a unit **director** that
undergoes rotational diffusion, and each vertex feels an active force equal to `v0 × ⟨director⟩`
averaged over its incident cells. Because the integrator is overdamped with μ = 1, this force
produces a per-step displacement `dt · v0 · ⟨director⟩` — a smooth, *sub-Δl_th* drift that stirs the
tissue without sabotaging the reconnection trigger.

```python
# after the bodies exist and Quality is configured:
tfv.MeshSolver.set_motility(v0=0.1, Dr=1.0, seed=7)
#   v0   : active speed (the model's "temperature"); 0 disables the drive (the default)
#   Dr   : rotational diffusion of the directors (default 1.0)
#   seed : RNG seed for reproducible directors (<0 keeps the current stream)

# inspect a cell's director (unit vector):
print(bodies[0].director)            # FVector3; {0,0,0} until motility is enabled

# read back the settings:
tfv.MeshSolver.get_motility_v0()     # -> 0.1
tfv.MeshSolver.get_motility_dr()     # -> 1.0
```

Then just `tf.step()` — the engine evolves the directors and applies the active force every step. No
per-step Python is needed. Keep `dt · v0` well below `reconnect_length` (e.g. `1e-3 · 0.1 = 0.1·Δl_th`).

> **Why not `tf.Force.random`?** A `tf.Force.random` (thermal, √dt) kick scales as √dt and is typically
> 10–45× Δl_th per step — it blows freshly-collapsing edges back over the trigger and *starves*
> reconnection. The active drive is dt-scaled (ballistic) and sub-Δl_th, so reconnection proceeds with
> no clamp. Use `tf.Force.random` only for long-time-diffusion studies, not as the sorting stir.

---

## 11. Running, reading state, and a custom step loop

**Run headless (batch):**
```python
for step in range(n_steps):
    tf.step()                       # advance one dt
    # ... read state, log metrics ...
```
`tf.step(until)` advances by a time interval; `tf.run()` runs the event loop with the renderer.

**Run interactively (with the renderer):** omit `windowless=True` in `tf.init`, then
`tf.show()` (blocks, opens a window). Camera helpers: `tf.system.camera_view_top()`,
`tf.system.camera_zoom_to(...)`.

**Read cell state** (handles stay valid across steps):
```python
b.volume        # enclosed volume
b.area          # total surface area
b.centroid      # FVector3 center of mass
b.surfaces      # bounding surfaces
b.vertices      # boundary vertices
b.connected_bodies   # cells in contact
b.director      # active-motility director (⭐)
```

**Iterate the live mesh** (e.g. for custom analysis / a Python-side event), robust to topology
changes:
```python
verts = [v for v in (mesh.get_vertex(i) for i in range(mesh.size_vertices))
         if v is not None and v.id >= 0]
```
`mesh.num_vertices` / `mesh.size_vertices` (and the `_bodies`/`_surfaces` analogues) give counts.

> ⚠️ **Re-fetch after topology changes.** A reconnection (or `become`/`destroy`) may invalidate cached
> pointers. A single quality pass can do paired I→H/H→I that *net* to zero vertex-count change, so
> don't use `num_vertices` as a staleness signal — re-fetch handles from the mesh each time you need
> them.

---

## 12. Stability & gotchas (the short list that saves hours)

- **`dt` is the master stability lever.** Overdamped descent overshoots near-collapsed faces at coarse
  `dt`, flipping a cell's winding (negative volume → runaway). If cells invert/inflate, **reduce
  `dt`**. The fork's orientation repair (automatic) absorbs transient flips, but small `dt` is still
  the first thing to try.
- **`density = 0` ⇒ vertex mass = 1 ⇒ μ = 1.** Setting a nonzero `density` changes the mobility and
  breaks the clean `dx = dt·F` mapping (and the §10 displacement calibration). Leave it 0 unless you
  know you want mass weighting.
- **Periodic box must equal `Universe.dim`.** The engine min-images at `dim`; a foam built in a
  sub-box is silently box-spanning. Build the packing at exactly `[L,L,L] == dim` and set
  `mesh.periodic_geometry = True`.
- **One `tf.init()` per process** (see §2).
- **Register types before `bind_adhesion`.** Call `.get()` on every spec, *then*
  `bind_adhesion([...])`.
- **Keep `reconnect_length` small** and per-step motion below it (§9, §10).
- **Disable stock 3D collapses** (`stock_quality_operations = False`) for sorting runs.
- **`become()` doesn't re-render** an existing surface's colour in some builds; set styles at type
  registration if appearance matters.

---

## 13. Complete worked example — periodic 3D two-type sorting

This mirrors the validated 3DVertVor/Manning reproduction (heterotypic tension + native reconnection +
active drive) at small scale. It uses the project helper `rnr.geometry.build_periodic_voronoi` for the
space-filling periodic packing.

```python
import os, sys
import numpy as np
import tissue_forge as tf
from tissue_forge.models.vertex import solver as tfv
from tissue_forge.models.vertex.solver.mesh_types import BodyTypeSpec, SurfaceTypeSpec

M, SIGMA, V0, A0, SEED = 6, 0.5, 1.0, 5.6, 7
L = float(M)

tf.init(windowless=True, dim=[L, L, L], cutoff=1.9, dt=1e-3)
tfv.init()
mesh = tfv.MeshSolver.get_mesh()
mesh.quality = None                 # we'll install a custom Quality below
mesh.periodic_geometry = True

from rnr.geometry import build_periodic_voronoi   # project helper (space-filling periodic foam)

class Iface(SurfaceTypeSpec):
    pass

class A(BodyTypeSpec):
    volume_lam = 10.0; volume_val = V0
    surface_area_lam = 1.0; surface_area_val = A0
    adhesion = {"A": 0.0, "B": SIGMA}

class B(BodyTypeSpec):
    volume_lam = 10.0; volume_val = V0
    surface_area_lam = 1.0; surface_area_val = A0
    adhesion = {"A": SIGMA, "B": 0.0}

stype, btA, btB = Iface.get(), A.get(), B.get()
BodyTypeSpec.bind_adhesion([A, B])

rng = np.random.default_rng(SEED)
seeds = (rng.random((M ** 3, 3)) * L).tolist()
bodies, _, stats = build_periodic_voronoi(seeds, [[0, L]] * 3, btA, stype)
tfv.MeshSolver.get().position_changed()

# random 50/50 mixed initial condition
rng2 = np.random.default_rng(SEED + 1)
for b in bodies:
    if rng2.random() < 0.5:
        b.become(btB)

# reconnection (3D T1)
q = tfv.Quality()
q.stock_quality_operations = False
q.reconnect_length = 1e-3
q.reconnect_hysteresis = 0.2
q.reconnect_energy_gate = False
q.reconnect_interval = 10
q.collision_2d = False
mesh.quality = q
mesh.periodic_geometry = True

# native active drive (the stir)
tfv.MeshSolver.set_motility(0.1, 1.0, SEED + 2)

# run + a simple sorting readout (heterotypic interface area fraction should DROP)
def het_area_fraction():
    het = tot = 0.0
    for b in bodies:
        for s in b.surfaces:
            bb = s.bodies
            if len(bb) == 2:
                a = s.area
                tot += a
                if bb[0].type().id != bb[1].type().id:
                    het += a
    return het / tot if tot else 0.0

print("hetA(0) =", round(het_area_fraction(), 4))
for step in range(1, 40001):
    tf.step()
    if step % 5000 == 0:
        vols = [b.volume for b in bodies]
        print(f"step {step}: hetA={het_area_fraction():.4f} "
              f"min_vol={min(vols):.3f} max_vol={max(vols):.3f}")
```

A correct run is **stable** (volumes stay bounded — no cell inverts or inflates) and the heterotypic
interface fraction **decreases** as like cells cluster. In the project, `rnr/metrics.py` provides a
ready `contact_summary(...)` / `demixing_index(...)`; `rnr/scripts/sort_periodic_oracle.py` is the full
harness (`NOISE_MODEL=native` runs exactly this).

---

## 14. Quick API index

**Init / solver**
`tf.init(dim=, cutoff=, dt=, windowless=, bc=, cells=)` · `tfv.init()` ·
`tfv.MeshSolver.get_mesh()` · `tfv.MeshSolver.get().position_changed()` · `tf.step()` · `tf.show()`

**Types** (`from ...mesh_types import BodyTypeSpec, SurfaceTypeSpec`)
subclass → set fields → `.get()` → `BodyTypeSpec.bind_adhesion([...])` · `body.become(type)`

**Actors** `tfv.VolumeConstraint(lam,constr)` · `tfv.SurfaceAreaConstraint(lam,constr)` ·
`tfv.Adhesion(lam)` · `tfv.EdgeTension(lam,order)` · `tfv.PerimeterConstraint(lam,constr)` ·
`tfv.BodyForce(F)` · `tfv.SurfaceTraction(F)` · `tfv.NormalStress(mag)`

**Binding** `tfv.bind.body(actor, body_or_type)` · `tfv.bind.surface(actor, surf_or_type)` ·
`tfv.bind.types(pair_actor, typeA, typeB)`

**Mesh builders** `tfv.create_hex3d_mesh(btype, stype, center, nx, ny, nz, dr, dz, ax1, ax2)` (3D) ·
`tfv.create_plpd_mesh(btype, stype, center, nx, ny, nz, sx, sy, sz, ax1, ax2)` (3D) ·
`tfv.create_hex2d_mesh(stype, center, nx, ny, dr, ax1, ax2)` (2D) ·
`tfv.create_quad_mesh(stype, center, nx, ny, sx, sy, ax1, ax2)` (2D)

**Mesh / quality** `mesh.periodic_geometry` ⭐ · `mesh.quality = tfv.Quality()` ·
`q.reconnect_length` ⭐ · `q.reconnect_hysteresis` ⭐ · `q.reconnect_interval` ⭐ ·
`q.reconnect_energy_gate` ⭐ · `q.stock_quality_operations` · `q.collision_2d` ·
`q.vertex_merge_distance` · `q.edge_split_distance` · `mesh.num_vertices`

**Active drive** ⭐ `tfv.MeshSolver.set_motility(v0, Dr=1.0, seed=-1)` ·
`tfv.MeshSolver.get_motility_v0()` · `tfv.MeshSolver.get_motility_dr()` · `body.director`

**Reading state** `body.volume` · `body.area` · `body.centroid` · `body.surfaces` · `body.vertices` ·
`body.connected_bodies` · `surface.area` · `surface.normal` · `surface.bodies` · `vertex.position` ·
`vertex.set_position(tf.FVector3(...))`

---

## 15. Where to look in the source / project

- Bundled examples: `tissue-forge/wraps/python/models/vertex/solver/examples/`
  (`cell_sorting.py` (2D), `cell_migration.py`, `cell_splitting.py`, `capillary_loop.py`).
- Type specs & binding: `.../vertex/solver/mesh_types.py`, `.../vertex/solver/bind.py`.
- Engine classes: `tissue-forge/source/models/vertex/solver/tf{Body,Surface,Vertex,Mesh,MeshSolver,MeshQuality}.{h,cpp}`.
- This project's RNR work, the native reconnection/motility rationale, and the cell-sorting harness:
  `rnr/PORTING_NOTES.md` (esp. §6f energetics, §6g periodic geometry, §6k orientation repair,
  §6n/§6o the active drive), `rnr/scripts/`, `rnr/metrics.py`, and the repo `CLAUDE.md`.

## 16. Citations

- TissueForge: Sego et al. (2023), *Sci. Rep.* 13:17886.
- RNR / I↔H reconnection: Okuda et al. (2013), *Biomech. Model. Mechanobiol.* 12:627–644.
- 3D vertex cell sorting (heterotypic tension): Zhang & Schwarz, *Phys. Rev. Research* 4:043148
  (arXiv:2204.07081); Manning-lab 3DVertVor.
- 2D cell-sorting benchmark (bundled example): Osborne et al. (2017), *PLoS Comput. Biol.* 13(2):e1005387.
