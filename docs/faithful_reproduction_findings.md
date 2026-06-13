# Faithful 3DVertVor / Manning-2024 reproduction — findings & the gap (2026-06-01)

Outcome of the session that started from `docs/phaseG_kickoff_plan.md` ("add a σ·A heterotypic
tension actor") and the question **"don't we need to rework how per-cell heterotypic adhesion is
implemented in TissueForge to replicate the PDFs / 3DVertVor?"**

**Short answer: no — the heterotypic adhesion does NOT need reworking.** The paper's model maps 1:1
onto TissueForge's *existing* actors, and TF's body-`Adhesion` *force* already IS the faithful
area-based σ·A interfacial tension. The real gap to the paper is the **reconnection regime**, not the
energetics. Adding the paper's thermal noise (the one ingredient we lacked) does **not** unlock
deeper sorting — it destabilizes the reconnection. This document records the evidence so the next
(architecturally significant) step can be chosen deliberately.

Companion records: `rnr/PORTING_NOTES.md` §6f (verified API/calibration detail), memories
`adhesion-force-is-already-area-tension`, `thermal-noise-destabilizes-reconnection`.

---

## 1. The paper's model (verified against both sources)

**Manning 2024 (PLOS Comp Biol), Eq. 3, p.5** and **3DVertVor `Energy/Interface.cpp` + `Energy/Volume.cpp`**:

```
E = Σ_cells [ K_A (A_i − A0)²  +  K_V (V_i − V0)² ]   +   Σ_{i≠j type} σ_ij · A_ij
```

- `K_A (A−A0)²` — per-cell **surface-area elasticity** (A = cell total surface area).
- `K_V (V−V0)²` — per-cell **volume elasticity** (`Volume.cpp`, kv=10).
- `σ_ij · A_ij` — **area-based** heterotypic interfacial tension on the shared interface AREA
  (Manning p.5 explicitly: "the additional edge cost is for the shared surface area ... rather than
  edges"); homotypic σ = 0.
- Dynamics: overdamped `dr/dt = μ∇E + η`, **μ = 1**, dt = 0.01, **white thermal noise kT = 0.1**,
  dimensionless shape index **s0 = A0/V0^(2/3) = 5.6** (fluid regime).
- `Interface.cpp:updateTension` builds each interface polygon's tension as `2(s_cell−s0) + σ_ij`
  (the surface-elastic part + the σ part) and `updatePolygonForces` applies `force = tension·∇A`.

## 2. It maps 1:1 onto EXISTING TissueForge actors

| paper / oracle term | TF actor | faithful? |
|---|---|---|
| `K_V (V−V0)²` | `VolumeConstraint` | ✅ energy **and** force |
| `K_A (A−A0)²` | `SurfaceAreaConstraint` | ✅ energy **and** force |
| `σ_ij · A_ij` (area-based, heterotypic) | **body-`Adhesion`** | ✅ **force**; energy() is perimeter but unused |

**The load-bearing finding — Adhesion's *force* is already area tension.** `Adhesion_force_Body`
(`tfAdhesion.cpp:68-104`) = `0.25·λ · Σ_{het surfaces of v} ftotal_loop`, where `ftotal_loop` is the
*identical* area-gradient cross-product loop used by `SurfaceAreaConstraint::force`
(`tfSurfaceAreaConstraint.cpp:38-65`). I derived and numerically verified (cos ≈ 0.9997) that
`ftotal_loop = −2·∇A` (TF's surface area uses centroid = vertex mean, `tfSurface.cpp:1061-1102`).
The solver sums `force(b1,v)+force(b2,v)` for each shared surface (`tfMeshSolver.cpp` VertexForce
iterates `v->getBodies()`), so per heterotypic surface the total force is

```
2 · 0.25·λ · (−2∇A_s)  =  −λ·∇A_s   ⇒   total = −∇(λ·A_het)
```

i.e. **exactly the area-tension force `−∇(σ·A_het)`** with σ = λ. The perimeter form
(`Adhesion_energy_Body = 0.5λΣ|edge|`) lives only in `energy()`, which the integrator **never calls**
— actor `energy()` is consumed only by C-API getters (`wraps/C/.../tfCMeshObj.cpp:112,159`). The
vertex dynamics is purely force-based, so the perimeter energy is inert.

**Consequence for the kickoff plan's Phase G.** A new σ·A actor whose force mirrors
`Adhesion_force_Body` is **force-identical** to `Adhesion(λ=σ)` and cannot change the sorting
dynamics. It is worth building ONLY for a *consistent* `energy()` (needed for energy reporting or an
energy-gated reconnection variant — and the paper's reconnection is purely geometric, so not for
faithfulness). **Deferred** — it is not the gap.

## 3. The thermal noise — `tf.Force.random` is faithful and usable

`Force.random(std, mean, duration)` → a `Gaussian` force (`tfForce.cpp:211-243`): isotropic random
direction, magnitude ~ N(mean, std), held `ceil(duration/dt)` steps then resampled.

- **Reaches vertices:** `MeshSolver::preStepJoin` adds mesh forces into the same particle buffer the
  engine integrates (`p->f += meshforce`, `tfMeshSolver.cpp:500-504`), so a force bound to
  `MeshParticleType` sums with the mesh forces. (Verified: free vertices diffuse under it.)
- **Calibration (measured on free vertices, this build):** mobility **μ = 1.000** (= the paper's μ
  exactly), diffusion **D = μ·kT, D ≈ 1.63e‑5·std²** (white). So request kT via `std = √(kT/1.63e‑5)`.
- **Faithful white noise config:** `duration = dt` (resample every step ⇒ delta-correlated) +
  `mean = 0`. (`duration ≫ dt` ⇒ colored/active-motility noise — the 3DVertVor checkout's regime,
  where the thermal line `Run.cpp:1344` is commented out for a `motility_` line.)
- **Caveat:** TF draws a Gaussian *magnitude* × isotropic *direction*, not three per-axis Gaussians.
  Same effective temperature / long-time diffusion; the microscopic per-kick shape differs. Exact
  per-axis Gaussian would need a `CustomForce`.

## 4. The experiment & result — noise does NOT unlock sorting

Harness: `rnr/scripts/sort_faithful_3dvertvor.py` (`pixi run sort-faithful-3dv`). Existing actors at
s0=5.6, σ = Adhesion λ_het, native geometric reconnection (energy gate OFF), centred cluster,
hardened stability (min_vol>0 AND max_vol<3·V0 AND no fling), + `tf.Force.random` white noise.

| regime | stable? | D plateau | het-area | reconnections |
|---|---|---|---|---|
| **σ=1, kT=0 (athermal)** | ✅ | **−0.065** | **0.375** | 2844 |
| σ=1, kT=0.02 | ❌ | −0.064 | 0.388 | 3054 |
| σ=1, kT=0.05 | ❌ | −0.057 | 0.396 | 3403 |
| σ=1, kT=0.1 (paper's kT) | ❌ | −0.031 | 0.429 | 4185 |
| σ=0.1, kT=0 (weak, athermal) | ✅ | −0.043 (FROZEN) | 0.462 | **0** |
| σ=0.1, kT=0.05, INT=10 | ❌ | −0.034 | 0.468 | 3376 |
| σ=0.1, kT=0.05, INT=30 (throttled) | ✅ | −0.057 | 0.479 | 1616 |

- **Strong σ:** athermal is the best and only-good run; noise → more reconnection churn → cells
  invert → *shallower* D (monotonic in kT).
- **Weak σ (paper's σ/K_A regime):** athermal is **frozen** (weak tension never shrinks het faces to
  the trigger); noise unfreezes it but the reconnections are essentially **random** (het-area rises),
  and only heavy throttling keeps it stable — with marginal D and flat het-area.

**No tested (σ, kT, throttle) point beats the athermal D = −0.065.** Noise is purely a destabilizer
here. (D = −0.065 is itself shallow but consistent with the periodic oracle, which also plateaus
~0.06–0.10 at matched σ — memory `oracle-comparison-ceiling-physical`.)

## 5. Diagnosis — the binding constraint is the RECONNECTION

Our reconnections displace vertices by ~`Δl_th·(1+hyst) ≈ 0.48` (≈70% of the 0.707 equilibrium
edge) **per event**; the oracle places new vertices ~`Δl_th = 1e‑3` apart (negligible). Root-cause
chain:

```
finite cluster + dt=1e-4  →  interior edges only collapse to ~0.3–0.45 (PORTING_NOTES §6e)
                          →  Δl_th MUST be ~0.4 to ever trigger
                          →  each reconnection violently perturbs the local mesh
                          →  noise's extra reconnection churn + jitter  →  cell inverts (volume → 0⁻)
```

The 3DVertVor oracle avoids all of this with **three stabilizers TF's vertex mesh lacks**
(PORTING_NOTES §6d, read-only against the GPL code, nothing copied):
1. **Periodic minimum-image mesh** — distances/forces/centroids wrapped to the nearest image; a
   vertex crossing a face is computed correctly across the boundary (no boundary jamming, no wrap
   fling). TF's vertex mesh has none.
2. **Quasi-static, tiny-Δl_th reconnection** — Δl_th = 1e‑3, checked every dtr = 10·dt, new vertices
   placed ±1e‑3 ⇒ each reconnection is a negligible perturbation.
3. **Orientation/volume repair** — `if(V<0){ V=|V|; flip polygon directions; }`, used consistently
   in volume *and* force ⇒ a transient sign-flip recovers instead of running away.

## 6. Where this leaves the prototype

- **Achievable ceiling (this prototype):** athermal, finite-cluster, existing-actor model demixes to
  **D ≈ −0.065** (het-area 0.48 → 0.37), stable to 10k steps. Artifact:
  `rnr/exports/sort_faithful_KT0_S1.{csv,png}`.
- **Reproducing the paper's sorting requires fixing the reconnection regime, not the energetics.**
  The next architecturally-significant investments (each was surfaced/partly explored; user to
  choose):
  - **(A) Periodic minimum-image vertex mesh** — the deepest unblock + most faithful; biggest lift
    (mesh geometry + force/centroid calcs across the vertex solver).
  - **(B) Gentler reconnection** — decouple placement scale from the trigger Δl_th (place new
    vertices ~1e‑3 apart regardless of when it fires) so each reconnection barely perturbs the mesh;
    medium native C++ change, directly targets the diagnosed root cause.
  - **(C) Volume/orientation robustness** — retry the oracle's abs+flip repair (§6d#3) so perturbed
    near-inversions recover; partially explored (§6c, mixed — converted eversion to inflation).
  - **(D) σ·A energy-faithful actor** — only if a faithful *energy* (reporting / energy-gated
    reconnection) is wanted; does not change sorting dynamics.

## 7. What was built this session

- `rnr/scripts/phase_f_baseline.py` (`pixi run phase-f-baseline`) — long-run baseline recorder
  (exact reconnection-event counting via `mesh.num_vertices`); showed the §6e stable config erodes /
  inverts a cell ~step 12k over a long run.
- `rnr/scripts/sort_faithful_3dvertvor.py` (`pixi run sort-faithful-3dv`) — the faithful harness
  (existing actors + calibrated `tf.Force.random` noise). The vehicle for §4's result; reusable for
  any future (σ, kT, regime) study once the reconnection is fixed.
- No engine (fork) changes. No GPL code copied — the oracle was read as a behavioral reference only.
