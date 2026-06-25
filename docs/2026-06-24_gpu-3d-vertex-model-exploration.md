# GPU acceleration of 3D vertex models (incl. RNR) — options & recommendation

*2026-06-24. Design exploration. Inputs: `cellGPU/`, `VertAX/`, `gpu_reference_papers/{cell_gpu_chaste,vertax}_preprint.pdf`, and the TissueForge fork internals (`tissue-forge/source/`). No code written yet — this scopes the work.*

---

## TL;DR

There is a clean, defensible literature gap: **nobody has run a 3D vertex model with
topology changes (T1 / reversible network reconnection) on the GPU.** cellGPU is 2D;
the Chaste/FLAME-GPU work is 3D but topology-free (overlapping spheres); VertAX is 2D and
isn't a dynamics engine. Filling that gap is the novel contribution.

The single hardest technical nut is **not** the forces — it's doing the **I↔H reconnection
in parallel on a GPU-resident mesh whose element counts change** (3D RNR creates/destroys
vertices and faces; the 2D T1 does not). cellGPU already solved the *conflict-free parallel
topology change* half (an atomic-reservation maximal-independent-set protocol). The new half
we'd add is *parallel allocation/compaction of created/destroyed mesh elements* plus the
Okuda reversible-placement and Condition-4 vetoes you already wrote on CPU.

**Recommended program (staged, de-risked):**

- **Stage 0 — SoA/CSR mesh + sync.** Replace the pointer-graph mesh with an index-based
  (CSR) Structure-of-Arrays mirror; prove it round-trips the existing mesh. This is the
  foundational decision everything else rests on, and it's where TissueForge fights us.
- **Stage 1 — GPU forces + geometry + integration, RNR stays on CPU.** The compute-dominant,
  embarrassingly-parallel phases go to the GPU; the (throttled, sparse) RNR stays serial on
  CPU and re-syncs the mirror only when topology actually changed. Big speedup, fully
  validatable against the current CPU oracle, low risk. This alone is "GPU-accelerated 3D
  vertex model" — publishable, and it de-risks the data structure + force kernels.
- **Stage 2 — GPU-native RNR (the novel result).** Port cellGPU's independent-set protocol
  to 3D I↔H, adding a parallel slot allocator for the count-changing surgery and your
  Appendix-1 placement + Condition-4 guards as per-candidate vetoes. This removes the last
  serial phase and is the literature-novel contribution.
- **Vehicle:** prototype the risky Stage-2 algorithm in **NVIDIA Warp** (Python, ergonomic
  atomics + preallocated pools, differentiable, reuses your `rnr/` tooling and visualizers —
  cheap iteration). Once the algorithm is validated, port it to **hand-written CUDA inside a
  TissueForge fork**, reusing the engine's existing `engine_flag_cuda` device-buffer/runner
  infrastructure — consistent with the native-RNR direction you've already committed to.
  (If you'd rather stay single-vehicle and minimize effort, an end-to-end Warp engine over
  the TissueForge mesh is itself a viable, novel first.)

The rest of this doc justifies each of those calls.

---

## 1. The literature gap (why this is novel)

Place the three references on three axes — **dimensionality**, **does topology change run on
the GPU**, and **dynamics vs. optimization**:

| Reference | Dim | Topology op | On GPU? | What it is | Vehicle |
|---|---|---|---|---|---|
| **cellGPU** (Sussman 2017, CPC 219:400; arXiv:1702.02939) | 2D | T1 edge flip (**count-preserving**) | **Yes — fully** | High-throughput dynamics (active vertex + self-propelled Voronoi) | Hand-written CUDA, SoA |
| **Chaste + FLAME GPU 2** (Leach/Heywood/Fletcher/Richmond, bioRxiv 2026.01.13.699201) | 2D & 3D | **none** (overlapping-spheres cell-centre) | Forces only; topology N/A | Hybrid GPU/CPU port of a CPU framework | FLAME GPU 2 (agent DSL → CUDA) |
| **VertAX** (Pasqui et al., arXiv:2604.06896) | 2D | T1 (count-preserving), procedural, **outside autodiff** | "Whatever XLA gives" — no custom kernels; ~20–100 cells | Differentiable inverse design / inference (not a dynamics engine) | JAX/XLA |
| **← our target →** | **3D** | **I↔H / RNR (count-CHANGING)** | **Yes** | High-throughput dynamics **+** (optionally) differentiable | TissueForge fork (CUDA) and/or Warp |

The empty cell — **3D + count-changing topology + on the GPU** — is the contribution. Tellingly,
the Chaste/FLAME-GPU paper itself cites cellGPU as the vertex-model GPU leader and does *not*
attempt a vertex model; it picked overlapping spheres precisely because that model has **no
topology operations** to parallelize.

---

## 2. What each reference actually teaches (and the trap in each)

### cellGPU — the crown jewel is the conflict-resolution protocol, not the forces

- **Reusable wholesale:**
  1. **SoA + a copy-on-access GPU array** (`GPUArray`, HOOMD-style: host/device dual buffers,
     `access_mode::overwrite` skips the upload). Dimension-agnostic.
  2. **Two-pass geometry→force** with a per-incidence stash: one thread/cell computes
     geometry and pre-bakes each vertex's local neighbors into a "force-set" slot; a stateless
     force kernel then reduces per vertex (`vertexModelBase.cu`, `vertexQuadraticEnergy.cu`).
     Generalizes directly to 3D (thread/cell → volume + face areas; force-set per (vertex,face)).
  3. **THE key idea — parallel topology change without corrupting connectivity.** cellGPU does
     T1s *on the GPU* via a GPU-resident **maximal-independent-set** loop
     (`vertexModelBase.cpp:830-895` + `.cu:172`): each candidate flip touches a *bounded set of
     cells* (4 in 2D); a one-thread-per-cell kernel uses `atomicExch` to **reserve all cells a
     flip needs**, commits only flips that won the reservation uncontended, applies that
     conflict-free batch in parallel, then **iterates** until no candidates remain (host reads a
     2-int "any-left?" flag per round). This is the answer to "how do many T1/I↔H happen at once
     without races," and it ports straight to 3D.
- **The trap (the 2D crutches that break in 3D):**
  - *Trivalence.* Every 2D vertex is exactly 3-valent → fixed stride-3 arrays
    (`Nvertices = 2·Ncells`, `3*i+j` addressing). 3D interior vertices are **4-valent** with
    variable-size incident faces/cells → **everything becomes ragged; the stride-3 shortcut is gone.**
  - *Count-preserving T1.* cellGPU's T1 *rotates and rescales an existing edge* — it never
    creates or destroys a vertex/face. The 3D I↔H **deletes an edge and births a triangular face
    (or vice-versa)** — element counts change. cellGPU's array surgery therefore does **not** cover
    the hard part; its "grow list" only widens a cell's row, it never allocates new vertices/faces
    in parallel.
  - *Crude placement.* "Rotate the edge 90°, double its length," guarded only by a no-triangle
    rule. Okuda reversibility (ΔU = O(Δl_th), Appendix-1 placement) is far more delicate.

### Chaste + FLAME GPU 2 — the hybrid "wrap a CPU framework" datapoint, and the Amdahl warning

- **What they did:** offloaded **only the force calculation + integration** of the *overlapping-spheres
  cell-centre* model to the GPU via FLAME GPU 2 (an agent-based DSL that emits CUDA from a C-like
  API — no hand-tuned kernels). Force + integrate both run on-device with no per-step round trip;
  everything else (cell cycle, division, "UpdateCellLocationsAndTopology") stays on CPU.
- **The numbers (V100):** **93.6× on force calc (3D)** but only **3.72× overall.** That gap *is* the
  lesson: forces were ~80% of runtime, so even infinite force speedup is **Amdahl-capped at ~4–5×**;
  CPU work (cell cycle + the AoS↔SoA "data translation") then dominates 70–90% of runtime.
- **Two lessons we must internalize:**
  - *AoS→SoA or pay forever.* They explicitly state: "If Chaste used a structure-of-arrays
    architecture rather than array-of-structures, much of the data-translation step could be
    avoided." **This is exactly TissueForge's situation** (pointer-graph AoS mesh) — Stage 0 below.
  - *Hybrid is a real, low-risk option*, but its ceiling is set by whatever you leave on the CPU.
    For us the serial remainder is the **RNR + geometry recache + transfer**, which — crucially —
    is *throttled* (RNR runs every Nth step, not every step) and sparse, so our Amdahl ceiling is
    much friendlier than Chaste's every-step cell-cycle case.
- **The trap:** FLAME GPU 2's spatially-partitioned agent + message-passing model fits
  *fixed-radius neighbor sums between point agents* beautifully — and fits a **mutating polygonal
  mesh with explicit shared faces and topology surgery** poorly. It was the right tool *because*
  overlapping spheres has no topology. It is **not** an obvious fit for our vertex model. (More in §6C.)

### VertAX — borrow the methodology, not the engine

- **Genuinely useful ideas:**
  - *Autodiff forces.* `jax.grad(energy)` gives vertex forces — **you never hand-derive the 3D
    force Jacobian** (volume + surface + heterotypic tension). Big ergonomic win; differentiability
    also buys inverse design / parameter inference later.
  - *Equilibrium Propagation (EP).* A way to get parameter gradients out of a **non-differentiable**
    forward simulator via repeated nudged relaxations — i.e. you could make a CUDA/Warp 3D engine
    "inverse-designable" *without* rewriting it in an autodiff framework. Worth remembering for a
    later phase.
  - *Topological loss (IAS / optimal-transport on the dual cell-adjacency graph)* to steer
    optimization *across* discrete reconnection barriers without differentiating the swap.
  - *Energy-gated T1 acceptance* (accept iff it lowers energy) — conceptually your Okuda Condition-3.
- **The trap (why this is not a path to a large dynamic 3D engine):**
  - It is an **optimizer/inference tool, not a time-stepped dynamics integrator** — no dt, no
    active drive, no thermostat. The "time loop" is energy minimization to equilibrium. (Your
    project already found active self-propulsion vs. thermal noise to be load-bearing; VertAX has
    neither.)
  - Its fixed-shape, no-re-jit trick **works only because a 2D T1 conserves element counts** → the
    JAX tables never resize. **3D I↔H changes counts**, so a JAX port must either pad every table to
    a worst-case max (ghost elements taxing every kernel forever) or re-jit on each topology change
    (the very thing JAX-style avoids). Plus the T1 sweep is a **serial `fori_loop`** (O(E) dependent
    steps) — fine at 20–100 cells, a throughput killer at scale. **JAX is the wrong substrate for
    high-throughput, count-changing 3D dynamics.**

---

## 3. The 3D problem, decomposed

A TissueForge vertex step (from the engine) has five phases. Their GPU-friendliness differs sharply:

| Phase | What it does | Parallelism | GPU difficulty |
|---|---|---|---|
| 1. Director update (active drive) | per-cell rotational diffusion | per-cell, trivial | **Easy** |
| 2. **Force computation** | per-vertex gather: volume grad, surface-area grad, tension | per-vertex / per-incidence | **Easy–medium** (the compute-dominant win) |
| 3. Integration | overdamped forward Euler `x += dt·f/μ` | per-vertex (per-particle) | **Easy** (already on the MD particle array, which already has a CUDA path) |
| 4. Geometry recache | per-surface/body centroid, area, **volume**, normals | per-surface/body | **Easy–medium** |
| 5. **RNR / `doQuality()`** | detect short edges → I↔H surgery on the mesh | detection parallel; **mutation serial** | **Hard — the crux** |

Two cruxes, and only two:

1. **The data structure.** 3D meshes are ragged at *every* level (vertex→{4 cells, variable
   faces}; face→variable polygon; cell→variable polyhedron). The right GPU representation is
   **CSR-style offsets + flat index arrays** (the proper generalization of cellGPU's "padded
   rectangle + count"). Index-based, not pointer-based. This decision dictates every kernel.

2. **Count-changing topology surgery in parallel.** Phase 5 is where novelty lives. Detection of
   trigger edges is a trivial parallel scan. The conflict-free *scheduling* of simultaneous ops is
   solved by cellGPU's independent-set protocol. The **new** piece is that I↔H **allocates and frees
   vertices/faces** — so the SoA arrays need a **parallel bump-allocator** (atomic counter into a
   preallocated pool) for births and a **free-list + periodic compaction** for deaths, all while the
   independent-set guarantee keeps concurrent ops from racing on shared slots.

Everything else (phases 1–4) is standard GPU particle/mesh work.

---

## 4. The TissueForge starting point — obstacle and head-start

**Obstacle — the mesh is an AoS pointer-graph.** `Vertex` holds `std::vector<Surface*>`; `Surface`
holds `std::vector<Vertex*>` + `Body* b1,b2`; `Body` holds `std::vector<Surface*>`. Connectivity is
heap pointers, not indices. This is the GPU-hostile substrate the Chaste paper warns about; **Stage 0
(an index-based CSR mirror) is unavoidable** for real performance.

**Head-start — TissueForge already has a CUDA path, just not for the vertex solver.** The molecular-
dynamics core (`mdcore/`) has the full pattern we'd imitate: a runtime flag (`engine_flag_cuda`),
persistent device SoA buffers, a runner that launches kernels and copies forces back
(`tfRunner_cuda.cu`, `tfEngine_cuda.cu`), gated and **single-precision-forced** under CUDA. Two
consequences:

- The **vertex *integration* already rides the MD particle array** (overdamped forward Euler,
  `tf_engine_advance.cpp:388`), which **already has a CUDA advance**. Phase 3 is nearly free.
- The **vertex *force* actors are CPU-only** and added into `p->f` between force-prep and advance.
  Phase 2 is the real porting work — and it must read connectivity, hence Stage 0 first.
- The vertex solver subtree has **zero** CUDA today — Stage 1/2 are greenfield kernels, but they
  plug into a proven on/off harness rather than inventing one.

**The native RNR is ~370 lines of serial, in-place pointer surgery** (`tfMeshQuality.cpp`), already
split check-half / mutate-half, throttled by `reconnectInterval`. That seriality is by design (the
code notes a parallel chain-builder races into cycles → stack overflow). It is the validated CPU
oracle we port *from* and validate *against* — not something to discard.

---

## 5. The sub-decisions (independent of vehicle)

**5a. Data layout: CSR index-based SoA.** Non-negotiable. Offsets + flat neighbor arrays for
cell→faces, face→vertices, vertex→{cells,faces}; positions in a flat `float3`(/`double3`) array
aligned with the particle ids. Spatial (Hilbert) reordering for locality, as cellGPU does.

**5b. Where does RNR run — CPU or GPU?** This is the staging axis, not a binary:

- *RNR on CPU (Stage 1):* keep the validated serial op; GPU does phases 1–4; re-sync the CSR mirror
  only when topology changed (i.e. after a reconnection pass — rare). Low risk; Amdahl-capped by how
  often RNR fires. **Right first milestone.**
- *RNR on GPU (Stage 2):* the novel result; needs the independent-set protocol + parallel slot
  allocator. **Right end-state.**

**5c. Precision.** TF's CUDA forces are fp32. RNR reversibility is O(Δl_th) and the energy-gap
guarantee is delicate — **fp32 may threaten reversibility/round-trip tolerance.** Flag as a
first-class validation risk; consider fp64 (or mixed: fp32 forces, fp64 placement) in the RNR path.

---

## 6. Vehicle options

### A. Hand-written CUDA in a TissueForge fork — *full* GPU (phases 1–5)
- **Pros:** maximal performance; fully in-engine; consistent with your native-RNR trajectory;
  reuses the existing `engine_flag_cuda` infrastructure; the cellGPU paradigm applied to 3D.
- **Cons:** highest effort; raw-CUDA iteration on a count-changing topology algorithm in a C++
  engine is the slowest possible debug loop; NVIDIA-locked; fp32 default.

### B. Hand-written CUDA in the fork — *hybrid* (phases 1–4 GPU, RNR on CPU)
- **Pros:** Option A minus the hardest part; reuses existing infra; directly validatable; realistic
  near-term win. The Chaste strategy, but with a *throttled* serial remainder so a friendlier Amdahl
  ceiling.
- **Cons:** per-RNR-pass host↔device round trip + mirror rebuild; you maintain two representations.
  Ceiling set by RNR frequency.

### C. FLAME GPU 2 wrapper (the Chaste approach)
- **Pros:** no specialist GPU code; proven to wrap a CPU framework; spatial partitioning for free.
- **Cons:** its **agent + message-passing model fits point-agent neighbor sums, not a mutating
  polygonal mesh with explicit shared faces and I↔H surgery.** The thing that made it a good fit for
  Chaste (overlapping spheres = no topology) is exactly what we *don't* have. **Weak fit for the
  vertex model; not recommended** beyond possibly the force phase.

### D. Portable GPU via a kernel DSL — **NVIDIA Warp** (or Taichi / Kokkos)
- **Warp (recommended for prototyping):** Python kernels with atomics, preallocated pools (no
  in-kernel malloc → you bump-allocate via atomic counters — *the same idiom raw CUDA needs*),
  built-in mesh/BVH, and **differentiability** (free inverse-design later, à la VertAX). Matches your
  existing Python `rnr/` prototype and visualizers → **cheapest iteration on the hard Stage-2
  algorithm.** Can run standalone or be driven from TissueForge's Python layer.
- **Taichi:** strong sparse/dynamic data structures (dynamic SNodes); also Python. Viable alternative.
- **Kokkos/Thrust:** C++ portable (CPU/GPU/AMD); fits an in-engine future without NVIDIA lock-in, but
  more ceremony than Warp.
- **Cons:** Warp/Taichi pull toward a **Python-orchestrated** loop — the opposite direction from the
  in-engine C++ native work; calling Python from the C++ hot loop is awkward, so this is most natural
  as a standalone engine or a Python-driven harness, not as code *inside* `tfMeshSolver`.

### E. Greenfield standalone GPU 3D vertex engine (cellGPU-style, but 3D)
- **Pros:** the data structure is right from day one; no AoS↔SoA impedance; no LGPL fork maintenance;
  cleanest design.
- **Cons:** abandons TissueForge's ecosystem (actors, rendering, IO, your validated RNR, the now-native
  active drive); large reimplementation; throws away real investment.

---

## 7. Recommendation

**End-state architecture:** an **index-based CSR Structure-of-Arrays 3D mesh**, GPU-resident, with
phases 1–4 (director, forces, integration, geometry) as GPU kernels and **phase 5 (RNR) as a
GPU-native independent-set operation** — cellGPU's atomic-reservation/maximal-independent-set/iterated-
batch protocol, extended with a **parallel slot allocator (atomic-bump births + free-list/compaction
deaths)** for the count-changing I↔H, and your **Okuda Appendix-1 placement + Condition-4 vetoes** as
per-candidate guards. Single GPU; fp32 forces with careful (possibly fp64) handling where RNR
reversibility demands.

**Why this and not the others:** it is the only option that fills the literature gap (3D + topology +
dynamics, on GPU); it reuses cellGPU's hardest-won idea instead of reinventing it; it keeps your
validated energetics + active drive + RNR semantics; and it degrades gracefully to a useful, low-risk
intermediate (Stage 1) if Stage 2 proves hard.

**Staged path:**

| Stage | Deliverable | Risk | Validates |
|---|---|---|---|
| **0** | CSR/SoA mesh mirror + bidirectional sync with the pointer-graph mesh; round-trip test | Low | the data-structure design |
| **1** | GPU forces + geometry + integration; **RNR on CPU**, mirror re-synced only on topology change | Low–med | force/geometry kernels vs. the CPU oracle (per-vertex); end-to-end Fig 1E/1F sorting unchanged. *This is already "GPU-accelerated 3D vertex model."* |
| **2** | **GPU-native RNR**: detection scan + independent-set reservation + parallel slot allocator + Okuda placement + Condition-4 vetoes | High (research) | reversibility round-trip (Okuda), and statistically-equivalent sorting vs. Stage 1 / CPU |

**Vehicle:** **two-vehicle plan.** Invent and debug the risky Stage-2 algorithm in **Warp** (fast
iteration, atomics + preallocated-pool idiom that transfers 1:1 to CUDA, differentiable, reuses your
Python tooling). Once validated, **port to hand-CUDA in a TissueForge fork**, reusing the existing
`engine_flag_cuda` device-buffer/runner pattern, for the production in-engine version (Stages 0/1 can
go straight into the fork since they reuse that infrastructure and don't need the hard algorithm). If
minimizing effort matters more than in-engine integration, a **standalone Warp engine end-to-end** is
itself a novel, publishable first and a legitimate stopping point.

**The one research problem to nail (everything else is engineering):** *parallel, conflict-free,
element-count-changing I↔H on a GPU-resident ragged mesh.* Concretely — reserve the full 3D
I-neighborhood (the two end-vertices, surrounding bodies, involved surfaces and outer vertices) with
atomics; bump-allocate the new vertices/triangular face from a preallocated pool; free + periodically
compact removed elements; apply Okuda Appendix-1 placement; veto via Condition-4 — all inside the
iterated independent-set batch loop.

---

## 8. Key risks / unknowns to validate early

1. **Parallel slot allocation/compaction** for created/destroyed elements — the genuinely new
   primitive cellGPU doesn't have. Prototype this *first* in Warp; it's the make-or-break.
2. **fp32 vs RNR reversibility.** Round-trip tolerance is O(Δl_th); single precision may break the
   energy-gap guarantee. Test the I↔H→H↔I round-trip in fp32 vs fp64 before committing.
3. **Conflict-neighborhood size.** The 3D I-neighborhood is larger than the 2D 4-cell set; if it's
   large, the independent set is small and the batch loop runs many serial rounds → measure conflict
   density and the resulting parallelism.
4. **Determinism.** GPU atomic ordering makes independent-set selection non-deterministic → you can't
   bit-match the CPU oracle. Plan a **statistical** validation (sorting indices, Fig 1E/1F
   distributions) rather than exact reproduction.
5. **AoS↔SoA sync cost** (Stage 1). If RNR fires often, mirror rebuilds + transfers erode the win
   (the Chaste data-translation tax). Measure; this sets the Stage-1 Amdahl ceiling and the urgency
   of Stage 2.
6. **Volume-gradient kernel correctness** under periodicity (the fork already does periodic min-image;
   confirm the divergence-theorem volume gradient matches the CPU actor per-vertex).

---

## 9. Open decisions for you

- **Scope of the win:** is the goal the *novel* result (Stage 2, GPU RNR) or a *practical* speedup
  soon (Stage 1)? They share Stage 0; Stage 1 is a clean stopping point if you want throughput now.
- **In-engine vs standalone:** keep pushing everything native into the TissueForge C++ fork
  (CUDA), or accept a Python-orchestrated Warp engine (faster, differentiable, but off the
  native path)? This is the biggest fork in the road.
- **Differentiability:** do you want the inverse-design / parameter-inference capability (VertAX's
  pitch) as an explicit goal? If yes, it tilts strongly toward Warp/JAX-style and changes the
  end-state; if no, raw CUDA is fine.
- **Hardware target:** single NVIDIA GPU (CUDA/Warp) enough, or do you need portability/multi-GPU
  (tilts toward Kokkos)?

---

---

## 10. Decisions & concrete plan (2026-06-24)

**Decisions made:**
- **Goal = the novel result** — GPU-native RNR (Stage 2). Stages 0/1 are the foundation it
  stands on, not the end. Everything orients around the one hard primitive.
- **Vehicle = Warp → CUDA-in-fork.** Prototype the algorithm in NVIDIA Warp (cheap iteration,
  atomic-bump-into-preallocated-pool idiom that transfers 1:1 to CUDA), then port to hand-CUDA in
  the TissueForge fork reusing `engine_flag_cuda`.
- **Forward simulation only** — no inverse design. *Simplifies the Warp prototype:* no tape /
  differentiability constraint, so we mutate in place and use atomics freely.

**Feasibility (this box):** NVIDIA **RTX 5090, 32 GB, compute cap 12.0 (Blackwell/sm_120)**,
driver 610.62; `nvidia-smi` works under WSL2 (GPU via `/dev/dxg`; absent `/dev/nvidia*` is normal).
Python 3.11. **Caveat:** sm_120 needs a recent Warp (Blackwell support) + CUDA 12.8+ runtime —
verify on install. **Env isolation:** keep Warp out of the TissueForge conda env (Warp bundles its
own CUDA runtime; mixing with TF's conda CUDA risks conflicts) — use a separate pixi feature/env.

**CSR/SoA mesh (the Stage-0 representation, in Warp arrays):**
- `vert_pos : array(vec3)`, plus `vert_alive : array(bool)` and a free-list/counter for births/deaths.
- Surface→vertices (ordered ring), CSR: `s2v_off`, `s2v_idx`; `surf_alive`, free-list.
- Surface→bodies: `s2b : array(vec2i)` (b1,b2).
- Vertex→surfaces and Vertex→bodies (the 4-valent neighborhood), CSR: `v2s_off/idx`, `v2b_off/idx`.
- Body→surfaces, CSR: `b2s_off/idx`. (+ per-cell `director` for the active drive.)
- Loader: dump the TF pointer-graph mesh to these arrays (extends the existing `rnr/` CSV round-trip).

**Build order — make-or-break first, round-trip test as the gate (mirrors the CPU RNR methodology):**

| Step | Build | Gate |
|---|---|---|
| **A** | CSR loader from the TF mesh (+ writer back) | **Gate A:** CSR round-trips the TF mesh exactly |
| **B** | **The make-or-break primitive:** parallel slot allocator (`wp.atomic_add` bump from a pool + free-list) + ONE I→H and its inverse H→I in a kernel, with Okuda Appendix-1 placement | **Gate B:** single I→H then H→I returns topology+geometry to original within tol (GPU analog of `test_roundtrip.py`). *Build this before any scheduler.* |
| **C** | Independent-set scheduler (cellGPU protocol): edge-detection scan → per-candidate I-neighborhood reservation via atomics (commit only if uncontended) → parallel batch apply → iterate on a device flag. Condition-4 vetoes as reservation-time predicates | **Gate C:** N non-conflicting I↔H apply in parallel; each individually round-trips; illegal patterns vetoed |
| **D** | Stream-compaction of dead vertex/surface slots (periodic, like cellGPU's grow list) | **Gate D:** arrays stay bounded over many reconnection passes |
| **E** | Stage-1 force kernels (volume grad, surface-area grad, tension) + overdamped integration; wire into a forward step | **Gate E:** per-vertex GPU forces match CPU (fp32 tol); end-to-end Fig 1E/1F sorting statistically matches the CPU oracle |

**Progress:**
- ✅ **Feasibility** (2026-06-24): Warp 1.14.0 initializes the RTX 5090 natively (sm_120,
  CUDA 12.9), the atomic-bump allocator + fp64 kernels run, and Warp coexists with TF in the
  pixi env. `warp-lang` added to `[pypi-dependencies]`.
- ✅ **Gate A** (2026-06-24): `rnr/gpu/csr_mesh.py` extracts TF's pointer-graph into the CSR/SoA
  layout; round-trips a minimal [I] config and a 91-cell Kelvin block exactly; verifier rejects
  corruption; SoA round-trips bit-exact through the 5090. Test `rnr/tests/test_gpu_csr_roundtrip.py`
  (4 tests, in the `pixi run test` gate → 52 passed). Demo: `pixi run gpu-csr`. Confirmed the 3D
  raggedness that mandates CSR: vertex valence 3–8, face size 3–6, cell size 7–14 (no fixed stride).
- ✅ **Gate B** (2026-06-24): the make-or-break count-CHANGING I↔H round-trip, proven on the
  host reference AND on the GPU. Split B1 substrate / B2 surgery / B3 GPU kernel.
  - ✅ **B1** (2026-06-24): `rnr/gpu/device_mesh.py` — the *padded mutable* working rep
    (fixed-width rows + per-row length + spare capacity), the **bump allocator** (births bump
    a high-water counter, deaths mark `alive=0`, reclaim deferred to Gate-D compaction — no
    concurrent free-list, the GPU-safe choice), and the local surgery primitives
    (`replace_v`/`insert_between`/`drop_v`/`attach_body`/`detach_body`) mirroring `reconnect.py`,
    each maintaining both sides of every adjacency. Test `rnr/tests/test_gpu_device_mesh.py`
    (4 tests): CSR↔padded round-trips exactly, allocator bumps, primitives mutate-then-revert
    restore the mesh, padded SoA uploads to the 5090 intact. (+`set_ring`/`ring_neighbors`/
    `from_warp` added for B2/B3.)
  - ✅ **B2** (2026-06-24): `rnr/gpu/reconnect_csr.py` — `i_to_h_csr`/`h_to_i_csr` as a direct
    translation of `reconnect.py` onto PaddedMesh primitives (births bump-allocated, deaths
    free-marked), `iconfig_to_indices` translator (via `csr_mesh.id_maps`), and the Okuda
    placement reused VERBATIM from `reconnect.place_*_xyz` (one formula, refactored into pure
    position-array cores — no CPU/GPU drift). Gate = **body-anchored slot-invariant fingerprint**
    (`csr_mesh.fingerprint`): vertex/surface SLOTS get relabelled by alloc/free/compact, so only
    a body-keyed invariant detects restoration. Test `rnr/tests/test_gpu_reconnect_roundtrip.py`
    (3): minimal (near-exact) + Kelvin round-trips restore the fingerprint, outer 6 verts byte-
    exact, recovered edge within O(dl_th), counts/consistency restored; +teeth test (fingerprint
    distinguishes [I] from [H]).
  - ✅ **B3** (2026-06-24): `rnr/gpu/reconnect_warp.py` — the B2 surgery as two Warp kernels
    (`i_to_h_kernel`/`h_to_i_kernel`, `dim=1`) with `d_*` device funcs mirroring the PaddedMesh
    primitives + `wp.atomic_add` bump allocation; placement **fp64**. Test
    `rnr/tests/test_gpu_reconnect_warp.py` (3): device round-trip matches the host reference
    **bit-for-bit** on integer connectivity (minimal + Kelvin) and to round-off on positions,
    and restores the fingerprint. **Precision (risk #2 SETTLED):** on-device fp64 placement ==
    numpy fp64 oracle to **0.0** (bit-exact, fully reversible); fp32 drifts **1.8e-7** (within
    the dl_th budget but NOT bit-reversible) → **fp64 is the RNR-path precision** (data-backed).
    Warp gotcha: a literal-init accumulator mutated in a dynamic loop must be `wp.int32(-1)`.
- ◐ **Gate C** (in progress): the cellGPU independent-set scheduler for many parallel I↔H.
  - ✅ **C0** (2026-06-24): `rnr/gpu/topology_csr.py` — index-based [I]-config detection on
    the PaddedMesh (the GPU analog of `topology.i_neighbourhood`/`find_short_edges`, NO TF
    handles — the scheduler must re-detect on the *mutated* device mesh). Emits the same
    `ICfgIdx` the surgery consumes. Test `rnr/tests/test_gpu_topology_csr.py` (3): detector
    matches the CPU oracle's site count on a Kelvin block, finds the minimal config's single
    edge, and every emitted config drives a clean round-trip (fingerprint restored).
  - ✅ **C1** (2026-06-24): `rnr/gpu/schedule_csr.py` — the host-reference scheduler:
    Condition-4 veto on indices (`i_to_h_veto_csr`, mirrors `conditions.py`), per-candidate
    **footprint** (2 end + 6 outer verts, 9 faces, 5 cells), greedy **maximal independent
    set**, `apply_batch`, and the iterated `reconnect_sweep_i_to_h`. Test
    `rnr/tests/test_gpu_schedule_csr.py` (5): independent sets are conflict-free; **THE
    parallel-safety property** — an independent batch applies to the SAME body-anchored
    fingerprint in ANY order (proven at scale: a ~10-reconnection Kelvin batch, fwd==rev);
    the veto fires on cap-contact; the iterate loop stays consistent.
    *Finding:* a static-mesh sweep does NOT converge — an I→H places triangle verts that
    form NEW short edges (1→3 measured), so reconnections cascade; in production, force
    relaxation between steps prevents this. So the Gate-C gate is "N non-conflicting I↔H in
    one parallel batch" (order-independent), not sweep-to-exhaustion. The bump allocator's
    +3 verts/op (no reclaim) also confirms **Gate D compaction** is needed for long runs.
  - ✅ **C2** (2026-06-24): `rnr/gpu/schedule_warp.py` + the batch kernel in
    `reconnect_warp.py` — the cellGPU conflict-resolution protocol in 3D, on the GPU.
    **THE NOVEL RESULT: parallel, conflict-free, element-count-CHANGING I→H on a
    GPU-resident ragged 3D mesh** — the empty cell in the literature table (§1).
    - **C2a reservation** (`reserve_kernel`/`check_kernel`, dim=N): each candidate
      `atomic_min`-claims its fixed 8-vert/9-surf/5-body footprint; wins iff it owns every
      element (lowest-id-wins → conflict-free by construction). Deterministic, so it matches
      the host reference (`schedule_csr.reserve_won_mask_host`) **bit-for-bit**.
    - **C2b parallel apply** (`i_to_h_batch_kernel`, dim=N winners): all winners run the
      count-changing I→H simultaneously; disjoint footprints ⇒ no races on existing
      elements, the shared `atomic_add` bump gives each thread distinct fresh slots.
    Test `rnr/tests/test_gpu_schedule_warp.py` (5): GPU reservation == host bit-for-bit;
    won set conflict-free + non-empty; disjoint candidates both admitted; **parallel apply
    == host sequential apply** (body-anchored fingerprint) on 2 disjoint configs and on a
    reserved Kelvin batch. 27 GPU tests total, all green.
  - ✅ **C2c — the iterated sweep, glued on the GPU** (2026-06-24):
    `schedule_warp.reconnect_sweep_warp(g, threshold, dl_th)` runs the cellGPU iterated-batch
    loop end-to-end on the device: each round `PaddedMesh.from_warp(g)` (slot-preserving)
    → host detect (`find_short_edges_csr` + Cond-4 veto) → GPU reserve (C2a) → GPU parallel
    apply (C2b, mutates `g`) → re-detect, bounded by `max_rounds`. Host mirror added:
    `schedule_csr.reconnect_sweep_reserve_host` (+`reserve_independent_set_host`). Test
    `rnr/tests/test_gpu_sweep.py` (3): **one GPU round == one host RESERVATION round**
    (body-anchored fingerprint) — round 1 shares the host's slot layout so it's exact by
    composition of C2a (bit-for-bit) + C2b (fingerprint); a bounded 3-round device sweep
    re-detects on the mutated mesh and stays consistent each round; a sub-threshold sweep is
    a 0-round no-op. **30 GPU tests total, all green.**
    - **Subtlety that mattered:** the per-round mirror is `reconnect_sweep_reserve_host`
      (reservation), NOT `reconnect_sweep_i_to_h` (greedy maximal). Greedy keeps a candidate
      if it is disjoint from the WINNERS so far; one reservation round keeps it only if it is
      disjoint from ALL lower-id candidates (winners *and* losers) — strictly more restrictive.
      Measured on the n=4 Kelvin block: **360 candidates → greedy 10, one reservation round 1.**
      Gating the GPU sweep (reservation) against the greedy host sweep would have compared
      1 reconnection vs 10 and failed.
    - **Efficiency finding (not correctness):** deterministic lowest-id-wins is very
      *non-maximal* on a DENSE candidate set (1/360 here) — it admits ≈one low-id "seed" per
      round, so a dense static batch resolves almost serially. This is a worst case of the
      static Kelvin block (uniform short edges ⇒ every edge a candidate). In production only
      a few edges fall below threshold per step (sparse, mostly disjoint) so one round admits
      most of them. If dense batches ever matter, cellGPU's **randomised per-round priorities**
      (re-rolled each round) give a near-maximal set in O(log n) rounds — but that trades the
      bit-for-bit host match (the current validation anchor) for a statistical one; deferred.
  - ✅ **C0′ — the reverse-direction [H] detector** (2026-06-24): `topology_csr`
    `h_neighbourhood_csr` + `find_small_triangles_csr` — the index-world mirror of
    `topology.h_neighbourhood`/`find_small_triangles` (no TF handles), emitting the same
    `HCfgIdx` that `h_to_i_csr` consumes. Condition-2 triggers on the **MAX** triangle edge
    (not min — Honda's wrong "condition H"). Test `rnr/tests/test_gpu_topology_h_csr.py` (3):
    one I→H makes one triangle the detector finds + reverses via the *detected* config to
    restore [I]; a fresh [I]-only Kelvin block (no triangular faces) yields zero sites; a
    batch of N I→H is detected and the canonical cap-cap sites reverse to restore the
    fingerprint. **33 GPU tests total, all green.**
    - **Reverse-direction cascade FINDING:** an I→H can collapse a *quad* side-face
      `[outer_top, v10, v11, outer_bot]` into a triangle `[outer_top, tri_k, outer_bot]` — a
      genuine, immediately-reverse-reconnectable [H] site. So one I→H yields the cap-cap
      triangle **plus** ≥0 side-collapse triangles (measured 1 extra per op on one Kelvin
      block ⇒ detector finds 2N, not N). They **share** the new tri vertex with their cap-cap
      triangle (overlapping footprints), so a reverse sweep must schedule them as conflicts —
      reversing only the cap-cap sites re-expands the side-faces back to quads. This is the
      H→I analogue of the forward C1 cascade; production force-relaxation separates the scales.
  - ✅ **C1′+C2′ — the H→I scheduler (the reverse mirror of C1/C2a/C2b/C2c)** (2026-06-24):
    the full reverse reconnection scheduler, host reference + GPU, with the round-trip gate.
    - **Host C1′** (`rnr/gpu/schedule_csr.py`): `h_footprint` (the reverse footprint —
      **9 verts** = 3 tri + 6 outer, **10 surfs** = the triangle + 3 side + 3 top + 3 bottom,
      **5 bodies**; the triangle + its 3 verts ARE existing here, so they join the footprint —
      that is what makes a cascade side-collapse triangle conflict with its parent cap-cap
      triangle, serialising them), `h_to_i_veto_csr` (mirror `conditions.h_to_i_veto`:
      caps share ≥2 faces / side-cell pair shares ≥2 faces / side-face pair shares ≥2 edges,
      the last via a new index helper `faces_share_multiple_edges_csr` = "an edge IS a cyclic
      vertex pair; ≥2 shared edges ⇒ veto"), `h_independent_set`, `h_reserve_won_mask_host`,
      `h_reserve_independent_set_host`, `h_batch_is_conflict_free`, `h_apply_batch`, and both
      sweeps (`reconnect_sweep_h_to_i` greedy + `reconnect_sweep_h_reserve_host` reservation,
      the latter the per-round GPU mirror, exactly as on the I-side).
    - **GPU C2′** (`rnr/gpu/reconnect_warp.py` + `schedule_warp.py`): `h_to_i_batch_kernel`
      (the `h_to_i_kernel` body indexed per-candidate by `tid`; births bump `n_used[0]` by 2,
      no surface alloc) + `apply_h_to_i_batch_warp`; `reserve_h_kernel`/`check_h_kernel` (the
      C2a kernels with `_HFV=9,_HFS=10,_HFB=5`), `pack_h_footprints`, `reserve_h_won_mask_warp`,
      `reserve_h_independent_set_warp`, and `reconnect_sweep_h_to_i_warp` (glued like C2c).
    - **Gates** (`rnr/tests/test_gpu_schedule_h_csr.py` ×5, `test_gpu_schedule_h_warp.py` ×5):
      host — h-independent-set conflict-free, veto fires on a double cap contact, reverse batch
      order-independent + forward-then-reverse restores fp0, reverse sweep consistent, pure-[I]
      no-op. GPU — H-reservation == host bit-for-bit (on the conflicting cap-cap+side-collapse
      set), parallel `h_to_i` apply == host sequential (fingerprint, restores fp0), **a full
      GPU round-trip: N parallel I→H then N parallel H→I restore the fingerprint**, GPU reverse
      sweep round 1 == host reservation round 1. **43 GPU tests total, all green** (was 33).
  - ✅ **On-GPU detection (the Condition-2 trigger scans)** (2026-06-24): `rnr/gpu/detect_warp.py`
    — the O(mesh) per-round scan moved off the host Python loop onto parallel Warp kernels.
    - **`scan_small_triangles_kernel`** (one thread / surface, single emit) + `scan_short_edges_kernel`
      (one thread / vertex; each edge emitted by its SMALLER endpoint so NO cross-thread dedup —
      only an O(k²) per-thread dedup of ring-neighbours + an O(k²) distinct-incident-body count,
      both done inline with no per-thread scratch array). Atomic-append compaction.
    - **Hybrid detect** (`detect_short_edges_hybrid`/`detect_small_triangles_hybrid`): GPU trigger
      scan (O(mesh), on device) → host `i_/h_neighbourhood_csr` gather on just the few candidates
      (O(cands)). A drop-in for `find_*_csr`: same sites, same canonical order.
    - **Wired into the sweeps**: `reconnect_sweep_warp` / `reconnect_sweep_h_to_i_warp` take
      `gpu_scan=False` (default = host Python scan, unchanged) / `True` (GPU scan). The host
      `from_warp(g)` mirror still happens each round (needed for the gather + reservation packing),
      so this removes the O(mesh) *Python* scan, not the device→host copy — the latter needs a full
      device gather (the remaining step toward "never returns to the host").
    - **Canonical-order fix (subtlety that mattered):** `find_short_edges_csr` returns sites in
      Python SET-iteration order, but the lowest-id-wins reservation is order-SENSITIVE, so the
      sorted GPU scan and the unsorted host scan picked DIFFERENT (both valid) winners → divergent
      fingerprints (same round sizes, different topology). Fixed by sorting detected sites by a
      canonical key ((v10,v11) / triangle idx) in ALL FOUR reservation sweeps (the 2 warp + their 2
      host mirrors) — makes the schedule reproducible across host/GPU detection AND keeps the C2c
      bit-for-bit round-1 gate exact (warp sweep and its host mirror both sort identically).
    - **Gates** (`rnr/tests/test_gpu_detect_warp.py` ×6): GPU trigger set == host trigger ref
      EXACTLY (both directions); hybrid detect == `find_*_csr` as a set AND in order, surgery-ready;
      **`gpu_scan=True` sweep == `gpu_scan=False` sweep bit-for-bit** (same round sizes + fingerprint),
      both directions. **49 GPU tests total, all green** (was 43).
  - ✅ **Gate D — stream-compaction of dead slots** (2026-06-24): the bump allocator never reclaims
    (+3 verts/+1 surf per I→H, +2 verts/−1 surf per H→I), so the high-water counters only grow.
    Compaction renumbers LIVE elements into a contiguous prefix [0, n_live) (ascending old-slot
    order) and resets the counters, keeping arrays bounded over long runs (cellGPU grow-then-compact).
    - **Host** `PaddedMesh.compact()` (`rnr/gpu/device_mesh.py`): in-place, same capacity; remaps
      s2v via vmap, v2s/b2s via smap, s2b (bodies) unchanged.
    - **Device** `rnr/gpu/compact_warp.py`: `wp.utils.array_scan` (exclusive prefix-sum of the alive
      flags → each live slot's new index) + scatter kernels into fresh arrays + an on-device n_used
      set — NO O(mesh) host work; swaps the compacted arrays into `g` in place.
    - **Gates** (`rnr/tests/test_gpu_compact.py` ×4): host compact preserves the fingerprint, drops
      the counters to the live counts, idempotent; bounded over many forward+reverse+compact passes;
      device compact == host compact (fingerprint + slot-for-slot); device compact after a device
      round-trip restores the bounds.
  - ✅ **Device GATHER — "never returns to the host" (both directions)** (2026-06-24): the
    neighbourhood gather (the hardest kernel) + fused Condition-4 veto on the device, so a whole
    round runs with NO `PaddedMesh.from_warp(g)` (only O(candidates) config data leaves the device).
    `rnr/gpu/gather_warp.py`:
    - **`gather_i_kernel`** = device `i_neighbourhood_csr` + `i_to_h_veto_csr`, FUSED: per candidate
      edge it classifies v10's bodies (cap_top = body at v10 not v11; side cells = bodies at both),
      finds cap_bot, the 3 arms (shared+consecutive+2-body side faces with their outer verts), the
      3 top + 3 bottom faces, and applies the veto (caps mustn't touch; no triangular side face; no
      side-cell pair sharing ≥2 faces). **No per-thread scratch** — results write straight to
      per-candidate output rows; set ops are O(k²) over bounded adjacency.
    - **`gather_h_kernel`** = the reverse mirror (device `h_neighbourhood_csr` + `h_to_i_veto_csr`):
      per triangle, side cell per edge, side face per tri vert (interface of its two flank cells
      containing it), outer verts by cap incidence, top/bottom faces, veto (caps share only the tri;
      side-cell pairs one face; no two side faces share ≥2 edges via `d_faces_shared_edges`).
    - **Fully-on-device sweeps**: `reconnect_sweep_warp_device` / `reconnect_sweep_h_to_i_warp_device`
      (`schedule_warp.py`) — scan→gather(+veto)→reserve→apply with no `from_warp`; capacities read
      from `g` via `reserve_*_independent_set_warp_g`.
    - **Gates** (`rnr/tests/test_gpu_gather_warp.py` ×7): device gather == host gather+veto
      per-candidate (caps/side cells/arms/top-bottom faces, normalised for free ordering), both
      directions; device detection == hybrid-after-veto, surgery-ready; **a fully-device sweep round
      == the host-scan sweep round by fingerprint**, both directions. (Round 1 is fingerprint-exact;
      the device gather may order arms differently → permuted tri-vertex positions, same topology.)
    - **Warp gotcha:** a cross-module `@wp.func` (here `d_vert_body_count` from `detect_warp`) MUST be
      imported into the calling module's namespace or the kernel fails to compile with
      "Referencing undefined symbol" (surfaced via `module.load(dev)`, not the bare pytest trace).
  - **60 GPU tests total, all green** (was 49).
  - ✅ **Gate E — Stage-1 physics: force/geometry/integration kernels + end-to-end sorting**
    (2026-06-24): the compute-dominant phases ported to Warp on the CSR/SoA, composed into a full
    forward step, validated against TF and a host reference. **The GPU 3D vertex engine now runs a
    complete step entirely on-device and SORTS.** Built host-reference-first (the `reconnect_csr` →
    `reconnect_warp` methodology):
    - **Host reference** `rnr/gpu/physics_csr.py` — geometry (surface centroid/area/unnormalized
      normal; body volume/area/centroid/orientSign, periodic min-image, first-vertex/first-surface
      floating origin) + the FOUR sorting-physics forces re-derived from the LGPL TF actors (read,
      not copied; each cites its actor): VolumeConstraint, SurfaceAreaConstraint (body variant),
      Adhesion (heterotypic σ, body variant = `0.25·λ·dA_s/dx` over het faces), and the active drive
      `v0·⟨incident directors⟩`; plus the overdamped integrator. TF's auto-bound mesh-hygiene
      regularizers (FlatSurfaceConstraint/ConvexPolygonConstraint, λ=0.1) are intentionally OMITTED
      (not the sorting physics). **Gate** `test_gpu_physics_csr.py` (7): geometry == TF
      `b.volume`/`b.area`/`b.centroid`/`s.area` to **float32 ε** (vol_rel 1.2e-7); each force == TF's
      directly-callable `actor.force(body,vertex)` to **float32** (vol 2.7e-5, area 5.3e-5, adhesion
      3.5e-5, active 4e-16). *Subtlety that mattered:* a monodisperse Kelvin foam has ~0 NET
      volume/area force at every vertex (Σ_cell dV/dx=0 by space-filling + Kelvin symmetry), so TF's
      float32 per-cell forces (~1e6) cancel to pure noise — the gate needs a **jittered polydisperse
      foam + v0/a0 at the cell mean** to be well-conditioned.
    - **GPU kernels** `rnr/gpu/physics_warp.py` — surface_geom / body_geom / force / integrate /
      director-update kernels, **fp64**. The per-body force sum is restructured as a per-(surface,body)
      sum (a body defines s ⟺ it's in s2b[s]) so NO body dedup is needed for the conservative forces;
      only the active mean needs the small O(valence²) first-occurrence dedup. **Gate**
      `test_gpu_physics_warp.py` (9): geometry/forces == host to **fp64** (~1e-14); integrator ==
      host + wraps into [0,L); active displacement == dt·v0·⟨n⟩ (5e-15); director rotational-diffusion
      Dr_eff=1.5·rate ≈ Dr (probe_native_calibration Part B, on-GPU). *Warp gotcha:* every fp64 literal
      meeting a `wp.float64` value must itself be `wp.float64(...)` ("Input types must be the same",
      which makes the referenced `@wp.func` "undefined").
    - **Composed engine** `rnr/gpu/engine.py` — `forward_step` = director → geometry → force →
      integrate → `reconnect_sweep_*_warp_device` (throttled, both directions) → `compact_warp`; plus
      the demixing order parameter `het_contact_fraction`. **Gate** `test_gpu_engine.py` (3): ~60-step
      run stays consistent/finite/positive-volume with slots bounded by compaction (47 reconnections);
      **the deterministic (Dr=0, no-RNR) GPU trajectory == the host reference to fp64 (~9e-16) over 12
      steps — THE "matches the CPU oracle" gate**; and from a MIXED IC the het-contact fraction
      demixes 0.48→0.42 under heterotypic tension (the GPU engine reproduces 3DVertVor cell sorting).
      Foam scaled to unit cells so production params (V0≈1, A0≈5.6) apply.
    - **79 GPU tests total, all green** (was 60). **Gate E DONE — the full staged GPU-port plan
      (A–E) is complete.** ▶ **next** (optional, post-plan): port the trigger-scan compaction to a
      device prefix-sum (O(cands) readback → none); a larger/longer sort for a publication-grade Fig
      1E/1F vs the TF oracle; then the hand-CUDA-in-fork port (the two-vehicle plan's second leg).

#### Decision (2026-06-25): orientation/closure repair — KEEP greedy, defer the tvm port

The RNR-at-scale balloon (commit `8e5b79e`) is healed by `orient_warp.orient_repair_warp`, a
fully-on-device **greedy closure-residual flip**. The prior handoff promoted porting `tvm`'s exact
`updatePolygonDirections` (per-cell BFS orientation propagation) as the recommended robustness
upgrade. **Decided to keep greedy** and treat the tvm port as *optional/later*:
- At the paper-scale gate `interval=round(0.01/dt)=1` ⇒ orient runs every step (100k/run). Greedy is
  on-device with ~no hit (n=10 = 5.6 ms/step); a host tvm port (full `from_warp` + Python BFS per
  step) is ~**3–6× slower** — it worsens the very host-copy bottleneck listed as the top perf item.
- Greedy's "in-principle stall" is **rare by construction** (I→H sets only 1 new winding; reserved
  batch winners are body-disjoint; degenerate initial faces are isolated single-bad-face cells) and
  is **already caught by the gate** as a volume-band balloon FAIL — not silent at the gate level.
- The tvm method's real edge (topological ⇒ correct for zero-area faces immediately; best C++-port
  reference) matters for the *eventual native C++ MeshQuality port*, not the GPU hot loop.

Optional follow-ups if revisited: (a) cheap — add a closure post-condition assert to orient; (b)
implement tvm `updatePolygonDirections` as a **host reference + pytest oracle / rare fallback** (not
in the hot loop). Revisit triggers + full pros/cons: `docs/2026-06-25_orientation-repair-options.md`;
memory `orientation-repair-greedy-kept`.

#### Perf (2026-06-25): foam build is cached to disk, not optimized (yet)

The O(N²) TF foam builder makes N=2000 setup ~10 min (the stepping itself is fast). Rather than
optimize `_setup_unit_foam`, the built unit-cell foam is now **cached to disk** (`rnr/gpu/foam_cache.py`):
build-once → save the scaled compact CSR + per-body phys state + box + (v0,a0) as npz (no TF needed
on load, headroom-independent) → repeat runs load in ~ms with a **de-novo fallback** if absent
(`--rebuild-foam` forces a rebuild). Optimizing the builder itself stays low priority.

#### Correctness (2026-06-25): periodic minimum-image in the I↔H Okuda placement — FIXED

The reconnection placement (`reconnect.place_i_to_h_xyz`/`place_h_to_i_xyz` + the 4 GPU
`reconnect_warp` kernels) computed the edge midpoint / triangle centroid / outer-neighbour
directions with **raw** coordinates, so a short edge / small triangle STRADDLING a periodic box
face was split through the box centre (teleporting the new vertices). Latent because the trigger
fires on small features (usually interior) and the round-trip/fingerprint gates use non-periodic
meshes. Fixed by differencing all positions under minimum-image (`d_minimg`) and wrapping results
into [0,L) (`d_wrapbox`); `box=None`/zero-box is the exact non-periodic path (interior sites are
bit-identical, so the existing 79 GPU + CPU round-trip/fingerprint gates are unaffected). New gate
`rnr/tests/test_periodic_placement.py` (5 tests: CPU straddle correctness both directions +
interior==non-periodic invariance + GPU==CPU periodic parity). Full gate **132 passed**;
`gpu-stability --n 4` STABLE with min-image active. (Closes prior-handoff item #3.)

**Risks to confront in this order:** (1) Warp on sm_120 actually initializes → verify *first*;
(2) the parallel slot allocator + count-changing surgery (Gate B) — the real research risk;
(3) fp32 vs reversibility tolerance (use fp64 placement if Gate B fails in fp32);
(4) determinism — atomic ordering makes the independent set non-deterministic → validate
statistically (Gate E), not by bit-matching the CPU oracle.

**Immediate next action:** stand up an isolated Warp env and confirm it initializes the RTX 5090
(resolves the sm_120/Blackwell unknown before any design investment), then Gate A.

---

### Citations
- Sussman, *cellGPU*, Comp. Phys. Comm. 219:400 (2017), arXiv:1702.02939.
- Leach, Heywood, Fletcher, Richmond, *GPU acceleration of cell-based simulations in Chaste using
  FLAME GPU 2*, bioRxiv 2026.01.13.699201.
- Pasqui et al., *VertAX*, arXiv:2604.06896.
- Okuda et al. (2013), *Biomech. Model. Mechanobiol.* 12:627–644 (RNR / I↔H).
- Sego et al. (2023), *Sci. Rep.* 13:17886 (TissueForge).
