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
  - ▶ **next**: (a) wire reserve+apply+re-detect into a single GPU iterated sweep
    (`reconnect_sweep_warp`, glue over C0/C2a/C2b — meaningful with dynamics, see the C1
    cascade finding); (b) on-GPU detection (currently host); (c) the H→I reverse direction
    in the scheduler. Then **Gate D** (stream-compaction of dead slots — the bump
    allocator's +3 verts/op makes this needed for long runs), then **Gate E** (force +
    integration kernels + Fig 1E/1F sorting validation vs the CPU oracle).

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
