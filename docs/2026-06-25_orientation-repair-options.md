# Orientation / closure repair — options, decision, and revisit triggers (2026-06-25)

**Status: DECIDED — keep the greedy closure-residual flip (`orient_warp.orient_repair_warp`).**
The tvm `updatePolygonDirections` port is deferred to an *optional, later* investigation, not
adopted now. This note records the analysis so the decision can be revisited deliberately.

## Background

The GPU 3D vertex engine keeps every cell **closed** (`Σ_faces sense·snorm = 0`, `sense=+1` iff
`b==b1`) so its divergence-theorem volume is correct + origin-independent. Faces can become wound
inconsistently with their `b1/b2` from two sources (see memory `gpu-rnr-scale-corruption`,
PORTING_NOTES §6p):

1. **near-degenerate INITIAL foam faces** (zero area ⇒ undefined normal ⇒ the foam builder stores
   an arbitrary, often `b1/b2`-inconsistent winding) — the *dominant* source; ~4 isolated cells at
   step 0 for n=8 seed7;
2. occasional `b1/b2`-inconsistent output from the parallel I↔H surgery (~60 over 3000 steps @ n=8).

Left unrepaired, a mis-wound face contributes the wrong sign to its cell's volume → the cell
**balloons** once the face grows (the bug fixed in commit `8e5b79e`).

The **current fix** is `orient_repair_warp` (`rnr/gpu/orient_warp.py`): a fully-on-device, parallel
**greedy closure descent** — a face is reversed iff flipping it (negating its snorm) strictly
reduces the closure residual `‖Σ sense·snorm‖` of **both** its incident cells; iterated ≤4×. It
runs in `engine.forward_step` after each reconnection sweep.

The **candidate replacement** is a faithful port of `tvm`'s `Cell::updatePolygonDirections()`
(`tvm/Cell/Cell.cpp:49-209`, called every reconnection at `Reconnection.cpp:111-114`): per cell,
BFS-propagate a consistent orientation across **shared edges** (each interior edge is shared by
exactly 2 of that cell's faces; the two faces traverse it in OPPOSITE directions in a consistent
outward orientation), then flip the whole cell if its signed volume is negative. Exact in one pass,
purely topological — works even for zero-area faces (they still have edges). It does **not** need
explicit Edge objects: reconstruct the per-cell edge→face adjacency from consecutive ring pairs
`(s2v[s,i], s2v[s,i+1])`.

## Facts established (2026-06-25 investigation)

- **Call frequency:** at the paper-scale gate (`--dt 0.01`) `interval = round(0.01/dt) = 1`, so
  orient runs **every step** — 100k calls per `gpu-stability` run.
- **Current cost:** greedy is fully on-device (no host copy); n=10 = 5.6 ms/step *including* orient.
- **A host tvm port would need a full `PaddedMesh.from_warp(g)` (device→host copy) every step**
  + a Python per-cell BFS over ~2000 cells → structurally a **3–6× per-step slowdown** of the gate
  (≈9 min → ≈30–60 min stepping). This cuts directly against the host-copy cost the design already
  flags as the wall-clock bottleneck.
- **The gate already catches the failure the tvm method guards against.** `gpu_stability.py` fails a
  run on any volume-band excursion / inversion / non-finite / consistency problem. A greedy *stall*
  reappears as exactly the balloon → the run goes red at the next audit. The stall is therefore
  **not silent at the gate level** — only silent *within* a single step (orient has no closure
  post-condition check today).
- **"Many inconsistent faces on one cell" is rare by construction.** Each I→H sets exactly ONE
  from-scratch winding (the new triangle `[tri0,tri1,tri2]`); modified top/bottom faces keep their
  already-consistent winding. The independent-set reservation makes batch winners **body-disjoint**
  (footprint includes both caps + 3 side cells), so no two reconnections in a round touch the same
  cell. The dominant source (degenerate initial faces) is isolated single-bad-face cells.
- **One genuine qualitative difference:** greedy is *geometry*-based, so a **zero-area** mis-wound
  face is invisible to it (flipping changes closure by `2·snorm ≈ 0`). It heals such faces *lazily*,
  once they grow enough to register — empirically sufficient (closure stays at round-off at paper
  scale). tvm is *topological*, so it orients them correctly from the start.

## Options

### A. Keep the greedy closure-residual flip *(CHOSEN)*
**Pros**
- Fast: fully on-device, ~no measurable hit; validated 100k-step paper-scale run + 127-test gate.
- Correct for every failure mode that actually occurs (isolated single mis-wound faces).
- Its theoretical stall is caught by the gate (reappears as the volume-band balloon), not silent in
  practice.
- Simplest to maintain; no new host round-trip, no new kernel.

**Cons**
- No *hard* guarantee; relies on the residual being dominated by individually-flippable faces.
- As written, a within-step stall returns silently (no closure post-assert) — you'd only see it as a
  balloon tens of steps later. *(Cheap to close — see "Cheap hardening" below.)*
- Not the faithful reference the eventual C++ `MeshQuality` port wants.

### B. Replace with tvm `updatePolygonDirections` on the host
**Pros**
- Exact in one pass, topological; no stall possible; correct for zero-area faces immediately.
- Cleanest, most readable C++-port reference (the native op is sequential-per-cell anyway).

**Cons**
- **3–6× per-step slowdown** of the gate (host copy + Python BFS every step) — the "ruin
  performance" risk, and it is real.
- Worsens the host-copy bottleneck already called out as the top perf item.

### C. tvm `updatePolygonDirections` on the GPU (per-cell thread)
**Pros**
- Exact + fast (cells are independent → parallelizable; no host copy).

**Cons**
- Most complex to write (BFS with per-cell edge→face adjacency + dynamic local arrays in a Warp
  kernel).
- A *weaker* C++-port reference than clean host code — the main faithfulness argument weakens.

### Cheap hardening (orthogonal; recommended if/when touched)
- Add a closure **post-condition** to `orient_repair_warp`: after the loop, a one-kernel `max‖clo‖`
  reduction; if it exceeds tol, `log`/flag (and optionally fall back to a host tvm pass *that one
  step*). Converts "silent within-step stall" → explicit signal at ~zero cost.
- Implement tvm `updatePolygonDirections` as a **host reference + pytest oracle** asserting greedy
  agrees with it on the degenerate-foam fixtures, and reuse it as the rare fallback above. Captures
  the C++-port faithfulness + a correctness proof for greedy **without** putting BFS in the hot loop.

## Decision

**Keep A (greedy) in the engine hot loop.** Replacing it wholesale (B) trades a validated, fast,
on-device pass for a 3–6× slower one to defend against a failure that (a) hasn't occurred, (b) is
rare by construction, and (c) the gate already catches. C is fast but the most code and the weakest
reference. The cheap-hardening items above are the parts of the robustness goal actually worth doing,
and are deferred as optional polish (not blocking).

## Revisit triggers — adopt B/C (or the cheap hardening) if any of these occur

- `gpu-stability` ever FAILs with a **volume-band balloon** that traces to a closure stall (i.e.
  greedy did not reach round-off) — especially at larger N / longer t than validated (n=10 / 100k).
- Moving to **lower `Lth` or denser foams** that produce many simultaneously-degenerate faces per
  cell (greedy's single-flip-descent assumption weakens).
- Starting the **native C++ `MeshQuality` port** — then implement the host tvm method as the
  faithful reference (option B/oracle), independent of the GPU hot-path choice.

## Pointers
- Current impl: `rnr/gpu/orient_warp.py`; call site `rnr/gpu/engine.py:48-57`.
- Oracle: `tvm/Cell/Cell.cpp:49-209` (`updatePolygonDirections`), `Reconnection.cpp:111-114`.
- History: memory `gpu-rnr-scale-corruption` (RESOLVED), PORTING_NOTES §6p, design doc
  `docs/2026-06-24_gpu-3d-vertex-model-exploration.md` §10, handoff
  `docs/sessions/2026-06-24-2345-gpu-rnr-winding-balloon-fixed.md`.
</content>
</invoke>
