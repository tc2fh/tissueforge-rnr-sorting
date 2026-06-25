# Session — GPU 3D RNR: paper-scale balloon FIXED (winding/closure repair) + why tvm doesn't need it

## Summary 2026-06-24 23:45 EDT

**Goal:** the prior handoff's priority #1 — fix the RNR-at-scale balloon that stopped the GPU sort
from completing at paper scale (N=1728/2000) — then validate (#3). **Both done.** The fix is
committed (`8e5b79e`) and the paper-scale gate now passes.

**Root cause was DIFFERENT from the prior handoff's hypothesis** (which said "I→H mis-winds the new
cap-cap triangle; the ring is set post-creation"). That was incomplete:
- The `[v0,v0+2,v0+1]` "reversed ring" signature the prior session saw was a **compaction-relabeling
  artifact**, not a post-creation reorder. Instrumented proof: right after creation the stored ring
  is ALWAYS `[v0,v0+1,v0+2]` (`ring_mismatch=0`); compaction permutes vertex *indices* (preserving
  cyclic order) so the monotonic labeling is lost — that's all.
- **Actual cause:** a face wound INCONSISTENT with its `b1/b2` (`s2b[:,0]/s2b[:,1]`) breaks cell
  closure (`Σ sense·snorm ≠ 0`, `sense=+1` iff `b==b1`), so the divergence-theorem volume
  (`physics_warp.body_geom_kernel:127-138`) is wrong + origin-dependent → balloon as the face grows.
  Two sources, the **dominant one a surprise**:
  1. **Near-degenerate INITIAL foam faces** — zero area ⇒ undefined normal ⇒ the foam builder stores
     an arbitrary, `b1/b2`-inconsistent winding. Deterministic repro: n=8 seed7 has surf 5553
     (cells 720/864) with `|snorm|=0`, edges `[0.0002, 0.0012, 0.0012]` — **two edges sit JUST above
     `lth=1e-3`, so the H→I small-triangle detector never collapses it**; it grows mis-wound and
     balloons cells 720/864 (closure crosses 1e-2 by ~step 60). `check_step0` found 4 such cells.
  2. Occasional `b1/b2`-inconsistent output from the parallel I↔H surgery (~60 over 3000 steps@n=8).

**Dead ends RULED OUT (don't retry):**
- Orienting the I→H cap-cap triangle at creation by the edge axis `uT=p10−p11`: only approximates
  the cap_top direction, WRONG for ~30% of configs (measured: 12/40 still mis-wound).
- …by the cap_top **centroid**: better, but still leaves faces needing repair AND doesn't touch the
  dominant source (degenerate INITIAL faces). Reverted (`git checkout` of the 3 `reconnect*` files).
- **ANY geometric per-face winding test** (`snorm·(scent−bcent[b1])` or `snorm·(bcent[b2]−bcent[b1])`):
  irregular/near-degenerate foam cells have correctly-CLOSED faces whose `snorm` points "into" the
  cell by every centroid test — a geometric repair **reversed 11 GOOD faces at step 0** and broke
  closure. **Winding correctness is topological, not geometric.**

**THE FIX — `rnr/gpu/orient_warp.py::orient_repair_warp(g)`** (new file, 122 lines), called in
`engine.forward_step` (`rnr/gpu/engine.py:48-52`) after the reconnect sweep + compaction. Uses the
EXACT closure residual (a topological invariant, not a geometry guess): a face is mis-wound iff
reversing its ring (positions `[1,L)`, keep `ring[0]`) strictly reduces BOTH incident cells'
`‖Σ sense·snorm‖`. Device kernels `_body_closure` (per-body atomic add) → `_flip_mark` (dual-cell
reduction test) → `_flip_apply` (reverse ring + negate snorm), iterated ≤4× (a flip touches 2 cells).
Lands closure at machine round-off (~4e-15); reverses only the few inconsistent faces (0-3/step);
~no perf hit (n=8 3.36 ms/step, n=10 5.6 ms/step).

**Why `tvm`/3DVertVor never need this (investigated the oracle source):**
- `tvm` computes volume the SAME way we do — signed tets weighted by a STORED per-face orientation
  bool `Cell::polygonDirections_` (`tvm/Cell/Cell.cpp:212-250`) ≈ our `sense`+`borient`. Not the diff.
- The diff: `tvm` **re-derives** all face orientations TOPOLOGICALLY after *every* reconnection
  (`updatePolygonDirections()`, `tvm/Cell/Cell.cpp:49-209`, called at `Reconnection.cpp:111-114`):
  seed a face → propagate consistent orientation across **shared edges** (using directed `Edge::vv_`)
  → flip the whole cell if its signed volume is negative. Geometry-independent ⇒ works for zero-area
  faces (they still have edges) and any surgery output. We store orientation only *implicitly*
  (ring winding + `b1/b2`), set it from the gather's **arbitrary arm order**, and never re-derive it.
- **Neither `tvm` NOR 3DVertVor remove degenerate faces** (grepped both — no cleanup/equilibration;
  same `Lth=1e-3`, raw pyvoro). 3DVertVor has the same degenerate Voronoi faces — they just don't
  balloon because `updatePolygonDirections` orients them *consistently* via topology. So our
  `orient_repair_warp` is the functional analogue of `tvm`'s `updatePolygonDirections` (we have its
  per-CELL volume-sign step via `borient`/§6k; we lacked its per-FACE topological derivation). Root
  reason = the data-model choice (CLAUDE.md: "TF has no explicit Edge object — differs from `tvm`").

**Validation:**
- `pixi run gpu-stability --n 10 --steps 100000 --dt 0.01 --sigma 0.5 --ic mixed` (the prior
  handoff's exact paper-scale command) → **STABLE**, 100k steps, **6010 I→H** (old failure: step
  ~1100 @ ~1046 I→H), vol `[0.90,1.05]` throughout, het demixing `0.497→0.447`, `problems=0`,
  5.6 ms/step. CSV: `rnr/exports/gpu_stability_paperscale.csv` (now GOOD data, replacing the prior
  false-pass; still a leave-it export).
- n=8 8000 steps (failed ~6000 before): STABLE, 1783 I→H, vol `[0.93,1.04]`.
- `pixi run test` = **127 passed** (5:41) at the exact code state now committed.

**Build / test / git state:**
- Branch `migrate/linux64-wsl2`. Fix committed as `8e5b79e` (only `rnr/gpu/orient_warp.py` +
  `rnr/gpu/engine.py`; the 3 `reconnect*` files were reverted to green, so `git diff` for them is
  empty). 127-gate green at that state.
- Memory updated (outside repo): `gpu-rnr-scale-corruption` → RESOLVED + `MEMORY.md` index.
- **Everything in `git status --short` is leave-it `rnr/exports/*`** (this + prior sessions; do NOT
  commit). Full list:
```
 M rnr/exports/dpmax.json
 M rnr/exports/fig1e_demixing_native.csv
 M rnr/exports/fig1e_demixing_native.png
 M rnr/exports/fig1f_stability_native.csv
 M rnr/exports/fig1f_stability_native.png
?? rnr/exports/gpu_stability_paperscale.csv                    # THIS session — GOOD paper-scale run
?? rnr/exports/native_*_mixed*.mp4                             # prior sessions
?? rnr/exports/sort_oracle_M8_*_native[_demixed].csv           # prior sessions
?? rnr/exports/vertex_motion_native.gif                        # prior session
```
(Scratchpad — gone next session: `catch_closure.py` [closure catcher], `catch_phase.py`,
`trace_winding.py`, `trace_op2.py`, `trace_cell.py`, `check_step0.py`, `trace_flip.py`,
`trace_repair.py` — the diagnosis chain; all under the session scratchpad, not in the repo.)

## Kickoff — next session

You are continuing a GPU port of TissueForge's 3D vertex model + RNR. Gates A–E are green (127
tests) and the **RNR-at-scale balloon is FIXED** (this session): the GPU sort now runs a faithful
100k-step paper-scale trajectory (N=2000) without degrading. Read memory `gpu-rnr-scale-corruption`
(RESOLVED) first; design context in `docs/2026-06-24_gpu-3d-vertex-model-exploration.md` §10 and the
prior handoff `docs/sessions/2026-06-24-2000-gpu-rnr-paperscale-stability-bug.md`.

**The fix in one line:** `orient_warp.orient_repair_warp(g)` (run after each reconnect sweep in
`engine.forward_step`) reverses any face whose ring-flip reduces BOTH its cells' closure residual —
restoring the per-face orientation invariant that `tvm`'s `updatePolygonDirections` maintains
natively but our implicit (ring+b1/b2) model loses.

**Priority-ordered next steps (core bug is closed; #1 is the RECOMMENDED robustness upgrade, rest optional):**

1. **Implement `tvm`'s `updatePolygonDirections` to REPLACE the closure-residual flip — recommended
   next step (robustness + C++-port faithfulness).** Our current `orient_repair_warp` is a *greedy*
   closure descent: it's validated to round-off at paper scale (only 0–3 flips/step), but it relies
   on the residual being dominated by individually-flippable faces, so it can *in principle* stall
   on a cell with many simultaneously-inconsistent faces. `tvm`'s method (`tvm/Cell/Cell.cpp:49-209`,
   called every reconnection at `Reconnection.cpp:111-114`) is **exact in one pass and topological**:
   per cell, BFS-propagate orientation across shared edges (each edge is shared by exactly 2 of that
   cell's faces) so the two faces traverse the shared edge in OPPOSITE directions, then flip the
   whole cell if its signed volume is negative — re-deriving the unique consistent orientation
   regardless of how many faces were wrong.
   - **It does NOT need explicit Edge objects** (TF/our model has none): reconstruct the per-cell
     edge→face adjacency from the vertex rings — an edge is a consecutive ring pair
     `(s2v[s,i], s2v[s,i+1])`; build a per-cell map `frozenset{a,b} → [(face, direction)]`, then BFS.
     ~O(faces·valence) per cell. This is the faithful reference for the eventual native C++
     `MeshQuality` op (CLAUDE.md goal: harden into C++ later).
   - Keep the cheap closure metric as a **post-condition assert** (closure ≈ round-off after the
     pass). Validate: `gpu-stability --n 10 --steps 100000` still STABLE + `pixi run test` = 127.
   - Document the choice (greedy-flip vs topological-propagation) in the design doc §10.
2. **(Perf, prior handoff #3 — still open)** The **O(N²) foam builder** makes N=2000 SETUP take
   ~10 min (the 100k stepping itself is fast, ~5.6 ms/step). This is now the wall-clock bottleneck
   for paper-scale runs. Profile/optimize `_setup_unit_foam` (the TF-foam→CSR build) if you need to
   iterate on large N. Not a correctness issue.
3. **(Latent correctness, prior handoff #4 — NOT the balloon)** I→H/H→I Okuda placement lacks
   periodic min-image (`r0=0.5(p10+p11)` / `r0=(p0+p1+p2)/3` break if verts straddle the box).
   Fold min-image into `reconnect_warp.py` (+ the CPU mirrors `reconnect_csr` / `reconnect`); keep
   the round-trip + CPU==GPU fingerprint gates green.
4. **(Hygiene)** The native Python sort path (`rnr/operator.py` + `reconnect.py`) was NOT re-checked
   for the same degenerate-initial-face seeds — it's a separate earlier-phase path with short tests,
   but if you ever push it to long/large runs, expect the same balloon (apply the analogous repair).
5. **(Prior CLAUDE.md polish, still pending)** Regenerate canonical `fig1e`/`fig1f` with the native
   active drive (point `run_overnight.py` at `native` + a `MODEL=native` fig selector).

**Commands / caveats:**
```
pixi run gpu-stability --n 10 --steps 100000 --dt 0.01 --sigma 0.5 --ic mixed   # paper-scale gate -> STABLE
pixi run gpu-stability --n 8  --steps 8000  --ic mixed                          # quick (~30s stepping) -> STABLE
pixi run test                                                                   # full gate (~5.5 min, expect 127)
pixi run python -m pytest rnr/tests/test_gpu_*.py -q                            # GPU subset (~40s)
```
- **`orient_repair_warp` runs inside `engine.forward_step` only** (after the sweep, when
  `reconnect` is on) — the build-time degenerate faces get healed at step 0's first sweep. The raw
  `reconnect_sweep_*_warp_device` functions and the single-op `i_to_h_warp` do NOT call it (so the
  fingerprint/round-trip gates are unaffected — they were left green).
- Warp recompiles on edit (~15-30s first launch). `g["box"]` is set by the foam setup; absent for the
  local fingerprint-test mesh (not relevant to the orient kernels, which read only `s2b/s2v/snorm`).
- **Scope/license:** GPU-port phase only; reimplement from Okuda 2013 / our `rnr/`; never copy GPL
  `tvm/`; `cellGPU/ VertAX/ gpu_reference_papers/` are read-only oracles (own `.git`, don't commit).
  `tf.init()` is one-per-process; standalone scripts mirror `conftest.vsolver`.
```
