# Reconnect-sweep optimization: Option A (device-resident) + the real bottleneck (per-surface I-scan)

## Summary 2026-06-25 09:57 EDT

Goal: make the GPU reconnect sweep faster (prior handoff's "Option A"). Delivered a **~3.0× end-to-end
speedup at paper scale (N=2000), bit-identical trajectory** — and corrected the prior handoff's
bottleneck diagnosis. All changes are pure performance; the math/topology is unchanged.

**What changed and why:**
- **Option A — device-resident gather→reserve→apply** (`schedule_warp.py`, `reconnect_warp.py`). The
  device gather already returns packed device arrays; they were being round-tripped to host `ICfgIdx`
  objects + re-packed twice. Added `won`-mask reserve/apply kernels (`reserve_i_won_device` /
  `reserve_h_won_device` in `schedule_warp.py:251+`; `apply_{i_to_h,h_to_i}_won_kernel` +
  `apply_*_device_warp` in `reconnect_warp.py:744+`) that consume the gather arrays directly — invalid
  rows and reservation-losers skip via the mask, so NO compaction/sort is needed (lowest-id-wins
  depends only on relative order). Both `reconnect_sweep_*_warp_device` (`schedule_warp.py:308,375`)
  rewritten: `find_*_warp` → device gather → device reserve → **1-int winner-count readback** →
  parallel device apply. Killed ~12 host round-trips/round.
- **SURPRISE: Option A barely moved the step** → the prior handoff misdiagnosed it. Micro-benchmarks
  (`/tmp/.../scratchpad/bench_*.py`) showed the real cost is the **I-detect scan**
  (`find_short_edges_warp`), not host syncs. At quiescence (0.03 reconnections/step) `recon_i` was
  STILL 2.1 ms — it's the per-step scan finding nothing.
- **Scan fixes** (`detect_warp.py`), all keeping `find_short_edges_warp` output **byte-identical**
  (interior-filtered, deduped, sorted) → no gate/contract change:
  1. body_count interior filter moved OFF the per-vertex hot kernel into a tiny per-candidate
     `filter_interior_short_edges_kernel` (its mere presence tanked occupancy: 3.9 vs 0.17 ms).
  2. `out_v10.numpy()[:k]` → `out_v10[:k].numpy()` (was copying the whole ~6.8 MB `cap` array).
  3. **THE lever (`scan_short_edges_kernel`): per-vertex → PER-SURFACE.** The per-vertex scan did a
     scattered `d_ring_neighbor`→`d_ring_pos` ring-search per (vertex, face, side) with no early-out —
     cheap warm (0.03 ms) but ~10× cold-cache-penalized in the loop. Per-surface emits each consecutive
     `s2v` ring pair (implicit edges ARE ring pairs → identical set), contiguous reads, no `d_ring_pos`
     (same access pattern as `compute_geometry_warp`). The H-scan was already per-surface (hence cheap).
- **Bottleneck re-diagnosis: it was never the host syncs — it's the per-step scan reading the mesh
  adjacency cold.** Do NOT re-chase syncs next time. (memory `reconnect-sweep-scan-bottleneck`.)

**Numbers** (300-step phase profiles inflate vs the real loop, which is faster):
- mid-sort step 10.06 → 3.68 ms; `recon_i` 7.70 → 1.44 ms. quiescent step 3.87 → 2.77 ms.
- **Real 100k/20k stability loop: 9.09 → 3.05 ms/step (~3.0×).** Before/after at matched 20k is
  **BIT-IDENTICAL** (het 0.46382583300146024 / 0.46041886195995785; recon_i/h 3292/2428 · 4010/3028;
  nv/ns/vol all match to the last digit) — proving a pure perf change. 100k run STABLE, `n_problems=0`.

**Follow-on after this handoff was first written — lever #2 done (commit `b260eb5`):** gated
`orient_repair` on `(ni+nh)>0` + a one-time initial heal (`_healed_initial` flag in `g`, fired at the
old step-0 orient point). orient is idempotent (a no-op that pays a full geometry recompute on a clean
mesh) and windings only break via initial-foam + surgery, so this is **bit-identical**. **20k stability
3.05 → 2.78 ms/step (≈3.3× over pre-session 9.09), trajectory bit-identical at every checkpoint, 132-gate
green.** Files: `engine.py` + the findings doc. Lever #3 below (orient) is therefore DONE; the remaining
primary lever is #1 (skip-scan), detailed in the Kickoff.

**Build / test / git state:**
- Branch `migrate/linux64-wsl2` (NOT main). Two commits this session: `a3ac037` (Option A device-resident
  sweep + per-surface I-scan) and `b260eb5` (orient gating). **Gate: `pixi run test` = 132 passed** after
  each. Branch is ahead of origin by 3 — **NOT pushed** (push needs an explicit ask).
- Committed: the 4 code files (`detect_warp`, `schedule_warp`, `reconnect_warp`, `engine`) +
  `docs/2026-06-25_reconnect-sweep-optimization.md` + this handoff. **Leave-it (NOT committed):** all
  `rnr/exports/*` (figures/CSVs/mp4s — regenerable). Memory updated (outside repo):
  `reconnect-sweep-scan-bottleneck`, MEMORY.md index.

## Kickoff — next session

You are continuing a GPU port of TissueForge's 3D vertex model + RNR (Warp on an RTX 5090). The engine
runs faithful paper-scale sorts (N=2000) and reproduces Manning Fig 1E/1F. This session made the reconnect
sweep device-resident, the I-detect scan per-surface, and gated `orient_repair` on reconnections →
**~3.3× faster (9.09 → 2.78 ms/step), trajectory bit-identical, 132-gate green** (commits `a3ac037`,
`b260eb5`; docs/2026-06-25_reconnect-sweep-optimization.md; memory `reconnect-sweep-scan-bottleneck`). The
bottleneck was NOT host syncs (the prior handoff's wrong guess) — it's the per-step detect SCAN reading mesh
adjacency cold. **`recon_i` (the per-surface I-scan, ~1 ms/step) + the H-scan (~0.29 ms) are the largest
remaining phase**, and they run EVERY step even though reconnections are rare (~0.06/step avg, →~0.001/step
late). Your task: skip them on the (majority) steps where no edge can be short.

**THE NEXT LEVER — skip the reconnect sweep on quiescent steps (exact, bit-identical version).**

Why it's safe to skip: a reconnection fires iff some edge is < `Lth` (=1e-3) after the integrate move
(this also covers H→I: a small triangle needs ALL its edges < `Lth`, so no short edge ⇒ no H-site either).
We can prove "no short edge this step" from work the step ALREADY does, without scanning:
- `surface_geom_kernel` (`physics_warp.py:74`, runs BEFORE integrate) already loads each edge's endpoints
  as `posc`/`posp` (lines 94-99). The edge length is `|posp - posc|` → add `atomic_min` to a global
  **`min_edge`** (pre-move min). Nearly free.
- `integrate_kernel` (`physics_warp.py:246`) applies `x += force[v]*dt` (unit mobility). The displacement
  `|force[v]*dt|` is already computed → add `atomic_max` to a global **`max_disp`**.

By the triangle inequality `min_edge_post >= min_edge_pre - 2*max_disp`. So the EXACT gate (in
`engine.forward_step`, after integrate, before the reconnect block at `engine.py:48`):
```
read min_edge, max_disp (1-2 tiny readbacks);  run the sweep ONLY IF  (min_edge - 2*max_disp) < Lth
```
When `min_edge - 2*max_disp >= Lth`, no edge is short post-move, so the scan would find nothing → skipping
is PROVABLY correct → **trajectory stays bit-identical** (NO faithfulness tradeoff — this is the key
improvement over the prior "fixed margin" idea, which only held empirically). The gate also lets you skip
gather/reserve/apply/compact/orient on those steps.

Implementation order:
1. `physics_warp.py` `surface_geom_kernel`: guard the atomic to dodge contention —
   `if edge_len < cap: atomic_min(min_edge, edge_len)` with a loose `cap` (~100*Lth); use a float32 global,
   init to a large sentinel each step (so "no edge below cap" ⇒ min_edge stays huge ⇒ gate skips). Return
   `min_edge` alongside the geometry dict (`compute_geometry_warp`).
2. `physics_warp.py` `integrate_kernel`: `atomic_max(max_disp, length(force[v]*dt))`; surface it from
   `integrate_warp`.
3. `engine.forward_step`: gate the `if reconnect and ... :` block on `(min_edge - 2*max_disp) < Lth`. Keep
   the one-time `_healed_initial` orient OUTSIDE the gate (the initial foam's near-degenerate faces sit just
   above Lth — don't rely on the gate to catch them on step 0).
4. If single-address `atomic_min` contention is measurable, fall back to a small per-block partial-min +
   reduction. Measure with the scratch profiler.

Expected: removes ~1.3 ms (I+H scan) on quiescent steps → quiescent step ~2.4 → ~1.1 ms (then `force` ~0.6 +
`geom` ~0.28 dominate). Active steps (min_edge near Lth) correctly still scan. Biggest remaining single win.

**Secondary levers (lower ROI):** dirty-region re-scan (the sweep re-scans the WHOLE mesh after every
applied round; only reconnection neighbourhoods gain short edges — exact, but only helps multi-round steps);
CUDA-graph / `force`-kernel fusion (`force` is real physics, low ROI).

**Validate (optimizations change NO math → trajectory must stay BIT-IDENTICAL; that IS the correctness test):**
```
pixi run test                                                          # 132 expected
pixi run python -m pytest rnr/tests/test_gpu_*.py -q                   # GPU subset (~35s)
pixi run gpu-stability --n 10 --steps 20000 --dt 0.01 --ic mixed --csv /tmp/x.csv
#   STABLE + het@10k = 0.46382583300146024, recon_i/h@20k = 4010/3028 MUST match to the last digit.
#   A single differing float = the gate skipped a real reconnection (bug). That is the unit test.
```
Re-profile with the scratch pattern (time each `forward_step` phase with `wp.synchronize_device` between;
load the cached n=10 foam via `foam_cache.load_or_build`, no TF; warmup ~2000 mid-sort or ~20000 quiescent).
The current committed baseline for a before/after is HEAD (`b260eb5`).

**Caveats / guardrails:**
- GPU-port phase only. Reimplement from Okuda 2013 / our `rnr/`; **never copy GPL `tvm/`**;
  `tvm/ 3DVertVor/ tissue-forge/ cellGPU/ VertAX/ gpu_reference_papers/` are read-only oracles (own
  `.git`, never commit). n=10 foams are cached (`rnr/exports/foam_cache/*.npz`, gitignored).
- The per-surface scan + device-resident sweep are EXACT (byte-identical `find_short_edges_warp` output,
  same winner set); the gated round-1 fingerprint tests (`test_gpu_gather_warp.py`,
  `test_gpu_detect_warp.py`) enforce this — keep them green when touching the scan/sweep.
- Warp recompiles on edit (~15-30s first launch). Cross-module `@wp.func` must be imported into the
  calling module's namespace.
