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

**Build / test / git state:**
- Branch `migrate/linux64-wsl2` (NOT main). **Gate: `pixi run test` = 132 passed** twice this session
  (after Option A, and again after the per-surface scan); the 3 `*.py` are unchanged since that green
  run (only docs/exports changed after), so this handoff's commit cites it rather than re-running.
- Committed by this handoff: the 3 code files + `docs/2026-06-25_reconnect-sweep-optimization.md` + this
  handoff. **Leave-it (NOT committed):** all `rnr/exports/*` (figures/CSVs/mp4s incl. the new
  `gpu_stability_optA.csv` — regenerable). Memory updated (outside repo): `reconnect-sweep-scan-bottleneck`,
  MEMORY.md index.
- Full `git status --short` (pre-commit): ` M rnr/gpu/{detect_warp,reconnect_warp,schedule_warp}.py`,
  `?? docs/2026-06-25_reconnect-sweep-optimization.md`, plus ` M`/`??` `rnr/exports/*` (leave-it, this +
  prior sessions — do NOT commit).

## Kickoff — next session

You are continuing a GPU port of TissueForge's 3D vertex model + RNR (Warp on an RTX 5090). The engine
runs faithful paper-scale sorts (N=2000), reproduces Manning Fig 1E/1F, and the reconnect sweep was just
made device-resident + the I-detect scan made per-surface → **~3.0× faster (9.09→3.05 ms/step), trajectory
bit-identical, 132-gate green** (docs/2026-06-25_reconnect-sweep-optimization.md, memory
`reconnect-sweep-scan-bottleneck`). The bottleneck is NOT host syncs — it's the per-step detect SCAN
reading mesh adjacency cold. `recon_i` is still the largest phase (~1 ms/step, mildly cold-cache-bound).

**Optional next levers (in priority order; each is a fresh optimization, not required — the perf target
is already met):**
1. **Skip the I-scan on quiescent steps.** Piggyback a global `min_edge` atomic-min onto
   `compute_geometry_warp` (`physics_warp.py:55-60` already iterates every ring edge → ~free), and in
   `engine.forward_step` (`engine.py:48`) skip the I-sweep when `min_edge >= Lth + margin`. CAVEAT:
   geometry is PRE-integrate, so a too-small margin delays a rare reconnection by 1 step and BREAKS the
   bit-identical trajectory — this is a faithfulness tradeoff, so **get user buy-in** and validate the
   het trajectory vs the committed baseline.
2. **Dirty-region re-scan.** The sweep re-scans the WHOLE mesh after every applied round; only the
   reconnection neighbourhoods gain new short edges. Exact (no faithfulness risk) but only helps the
   minority of multi-round steps.
3. **Gate `orient_repair` on `(ni+nh)>0`** (`engine.py:57`, 0.37 ms/step). Windings only change on
   reconnection, so it's redundant on no-reconnection steps — BUT the initial foam has degenerate
   windings it heals at startup, so fold that initial heal into the foam-cache build first.

**Validate (all must stay green; optimizations change NO math → trajectory must stay bit-identical):**
```
pixi run test                                                          # 132 expected
pixi run python -m pytest rnr/tests/test_gpu_*.py -q                   # GPU subset (~35s)
pixi run gpu-stability --n 10 --steps 20000 --dt 0.01 --ic mixed --csv /tmp/x.csv  # STABLE; het@10k=0.46382583300146024, recon_i/h@20k=4010/3028 must match BIT-for-bit
```
Re-profile with the scratch pattern (time each `forward_step` phase with `wp.synchronize_device` between;
load the cached n=10 foam via `foam_cache.load_or_build`, no TF). The committed pre-opt baseline for a
clean before/after is `git show 91b474b:rnr/gpu/detect_warp.py` etc.

**Caveats / guardrails:**
- GPU-port phase only. Reimplement from Okuda 2013 / our `rnr/`; **never copy GPL `tvm/`**;
  `tvm/ 3DVertVor/ tissue-forge/ cellGPU/ VertAX/ gpu_reference_papers/` are read-only oracles (own
  `.git`, never commit). n=10 foams are cached (`rnr/exports/foam_cache/*.npz`, gitignored).
- The per-surface scan + device-resident sweep are EXACT (byte-identical `find_short_edges_warp` output,
  same winner set); the gated round-1 fingerprint tests (`test_gpu_gather_warp.py`,
  `test_gpu_detect_warp.py`) enforce this — keep them green when touching the scan/sweep.
- Warp recompiles on edit (~15-30s first launch). Cross-module `@wp.func` must be imported into the
  calling module's namespace.
