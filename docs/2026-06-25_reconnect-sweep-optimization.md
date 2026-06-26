# Reconnect-sweep optimization (Option A + the real bottleneck) — 2026-06-25

Goal of the session: make the GPU reconnect sweep faster. The engine runs faithful paper-scale
sorts (N=2000) but was **host-bound** — the prior handoff profiled `reconnect_I` at ~72% of each
step and attributed it to **~12 host↔device syncs/step** in the gather→reserve→apply round-trip,
and prescribed "Option A: make the sweep device-resident."

## What was done

### 1. Option A — device-resident gather→reserve→apply (DONE; the prescribed change)
The device gather (`gather_*_configs_warp`) already returns packed device arrays
(`valid/caps/side/arms/top/bot`). They were being round-tripped to host `ICfgIdx` objects, re-packed
by `pack_footprints`, and re-packed again by `apply_*_batch_warp`'s `col`/`mat` lambdas. Now:
- `reconnect_warp.py`: `apply_i_to_h_won_kernel` / `apply_h_to_i_won_kernel` (+ `apply_*_device_warp`
  wrappers) — the gated `*_batch_kernel` bodies plus a `won[i]==0` skip guard, reading the gather's
  device arrays directly and dropping the unused output readback.
- `schedule_warp.py`: `reserve_i_won_device` / `reserve_h_won_device` (+ device reserve/check
  kernels) run the lowest-id-wins reservation straight off the gather arrays (invalid rows skip);
  both `reconnect_sweep_*_warp_device` loops rewritten to `find_*_warp` → device gather → device
  reserve → **1-int winner-count readback** → parallel device apply, with an empty-step fast path.

**No compaction or on-device sort needed:** compaction is order-preserving and lowest-id-wins
depends only on relative order, so launching reserve/apply over ALL M gathered candidates and
skipping invalid/non-winner rows yields the SAME winner set the host reference picks. Round-1
body-anchored fingerprint stays identical (gated by `test_fully_device_{,h_}sweep_round_matches_host`).

**Result:** the gather→reserve→apply portion dropped from the profiled host-bound cost to ~0.8 ms/step.
But the end-to-end step barely moved — because the handoff's diagnosis was wrong.

### 2. The real bottleneck — the I-detect SCAN, not the host syncs (RE-DIAGNOSED)
Isolated micro-benchmarks at the warmed-up N=2000 (n=10 foam) mid-sort state:
- `find_short_edges_warp` (I-detect scan): **3.95 ms/call** — THE bottleneck, run every step + every round.
- `find_small_triangles_warp` (H-detect scan): 0.24 ms — cheap (simple per-surface kernel).
- gather + reserve + apply (post-Option-A): ~0.8 ms/step combined.

Splitting the I-scan kernel: `d_vert_body_count` **alone = 3.51 ms**, edge-loop alone = 0.027 ms.
The cost is the O(L²) **double-indirect** distinct-body count (`s2b[v2s[v,i2],slot2]`) run over all
~12.7k vertices — **memory-latency-bound** (removing the arithmetic/dedup did nothing). And it's the
per-vertex kernel's *register pressure* that hurts: even calling `d_vert_body_count` lazily (~16×,
confirmed by an invocation counter) left the scan at 3.9 ms, because its mere PRESENCE in the
per-vertex kernel tanks occupancy (an identical kernel WITHOUT it: 0.17 ms).

### 3. Scan optimization (DONE) — two-kernel split + device-slice readback
- `scan_short_edges_kernel` is now a pure **length trigger** (no `d_vert_body_count`) → high occupancy.
- `filter_interior_short_edges_kernel` applies the interior (4-cell) filter over just the few emitted
  candidates (body_count off the per-vertex hot path).
- The in-kernel w-dedup was also removed (it emits an edge once per incident face; the host dedups
  via `np.unique(axis=0)` after the canonical sort).
- **Readback fix:** `out_v10.numpy()[:k]` copied the ENTIRE `cap = 2·n_v·MAX_VS` array (~6.8 MB ×2)
  before slicing; `out_v10[:k].numpy()` slices on-device first → copies only `k`.

`find_short_edges_warp`'s output (interior-filtered, deduped, (v10,v11)-sorted) is **byte-identical**
to before, so NO gate/contract change (`test_i_scan_matches_host_trigger` etc. pass unchanged).

**Result:** `find_short_edges_warp` 3.95 → **0.50 ms** isolated (~8×).

### 4. Per-surface I-scan (DONE) — the "remaining lever", investigated and resolved
After (1)–(3) `recon_i` was still ~the dominant phase, and a quiescence profile (warmup 20k, only
0.03 reconnections/step) showed `recon_i` = **2.1 ms** even with ~zero reconnections — i.e. it is the
per-step SCAN finding nothing, not the reconnection work. The scan was cheap warm (0.03 ms) but ~10×
**cold-cache-penalized** in the dynamic loop, because the per-VERTEX kernel did a scattered
`d_ring_neighbor`→`d_ring_pos` ring-search for each (vertex, incident-surface, side) with no early-out.
Contrast `recon_h` = 0.29 ms: the H-scan is per-SURFACE.

Fix: restructure `scan_short_edges_kernel` to **one thread per surface**, emitting each implicit edge
(consecutive `s2v` ring pair) under threshold. Implicit edges ARE consecutive ring pairs, so this finds
the IDENTICAL edge set — but reads `s2v[s,:]` contiguously and needs no `d_ring_pos` (same access
pattern as `compute_geometry_warp`, which is ~0.29 ms). Each edge is shared by two surfaces → emitted
twice; the host already dedups (`np.unique`) + the interior filter (`filter_interior_short_edges_kernel`)
runs on the smaller endpoint. `find_short_edges_warp` output stays **byte-identical** → no gate change.

**Result:** scan ~2.1 → ~1.0 ms in the dynamic loop; static `reconnect_sweep_warp_device` 2.07 → 0.98 ms.

### 5. Gate `orient_repair` on reconnections (DONE) — ~0.3 ms/step, exact
`orient_repair_warp` ran every interval step (`engine.py`) and pays a full `compute_geometry_warp`
recompute (orient_warp.py:97) even though it is IDEMPOTENT — a no-op (0 flips, `s2v` unchanged) on a
consistently-wound mesh. Windings only go inconsistent from (a) the initial foam's near-degenerate
faces and (b) reconnection surgery — NOT the per-step integrate — so the per-step orient on
no-reconnection steps was pure redundant cost (~0.3–0.38 ms). Gated it on `(ni+nh)>0` plus ONE initial
heal (`_healed_initial` flag in `g`, fired at the end of the first step exactly where the old code's
step-0 orient ran → same trajectory point). Skipping the no-op orients is **bit-identical**.

## Net result
End-to-end `forward_step` at N=2000, σ=0.5, dt=0.01 (300-step phase profiles inflate vs the real loop):
- **mid-sort step 10.06 → 3.68 ms/step**; `recon_i` 7.70 → 1.44 ms.
- **quiescent step 3.87 → 2.77 ms/step** (pre orient-gating); `recon_i` 2.10 → 1.04 ms.
- **Real 20k stability loop: 9.09 → 3.05 ms (per-surface) → 2.78 ms (orient-gated) — ~3.3×.** More at
  quiescence, where every step skips orient. 100k STABLE; before/after at matched 20k is BIT-IDENTICAL
  at every checkpoint (het + reconnection counts + volumes match to the last digit) — pure perf.

All gates green (132-test gate; 79 GPU tests incl. round-1 fingerprint equivalence + engine forward-step).

## Remaining bottleneck + next levers (NOT done)
`recon_i` (the per-surface scan, ~1 ms/step, mildly cold-cache-bound) is the largest phase left; `force`
(~0.6 ms, the 4 actors — real physics, not overhead) is next. Further wins, in rough order:
1. **Skip the scan on quiescent steps.** Piggyback a global `min_edge` atomic-min onto
   `compute_geometry_warp` (already iterates every ring edge, ~free) and skip the I-sweep when
   `min_edge >= Lth + margin`. CAVEAT: geometry is PRE-integrate, so it needs a conservative margin
   (≥ max per-step displacement) to never miss a reconnection; a too-small margin delays a rare
   reconnection by 1 step (breaks the bit-identical trajectory). Faithfulness call — get buy-in.
   This is the biggest remaining single win (~1 ms on the majority quiescent steps).
2. **Dirty-region re-scan.** The sweep re-scans the WHOLE mesh after every applied round; only the
   reconnection neighbourhoods gain new short edges. Exact, but only helps multi-round (minority) steps.
3. **CUDA graphs / `force`-kernel fusion** — secondary; `force` is real work, low ROI.

## Update (later sessions — these "next levers" are now superseded; see memory `reconnect-sweep-scan-bottleneck`)
- **Lever 1 (skip-scan) is DEAD** — measured: the foam ALWAYS has a sub-Lth edge (0/3000 steps had
  `found==0`); short edges persist but are mostly Condition-4-BLOCKED, so the scan can never be skipped.
  Also the in-kernel `min_edge` atomic perturbs FP codegen ~1 ULP → breaks bit-identicality.
- **What WAS done instead (all bit-identical):** compact double-buffer (3.56→0.19 ms at n=16), detect
  scan-buffer reuse + tighten, and **device-resident I→H detect (2026-06-25)** — `find_short_edges_device`
  keeps the candidate list on-device through the round (on-device dedup/lex-sort reproducing
  `np.unique(axis=0)` via int64-key `radix_sort_pairs` + `array_scan`; only the count M is read back),
  consumed by `gather_i_configs_warp_device` with no h2d. Concurrency K=16 throughput +12% (202→226
  sim-steps/s), util 43→46%; single-sim per-step −4%. 133-gate green; 20k recon 4010/3028, het@10k=0.4638.
