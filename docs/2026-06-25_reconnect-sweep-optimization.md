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

## Net result
End-to-end `forward_step` at N=2000, σ=0.5, dt=0.01 (300-step phase profiles; the per-phase syncs
inflate vs the real loop, which is faster):
- **mid-sort step 10.06 → 3.68 ms/step (~2.7×)**; `recon_i` 7.70 → 1.44 ms.
- **quiescent step 3.87 → 2.77 ms/step**; `recon_i` 2.10 → 1.04 ms.
- Real 100k stability loop (pre-session): 9.09 ms/step → (this session) measured below.

All gates green (132-test gate; 79 GPU tests incl. round-1 fingerprint equivalence). 100k stability
STABLE; before/after at matched 20k steps is BIT-IDENTICAL (het + reconnection counts) — pure perf.

## Remaining bottleneck + next levers (NOT done)
`recon_i` is still the largest phase (~37–39%), now ~1 ms/step — the per-surface scan ran ~1×/step
(plus per-round re-scans) and is still mildly cold-cache-bound. Further wins, in rough order:
1. **Skip the scan on quiescent steps.** Piggyback a global `min_edge` atomic-min onto
   `compute_geometry_warp` (already iterates every ring edge, ~free) and skip the I-sweep when
   `min_edge >= Lth + margin`. Caveat: geometry is PRE-integrate, so it needs a conservative margin
   (≥ max per-step displacement) to never miss a reconnection; a too-small margin delays a rare
   reconnection by 1 step (breaks the bit-identical trajectory). Faithfulness call — get buy-in.
2. **Dirty-region re-scan.** The sweep re-scans the WHOLE mesh after every applied round; only the
   reconnection neighbourhoods gain new short edges. Exact, but only helps multi-round (minority) steps.
3. **Reduce `orient_repair` frequency** (0.37 ms/step, runs every interval step). Windings only change
   on reconnection, so it is redundant on no-reconnection steps — BUT the initial foam has degenerate
   windings it heals at startup, so gating on `(ni+nh)>0` needs the initial heal handled (e.g. fold
   into the foam-cache build).
