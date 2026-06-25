# Reconnect-sweep, round 2: the skip-scan dead-end + gather buffer reuse — 2026-06-25

Continues `docs/2026-06-25_reconnect-sweep-optimization.md`. The prior session left two "next
levers": (1) skip the detect scan on quiescent steps, (2) make the gather/reserve/apply cheaper.
This session **killed lever (1) as infeasible** and instead **cut the gather's per-round allocation
overhead** (bit-identical, ~6–7% faster).

## Lever (1) — "skip the scan on quiescent steps" is DEAD (do not re-attempt)

The handoff's premise was that most steps are quiescent (no short edge) so the scan can be skipped.
Measured on the deterministic baseline (n=10, dt=0.01, σ=0.5 mixed):

- **`found==0` (no short edge present → the only safe skip): 0/3000 early AND 0/1500 late** (after a
  25k warmup). The global minimum edge length is *always* below Lth=1e-3 (median 1.5e-4, max 8.3e-4).
- Reconnections ARE rare (93% of late steps apply nothing), but **short edges are not** — they are
  mostly **blocked** (Condition-4 veto / not a valid 4-cell I-site) and persist sub-Lth indefinitely.
  So the scan always finds candidates; it can never be skipped.
- The originally-proposed *global* gate `min_edge − 2·max_disp ≥ Lth` is doubly dead: `max_disp` is a
  global max ≈ 14×Lth (stiff kv=10 volume forces on some far-off vertex), so the gate would fire on
  0/3000 steps even ignoring the always-short-edge fact.
- **Gotcha:** adding the `min_edge`/`max_disp` reductions *inside* the hot `surface_geom_kernel` /
  `integrate_kernel` perturbed their FP codegen ~1 ULP → broke the bit-identical trajectory even with
  the gate inert (passed the `<1e-9` single-step GPU==host tests, but compounded over ~5k steps). Any
  future in-kernel reduction on those hot kernels MUST live in a separate kernel.

## Profile: where the ~2.78 ms/step actually goes (mid-sort, warmup 2000)

Per-phase (sync between phases inflates absolutes ~25%; relative split holds):

| phase | ms | note |
|---|---|---|
| recon_I | 1.44 | scan 0.56 + **gather/reserve/apply 0.88** |
| force | 0.55 | 4 actors — real physics |
| recon_H | 0.50 | scan 0.26 + gather/apply 0.24 |
| compact | 0.48 | reconnection steps only (~1.1 ms each) |
| geom 0.25 · orient 0.16 · integ+director 0.11 | | |

The scan (0.82 ms I+H) is **no longer the bottleneck** (the prior per-surface rewrite shrank it).
Fine breakdown of recon_I: scan 0.56, **gather 0.27, reserve 0.08**, count-readback ~0.05, apply 0.03.
Round-1 candidates: **I: M≈16.6/step, 97% gathered-then-rejected** (valid≈0.43, all valid win); H≈0.3.

A micro-benchmark settled the gather's cost: **74% of it (0.197 ms) is per-round alloc+upload+fill of
its 9–11 output arrays**; the kernel itself is only 0.070 ms. The reserve's 0.08 ms is entirely 3
capacity-sized fills (cap_v 16202 / cap_s 18202 / nb 2000) + 2 launches, every round.

Crucially, **caching/​filtering blocked candidates would change candidate row-indices → change the
parallel apply's bump-allocation order → diverge the trajectory** (not bit-identical). So the
bit-identical rework is **buffer reuse** (eliminate the per-round allocs/fills), NOT candidate caching.

## What was done — gather output-buffer reuse (DONE, bit-identical)

`gather_warp.py`: `_ensure_gather_buf(g, key, m, with_tri)` allocates the 9–10 int32 output arrays
ONCE on `g` (grown ×2 on demand) and reuses them across rounds/steps; `gather_i_configs_warp` /
`gather_h_configs_warp` gained an optional `buf=` arg that, when present, writes into the reused
arrays, drops the end `wp.synchronize_device`, and returns `[:m]` views. `schedule_warp.py`'s two
device sweeps pass the buffers. Correctness (bit-identical): the kernel writes `valid[i]` for every
launched row; reserve/apply read the other fields only for valid rows and launch over `m` (the
returned views' `shape[0]`), so stale data in invalid rows or rows ≥ m is never consumed. The
fresh-allocation path (no `buf`) is unchanged for tests / `gather_*_to_list`.

**Result: 20k stability bit-identical at every checkpoint** (het@10k=0.46382583300146024,
recon_I/H@20k=4010/3028 — match to the last digit) and **2.83 → 2.64 ms/step (~6.6% wall)**.

## Remaining levers (NOT done)

- **Reserve owner-array reuse** (~0.04 ms, ~1.5%): pre-allocate vown/sown/bown once at a FIXED
  sentinel (any value > max candidate id is bit-identical), and after reserve+check reset only the
  footprint to sentinel via a small kernel (mirrors the reserve's atomic targets). Replaces 3
  capacity fills/round. Same overhead-elimination pattern as the gather.
- **force kernel** (0.55 ms, bit-identical-able): the O(valence²) active-drive distinct-body dedup +
  the per-surface ring-index re-search. Gated by GPU==host `<1e-9`.
- **compact** (0.48 ms): throttle to every-K-reconnections / a slot-usage threshold. NOT
  bit-identical (slot renumbering shifts reservation order) → a faithfulness tradeoff (re-validate
  het curve + stability), like the dt lever.
- The scan (0.82 ms) is near-irreducible: O(surfaces) every step, already per-surface, can't be
  skipped (always a short edge).
