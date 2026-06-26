# GPU 3D-vertex engine — scalability sweep (2026-06-26)

Two sweeps of the Warp engine (`rnr/gpu`, eager `engine.forward_step`, faithful production
params: K_V=10, K_A=1, σ=0.5, v0=0.1, Dr=1, Lth=1e-3, dt=0.01 → reconnect every step) on the
**RTX 5090** (32 GB, sm_120). Harness: `rnr/scripts/gpu_scalability.py`. Each config does
`--warmup 300` untimed steps (kernel JIT + initial orient-heal + reconnection ramp) then **times
2000 steps** with a CUDA sync around the window — ms/step / util / VRAM are *rates* that converge
in a few hundred steps, so the window equals a 100k run for these metrics at ~50× less cost.
Foam = two-type BCC (Kelvin) cells, `cells = 2·n³`, mixed IC, unit-cell scaled. Total run 38.6 min.

> **Timing hygiene.** `gpu_stability.py`'s per-step number is *contaminated* by its per-checkpoint
> audit — `het_contact_fraction` is a Python loop over **every** surface (200k+ at scale) plus a
> full device→host copy and numpy geometry recompute. That audit, not the GPU step, dominated the
> "550 ms/step at n=24" we first saw. This harness times pure stepping; the true cost is ~5.7 ms.

## Headroom must scale with mesh size (fixes the n≥36 "illegal memory access")

The bump-allocated vertex/surface slots have capacity `cap = n_used + headroom`. One **parallel**
reconnection round reserves slots for *all* simultaneously-short edges, which grows with mesh size.
The fixed default `headroom=4000` overflows above ~n=32 → **Warp CUDA error 700 (illegal memory
access)** (surfaces at the next d2h/free). This is NOT a fundamental ceiling — the harness sets
`headroom = max(4000, 0.10·n_verts)` and the engine then runs cleanly to **n=48 (221k cells,
1.35M verts)**. (`gpu_stability.py` keeps the fixed default and so still crashes past n=32; if we
want it to scale too, lift the same rule into its `--headroom`.)

## Study 1 — serial scalability (one sim)

| n | cells | verts | ms/step | steps/s | GPU util (mean) | VRAM peak | VRAM (sim) |
|--:|------:|------:|--------:|--------:|----------------:|----------:|-----------:|
| 8  | 1,024   | 6,253     | 2.02  | 495 | 46% | 3,343 | 533  |
| 12 | 3,456   | 21,?      | 2.63  | 381 | 44% | 3,379 | 569  |
| 16 | 8,192   | 49,894    | 3.73  | 268 | 52% | 3,411 | 601  |
| 20 | 16,000  | 97,?      | 4.92  | 203 | 62% | 3,507 | 697  |
| 24 | 27,648  | 168,437   | 5.99  | 167 | 70% | 3,661 | 851  |
| 28 | 43,904  | 267,?     | 7.86  | 127 | 77% | 3,795 | 985  |
| 32 | 65,536  | 399,189   | 10.20 | 98  | 81% | 4,019 | 1,209 |
| 36 | 93,312  | 568,402   | 16.97 | 59  | 70% | 4,303 | 1,493 |
| 40 | 128,000 | 779,752   | 17.35 | 58  | 86% | 4,717 | 1,907 |
| 44 | 170,368 | 1,037,983 | 21.89 | 46  | 87% | 5,071 | 2,261 |
| 48 | 221,184 | 1,347,355 | 32.45 | 31  | 80% | 5,614 | 2,804 |

(MiB; VRAM(sim) = peak − 2,810 base. n=36/n=44 reconfirmed from warm cache within 2%.)

- **Per-step time** is sub-linear in cells at small n (host/launch-bound, util < 55%) and ~linear
  once GPU-bound (util 80–90%). Throughput falls from 495 → 31 steps/s across 216× the cells.
- **The host→GPU transition is ~n=24–32** (util crosses 70→80%). Below it the GPU is starved by
  the per-step host launch loop (incl. `forward_step`'s `n_used.numpy()` sync); above it the
  reconnection/geometry kernels saturate the SMs.
- A mild **kink at n=36** (10.2→17.0 ms from n=32, then flat to n=40) is reproducible — a kernel
  occupancy / candidate-buffer-growth threshold around 0.4–0.6M verts, not measurement noise.
- **VRAM is linear** in cells (~13 MiB per 1,000 cells over base): 0.5 GB at 1k cells → 2.8 GB at
  221k. Memory is *not* the limit for a single sim — a 32 GB card could hold ~10× more.

## Study 2 — concurrency scalability (K independent sims at once)

| n | K | per-sim ms/step | per-sim steps/s | **aggregate steps/s** | speedup vs K=1 | total VRAM | util |
|--:|--:|----------------:|----------------:|----------------------:|---------------:|-----------:|-----:|
| 16 | 1  | 3.76   | 266 | **266** | 1.00 | 3,409  | 53% |
| 16 | 2  | 6.64   | 151 | **301** | 1.13 | 4,036  | 82% |
| 16 | 4  | 12.55  | 80  | **319** | 1.20 | 5,284  | 91% |
| 16 | 8  | 25.80  | 39  | **310** | 1.17 | 7,721  | 92% |
| 16 | 24 | 79.97  | 12.5| **300** | 1.13 | 17,358 | 92% |
| 24 | 1  | 5.99   | 167 | **167** | 1.00 | 3,628  | 70% |
| 24 | 4  | 22.43  | 45  | **178** | 1.07 | 6,052  | 95% |
| 24 | 24 | 138.0  | 7.3 | **174** | 1.04 | 22,040 | 95% |
| 32 | 1  | 10.15  | 99  | **99**  | 1.00 | 4,069  | 82% |
| 32 | 4  | 40.79  | 25  | **98**  | 0.99 | 7,679  | 97% |
| 32 | 12 | 122.2  | 8.2 | **98**  | 1.00 | 17,354 | 97% |
| 32 | 24 | — | — | *skipped* (VRAM guard: 31.8 GB > 93% of 32.6) | | | |

(full grids K∈{1,2,4,6,8,12,24} in `gpu_scalability_study2.csv`.)

**Concurrency buys essentially nothing here — the GPU is already saturated by one sim at scale.**

- **n=32:** aggregate throughput is dead flat at ~98 steps/s for *all* K (speedup 0.99–1.00). A
  single sim already runs the SMs at 82%; extra sims just time-share, so per-sim latency scales
  ~linearly with K (10 → 122 ms at K=12) while total work/s is unchanged.
- **n=24:** flat ~167–178 steps/s; best case K=4 = 1.07×. Negligible.
- **n=16:** the only regime with headroom — a single sim leaves the GPU ~50% idle (host-bound), so
  K=2–4 recovers ~20% via launch-overlap (**peak 1.20× at K=4**), then plateaus immediately as the
  shared host-launch path becomes the bottleneck. Never approaches linear (ideal would be 24×).
- **VRAM scales linearly** with K (~600 MiB/sim at n=16, ~1.2 GB/sim at n=32 + 2.8 GB shared base).
  The card holds 24-way at n=16/24 but only ~12-way at n=32 (the guard skips n=32×K=24).

### Takeaway for ensembles

Run sort ensembles **serially**, not concurrently. After the 2026-06 optimizations (compact/detect
buffer reuse, captured step), a single sim is fast enough (266 steps/s at n=16) that it already
*exceeds* the old 16-way aggregate (215 steps/s, memory `reconnect-sweep-scan-bottleneck` /
`cuda-graph-experiment`). Concurrency only helped back when one sim left the GPU idle; that gap is
now closed. The lever for more aggregate throughput is a **faster single step** (kill the per-step
`n_used` readback; the captured-while step), not more concurrent processes. Concurrency is still
useful only to hide *foam-build/load latency* or at small n (≤8k cells) where K=2–4 nets ~20%.

## Captured step vs eager under concurrency — "did the recent optimizations ruin concurrency?"

Study 2 above used the EAGER `forward_step`. The most recent optimization is the CUDA-graph
`CapturedStep` (no per-step host readback). Head-to-head, aggregate steps/s at matched (n,K)
(`gpu_scalability_study2_{captured,eager}.csv`, timed=1500):

| n | K | eager agg | captured agg | captured util | eager util |
|--:|--:|----------:|-------------:|--------------:|-----------:|
| 16 | 1  | 265 | **369 (+39%)** | 95% | 54% |
| 16 | 4  | 312 | **332 (+6%)**  | 99% | 90% |
| 16 | 8  | 301 | **332 (+10%)** | 100%| 91% |
| 16 | 16 | 295 | **334 (+13%)** | 100%| 91% |
| 24 | 1  | 168 | **204 (+21%)** | 99% | 70% |
| 24 | 4  | 178 | **183 (+3%)**  | 99% | 95% |
| 24 | 16 | 175 | **183 (+5%)**  | 100%| 95% |

**Captured aggregate ≥ eager at every (n,K) — the optimizations did NOT reduce concurrent
throughput; they raised it everywhere.** What changed is *why* concurrency was ever helping:

- The eager step does a `g["n_used"].numpy()` **host sync every step** → the GPU stalls on the
  host round-trip (eager K=1 util only 54%). Concurrency *filled that idle* (eager K=1→K=4 lifts
  265→312). The captured step has **no per-step host sync** → one sim already runs the SMs at
  **95–100%**. There is no idle left to fill.
- So **captured reaches the GPU roofline with a SINGLE sim**; eager needed ~4 concurrent sims to
  reach the *same* roofline (n=16 plateau ≈ 330 captured / ≈ 305 eager; n=24 ≈ 183 / ≈ 177).
- Consequently **concurrency is mildly COUNTERPRODUCTIVE for the captured path**: captured K=1
  (369) > captured K=4 (332) — extra sims only add memory/context contention with zero compute
  headroom to recover. The single best single-GPU throughput is **serial captured, K=1**.

Reconciliation with memory: the "+42% ensemble" (`cuda-graph-experiment`) is the *single-sim*
gain (here +39% at n=16 K=1); the concurrent-ensemble gain is smaller (+4–13%) because the eager
ensemble already clawed back throughput via concurrency. The "+76%" (`reconnect-sweep-scan-
bottleneck`) was the optimization speeding the K=16 ensemble, not concurrency out-scaling serial —
even then K=16 agg (215) ≈ single-sim (216).

**Practical upshot: run ensembles SERIALLY with the captured step.** Captured K=1 beats every
eager config and every concurrent captured config. Concurrency only helps the *eager* path (to
recover its host-sync idle), and even then only up to the roofline captured hits alone. The lone
residual concurrency *cost* of the recent work is VRAM: buffer-reuse made scratch persistent
(~150 MB/sim, memory `reconnect-sweep-scan-bottleneck`) and capture adds a graph (~tens of MB/sim),
lowering max-K — but that trades against not needing concurrency at all. (Measured at n=16/24,
the host-bound regime where the gap is widest; at n≥32 eager is already ~GPU-bound so the
captured-vs-eager single-sim gap narrows, but the ordering holds.)

Artifacts: `rnr/exports/gpu_scalability_study{1,2}.csv` + `…_study{1,2}.png` +
`gpu_scalability_study2_{captured,eager}.csv`; per-worker logs in `rnr/exports/scalability_logs/`.
