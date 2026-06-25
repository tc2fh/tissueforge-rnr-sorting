# GPU 3D-vertex sort — single-run scaling on the RTX 5090 (32 GB)

**Date:** 2026-06-25 · **Hardware:** RTX 5090 (32 GB, sm_120), WSL2 · **Engine:** fork
`feat/native-rnr-reconnection` + Warp GPU port (`rnr/gpu/`).

After the foam build was made O(N) this session (batch `Vertex.create` + geometric mesh-pool
growth in `tfMesh.cpp` — see `foam-build-scaling` memory), this sweep measures how a **single,
non-batched** simulation scales: build time, per-step wall time, peak VRAM, and GPU utilization.

## Setup

- One foam per run (NOT batched). `n` = BCC seeds per axis → **cells = 2·n³**, verts ≈ 6.1·cells.
- Faithful mixed sort: K_V=10, K_A=1, V0=1, A0≈5.4, σ=0.5, active v0=0.1, Dr=1, L_th=1e-3,
  dt=0.01, reconnect **every step**. Early-phase timing (150 steps after 40 warmup), the
  high-reconnection regime.
- **Capacity ≈ 1.3·nv** (auto-headroom = 0.3·nv): a fixed headroom overflows the reconnection
  bump-allocator at large N (see findings), so headroom is scaled with N for a consistent
  capacity/cell ratio.
- VRAM = absolute device `memory.used` (nvidia-smi); util = mean `utilization.gpu` over the timed
  window. Repro: `scratchpad/scale_probe.py <n> <steps> <headroom|0=auto>`, plot
  `scratchpad/plot_scale.py` → `rnr/exports/gpu_scale_sweep.png`.

## Results

| n  | cells   | verts     | ms/step | peak VRAM | GPU util | host build |
|----|---------|-----------|---------|-----------|----------|------------|
| 10 | 2,000   | 12,202    | 4.7     | 4.95 GB   | 26%      | (cached)   |
| 16 | 8,192   | 49,894    | 10.1    | 5.02 GB   | 33%      | (cached)   |
| 20 | 16,000  | 97,463    | 31.9    | 5.08 GB   | 21%      | (cached)   |
| 24 | 27,648  | 168,437   | 52.3    | 5.23 GB   | 21%      | (cached)   |
| 28 | 43,904  | 267,351   | 80.2    | 5.48 GB   | 22%      | (cached)   |
| 32 | 65,536  | 399,189   | 120.7   | 5.70 GB   | 23%      | (cached)   |
| 40 | 128,000 | 779,752   | 136.2   | 6.39 GB   | 35%      | 111.5 s    |
| 48 | 221,184 | 1,347,355 | 238.0   | 7.51 GB   | 35%      | 191.3 s    |
| 64 | 524,288 | ~3.2M     | —       | —         | —        | **CRASHED (host build)** |

(Build times shown only for fresh builds; n≤32 loaded from the foam cache. Fresh host build ≈
**0.87 ms/cell**, O(N).)

## Findings

1. **VRAM is not the constraint.** Linear at **~11.6 KB/cell over a ~4.93 GB fixed base** (CUDA
   context + Warp kernels). 32 GB extrapolates to a **~2.3M-cell ceiling** — unreachable in practice.

2. **The GPU is latency/launch-bound: 21–35% utilization throughout.** Compute is mostly idle
   waiting on kernel launches + per-step host syncs. Doubling cells barely moves utilization (and
   65k→128k cells was *sub-linear* in per-step time, 121→136 ms — the larger mesh amortizes launch
   overhead better). Big throughput headroom is being left on the table.

3. **Per-step ≈ 1–1.8 µs/cell** (early mixed sort, capacity 1.3·nv). A 100k-step production sort:
   ≈ 3.3 h at 65k cells, ≈ 6.6 h at 221k. **Per-step scales with CAPACITY, not just used count** —
   the tight-headroom run was ~2× faster per step than capacity=1.3·nv (some kernels are
   O(capacity)), so headroom must not be over-provisioned.

## Bottleneck ranking (what actually stops you)

1. **Host foam builder** — crashed at n=64 (**~524k cells / ~3.2M verts**); CPU-side TF object
   construction runs out of room before the GPU does. Current practical ceiling. Fix:
   TF-free direct-CSR builder (the handoff's "Option B").
2. **Per-step latency** — production runtime, not memory: ~½M cells ≈ 15+ h/run.
3. **VRAM** — ~2.3M cells; never the binding constraint in any practical regime.

**Largest confirmed single run: n=48 = 221,184 cells** (238 ms/step, 7.5 GB). The GPU side would go
further (VRAM ~11 GB at 524k); the host builder is the wall.

## Robustness gap (worth fixing for scale)

The reconnection bump-allocator overflows a **fixed** headroom at large N → an illegal memory access
that corrupts the CUDA context (this is why n=32 first failed at headroom=4000; it ran fine at 40000).
It should scale headroom with N (the per-round reconnection burst grows with mesh size) or fail
gracefully — and since per-step cost scales with capacity, the right fix is a "just enough"
auto-growing headroom, not a large static one.

---

# Concurrency: how many n=16 sims (8192 cells each) fit + scale on one GPU

Companion sweep: instead of one big foam, run K **independent** n=16 sims at once (K copies of the
cached foam uploaded as K device meshes, distinct active-drive seeds, time-multiplexed — the engine
syncs per step, so this is round-robin, not CUDA-stream-parallel). Repro:
`scratchpad/concurrency_probe.py <vram_budget_gb> <headroom>`, plot `scratchpad/plot_concurrency.py`
→ `rnr/exports/gpu_concurrency_n16.png`. headroom=4000.

| K (sims) | total cells | peak VRAM | ms/step/sim | agg steps/s | GPU util |
|----------|-------------|-----------|-------------|-------------|----------|
| 1        | 8,192       | 2.72 GB   | 11.1        | 90          | 9–18%    |
| 4        | 32,768      | 2.84 GB   | 8.7         | 115         | 32%      |
| 16       | 131,072     | 3.28 GB   | 8.2         | 122         | 34%      |
| 48       | 393,216     | 4.44 GB   | 8.3         | 120         | 34%      |
| 96       | 786,432     | 6.36 GB   | 8.6         | 117         | 34%      |
| 128      | 1,048,576   | 7.41 GB   | 11.1        | 90          | 29%      |
| 140      | 1,146,880   | 7.88 GB   | 11.6        | 87          | 28%      |
| 152      | 1,245,184   | 8.32 GB   | 9.6         | 104         | 30%      |

## Concurrency findings

1. **VRAM per concurrent sim is tiny: ~38 MB/sim over a ~2.69 GB base.** Dead linear →
   **8 GB holds ~143 concurrent n=16 sims** (measured: 140 = 7.88 GB, 152 = 8.32 GB). (Note: the
   ~2.7 GB context base here is smaller than the single-run sweep's ~4.9 GB — that run's longer
   trajectories grew the retained Warp mempool.)

2. **Concurrency *helps* — up to a point.** Per-sim cost drops 11.1 ms (K=1) → ~8.2 ms (K≥4) and
   util rises 9–18% → ~34% as independent sims fill each other's launch/sync gaps. Aggregate
   throughput rises 90 → ~122 sim-steps/s. **All gains saturate by K≈16.**

3. **Aggregate GPU throughput is fixed at ~122 sim-steps/s** (n=16). Past K≈16 the per-sim cost just
   tracks 1/K — running 16-at-a-time in 9 batches or 143-at-once costs the **same total wall-clock**.
   The only reason to pack more is fewer launches; the only reason not to is memory pressure (below).

4. **Beyond ~96 sims throughput DEGRADES** (per-sim 8 → 11 ms, util 34 → 28%) — memory/cache pressure
   from the large working set. So the VRAM-fit max (~143) is past the efficient operating range.

5. **The ~34% util ceiling persists even with 100+ concurrent sims** → the bottleneck is the
   **per-step host sync** (every `forward_step` reads device counters back to host), not compute or
   occupancy. Concurrency cannot break past it; removing/​batching the per-step readback is the real
   lever to reach the GPU's idle ~65%.

**Bottom line:** ~143 n=16 sims fit in 8 GB, but the **throughput sweet spot is ~16–48 concurrent**
(max efficiency, 3–4 GB, big headroom). More concurrency ≠ more total throughput — the GPU's n=16
capacity is ~122 sim-steps/s regardless, gated by the per-step host sync.
