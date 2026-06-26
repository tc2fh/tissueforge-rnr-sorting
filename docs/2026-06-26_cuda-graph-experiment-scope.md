# CUDA-graph capture of `forward_step` — experiment scope

**Date:** 2026-06-26 · **Status:** P0 + capture_while de-risk + **P1 + P2 DONE** (all bit-identical, 134-gate); **P3 next** · **Branch:** `migrate/linux64-wsl2`

## Progress (2026-06-26)

- **Phase 0 DONE** — verdict PROCEED, bottleneck localized to the reconnect path (see results below).
- **capture_while de-risk DONE** — `scratchpad/test_capture_while.py`: `wp.capture_while` (eager + inside
  a captured graph, replayed 2× with different device conditions → looped 3 then 7 times) and
  `wp.capture_if` (fires/skips on a device flag) all **PASS** on this RTX 5090 / WSL2 / driver. CUDA
  conditional graph nodes work here → P3's device-side round loop is viable.
- **P1 DONE** (alloc-free step path, commit pending) — `physics_warp._ensure_step_buffers` + persistent
  geometry/force/surface-geom buffers on `g` (zero_ in place); `orient_warp` persistent snw/clo/flip/
  counter (snw copied not cloned). Byte-identical (134-gate, 2k/20k). 0 perf change (mempool already free)
  — purely the capture prerequisite (no allocation allowed inside a capture region).
- **P2 DONE** (pointer-stable compact, commit pending) — `compact_warp` copies the compacted scratch back
  into g's canonical fixed-address arrays instead of the pointer ping-pong (compact_warp.py tail), so a
  captured graph's device addresses stay valid across compacts. Byte-identical (134-gate, 2k). Costs one
  device→device SoA copy/compact (~tens of µs) — the price of capture-compatibility.
- **P3 NEXT** — the load-bearing phase (see plan below). Remaining reconnect-path allocs (reserve/apply
  owner+won arrays per round) still need pre-allocation as part of P3's capture region.



## Phase 0 RESULTS (2026-06-26) — mechanism confirmed, bottleneck localized to the reconnect path

Probe `scratchpad/prof_graph_phase0.py` (static 5-kernel relaxation step `director→surface_geom→
body_geom→force→integrate`, **reconnection OFF**, pre-allocated scratch; baseline host-launches vs
`wp.ScopedCapture` replay, identical per-thread work):

| mode | K | round_ms | util |
|---|---|---|---|
| baseline | 1 | 1.214 | 79% |
| graph | 1 | 1.076 (−11%) | 98% |
| baseline | 16 | 18.879 | **99%** |
| graph | 16 | 18.440 (−2%) | **99%** |

- **Mechanism works:** Warp capture/replay runs clean; `capture_while`/`capture_if` confirmed present.
- **Single-sim:** graphing the prefix is a real but small win (−11% round_ms, util 79→98) — launch
  overhead matters when one small sim can't keep the GPU fed between launches.
- **THE KEY RESULT — K=16 prefix is already at 99% util.** The static prefix is **compute-bound** (the
  force kernel ≈ 0.9 ms dominates) and fully saturates the GPU with 16 sims round-robin. Graphing it
  adds nothing at K=16 (−2%, in the noise).
- **Therefore the ~47% full-step ceiling is NOT the prefix — it is entirely the RECONNECT PATH.** The
  concurrency probe steps sims *sequentially* (`for i: forward_step(sim_i)`), and each sweep round does a
  **full-device `wp.synchronize_device(dev)`** (schedule_warp.py:533) to read `M`/`n_win` — a global
  barrier that blocks ALL sims' overlap, idling the GPU while one sim's host decides its next round.
- **VERDICT: PROCEED — but redirect the effort.** Don't graph the prefix standalone (already optimal).
  The entire payoff is in (a) eliminating the reconnect path's per-round full-device syncs via a
  **device-side round loop** (`capture_while`, P3) and (b) **cross-sim overlap** (P4 / a batched driver).
  Phase 0 de-risked the mechanism and pinpointed the target; the expensive phases are justified.

**Plan refinement from P0:** consider a cheaper intermediate before full `capture_while` — a **batched
multi-sim driver** that advances all K sims through each reconnect round together (stream-parallel
launches + ONE sync/round instead of one per sim-round → a K× cut in sync count). It may recover much of
the util without the conditional-node restructure; if it doesn't, graphs are the answer. Test it first.

---


## Goal & the decision it informs

Push K=16 concurrency utilization from the measured **~47%** toward saturation by collapsing the
**~10 host syncs/step** into a device-resident CUDA-graph replay. The throughput number this produces is
**the go/no-go input for the C++/CUDA port question**: if Warp graphs get util to ~80–90% (≈1.7–1.9×
ensemble throughput), C++ is *not* needed for performance and the port reduces to the TF-integration
milestone (do it when the algorithm is frozen). If graph capture proves too constrained in Warp 1.14, the
specific failure is the concrete signal that raw CUDA/C++ is required.

## Why it should work, and why it's a Warp (not C++) job

- The kernels are **already native CUDA** (Warp JITs to PTX). The ceiling is host orchestration, not compute
  — `concurrency_probe` shows util ~47% at K=16 (GPU half-idle) because each `forward_step` is a host-driven
  sequence of ~10 syncs (detect `k`/`M` per round × 2 sweeps, `won.sum`, `n_used`, orient `counter`).
- Warp 1.14 (confirmed installed) exposes the full graph API: `wp.capture_begin/end/launch`,
  `wp.ScopedCapture`, **and `wp.capture_while` / `wp.capture_if`** (CUDA 12.4+ *conditional graph nodes*).
  So even the data-dependent round loop and the conditional compact/orient can run **device-side** inside the
  graph — no per-round host trip. (Requires a CUDA 12.4+ driver; this box reports CUDA 12.8 — verify in P3.)

## What blocks naive capture today — the three prerequisites

Graph capture has two hard rules: **(a) no memory allocation during capture, (b) no host syncs during
capture**, and **(c) replay reuses the exact device addresses recorded at capture time**. Current code
violates all three:

1. **Per-step allocations.** `compute_geometry_warp` (physics_warp.py:316, 7× `wp.zeros`),
   `compute_forces_warp` (:340, 1× `wp.zeros`), and `orient_repair_warp` (orient_warp.py:99–101 + its
   `compute_surface_geom_warp` 3×) allocate every call. detect/gather/compact buffers are *already*
   pre-allocated (this is why the prior buffer-reuse work matters here even though it gave **0 latency win
   standalone** — it is a hard **capture prerequisite**, now resurrected).
   → Move geometry/forces/orient scratch to pre-allocated buffers on `g` (zero_ in place to match `wp.zeros`).
2. **Pointer instability across compact.** `compact_warp` ping-pongs — `g[k] = dst[k]` (compact_warp.py:160–163)
   replaces the mesh arrays with the alternate buffer set, so their **addresses change every compact**. A graph
   captured once goes stale the moment compact runs.
   → Make compact write back into the **canonical fixed-address arrays** (one extra device→device copy,
   async, no sync). (Alternative for `interval=1` only: capture 2 alternating graphs for the deterministic
   ping-pong parity — fragile, not recommended.)
3. **Host-readback control flow.** The sweep round loops read `M` (find_short_edges_device, schedule_warp.py:527)
   and `n_win` (`won.numpy().sum()`, :534) to host each round and `break` on them; compact/orient are host
   `if (ni+nh)>0` (engine.py:52,61); `n_used` is read every step (:64).
   → detect writes `M` to a **device scalar** (not host); the round loop becomes `wp.capture_while` on a device
   "candidates remain" flag; gather/reserve/apply launch over a **fixed `MAX_CAND` dim and self-mask** on the
   device-`M`; compact/orient gated by `wp.capture_if` on a device "reconnected" flag. The recon-count/`n_used`
   readback for stats + the slot-exhaustion safety check moves to **one** post-replay sync (or every ~500 steps).

## Bit-identicality argument (the gate is byte-identical trajectories)

- `wp.capture_while` loops the **same rounds** as the host `while` (condition ≡ "candidates remain", the same
  predicate as `m>0`/`n_win>0`), so the executed work is identical — only the driver moves device-side. No
  arithmetic is reordered.
- Fixed-`MAX_CAND` launches with device-`M` masking do the **same work over the same rows** (threads ≥ M
  early-return). Same as today, just a static launch dim.
- **Risk:** `MAX_CAND` must upper-bound candidates/round (raw emit `k` ~150 at n=16; pick 4096–8192 with
  margin). Add a **device overflow flag** (assert) so an exceedance trips the gate rather than silently
  dropping candidates → non-bit-identical.
- Validated by the existing gates throughout: `pixi run test` (134) + 2k/20k byte-identical trajectory
  (recon I/H=4010/3028, het 0.4604).

## Phased plan — cheapest signal first, with go/no-go gates

- **Phase 0 — mechanism + ceiling probe (~0.5–1 day, NO restructure).** Run **reconnection OFF** (pure
  relaxation: director→geometry→forces→integrate = 5 pure kernels, fixed dims, no compact → pointers stable;
  pre-alloc just the geom/force scratch). Wrap the 5-kernel step in `wp.ScopedCapture`, replay per step, probe
  K=16. **GO/NO-GO:** if util barely moves with 5 static kernels graphed, launch overhead is *not* the
  bottleneck → graphs won't save us → STOP and reconsider (kernel occupancy / multi-stream only). If util
  jumps, the mechanism works for our kernel sizes → proceed.
- **Phase 1 — geometry/forces/orient buffer-reuse (~1 day).** The P0 prerequisite, generalized. Independently
  bit-identical + 134-gate-checkable; no behavior change.
- **Phase 2 — pointer-stable compact (~2–3 days).** compact copies back into canonical fixed-address arrays.
  Re-validate byte-identical (recon 4010/3028). Unblocks capturing the recon path.
- **Phase 3 — device-side control flow + full-step capture (~3–5 days).** detect→device-`M`; `capture_while`
  round loops; `capture_if` compact/orient; `MAX_CAND` masked launches + overflow flag; one post-replay
  readback for stats/safety. Capture full `forward_step`, replay per step. Validate byte-identical; measure
  single-sim + K=16. **This is the load-bearing phase** (the `capture_while` authoring is the novel/risky part
  — watch Warp's capture constraints: no syncs/allocs in the body, `force_module_load=True` before capture).
- **Phase 4 — multi-stream the K sims (~2–3 days).** Each sim → its own captured graph + `wp.Stream`; replay
  all K, sync once. Independent graphs **overlap** instead of round-robin time-multiplexing — this is where
  ensemble throughput actually materializes. Re-measure util/throughput.

## Effort / risk

~**1.5–2.5 weeks** for a solid K=16 number. Risk **medium**: prerequisites P1/P2 are bounded and
independently validatable against the byte-identical gate; the novel risk is P3 (`capture_while` semantics +
making the loop body alloc/sync-free). De-risked by the P0 early read — if graphs don't help the pure-relaxation
5-kernel step, abandon before the expensive phases.

## What it tells us about the C++ port

- **Util → ~80–90% / meaningful K=16 multiplier:** C++ NOT needed for perf. Port = TF-integration milestone
  only (native `MeshQuality` op), done when the algorithm is frozen.
- **`capture_while`/conditional nodes prove too constrained in Warp 1.14** (capture fails; body can't be made
  alloc/sync-free): that's the concrete, proven trigger that device-side round control flow needs raw CUDA/C++.
- **Correction to the 2346 handoff note:** Warp *does* expose conditional graph nodes, so the "C++ uniquely
  enables device-side round control" claim is downgraded to "verify in P3." Don't pre-commit to C++ on that
  basis.

## Measurement protocol

- Reuse `scratchpad/prof_perstep.py 16 150` (single-sim) + `scratchpad/concurrency_probe.py 5.0 4000` (K=16,
  run 2× — ±5% noise). **NB:** the per-phase *bracketed* profiler is meaningless under capture (you replay the
  whole graph as one unit) — judge by **natural per-step + concurrency util** only.
- Gate every `rnr/gpu/*.py` change: `pixi run test` (134) + the 2k/20k byte-identical checks.

See [[reconnect-sweep-scan-bottleneck]] for the sync inventory this attacks; prior design notes
`docs/2026-06-25_reconnect-sweep-optimization.md`, `docs/2026-06-24_gpu-3d-vertex-model-exploration.md` §10.
