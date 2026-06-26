# CUDA-graph capture of `forward_step` — experiment scope

**Date:** 2026-06-26 · **Status: ★ P3 RESOLVED** — full forward_step graph capture WORKS; **captured-sequential replay SATURATES at production scale (util 90% @ n=16, K=8; single-sim −33%)**. GO/NO-GO ANSWER: **C++ NOT needed for performance** — port = TF-integration milestone only. The n=10 72% was a small-n occupancy artifact; multi-stream overlap (P4) + its shared-CUB-workspace blocker are MOOT for production. (P0/capture_while/P1/P2 done; batched-driver +19%/63% superseded.) · **Branch:** `migrate/linux64-wsl2`

## ★★ PRODUCTIONIZED (2026-06-26 PM) — `rnr/gpu/capture_warp.py` + the `max_rounds` THROUGHPUT LEVER

The captured path is now a real, byte-identical module (`rnr/gpu/capture_warp.CapturedStep`), NOT just a
scratchpad proto — and productionizing it surfaced a **correction to the perf story the util number hid.**

**THE CORRECTION (util ≠ throughput): graph capture at the proto's `max_rounds=8` SATURATES the GPU but
REGRESSES throughput vs the variable-round production baseline.** The fixed-R=8 path runs 8 reconnect rounds
every step, but the variable sweep uses **≤2 applying rounds** at n=16 AND n=10, σ=0.5
(`scratchpad/proto_roundcount.py`: I/H mean ≈1.0, max 2) — so ~6 of the 8 rounds are no-ops, each still
paying a full detect (scan + radix_sort over CAP=8192). The 90% util is real but **includes wasted work.**

| K=16, n=16 | util | agg steps/s | note |
|---|---|---|---|
| prod_eager (variable, real baseline) | 51% | 262 | host-sync-bound, but does minimal work |
| captured_seq **max_rounds=8** (proto config) | 91% | **210** | saturates util, **−20% throughput** ✗ |
| captured_seq **max_rounds=3** (lean) | **93%** | **367** | **+40% throughput** ✓ |

**THE LEVER: capture with `max_rounds = (regime's max applying rounds) + 1`** (=3 for σ=0.5). Still
byte-identical (the variable path never exceeds 2 rounds → fixed-3 == variable-8), now WITHOUT the wasted
no-op detects → +40% ensemble throughput AND 93% util. Same at K=8 and K=16 (per-sim occupancy dominates,
as the scope predicted). **So the go/no-go answer holds — C++ NOT needed — but the win requires `max_rounds`
tuning, not just capture.** The handoff's "90% util → resolved" was right on the conclusion, wrong on the
mechanism: at mr=8 the 90% was throughput-negative.

**SAFETY (two device-flag guards make the perf-tuned small params bit-safe, read post-replay, no hot-loop
sync):**
- **`check_overflow`** — a round's deduped candidate count exceeded `MAX_CAND` (=512; observed max M=125 at
  n=16, 4× margin) → silently-dropped candidates → not bit-identical.
- **`check_underconverged`** — the LAST fixed round still applied a winner ⟺ the variable sweep would NOT
  have broken (it breaks iff m==0 or n_win==0 ⟹ won.sum()==0) ⟺ `max_rounds` too small. An UNSET flag
  GUARANTEES byte-identicality; rigor proven: mr=1 (no margin) → flag trips + trajectory diverges (het
  0.49154**194** vs **312**), mr=3 → flag clear + byte-identical.

**Validated:** byte-identical 2k AND 20k trajectories vs production at n=10 (het 0.4604 @ 20k matches the ref),
captured-vs-production at mr=3 and mr=8 (`scratchpad/proto_fixed_traj.py`, `proto_captured_traj.py`); 134-gate
green; the device-step-seed director (`physics_warp.set_director_step`/`_launch_director_update`, a 1-int
device scalar bumped per replay) varies the captured RNG per step (`proto_step_seed.py`) and stays byte-
identical eager.

### ★★★ `capture_while` IMPLEMENTED (2026-06-26 PM, user-chosen) — the RECOMMENDED path: no `max_rounds` tuning

The `max_rounds=3` lever above WORKS but needs per-regime tuning + the under-convergence guard. `capture_while`
(a CUDA conditional-graph node, device-side round loop) removes both: `CapturedStep.reconnect_sweep_*_while`
runs ONE fixed-dim round per iteration and re-reads a device `cond` (= "this round applied a winner", EXACTLY
the variable sweep's m==0 / n_win==0 break), looping until convergence or a `max_rounds` SAFETY cap (never hit
at ≤2 rounds). So it does EXACTLY the needed rounds — **byte-identical BY CONSTRUCTION, no tuning, no
under-convergence guard, no wasted no-op rounds.**

**THE BLOCKER it hit + the fix (the concrete "Warp-constrained" signal the scope wanted):** `wp.capture_while`'s
conditional-graph body REJECTS memory allocation, and **`warp.utils.array_scan` (CUB) allocates a workspace**
inside it → "unsupported operation (memory allocation)". Isolated (`scratchpad/proto_while_isolate.py`):
`array_scan` is the ONLY offender — `radix_sort_pairs`, `wp.copy`, slicing all capture fine. Fix: replace the
I-side dedup's `array_scan` with a single-thread serial inclusive-scan kernel (`_serial_inclusive_scan_kernel`,
byte-identical, n=CAP≈8192 small/off-critical-path; both the fixed-R and capture_while paths use it now, both
re-validated byte-identical). Capture needs `force_module_load=True`.

**RESULT (K=16, n=16): captured capture_while = 99% util / 369.7 steps/s = +42% vs prod_eager (260)** — ties
the mr=3 fixed-R throughput (367) but with HIGHER util and ZERO tuning/guard burden. **So `CapturedStep`
default = `use_capture_while=True, max_rounds=8` (cap) is the production config:** byte-identical for ANY regime,
saturating, no per-σ round-count measurement. (`max_rounds` fixed-R + the two guards remain as a validated
fallback `use_capture_while=False`.)

### ★ WIRED INTO THE DRIVERS (2026-06-26 PM) — `--captured` opt-in; + the concurrency-model nuance

`CapturedStep` is now wired into the production faithfulness/figure path as an **opt-in `--captured`** flag:
- **`gpu_stability.py --captured`** — drives the loop with `CapturedStep` (capture_while) instead of eager
  `forward_step`; warms up → resumes at `cs.next_step`; recon I/H untracked under capture; slot+overflow guards
  + the audit move to the checkpoint (`read_stats()` = the only sync). VALIDATED byte-identical to the eager run
  (het to full precision, nv/ns/vol/n_problems exact @ n=10, 2k). interval>1 (dt=0.002→interval=5) prefix-graph
  path also byte-identical (`scratchpad/proto_captured_traj.py PROTO_DT=0.002`).
- **`gpu_fig_runs.py --captured`** — forwards `--captured` to each per-sim subprocess. End-to-end smoke green.

**THE NUANCE (concurrency model matters): the captured win shows up SINGLE-SIM / IN-PROCESS, NOT in the
process-pool the figure pipeline uses.** Single-sim @ n=10: eager 3.51 → captured 2.74 ms/step = **+22%** (n=16
in-process K-sim was +42%, higher occupancy). But `gpu_fig_runs` runs K **separate processes** (CONC≈6), and OS
process concurrency ALREADY saturates the GPU (the host overhead of one process overlaps another's GPU work):
measured **eager pool 82.7s vs captured pool 81.05s** (4 jobs ×12k steps, sequential A/B) ≈ neutral. So
`--captured` there is byte-identical + harmless but adds ~no aggregate throughput at production concurrency;
its real value is single-sim / low-concurrency runs (where the GPU would otherwise idle between host trips), and
it lets you hit the same throughput at LOWER CONC (less CPU/host load). **Banking the +42% would need switching
the pipeline to the IN-PROCESS K-sim model** (one process, K `CapturedStep`s, sequential replay —
`scratchpad/proto_ensemble_captured.py`); but the process-pool is already GPU-saturated, so that's a
nice-to-have, not a throughput necessity. Multi-stream P4 stays MOOT.

**REMAINING (optional, next session):** (a) regenerate the canonical Fig 1E/1F with `--captured` if desired
(byte-identical, so figures are unchanged — purely a speed/validation exercise); (b) IF a single big-N
(n≥16) in-process ensemble is ever wanted, build a driver around `proto_ensemble_captured.py` to bank +42%.

## Progress (2026-06-26)

- **Batched-driver intermediate TESTED** (`scratchpad/batched_driver.py`, the P0 "cheaper
  intermediate"): advance all K sims through each reconnect round TOGETHER → ONE
  `wp.synchronize_device` per phase-round instead of per-sim-round (the K× sync cut), reusing the
  exact kernels (detect/gather/reserve/apply), syncs stripped from `apply_*` + deferred from detect.
  **Result @ K=16, n=16, dt=0.01, σ=0.5 (reconnection-ACTIVE: ~9.8 recon/sim-step), fair fresh-mesh
  per mode, 2 reps each, ROCK-STABLE:** sequential 62.2 ms/round, util 51%, 257 agg steps/s vs
  **batched 52.3 ms/round, util 63%, 306 agg steps/s = +19% throughput, +12 pp util, −16% wall.**
  Reconnection total IDENTICAL (62795) both modes/reps → pure scheduling win, NOT less work.
  **VERDICT: real bankable win, but PLATEAUS below the 80–90%/1.7–1.9× saturation target** → per the
  scope's own go/no-go, batching alone does NOT retire the C++ question; **P3 graphs still justified.**
  Remaining ceiling above 63% = (a) per-round phase barriers still serialize (I-sweep 3, H-sweep 2)
  and (b) Python per-launch overhead (~160 kernels/step from a Python loop at K=16) — only graph
  CAPTURE kills (b). Next cheap read before full P3: fixed-`MAX_CAND` masked dedup/gather launches to
  drop the k/M reads → 1 sync/round; if util→~75% the barriers were the issue, if still ~63% it's
  launch overhead (→ capture essential).
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

## ★ P3 RESOLVED (2026-06-26) — graph capture SATURATES at production scale; C++ NOT needed for perf

**The decisive result: captured-sequential replay reaches util 90% at n=16 (production cell scale), K=8 —
saturation, with NO multi-stream overlap.**

| n | mode | K | util | steps/s | single-sim capture |
|---|---|---|---|---|---|
| 10 | captured_seq | 16 | 72% | 281 | −30% |
| **16** | seq_eager | 8 | 67% | 161 | — |
| **16** | **captured_seq** | **8** | **90%** | **232 (+44% vs eager)** | **−33%** |

The earlier 72% was a **small-n artifact**: at n=10 (2000 cells/sim) the per-kernel occupancy is low, so a
single sim underfills the GPU and overlap would matter. At n=16 (8192 cells/sim) each kernel fills the GPU,
so **sequential captured-graph replay alone saturates (90%)** — even at LOWER K (8 vs 16). Per-sim occupancy
(n), not cross-sim overlap, is the dominant factor at production scale.

**→ GO/NO-GO ANSWER: Warp graph capture delivers GPU saturation (~90%) at production scale. C++ is NOT needed
for performance.** The port reduces to the TF-integration milestone (native `MeshQuality` op), to be done when
the algorithm is frozen — exactly the outcome the scope's go/no-go predicted for an 80–90% result. The
multi-stream-overlap path (P4) and its blocker below are MOOT for production (only relevant at tiny n).

**P4 multi-stream blocker (documented, now moot):** concurrent replay of K full-step graphs on K `wp.Stream`s
faults (CUDA 700) — isolated via fresh-process tests (an illegal access POISONS the context, so each mode
needs its own process): single pieces (prefix, compact, ONE I-round) replay concurrently fine, but **8
I-rounds fail** → the `radix_sort_pairs`/`array_scan` **shared per-device CUB workspace** is a probabilistic
race that accumulates over many library-op calls (1 call lucky, 16 calls reliably corrupts). Fixing it (only
needed if small-n ensembles ever matter) = custom per-`g`-scratch scan/sort kernels, or a dedup/sort-free
perf variant. This is the concrete "Warp 1.14 multi-stream + captured library ops is constrained" signal — but
it does NOT gate the perf goal, which captured-sequential already meets.

### Earlier readings (superseded by the n=16 result above)
## P3 CAPTURE RESULTS (2026-06-26) — graph capture WORKS; captured-sequential hits util 72% @ K=16

Full fixed-dim `forward_step` (prefix + fixed-R=8 I-rounds + fixed-R=8 H-rounds + compact) assembled
(`scratchpad/proto_capture_step.py`) and captured with plain `wp.ScopedCapture`. All de-risks GREEN:
fixed-dim masked detect + winners byte-identical (I+H, `proto_fixeddim_detect.py`/`proto_capture_round.py`);
`radix_sort_pairs`+`array_scan`+compact all capturable (`proto_capture_smoke.py`); full step captures+replays.

**Numbers (n=10, dt=0.01, σ=0.5, RTX 5090):**
- **Single-sim: captured −30%** (5.4 → 3.75 ms/step) — the full step (not just the prefix, which P0 said
  graphs barely help) wins big because the reconnect path's MANY small kernels carried heavy per-launch
  host overhead that capture erases.
- **K=16, captured-SEQUENTIAL replay (no overlap): util 50%→72%, throughput 178→281 steps/s (+58%)** vs the
  same fixed-R work launched eagerly. **This already EXCEEDS the batched driver's 63%** and approaches the
  80–90% saturation bar — WITHOUT multi-stream overlap. (seq_eager here does fixed-R=8 always, so its
  baseline is below production's variable-round path; the +58% is the pure host-overhead elimination.)

**P4 multi-stream overlap is BLOCKED by a Warp internals limit (the concrete "Warp too constrained" signal
the scope predicted).** Replaying the K captured graphs concurrently on K `wp.Stream`s → CUDA error 700
"illegal memory access". Root cause: `array_scan`/`radix_sort_pairs` call Warp's C++ runtime
(`wp_array_scan_int_device`/`wp_radix_sort_pairs_*_device`), which uses a **shared per-device CUB
workspace**; captured into K graphs they bake the SAME workspace address, so concurrent replay races on it.
Sequential replay reuses it safely (→ captured_seq is correct). Also: capturing on a CUSTOM stream fails
outright ("invalid device ordinal" in scan_device) — so graphs must be captured on the DEFAULT stream and
replayed elsewhere. **To unblock P4 → 80–90%:** replace array_scan/radix_sort in the captured region with
custom Warp kernels using PER-`g` scratch (no shared workspace) — significant but stays in Warp; OR accept
captured_seq's 72%. This is the specific evidence for whether the FINAL overlap increment needs raw CUDA.

**GO/NO-GO READ:** Warp graph capture delivers util ~72% @ K=16 (sequential replay) — **C++ is NOT needed to
get most of the way to saturation.** The last ~10–20pp (multi-stream overlap) hits a Warp shared-CUB-workspace
limit; closing it needs per-`g` scan/sort scratch (Warp-side, doable) — not a wholesale C++ rewrite.

## P3 SIMPLIFICATION (2026-06-26) — fixed-R unrolled + plain capture; `capture_while` is OPTIONAL

The scope below assumed `capture_while` (data-dependent device round loop) is REQUIRED. It is not, for
the first saturation measurement. A simpler, byte-identical path:

- **Run a FIXED `max_rounds` (e.g. 8) unrolled, NO host break.** With fixed-dim launches + a device-scalar
  `M` masking the tail, an EMPTY round (M==0) is an all-threads-early-return no-op. The host `while` already
  bounds rounds at `max_rounds` and breaks early; running exactly `max_rounds` with masked no-op tail rounds
  reaches the SAME converged state (extra rounds mutate nothing) → byte-identical. So the round loop needs no
  device control flow — it's a fixed unrolled sequence, capturable with a **plain `wp.ScopedCapture`** (the
  P0-proven mechanism). `capture_while` becomes a LATER optimization (skip the masked no-op rounds' launch
  cost), not a prerequisite — de-risks the load-bearing phase.
- The real work either way is the **fixed-dim masked reconnect** (device-scalar `M` guard); that's unchanged.

**FOUNDATION VERIFIED (2026-06-26, `scratchpad/proto_fixeddim_detect.py`): the fixed-dim masked
device-scalar-M I-detect is BYTE-IDENTICAL to production `find_short_edges_device`** across 200 reconnecting
steps (199 with candidates, maxM=27, 0 mismatches). Key finding: the detect kernels ALREADY carry the
sentinel mechanism (`_SENTINEL_KEY` for `keep==0` rows → sort to tail → never emitted), so launching
scan/build_keys/`radix_sort_pairs(…, CAP)`/mark/`array_scan(…[:CAP])`/scatter over the FIXED buffer cap
self-masks — the ONLY new kernel needed is a **guarded `filter`** (`tid >= count[0] → keep=0`, else the exact
interior test), because filter alone would run `d_vert_body_count` on the stale tail. `M = out_pos[CAP-1]` is
then a device scalar (no host `k`/`M` read). The radix sort over fixed CAP (sentinel tail sorts last) leaves
the real keys' order identical → same deduped (v10,v11)-ascending set + same M.

**Remaining P3 steps (each follows the SAME fixed-dim+mask pattern, lower risk than detect):**
1. **gather** fixed-dim: launch `gather_i_kernel` over a fixed `MAX_CAND`, add a device-`M` guard
   (`tid >= M → valid=0`); reserve/apply already self-skip on `valid`/`won`, so only gather needs the guard.
   Verify byte-identical (read-only, no mesh clone needed) — mirror `proto_fixeddim_detect.py`.
2. **reserve** owners pre-allocated: `vown/sown/bown/won` are `wp.full/wp.zeros` per round (the last
   in-region allocs) → pre-allocate on `g`, `fill_(MAX_CAND)`/`zero_` in place each round; launch over fixed
   `MAX_CAND`. (`m = valid.shape[0]` is already host-known, becomes the fixed `MAX_CAND`.)
3. **apply** fixed-dim: launch `apply_*_won_kernel` over `MAX_CAND` (already skips `won[i]==0`); strip the
   trailing `wp.synchronize_device`. Verify a single fixed-dim ROUND byte-identical vs the production sweep's
   round (needs a `g` snapshot/clone to compare the mutation).
4. **H-side** mirror (simpler: no dedup, `M==k`; `find_small_triangles_device` + gather_h/reserve_h/apply_h).
5. **fixed-R captured step**: pre-fill empty-round safety (masked no-ops), wrap prefix + R I-rounds + R
   H-rounds + compact + orient(fixed-iter, no counter readback) in `wp.ScopedCapture`; replay per step.
   Director seed is `step`-varying (engine.py:42 `step*nb`) → feed `step` via a device array the graph reads.
6. **measure** single-sim + multi-stream K=16 (P4: each sim its own graph + `wp.Stream`). Compare util/
   throughput to the batched driver's 63% / 306 steps/s. **THE go/no-go number** for the C++-port question.

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
