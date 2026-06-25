# GPU reconnect perf: gather buffer reuse (~6.6%) + the skip-scan dead-end

## Summary 2026-06-25 11:41 EDT

Goal: continue the GPU reconnect-sweep optimization from the prior handoff. That handoff named
"skip the detect scan on quiescent steps" as the biggest next lever. **This session proved that lever
INFEASIBLE, reverted it, re-profiled, and instead cut the gather's per-round allocation overhead —
bit-identical, ~6.6% faster.** Engine now 2.83 → 2.64 ms/step at N=2000.

**1. Skip-scan lever is DEAD (tried, reverted — do NOT re-attempt).** The premise (most steps are
quiescent → skip the scan) is false. Measured on the deterministic baseline (n=10, dt=0.01, σ=0.5
mixed):
- `found==0` (no short edge present → the only provably-safe skip) occurred on **0/3000 early AND
  0/1500 late** steps (warmup 25k). Global min-edge median 1.5e-4, MAX 8.3e-4 — always < Lth=1e-3.
- Reconnections ARE rare (`applied==0` on 44.8% early → 93% late) but **short edges are not** — they
  are mostly BLOCKED (Condition-4 veto / not a valid 4-cell I-site) and persist sub-Lth. So the scan
  always finds candidates; it can never be skipped.
- The proposed global gate `min_edge − 2·max_disp ≥ Lth` is doubly dead: `max_disp` is a global max
  ≈ 14×Lth (stiff kv=10 forces on a far-off vertex) → fires 0/3000 even ignoring the above.
- GOTCHA: putting the `min_edge`/`max_disp` reductions INSIDE the hot `surface_geom_kernel` /
  `integrate_kernel` perturbed their FP codegen ~1 ULP → broke bit-identicality even with the gate
  inert (passed the `<1e-9` single-step GPU==host tests, compounded over ~5k steps). Any future
  in-kernel reduction on those hot kernels MUST be a SEPARATE kernel.

**2. Re-profiled (the picture shifted).** Per-step (mid-sort, n=10; sync-between-phases inflates
absolutes ~25%, relative holds): recon_I 1.44 (scan 0.56 + **gather/reserve/apply 0.88**), force 0.55,
recon_H 0.50, compact 0.48, geom 0.25, orient 0.16. **The scan is no longer the bottleneck** (the
prior per-surface rewrite shrank it). gather/reserve/apply is, and it's **overhead-bound**: only
~16.6 I-candidates/step (97% gathered-then-rejected), and a micro-bench showed **74% of the gather
(0.197 ms) is per-round alloc+upload+fill of its 11 output arrays, NOT compute** (kernel only 0.070 ms).
Reserve's 0.08 ms is entirely 3 capacity-sized fills (cap_v 16202 / cap_s 18202 / nb 2000)/round.

**3. Reworked the gather → buffer reuse (DONE, bit-identical).** Caching/FILTERING blocked candidates
would change candidate row-indices → change the parallel apply's bump-allocation order → DIVERGE the
trajectory; so the bit-identical fix is **pre-allocate + reuse the gather output buffers**, not
candidate filtering. `gather_warp.py:35` `_ensure_gather_buf(g,key,m,with_tri)` allocates the 9-10
int32 output arrays ONCE on `g` (grown ×2), reused across rounds/steps; `gather_i_configs_warp` /
`gather_h_configs_warp` gained a `buf=` arg (write into reused arrays, drop the end `wp.synchronize`,
return `[:m]` views); `schedule_warp.py:526,597` pass the buffers. Safe: the kernel writes `valid[i]`
for every launched row, and reserve/apply read other fields only for valid rows + launch over m, so
stale data in invalid rows / rows ≥ m is never consumed; the fresh-alloc path (no `buf`) is unchanged
for tests / `*_to_list`.

**Build / test / git state:**
- Branch `migrate/linux64-wsl2` (NOT main). HEAD before this session: `68ec341`.
- **Gate: `pixi run test` = 132 passed** (ran at exactly the committed code state; gather_warp.py +
  schedule_warp.py unchanged since). Only the findings doc + memory (outside repo) changed after.
- **20k stability BIT-IDENTICAL** vs baseline at every checkpoint: het@10k=0.46382583300146024,
  recon_I/H@20k=4010/3028 — match to the last digit. **2.83 → 2.64 ms/step (~6.6% wall).**
- The skip-scan attempt (`physics_warp.py`, `engine.py`) was FULLY REVERTED — they match HEAD.
- Committing this session: `rnr/gpu/gather_warp.py`, `rnr/gpu/schedule_warp.py`, the two docs
  (`docs/2026-06-25_gather-overhead-and-skip-scan-deadend.md` + this handoff). **Leave-it (NOT
  committed): all `rnr/exports/*`** (regenerable figs/CSVs/mp4s from prior + this session's probes).
  Memory updated (outside repo): `reconnect-sweep-scan-bottleneck` (skip-scan dead + gather win).

Full `git status --short` at handoff (exports intentionally uncommitted):
```
 M rnr/exports/dpmax.json                          (+ ~60 more ?? rnr/exports/* : regenerable, LEAVE)
 M rnr/exports/fig1e_demixing_native.{csv,png}
 M rnr/exports/fig1f_stability_native.{csv,png}
 M rnr/gpu/gather_warp.py        <- COMMIT
 M rnr/gpu/schedule_warp.py      <- COMMIT
?? docs/2026-06-25_gather-overhead-and-skip-scan-deadend.md   <- COMMIT
?? docs/sessions/2026-06-25-1141-gather-buffer-reuse-and-skip-scan-deadend.md  <- COMMIT (this file)
```

## Kickoff — next session

You are continuing a GPU port of TissueForge's 3D vertex model + RNR (NVIDIA Warp on an RTX 5090). The
engine runs faithful paper-scale sorts (N=2000), reproduces Manning Fig 1E/1F, and is now at **2.64
ms/step** after this session's gather buffer-reuse (bit-identical; commit on branch
`migrate/linux64-wsl2`; docs/2026-06-25_gather-overhead-and-skip-scan-deadend.md; memory
`reconnect-sweep-scan-bottleneck`). The science goal is already met — further perf is **optional polish
with diminishing returns**; pick a lever only if perf matters for upcoming production runs.

**DO NOT re-attempt "skip the scan on quiescent steps" — proven infeasible this session** (there is
always a sub-Lth blocked edge; `found==0` on 0/3000 + 0/1500 steps; details in memory + the doc).

**Remaining perf levers, priority order (all measured against HEAD = this session's commit):**
1. **force kernel** (~0.55 ms/step, the 2nd-biggest phase, BIT-IDENTICAL-able). `physics_warp.py`
   `force_kernel`: the active-drive distinct-body dedup is O(valence²) (lines ~203-222), and the
   conservative loop re-searches v's ring index per surface (~172-175). Optimize while keeping
   GPU==host `<1e-9` (gated by `test_gpu_physics_warp.py::test_force_*`). Bit-identical 20k trajectory
   is the real test.
2. **reserve owner-array reuse** (~1.5%, same pattern as the gather, low risk). `schedule_warp.py`
   `reserve_{i,h}_won_device`: pre-allocate vown/sown/bown ONCE on `g`, reuse. Either re-fill each
   round (keeps sentinel=m, bit-identical by construction) or footprint-reset (fixed sentinel — any
   const > max candidate id is bit-identical). Replaces 3 capacity-sized fills/round.
3. **compact throttle** (~0.2-0.4 ms): run compaction every-K-reconnections / on a slot-usage
   threshold instead of every reconnection step. NOT bit-identical (slot renumbering shifts reservation
   order) → a faithfulness tradeoff; re-validate the het curve + stability stay green, like the dt
   lever. Get buy-in.

**Validate ANY bit-identical perf change (this IS the correctness test):**
```
pixi run test                                                          # 132 expected
pixi run gpu-stability --n 10 --steps 20000 --dt 0.01 --ic mixed --csv /tmp/x.csv
#   STABLE + het@10k = 0.46382583300146024, recon_I/H@20k = 4010/3028 MUST match to the last digit.
#   A single differing float = the change is NOT bit-identical (a bug, OR a deliberate faithfulness
#   tradeoff like compact-throttle that needs the het-curve re-validated instead).
```
Re-profile with the scratch pattern (time each forward_step phase with `wp.synchronize_device` between;
load the cached n=10 foam via `foam_cache.load_or_build`, no TF; warmup ~2000 mid-sort). Examples this
session: scratchpad `profile_phases.py`, `profile_recon_fine2.py`, `bench_gather_alloc.py`.

**Caveats / guardrails:**
- GPU-port phase only. Reimplement from Okuda 2013 / our `rnr/`; **never copy GPL `tvm/`**;
  `tvm/ 3DVertVor/ tissue-forge/ cellGPU/ VertAX/ gpu_reference_papers/` are read-only oracles (own
  `.git`, never commit). n=10 foams are cached (`rnr/exports/foam_cache/*.npz`, gitignored).
- Warp recompiles only when a `@wp.kernel`/`@wp.func` source changes (~15-30s); Python-wrapper-only
  changes (like this session's gather buffer reuse) need no recompile.
- The gated round-1 fingerprint tests (`test_gpu_gather_warp.py`, `test_gpu_detect_warp.py`) enforce
  device==host equivalence of the scan/gather/reserve — keep them green when touching that path.
