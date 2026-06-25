# Session — GPU Fig 1E/1F reproduced + min-image fix + foam cache; NEXT: reconnect-sweep optimization (Option A)

## Summary 2026-06-25 08:24 EDT

Four pieces of work landed; the forward focus is **Option A** (make the host-bound reconnect sweep
device-resident — the profiled ~80%-of-step bottleneck). All gated changes are code/docs; every
`rnr/exports/*` is a leave-it generated artifact.

**1. Orientation/closure repair — DECIDED to KEEP greedy (not port tvm `updatePolygonDirections`).**
At paper scale orient runs every step; a host tvm port would be ~3–6× slower for a stall that is rare
by construction AND already caught by the gpu-stability volume gate. Kept `orient_warp.orient_repair_warp`
as-is. Full pros/cons + revisit triggers: `docs/2026-06-25_orientation-repair-options.md`; design doc
`docs/2026-06-24_gpu-3d-vertex-model-exploration.md` §10; memory `orientation-repair-greedy-kept`.

**2. Foam disk cache (`rnr/gpu/foam_cache.py`, new).** The O(N²) TF builder (~10 min for N=2000) is now
build-once → save → load (de-novo fallback, `--rebuild-foam` to force). Cached artifact = scaled compact
CSR + phys state + box + (v0,a0); **no TF on load**, headroom-independent. `_setup_unit_foam` split into
`_build_unit_foam_host` (TF half) + `foam_cache.upload_unit_foam` (device half, shared) —
`test_gpu_engine.py:35`. `gpu_stability.py` loads via `foam_cache.load_or_build` and skips TF init on a
cache hit. Gotcha fixed: np temp name must end `.npz` or `savez_compressed` appends it
(`foam_cache.py` save_host). Cache dir gitignored. n=10 mixed+demixed foams are built + cached.

**3. Periodic min-image in the Okuda I↔H placement (latent correctness, prior handoff #3 — DONE).** A
short edge / small triangle straddling a periodic face was split through the box CENTRE (raw
`0.5(p10+p11)` / `(p0+p1+p2)/3`). Fixed with minimum-image differencing + `[0,L)` wrap, anchored on one
vertex: `reconnect.py` `place_i_to_h_xyz`/`place_h_to_i_xyz` (+ `_minimg`/`_wrapbox`, optional `box=`),
`reconnect_csr.py` pass-through, and the **4** GPU kernels in `reconnect_warp.py` (+ `_box_of(g)`, imported
`d_minimg`/`d_wrapbox`). `box=None`/zero-box = exact non-periodic path → interior sites bit-identical →
existing gates untouched. New gate `rnr/tests/test_periodic_placement.py` (5 tests incl. GPU==CPU
periodic parity). **Full gate 132 passed** at this state; `gpu-stability --n 4` STABLE with min-image active.

**4. GPU Fig 1E/1F reproduced at paper scale (N=2000) + video + a finding.** New pipeline:
`gpu_dpmax.py` (DP_max=0.789), `gpu_fig_runs.py` (concurrency-capped ensemble, resumable),
`gpu_fig1e1f.py`, `gpu_video_cells.py`. Ran 12 runs (3σ×mixed + σ=0.5 demixed, 3 seeds, dt=0.01, 400k
steps/t=4000, 6-concurrent) in ~3.3h. **Fig 1F faithful** (demixed σ=0.5 holds DP/DP_max≈1.0).
**Fig 1E:** σ=0.1→0.59, σ=0.2→0.64 (strong, σ-ordered — *closes* the old finite-N flat-DP gap), but
**σ=0.5 plateaus at 0.16**. Diagnosed: at dt=0.01 the stiffest tension over-damps → RNR trigger stalls
(15.6k vs 435k reconnections) = **TIMESTEP artifact, not a bug**. Confirmed by a dt=2e-3 re-run of σ=0.5:
reconnections 11k→~500k, DP 0.11→0.55 at t=1000 → σ=0.5 becomes highest (recovers paper ordering). A
fully-faithful Fig 1E needs all σ at dt≤2e-3 (NOT regenerated — user opted to stop at the finding).
Writeup `docs/2026-06-25_gpu-fig1e1f-reproduction.md`; memory `gpu-fig1e1f-dt-arrest`
(+ `m8-count-dp-still-scale-limited` marked RESOLVED). Deliverables (leave-it): `fig1e_gpu.png`,
`fig1f_gpu.png`, `gpu_cells_sort_mixed.mp4` (100k-step ~2000-cell turntable, no clip, 0.75°/frame).

**5. Bottleneck profile (planning for Option A).** Per-step at N=2000 (scratch profiler, mid-sort):
reconnect_I **72%** / reconnect_H 9% / compact 7% / orient 3% / physics (director+geom+force+integrate)
only **8%**. GPU **1%** utilized. reconnect_I is ~7.5 ms **regardless of σ** (0.2 vs 0.5) and averages
**0.69 rounds/step** (NOT a cascade) → the cost is ~12 tiny device↔host syncs/step in
detect→`gather_i_configs_to_list`(~11 `.numpy()`+Python `ICfgIdx` rebuild)→reserve(`won.numpy()`)→
apply(`out.numpy()`). It's **latency-bound on host round-trips**, not compute/rounds. (Concurrency already
exploits this: ~3× aggregate for ensembles, free.)

**Build / test / git state:**
- Branch `migrate/linux64-wsl2` (NOT main). **Gate: `pixi run test` = 132 passed** earlier this session
  (after the min-image fix); the new `gpu_*.py` scripts are standalone (not imported by the suite). This
  handoff's commit re-runs the gate to be safe (see commit message).
- Committed by this handoff: the code + docs + new scripts/test below. **Leave-it (NOT committed):** all
  `rnr/exports/*` (figures, CSVs, mp4s, logs, frames — regenerable; the foam cache + fig_run_logs +
  gpu_cells_frames are now gitignored).
- Memory updated (outside repo): `orientation-repair-greedy-kept`, `gpu-fig1e1f-dt-arrest`,
  `m8-count-dp-still-scale-limited` (resolved), MEMORY.md index.

Full `git status --short` at handoff (pre-commit): `.gitignore`, `docs/2026-06-24_…§10`,
`docs/2026-06-25_{gpu-fig1e1f-reproduction,orientation-repair-options}.md`, `rnr/gpu/foam_cache.py`,
`rnr/gpu/{reconnect_csr,reconnect_warp}.py`, `rnr/reconnect.py`, `rnr/scripts/gpu_stability.py`,
`rnr/scripts/{gpu_dpmax,gpu_fig1e1f,gpu_fig_runs,gpu_video_cells}.py`, `rnr/tests/test_gpu_engine.py`,
`rnr/tests/test_periodic_placement.py` — all committed. Remaining `?? rnr/exports/*` = leave-it artifacts
(this + prior sessions; do NOT commit).

## Kickoff — next session

You are continuing a GPU port of TissueForge's 3D vertex model + RNR. The engine runs faithful
paper-scale sorts (N=2000) and reproduces Manning2024 Fig 1E/1F (memory `gpu-fig1e1f-dt-arrest`). It is
**host-bound, not compute-bound**: the GPU sits at ~1% and **~80% of each step is host orchestration of
the reconnect sweep**. Your job: **Option A — make the reconnect sweep device-resident** (a pure plumbing
change; no algorithm/math change → must stay faithful + robust).

**Why (profiled this session):** reconnect_I = 72% of the step, reconnect_H = 9%, real physics = 8%.
reconnect_I is ~7.5 ms/step independent of σ and averages **0.69 rounds/step** (not a cascade) — the cost
is **~12 tiny device↔host syncs per step**: detect scan → `gather_i_configs_to_list` (~11 `.numpy()` +
Python `ICfgIdx` rebuild) → `sort` → `pack_footprints` → reserve (`won.numpy()`) → pack winners → apply
(`out.numpy()`). The device gather (`gather_warp.gather_i_configs_warp`) ALREADY returns packed device
arrays; the reservation footprints and apply inputs are just slices of them — they get needlessly
round-tripped through host Python objects.

**The plan (priority order):**
1. **Empty-step fast path.** Have the detect trigger scan return just a **count (1 int)**; if 0 short
   edges, skip the gather/reserve/apply entirely. (~31% of steps do 0 reconnections but still pay the
   gather today.)
2. **Device-resident detect→reserve→apply.** Feed `gather_i_configs_warp`'s device arrays
   (`valid,v10,v11,cap_top,cap_bot,side,arm_side,arm_otop,arm_obot,top,bot`) **directly** into the
   reservation + apply kernels — delete `gather_i_configs_to_list`, the host `ICfgIdx` list, `sort`,
   `pack_footprints`, and the `col`/`mat` host packing in `apply_*_batch_warp`. Compact the `valid`
   candidates on-device with a prefix-sum (mirror `compact_warp`’s `wp.utils.array_scan`). Keep the host
   while-loop, but its only per-round readback is the tiny 1-int "remaining candidates" count.
3. **Mirror for H→I** (`detect_small_triangles_device` / `reserve_h_*` / `apply_h_to_i_batch_warp`).
4. **Preserve determinism:** the reservation is lowest-id-wins and round-1 must still bit-match the host
   reference — keep candidates in the SAME canonical order the host used (`sort` by `(v10,v11)` for I→H,
   by `triangle` for H→I). The trigger scan already emits surface-ascending; verify the device candidate
   order matches, or sort the small candidate index array on-device, before reserving.

**Files:** `rnr/gpu/schedule_warp.py` (the two `reconnect_sweep_*_warp_device` loops + `reserve_*_warp_g`
/ `reserve_won_mask_warp` / `pack_footprints` — add device-array variants), `rnr/gpu/detect_warp.py`
(`detect_short_edges_device` / `detect_small_triangles_device` — return device arrays + a count),
`rnr/gpu/gather_warp.py` (`gather_i_configs_warp` + the H gather already return device arrays; add the
device compaction of `valid`), `rnr/gpu/reconnect_warp.py` (`apply_i_to_h_batch_warp` /
`apply_h_to_i_batch_warp` — variants taking device arrays directly). Engine call site: `engine.py:48-57`.

**Validate (faithfulness + robustness — all must stay green; the optimization changes NO math):**
```
pixi run test                                                          # 132 expected (round-trip + CPU==GPU fingerprint + periodic placement)
pixi run python -m pytest rnr/tests/test_gpu_*.py -q                   # GPU subset (~35s)
pixi run gpu-stability --n 10 --steps 100000 --dt 0.01 --ic mixed      # STABLE; het trajectory must match the pre-opt run
```
The order-invariant body-anchored fingerprint validates the parallel apply; round 1 must remain
bit/fingerprint-equal to the host sweep (that's the determinism check). Re-profile to confirm reconnect_I
dropped (scratch profiler pattern: time each `forward_step` phase with `wp.synchronize_device` between).
Target: reconnect_I 7.5 ms → ~1–2 ms ⇒ step ~11 ms → ~4 ms (~2.5–3× single-run; physics is the ~8% floor).

**Caveats / guardrails:**
- GPU-port phase only. Reimplement from Okuda 2013 / our `rnr/`; **never copy GPL `tvm/`**;
  `tvm/ 3DVertVor/ tissue-forge/ cellGPU/ VertAX/ gpu_reference_papers/` are read-only oracles (own
  `.git`, never commit). `tf.init()` is one-per-process; standalone scripts mirror `conftest.vsolver`
  or use `foam_cache.load_or_build` (no TF).
- Warp recompiles on edit (~15–30s first launch). Cross-module `@wp.func` MUST be imported into the
  calling module's namespace (design doc §10 gotcha).
- n=10 foams are cached (`rnr/exports/foam_cache/foam_n10_{mixed,demixed}_*.npz`, gitignored).
- **Free win, orthogonal:** for ensemble/figure runs use **concurrency** (`gpu_fig_runs.py` runs 6 at
  once → ~3× aggregate; host-bound). It does NOT help a single long run (the video) — that's what
  Option A is for.

**Lower-priority / optional (after Option A):** CUDA graphs for the physics kernel sequence (~8% → small);
regenerate Fig 1E with all σ at dt=2e-3 for a clean σ-ordered figure (σ=0.5 dt=2e-3 CSVs already exist);
native Python sort path degenerate-face hygiene (prior handoff #4); fig1e/1f native-drive regen
(prior #5).
</content>
