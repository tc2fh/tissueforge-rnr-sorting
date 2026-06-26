# P3 RESOLVED — CUDA-graph capture of forward_step SATURATES the GPU; C++ not needed for perf

## Summary 2026-06-26 07:23 EDT

**Goal:** attack the K=16 ensemble concurrency ceiling (util ~47%, sync-bound) — the go/no-go input for the
C++/CUDA-port question. Resumed at the CUDA-graph experiment (P1/P2 done). This session ran the planned
batched-driver intermediate, then took P3 (graph capture) all the way to a **definitive answer.**

**Outcome: graph capture of the full `forward_step` SATURATES the GPU at production scale → C++ is NOT needed
for performance.** The port reduces to the TF-integration milestone (native `MeshQuality`), done when the
algorithm is frozen.

### The arc + the decisions/surprises (not a file dump)
- **Batched-driver intermediate (priority-1):** `scratchpad/batched_driver.py` — advance all K sims through
  each reconnect round together, ONE `wp.synchronize_device`/phase-round instead of per-sim. **+19% throughput
  / util 51%→63% @ K=16, identical recon work (pure scheduling win).** Plateaus below the 80–90% bar → user
  OK'd proceeding to P3. *(Superseded by P3 but a valid, simpler standalone lever.)*
- **P3 reframe that unlocked it:** `capture_while` is NOT required. Run a **fixed `max_rounds` unrolled (no host
  break)**; fixed-dim launches + a **device-scalar `M`** mask make empty rounds all-threads-early-return no-ops
  → same converged state → byte-identical → capturable with **plain `wp.ScopedCapture`**. (`capture_while` =
  later opt to skip no-op-round launch cost.)
- **Every capture risk retired, each verified byte-identical/working:**
  - Fixed-dim masked I-detect byte-identical (`scratchpad/proto_fixeddim_detect.py`, 200 steps, maxM=27, 0
    mismatch). KEY: the detect kernels ALREADY sentinel-mask (`_SENTINEL_KEY` for `keep==0` → sorts to tail →
    never emitted), so scan/build_keys/`radix_sort(…,CAP)`/mark/`array_scan(…[:CAP])`/scatter over the FIXED
    buf cap self-mask — the ONLY new kernel is a **guarded `filter`** (`tid>=count[0]→keep=0`). `M=out_pos[CAP-1]`
    is the device scalar.
  - Fixed-dim **winners** byte-identical I+H (`scratchpad/proto_capture_round.py`): detect→gather→reserve gives
    the identical `won` mask → apply (same kernel, same winners) is identical by construction. New tiny kernels:
    `mask_tail_valid` (I+H) + `clamp_tail` (H sorts surface indices, so its stale tail must be sentinel-padded
    before sort then clamped to a safe index before gather). gather/reserve/apply reused UNMODIFIED.
  - `radix_sort_pairs`+`array_scan`+`compact` all capturable (`scratchpad/proto_capture_smoke.py`).
- **Full `forward_step` captured + measured** (`scratchpad/proto_capture_step.py`): prefix + fixed-R=8 I-rounds
  + fixed-R=8 H-rounds + compact, plain `ScopedCapture`. Reserve owners (`vown/sown/bown/won`) pre-allocated
  (last per-round allocs). **single-sim −30% (n=10) / −33% (n=16).**
- **★ THE DECISIVE NUMBER — captured-SEQUENTIAL replay SATURATES at production scale:**

  | n | mode | K | util | steps/s |
  |---|---|---|---|---|
  | 10 | captured_seq | 16 | 72% | 281 |
  | **16** | seq_eager | 8 | 67% | 161 |
  | **16** | **captured_seq** | **8** | **90%** | **232 (+44%)** |

  The n=10 72% was a **small-n occupancy artifact** — tiny meshes underfill the GPU so single-sim can't
  saturate. At n=16 (8192 cells/sim) each kernel fills the GPU, so **sequential captured-graph replay alone
  hits 90%**, even at lower K. **Per-sim occupancy (n) dominates, not cross-sim overlap.**
- **Multi-stream overlap (P4) is BLOCKED but MOOT.** Concurrent replay of K full-step graphs on K `wp.Stream`s
  faults (CUDA 700 illegal access). Isolated via **fresh-process** tests (an illegal access POISONS the CUDA
  context, so each mode needs its own process; `scratchpad/proto_multistream_diag.py`): prefix / compact / ONE
  I-round replay concurrently fine, but **8 I-rounds fail** → `radix_sort_pairs`/`array_scan` share a
  **per-device CUB workspace**, a probabilistic race that accumulates over many lib-op calls (1 call lucky, 16
  reliably corrupt). Also capturing on a CUSTOM stream fails ("invalid device ordinal" in scan_device) → must
  capture on the DEFAULT stream, replay elsewhere. **Irrelevant for production** (captured-sequential already
  saturates); a fix (custom per-`g`-scratch scan/sort) is only needed if tiny-n ensembles ever matter.
- **Gotchas burned:** `g.setdefault(k, wp.zeros(...))` allocates EVERY call (Python evals the default) → alloc-
  in-capture crash — use `if k not in g`. Director seed is BAKED into the graph (Python int) — fine for the
  perf delta (both paths pay it) but NOT production-faithful; productionizing needs a device-step array.

### Build / test / git state
- **Branch `migrate/linux64-wsl2`.** Engine fork `tissue-forge` (`feat/native-rnr-reconnection`) UNCHANGED this
  session. **Nothing pushed.**
- **Gate NOT re-run this session — basis stated:** no tracked `*.py`/`conftest`/`pixi.toml` changed (ALL code
  work was in untracked `scratchpad/` perf/correctness probes; the only tracked edit is the scope `.md`). The
  **134-test gate was green at last session's end** (commits `877560d` P1 / `10b24ec` P2) and the tracked
  codebase is byte-identical since → still green.
- **This session's commit stages ONLY:** `docs/2026-06-26_cuda-graph-experiment-scope.md` (the ★ P3 RESOLVED
  results) + this handoff. The `scratchpad/` protos are untracked/ephemeral (convention: LEAVE — they persist
  on disk for next session as the productionization reference) and the `rnr/exports/*` blobs are
  prior-session regenerable artifacts (LEAVE).
- Memory updated: `cuda-graph-experiment` (now "★ RESOLVED") + MEMORY.md line.

```
 M docs/2026-06-26_cuda-graph-experiment-scope.md   <- COMMIT (this session)
 M rnr/exports/dpmax.json                           <- LEAVE (prior-session blobs)
 M rnr/exports/fig1e_demixing_native.{csv,png}      <- LEAVE
 M rnr/exports/fig1f_stability_native.{csv,png}     <- LEAVE
?? rnr/exports/ (many gpu_*/native_*/sort_oracle_*/fig*_gpu* / *.mp4 / *.gif)  <- LEAVE (prior-session)
?? scratchpad/ (batched_driver, proto_fixeddim_detect, proto_capture_round,
                proto_capture_smoke, proto_capture_step, proto_multistream_diag)  <- LEAVE (ephemeral protos)
```
(Full `git status --short` was ~70 `rnr/exports/*` lines — all prior-session regenerable blobs; none are this
session's work.)

## Kickoff — next session

You are continuing the Warp/CUDA GPU port of TissueForge's 3D vertex model + RNR (RTX 5090, 32 GB) on branch
`migrate/linux64-wsl2`. **The perf question is ANSWERED: CUDA-graph capture of `forward_step` saturates the GPU
(~90% util at n=16) → C++ is NOT needed for performance.** Read `docs/2026-06-26_cuda-graph-experiment-scope.md`
first (the ★ P3 RESOLVED section at top + the P3 SIMPLIFICATION + Bit-identicality sections). The proof-of-concept
lives in **untracked `scratchpad/` protos** — they are the reference + starting point (all are perf/correctness
probes, NOT gated production code):
- `proto_capture_step.py` — the full captured step + measurement (run: `pixi run python scratchpad/proto_capture_step.py 8 150` ; `PROTO_N=16` env for production scale)
- `proto_fixeddim_detect.py` / `proto_capture_round.py` — byte-identical fixed-dim detect + I/H winners
- `proto_capture_smoke.py` — capturability of radix_sort/array_scan ; `proto_multistream_diag.py` — the multi-stream blocker isolation

### Task: PRODUCTIONIZE the captured path (device-step-seed + byte-identical gate + engine wiring)
Bank the ~90% saturation in real ensemble runs. **Replay SEQUENTIALLY** (per-sim captured graph, replayed each
step) — that already saturates at production scale; do NOT attempt concurrent multi-stream (the shared-CUB-
workspace race; moot anyway). Priority order:

1. **Device-step-seed for the director** so a captured graph varies the RNG per step (today it's a baked Python
   int → frozen noise). `physics_warp.director_update_warp` (engine.py:41–42 passes `seed,step`, used via
   `step*nb`): make the kernel read `step` from a small device array the host bumps each step (or a captured
   graph reuses one step's seed every replay). Validate: trajectory varies per step like the eager path.
2. **Productionize the fixed-dim masked reconnect in `rnr/gpu/`** (the protos proved it byte-identical). Add a
   device-`M`-guarded fixed-dim path to detect/gather/reserve/apply + the guarded `filter`/`mask_tail_valid`/
   `clamp_tail` kernels; pre-allocate reserve owners (`vown/sown/bown/won`) on `g`. Keep `MAX_CAND` an asserted
   upper bound (256 fine at n=10; **at n=16+ verify maxM < MAX_CAND** — raw candidate count grows with n; the
   protos carry the overflow assert). **GATE every `rnr/gpu/*.py` edit:** `pixi run test` (expect 134) + the
   2k/20k byte-identical trajectory (REF recon I/H=4010/3028, het 0.4604 — see the scope doc's measurement
   protocol + `scratchpad/REF_traj.csv`).
3. **Capture + sequential-replay wrapper** around `forward_step` (fixed-R unrolled + `compact`; orient at fixed
   iters or post-replay, no counter readback — it has a host `.numpy()` today). Warm up once before capture so
   lazy allocs (detect/gather/owners/compact bufs) happen OUTSIDE the capture region.
4. **Wire into the ensemble driver** (the K-sim runner: `concurrency_probe` / `run_overnight` / `gpu_fig_runs`):
   capture each sim's step graph once, replay per step. **Measure at production scale (n=16+) to confirm the
   ~90% holds in the real pipeline.**

Caveats: orient's `counter.numpy()` loop (orient_warp.py:119) is NOT capture-safe → fixed-iter or move outside
the captured region. The per-phase bracketed profiler is meaningless under capture — judge by natural per-step
+ `nvidia-smi` util. If you ever need tiny-n ensemble overlap, the multi-stream fix is custom per-`g`-scratch
scan/sort (drops the shared CUB workspace) — but it's out of scope for the perf goal.

**Scope + license guardrails:** GPU-port phase only; reimplement from Okuda 2013 / our `rnr/`, NEVER copy GPL
`tvm/`. Read-only oracles (own `.git`, never commit): `tvm/ 3DVertVor/ cellGPU/ VertAX/ gpu_reference_papers/`.
`tissue-forge/` is our ACTIVE fork — engine changes commit to ITS repo (`feat/native-rnr-reconnection`). Don't
scope-creep into growth/morphogenesis. Commit at handoff (standing auth); push only on explicit ask.
