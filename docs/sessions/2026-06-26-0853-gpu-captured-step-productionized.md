# Productionized the captured forward_step (`CapturedStep`) + capture_while; found the max_rounds throughput lever

## Summary 2026-06-26 08:53 EDT

**Goal:** productionize the proven CUDA-graph captured path (prior session's `scratchpad/` protos) into a
byte-identical `rnr/gpu` module and bank the GPU saturation in real ensemble runs. Did that, and the
productionization surfaced a correction to the handoff's perf conclusion + the user picked `capture_while`.

### What changed and why (decisions / surprises — not a file dump)
- **NEW `rnr/gpu/capture_warp.py`** — the captured forward step, byte-identical to `engine.forward_step`. It
  ONLY ADDS: imports the UNMODIFIED production detect/gather/reserve/apply/compact kernels and assembles them;
  the only new kernels are tiny device-side helpers. So no existing module changed except step-1 below → the
  134-gate cannot regress from it (no test imports it). Key pieces:
  - `CapturedStep` — warms up (advances g, allocs all lazy buffers OUTSIDE capture) → captures the step graph →
    `.step(n)` replays per step. Pointer-stable canonical arrays (P1/P2 from prior session) let ONE capture
    replay as the mesh content evolves; launch dims are all capacity constants.
  - fixed-dim, device-`M`-masked reconnect rounds (`_round_{i,h}_fixed`, `_detect_{i,h}_fixed`) — the variable
    sweep's host `M`/`n_win` readbacks become a device scalar + tail masks; reserve owners pre-allocated.
  - **capture_while sweeps** (`reconnect_sweep_{i,h}_while`) — device-side round loop (CUDA conditional graph
    node): body sets `cond = (this round applied a winner)` (EXACTLY the variable break m==0/n_win==0) + a
    round-count cap. Does EXACTLY the needed rounds → **byte-identical BY CONSTRUCTION, no tuning, no guard.**
  - capture-safe `_orient_repair_fixed` (fixed `max_iter`, no `counter.numpy()` readback — converged iters are
    no-ops → byte-identical); compact+orient UNCONDITIONAL on a reconnect step (no-op on gap-free / idempotent
    on clean → == production's `if (ni+nh)>0` gate, no count readback).
- **`rnr/gpu/physics_warp.py` (step 1, device-step-seed)** — `director_update_kernel` now reads the per-step RNG
  key from a 1-int device scalar `g['_step_dev']` instead of a baked Python int; `set_director_step` (host bump,
  OUTSIDE capture) + `_launch_director_update` (the captured launch). So a captured graph varies its director
  noise per replay (was frozen). Byte-identical eager (same `step*nb`). `physics_warp.py:257,289`.
- **★ THE CORRECTION (util ≠ throughput) — the handoff resolved perf on "90% util" but that was misleading.**
  `scratchpad/proto_roundcount.py`: the variable sweep uses **≤2 applying rounds** (n=10 AND n=16, σ=0.5). The
  proto's `max_rounds=8` does ~6 no-op rounds/step (each a full detect incl. radix_sort over 8192). So at mr=8
  the captured path SATURATES util (91% @ K=16) but **REGRESSES throughput** (210 vs prod 262 steps/s). THE
  LEVER: `max_rounds=3` (=(max 2)+1) → byte-identical AND +40% (93% util). Two device-flag guards make a small
  mr bit-safe: `check_overflow` (M>MAX_CAND=512; observed max M=125 @ n=16) + `check_underconverged` (last fixed
  round `won.sum()>0` ⟺ mr too small; UNSET ⟺ byte-identical — proven: mr=1 trips+diverges, mr=3 clear).
- **★ capture_while (user-chosen) — the RECOMMENDED path, removes the mr-tuning footgun.** Hit a real Warp
  constraint: the conditional-graph body FORBIDS memory allocation, and **`warp.utils.array_scan` (CUB)
  allocates a workspace** there → "unsupported operation (memory allocation)". Isolated
  (`scratchpad/proto_while_isolate.py`): `array_scan` is the ONLY offender (`radix_sort_pairs`/`wp.copy`/slicing
  capture fine). FIX: I-side dedup's `array_scan` → a single-thread `_serial_inclusive_scan_kernel` (byte-
  identical inclusive prefix sum; CAP≈8192 small, off critical path; both paths use it). Capture needs
  `force_module_load=True`. **RESULT K=16/n=16: 99% util / 369.7 steps/s = +42% vs prod 260**, byte-identical,
  NO tuning. → `CapturedStep` default = `use_capture_while=True, max_rounds=8` (safety cap, never hit).

### Numbers (K=16, n=16, σ=0.5, dt=0.01, RTX 5090)
| mode | util | agg steps/s |
|---|---|---|
| prod_eager (variable, real baseline) | 51% | 260 |
| captured fixed-R **mr=8** (proto config) | 91% | 210 (−20%, saturates but slower) |
| captured fixed-R **mr=3** (lean+guards) | 93% | 367 (+40%) |
| captured **capture_while** (default) | **99%** | **369.7 (+42%)** |

### Build / test / git state
- **Branch `migrate/linux64-wsl2`.** Engine fork `tissue-forge` UNCHANGED this session (only `rnr/gpu` Python +
  docs). Nothing pushed.
- **Gate: 134 passed** — green twice mid-session after the step-1 `physics_warp.py` edit (`5:23`, `5:26`); a
  third re-run was kicked off at handoff to back the commit honestly (only `capture_warp.py`, not imported by any
  test, changed since the 2nd green run). capture_warp validated byte-identical by the protos below (2k+20k vs
  production + REF_traj.csv; mr=3, mr=8, capture_while all ✅; mr=1 correctly FAILS the guard).
- **Commit stages ONLY:** `rnr/gpu/physics_warp.py`, `rnr/gpu/capture_warp.py`, `docs/2026-06-26_cuda-graph-
  experiment-scope.md`, + this handoff. LEAVE (not this session's work): all `rnr/exports/*` (prior-session
  blobs) + `scratchpad/` (ephemeral protos — kept on disk as the productionization reference).
- Memory updated: `cuda-graph-experiment` (now "RESOLVED + PRODUCTIONIZED"; capture_while + the lever) + MEMORY.md.

```
 M docs/2026-06-26_cuda-graph-experiment-scope.md   <- COMMIT
 M rnr/gpu/physics_warp.py                           <- COMMIT (step-1 device-step-seed)
?? rnr/gpu/capture_warp.py                           <- COMMIT (the captured path)
 M rnr/exports/dpmax.json + fig1e/1f_*native.{csv,png}   <- LEAVE (prior-session blobs)
?? rnr/exports/ (many gpu_*/native_*/sort_oracle_*/*.mp4/*.gif)  <- LEAVE (prior-session)
?? scratchpad/ (proto_step_seed, proto_fixed_traj, proto_captured_traj, proto_maxcand_n16,
                proto_roundcount, proto_ensemble_captured, proto_while_isolate, + prior protos)  <- LEAVE
```

## Kickoff — next session

You are continuing the Warp/CUDA GPU port of TissueForge's 3D vertex model + RNR (RTX 5090) on branch
`migrate/linux64-wsl2`. **The captured forward step is DONE and productionized: `rnr/gpu/capture_warp.CapturedStep`
(default `use_capture_while=True`) is byte-identical to `engine.forward_step` and delivers 99% util / +42%
ensemble throughput at n=16, with NO max_rounds tuning.** Read `docs/2026-06-26_cuda-graph-experiment-scope.md`
top sections (★★ PRODUCTIONIZED + ★★★ capture_while) first; the gate protos are `scratchpad/proto_captured_traj.py`
(byte-identical 2k vs production+REF; `PROTO_MR` sets max_rounds) and `scratchpad/proto_ensemble_captured.py`
(the K-sim util/throughput template).

### Task: WIRE `CapturedStep` into the production ensemble figure drivers
Bank the +42% in real Fig-1E/1F runs. Priority order:

1. **Add an opt-in `--captured` path to `rnr/scripts/gpu_stability.py`** first (single-sim, it's the faithfulness
   gate) — build `CapturedStep(g, phys, params, dt, dr, seed, threshold=lth, dl_th=lth, interval=interval)` once,
   then `for step: cs.step(step)`; audit via `PaddedMesh.from_warp` at checkpoints + `cs.read_stats()` for nv/ns
   + the overflow flag. Keep the eager `E.forward_step` path as the DEFAULT. Confirm the audit timeline matches
   the eager run (it's byte-identical).
2. **Then the K-sim drivers `run_overnight` / `gpu_fig_runs`** — per-sim `CapturedStep`, replay per step. Use
   `proto_ensemble_captured.py` as the template. **Measure at n=16+ to confirm 99%/+42% holds in the real
   pipeline**, then regenerate the canonical Fig 1E/1F (`fig1e_*`/`fig1f_*`) if the user wants.
3. **Validate the interval>1 prefix-graph path** — for dt<0.01 (interval=round(0.01/dt)>1) `CapturedStep`
   captures a separate prefix-only graph for non-reconnect steps; the 2k gate only exercised interval=1. Add a
   `proto_captured_traj.py` run at dt=0.002 (interval=5) and confirm byte-identical.

Commands: `pixi run test` (134 gate) · `PROTO_MR=8 pixi run python scratchpad/proto_captured_traj.py 2000 500`
(capture_while byte-identical gate) · `PROTO_N=16 pixi run python scratchpad/proto_ensemble_captured.py 16 250`
(K=16 util/throughput) · `PROTO_N=16 pixi run python scratchpad/proto_roundcount.py 600` (round-count per regime).

Caveats: a captured graph BAKES the director `seed` (constant per sim, fine) but reads `step` from `g['_step_dev']`
(host-bumped per replay) — keep `set_director_step` OUTSIDE any capture. `read_stats()` and audits are the ONLY
host syncs; don't add per-step readbacks. capture needs `force_module_load=True`. If a NEW regime's round count
ever exceeds the `max_rounds=8` cap (it won't at ≤2), capture_while just stops at the cap (the count-cap kernel);
re-measure with `proto_roundcount.py` to confirm. Multi-stream P4 stays MOOT (sequential replay already saturates).

**Scope + license guardrails:** GPU-port phase only; reimplement from Okuda 2013 / our `rnr/`, NEVER copy GPL
`tvm/`. Read-only oracles (own `.git`, never commit): `tvm/ 3DVertVor/ cellGPU/ VertAX/ gpu_reference_papers/`.
`tissue-forge/` is our ACTIVE fork — engine changes commit to ITS repo (`feat/native-rnr-reconnection`); none this
session. Don't scope-creep into growth/morphogenesis. Commit at handoff (standing auth); push only on explicit ask.
