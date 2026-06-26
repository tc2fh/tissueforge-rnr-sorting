# Wired CapturedStep into the drivers (--captured) + the concurrency-model nuance

## Summary 2026-06-26 09:16 EDT

**Continuation of `2026-06-26-0853-gpu-captured-step-productionized.md`** (commit `1744f2e`: the captured
`forward_step` / `CapturedStep` + capture_while). This block did the **driver wiring** that handoff teed up
(its priorities 1–3) and measured the real-pipeline impact.

### What changed and why
- **`rnr/scripts/gpu_stability.py` — opt-in `--captured`** (default OFF; eager `forward_step` stays the
  default). Builds one `CapturedStep` (capture_while), then `cs.step(step)` per step. Because `CapturedStep`
  warms up a few steps during construction, the loop resumes at `cs.next_step`. Under capture there's NO
  per-step host readback → recon I/H are not tracked (logged `n/a`, frozen-heuristic skipped) and the
  slot-capacity + MAX_CAND-overflow guards + the audit move to the checkpoint (`cs.read_stats()` = the one
  sync). VALIDATED byte-identical to the eager run: het to full precision, nv/ns/vol_min/vol_max/n_problems
  EXACT at every checkpoint (n=10, 2k). `gpu_stability.py:48,164` (the branch).
- **`rnr/scripts/gpu_fig_runs.py` — opt-in `--captured`** forwards `--captured` to each per-sim
  `gpu_stability` subprocess. End-to-end smoke green (4 concurrent captured subprocesses → rc=0 OK).
- **interval>1 prefix-graph path VALIDATED** — `CapturedStep` captures a separate prefix-only graph for
  non-reconnect steps when interval>1; the prior gate only hit interval=1. `scratchpad/proto_captured_traj.py`
  parametrized by `PROTO_DT`; at dt=0.002 (interval=5) the captured run is byte-identical to production
  (`prefix_graph=yes`, overflow/underconverged False).

### ★ THE NUANCE (the key finding of this block) — captured wins SINGLE-SIM, NOT the process-pool
- **Single-sim @ n=10: eager 3.51 → captured 2.74 ms/step = +22%** (the n=16 in-process K-sim was +42% —
  higher per-sim occupancy). The capture removes the per-step host overhead that idles the GPU between trips.
- **BUT `gpu_fig_runs` runs K SEPARATE PROCESSES (CONC≈6), and OS process concurrency ALREADY saturates the
  GPU** (one process's host overhead overlaps another's GPU work). Measured sequential A/B, 4 jobs ×12k steps:
  **eager pool 82.7s vs captured pool 81.05s ≈ neutral (~2%).** So `--captured` in the figure pipeline is
  byte-identical + harmless but adds ~no aggregate throughput at production concurrency. Its real value is
  single-sim / lower-CONC runs (GPU otherwise idles), or hitting the same throughput at LOWER CONC (less host/
  CPU load). **Banking the +42% would require switching to the IN-PROCESS K-sim model** (one process, K
  `CapturedStep`s, sequential replay — `scratchpad/proto_ensemble_captured.py`), but the process-pool is
  already GPU-saturated → that's a nice-to-have, not a throughput necessity.

### Build / test / git state
- **Branch `migrate/linux64-wsl2`.** Engine fork UNCHANGED. Nothing pushed.
- **Gate:** 134 passed green earlier this session (3×, incl. commit `1744f2e`); a re-run was kicked off at this
  handoff to back the commit (only `gpu_stability.py` + `gpu_fig_runs.py` — CLI scripts NOT imported by any
  test, verified — + docs changed since; both scripts exercised directly + byte-identical). `capture_warp.py`
  unchanged since `1744f2e`.
- **Commit stages ONLY:** `rnr/scripts/gpu_stability.py`, `rnr/scripts/gpu_fig_runs.py`,
  `docs/2026-06-26_cuda-graph-experiment-scope.md`, + this handoff. LEAVE: `rnr/exports/*` (prior-session
  blobs) + `scratchpad/` (ephemeral). Memory `cuda-graph-experiment` + MEMORY.md updated (the wiring + nuance).

```
 M docs/2026-06-26_cuda-graph-experiment-scope.md   <- COMMIT
 M rnr/scripts/gpu_fig_runs.py                       <- COMMIT
 M rnr/scripts/gpu_stability.py                      <- COMMIT
?? docs/sessions/2026-06-26-0916-...md               <- COMMIT (this handoff)
 (rnr/exports/* + scratchpad/*  <- LEAVE)
```

## Kickoff — next session

You are continuing the Warp/CUDA GPU port of TissueForge's 3D vertex model + RNR (RTX 5090) on branch
`migrate/linux64-wsl2`. **The CUDA-graph captured step (`rnr/gpu/capture_warp.CapturedStep`, capture_while) is
DONE, byte-identical, and WIRED as opt-in `--captured` into `gpu_stability.py` + `gpu_fig_runs.py`.** Read
`docs/2026-06-26_cuda-graph-experiment-scope.md` (the ★★/★★★/★ WIRED sections at top). The perf question is
ANSWERED: C++ not needed; the captured win is single-sim/in-process (+22–42%), and the production figure
process-pool is already GPU-saturated (so `--captured` there is byte-identical but throughput-neutral).

The GPU-graph thread is essentially complete. Remaining items are OPTIONAL — pick by what the user wants:
1. **(optional) Regenerate canonical Fig 1E/1F** — runs are byte-identical with/without `--captured`, so the
   figures are unchanged; only worth it as a speed/validation exercise. `pixi run python rnr/scripts/gpu_fig_runs.py
   400000 6 7,8,9 0.01 [--captured]` then `pixi run python rnr/scripts/gpu_fig1e1f.py` (check that script's args).
2. **(optional) In-process K-sim captured driver** — only if a single large-N (n≥16) ensemble in ONE process is
   wanted; build it around `scratchpad/proto_ensemble_captured.py` (CapturedStep per sim, sequential replay) to
   bank +42%. Not needed for the figures (process-pool already saturates).
3. **The actual next milestone (per CLAUDE.md): the native C++ `MeshQuality` TF-integration** — the GPU work
   proved perf doesn't need C++, so the port is now a TF-integration milestone, to be done when the RNR
   algorithm is frozen. That's a separate, larger effort — confirm scope with the user before starting.

Commands: `pixi run test` (134 gate) · `pixi run gpu-stability --n 10 --steps 5000 --dt 0.01 --sigma 0.5 --ic
mixed [--captured]` (single sim) · `PROTO_DT=0.002 PROTO_MR=8 pixi run python scratchpad/proto_captured_traj.py
1500 500` (interval>1 byte-identical gate) · `PROTO_N=16 pixi run python scratchpad/proto_ensemble_captured.py 16
250` (in-process K-sim util/throughput).

Caveats: `--captured` warms up → the loop starts at `cs.next_step` (recon I/H untracked; slot/overflow checked at
`--check-every`, so use a smaller value for tighter slot guarding on long runs). capture needs
`force_module_load=True` (handled). Don't add per-step host readbacks under capture.

**Scope + license guardrails:** reimplement from Okuda 2013 / our `rnr/`, NEVER copy GPL `tvm/`. Read-only oracles
(own `.git`, never commit): `tvm/ 3DVertVor/ cellGPU/ VertAX/ gpu_reference_papers/`. `tissue-forge/` is our
ACTIVE fork (`feat/native-rnr-reconnection`); none changed this session. Commit at handoff (standing auth); push
only on explicit ask.
