# CUDA-graph experiment — Phase 0 + capture_while de-risk + P1/P2 prerequisites

## Summary 2026-06-26 01:48 EDT

Goal: attack the K=16 concurrency ceiling (util ~47%, sync-bound) by capturing `forward_step` into a CUDA
graph. This session: scoped the experiment, ran the cheap Phase-0 probe, de-risked the load-bearing
mechanism, and landed the two bit-identical prerequisites (P1, P2). Full plan +
running status: `docs/2026-06-26_cuda-graph-experiment-scope.md`; memory `cuda-graph-experiment`.

**Why graphs, why Warp (not C++):** the kernels are already native CUDA (Warp JITs to PTX) — no Python→C++
multiplier; the bottleneck is host-orchestration syncs. The C++ port's real trigger is TF *integration* +
algorithm freeze, not speed (see the answer recorded in the scope doc's intro).

**Phase 0 — verdict PROCEED, bottleneck localized** (`scratchpad/prof_graph_phase0.py`, static 5-kernel
relaxation step, reconnection OFF, baseline vs `wp.ScopedCapture` replay):
- **K=16 the prefix is ALREADY 99% util** — it's compute-bound (the force kernel), so graphing it adds
  nothing (−2%). The surprise that reframes everything: the ~47% full-step ceiling is **entirely the
  reconnect path**, specifically the per-round full-device `wp.synchronize_device` (schedule_warp.py:533)
  reading M/n_win — a global barrier blocking cross-sim overlap. (K=1 graphing helps −11%, util 79→98, but
  K=1 isn't the target.) → don't graph the prefix standalone; the payoff is graphing the reconnect path
  device-side + cross-sim overlap.

**capture_while de-risk — PASS on this box** (`scratchpad/test_capture_while.py`): `wp.capture_while`
(eager + inside a captured graph, replayed 2× with device conditions 3 then 7 → looped exactly 3 then 7,
re-reading the device flag each replay) and `wp.capture_if` all work on RTX5090/WSL2/CUDA-12.8. So P3's
device-side round loop is viable here — this was the single biggest unknown.

**P1 — alloc-free step path** (bit-identical, 134-gate + 2k/20k byte-identical). Graph capture forbids
allocation in-region, so `physics_warp._ensure_step_buffers` pre-allocates the geometry/force/surface-geom
outputs on `g` (zero_ in place → byte-identical to per-call `wp.zeros`); `orient_warp` uses persistent
snw/clo/flip/counter (snw `wp.copy` not `wp.clone`; the mark kernel writes every `flip[s]` each launch so
reuse is safe — orient_warp.py:52). **0 perf change** (mempool already made `wp.zeros` free, measured last
session) — this is PURELY the capture prerequisite, resurrecting the "dead" buffer-reuse lever for a new reason.

**P2 — pointer-stable compact** (bit-identical, 134-gate + 2k). Graph capture pins device addresses, but
`compact_warp` ping-ponged `g[k]=dst[k]` (different addresses each compact → a captured graph goes stale).
Now it COPIES the compacted scratch back into g's canonical fixed-address arrays (compact_warp.py tail).
Costs one device→device SoA copy/compact (~tens of µs) — the price of capture-compatibility; the swap was free.

**Build/test/git state:**
- Branch `migrate/linux64-wsl2`. Commits this session so far: `9a7028a` (orient surf-only), `8eedf8f`
  (detect n_s→cap_s) from the earlier handoff; P1/P2 commits added below. Fork `tissue-forge`
  (`feat/native-rnr-reconnection`) UNCHANGED. **Nothing pushed.**
- **Gate: `pixi run test` = 134 passed** (run after both P1 and P2 code edits; only the scope doc + memory
  changed since → not re-run). 2k trajectory byte-identical vs `scratchpad/REF_traj.csv`; 20k recon
  I/H=4010/3028.
- This batch stages ONLY: `rnr/gpu/{physics_warp,orient_warp,compact_warp}.py` + the scope doc + this
  handoff. Everything else is prior-session/ephemeral, LEAVE:
```
 M rnr/gpu/{compact_warp,orient_warp,physics_warp}.py   (committed: P1/P2)
?? docs/2026-06-26_cuda-graph-experiment-scope.md        (committed: scope)
 M/?? rnr/exports/*      (prior-session regenerable blobs — LEAVE)
?? scratchpad/           (this session's prof_graph_phase0.py, test_capture_while.py, prof_geom_split.py,
                          + prior profilers/REF_traj — ephemeral — LEAVE)
```
- Memory: new `cuda-graph-experiment` + MEMORY.md line.

## Kickoff — next session

You are continuing the Warp/CUDA GPU port of TissueForge's 3D vertex model + RNR (RTX 5090, 32 GB) on branch
`migrate/linux64-wsl2`. The active workstream is the **CUDA-graph capture of `forward_step`** to break the
K=16 concurrency ceiling (util ~47%, sync-bound). Read `docs/2026-06-26_cuda-graph-experiment-scope.md` first
— it has the full plan + status. P0 (verdict: proceed; bottleneck = the reconnect path's per-round
full-device sync, NOT the prefix which is already 99% util), the `capture_while` de-risk (PASS on this box),
**P1 (alloc-free step path)** and **P2 (pointer-stable compact)** are DONE and bit-identical. All perf gains
are still GATED on P3 — P1/P2 are 0-benefit prerequisites.

**Validation gate (RE-RUN `pixi run test` on ANY `rnr/gpu/*.py` change):**
```
pixi run test                                                     # expect 134
pixi run gpu-stability --n 10 --steps 20000 --dt 0.01 --ic mixed  # REF: recon I/H=4010/3028, het 0.4604
pixi run gpu-stability --n 10 --steps 2000 --dt 0.01 --ic mixed --check-every 500 --csv /tmp/t.csv
cut -d, -f1-10 /tmp/t.csv | diff - scratchpad/REF_traj.csv        # byte-identical (strip col 11 = wall sec)
```

**Priority order:**
1. **Consider the cheaper intermediate FIRST (from the P0 finding):** a **batched multi-sim driver** that
   advances all K sims through each reconnect round together — stream-parallel launches + ONE
   `wp.synchronize_device`/round instead of one per *sim*-round (a K× sync cut). The current
   `concurrency_probe` steps sims sequentially (`for i: forward_step(sim_i)`); a batched driver may recover
   much of the util WITHOUT the P3 restructure. Prototype in `scratchpad/`, measure K=16 util via
   `concurrency_probe.py 5.0 4000`. If it gets util high, you may not need full graphs.
2. **P3 — device-side reconnect loop + full-step capture** (the load-bearing phase, ~3–5 days). detect writes
   M to a device scalar (stop the host readback); reconnect round loops → `wp.capture_while` on a device
   "candidates remain" flag; gather/reserve/apply launch over a fixed `MAX_CAND` with device-M masking +
   an overflow flag (raw emit k~150 at n=16 → MAX_CAND 4096–8192); **reserve/apply still allocate owner+won
   arrays per round — pre-allocate them** (the remaining in-region allocs); compact/orient gated by
   `wp.capture_if`. Capture full `forward_step`, replay per step. Then **P4 — multi-stream the K sims**
   (each its own graph + `wp.Stream`) for the cross-sim overlap that actually lifts ensemble throughput.
3. Watch for: `capture_while` body must be alloc/sync-free (P1/P2 cleared the prefix+compact; reserve/apply
   are the last allocs); `force_module_load=True` before capture; the director RNG seed is `step`-varying
   (engine.py:42 via `step*nb`) — baked into a captured graph, so feed `step` via a device array the graph
   reads, or it'll reuse one step's seed every replay.

**Measurement caveat:** the per-phase *bracketed* profiler (`prof_perstep.py`) is meaningless under capture
(you replay the whole graph) — judge by natural per-step + `concurrency_probe` util only.

**Scope + license guardrails:** GPU-port phase only; reimplement from Okuda 2013 / our `rnr/`, NEVER copy GPL
`tvm/`. Read-only oracles (own `.git`, never commit): `tvm/ 3DVertVor/ cellGPU/ VertAX/ gpu_reference_papers/`.
`tissue-forge/` is our ACTIVE fork — engine changes commit to ITS repo (`feat/native-rnr-reconnection`).
Don't scope-creep into growth/morphogenesis. Commit at handoff (standing auth); push only on explicit ask.
