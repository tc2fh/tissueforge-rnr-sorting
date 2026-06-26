# GPU per-step throughput: compact + find_short_edges buffer-reuse (host-sync hunt)

## Summary 2026-06-25 22:18 EDT

Goal (kickoff from the 1640 handoff): attack the per-step **host-sync** capping GPU util at ~34%.

**The kickoff's hypothesis was WRONG, twice — measure, don't assume.** A per-phase profile of
`forward_step` (`scratchpad/prof_perstep.py`, n=16, reconnect every step) showed:
- The unconditional readback `engine.py:64` `g["n_used"].numpy()` = **0.6%** (0.06 ms). Deferring it
  saves nothing.
- The real per-step cost: **compact 37%** + **I→H reconnect sweep 57%** (forces 9%, the rest small).
- Within the I→H sweep (`prof_isweep2.py`, EARLY active phase — the cost is state-dependent, heavy only
  while sorting): **74% is `find_short_edges_warp`** (detect), NOT the `won.numpy().sum()` the kickoff
  blamed (4%), nor gather/reserve/apply (22%).

**Root cause = per-step BUFFER ALLOCATION (not syncs).** Both hot phases re-allocated big scratch every
call:
- `compact_warp.py`: 91% of compact was per-call `wp.zeros`/`np.full(-1)`+h2d of ~9M-int pad arrays
  (`cap_v×MAX_VS` / `cap_s×MAX_RING`), rebuilt on the HOST every step (`prof_compact.py`).
- `find_short_edges_warp` (`detect_warp.py`): 59% of detect was per-call `wp.zeros(cap)` of the two
  `cap = n_s×MAX_RING` (~4.2M-int) emit arrays — even though the raw emit `k` is only ~150
  (`prof_detect.py`).

**Three fixes, all BIT-IDENTICAL (recon 4010/3028 + het@10k=0.4638 to the last digit, the cached-foam
reference), 132-gate green:**
1. `a5f277d` — **compact double-buffer**: alloc the alt array set + scan buffers ONCE, ping-pong, reset
   on-device (`fill_(-1)`/`zero_()`) not host np.full+h2d, drop the `wp.synchronize` (same-stream order
   makes the pointer swap safe). compact **3.56→0.19 ms (18.6×)**.
2. `adbb3be` — **reuse find_short_edges scan buffers** (`_ensure_detect_buf`, grown ×2 like
   `_ensure_gather_buf`); scan writes only `[0,count)`, host reads only `[:k]`, so stale tail is never
   consumed → reuse needs only `count.zero_()`. detect **3.23→1.08 ms**; I→H sweep early 4.34→2.12 ms.
3. `4594b66` — **tighten the detect buffer** (it was the n_s×MAX_RING worst case = ~50 MB/sim PERSISTENT
   though k~150): start at `_DETECT_BUF_START=8192`, grow ×2 on overflow. `scan_short_edges_kernel`
   bounds-guards its write (`if idx < out_v10.shape[0]`) so the atomic count stays exact on overflow;
   `find_short_edges_warp` detects `k > cap` → grow + rescan (rare; normally one scan). FP length trigger
   untouched (no codegen perturbation). Forced-overflow test (`verify_detect_overflow.py`, tiny start) =
   identical recon/het with the buffer grown via rescan.

**Cumulative result:** per-step n=16 **9.53 → 4.63 ms (51%)**; concurrency K=16 aggregate **122 → ~208
sim-steps/s (+70%)**, util **34% → 45%** (the ceiling moved). **TRADEOFF:** reused buffers are persistent
per-sim; the detect part was reclaimed by fix 3 (K=16 VRAM 6.04→4.41 GB), the remaining ~37 MB/sim is
compact's double-buffer (a 2nd full mesh — inherent to the ping-pong, can't reclaim without giving back
the per-step alloc). So fewer sims fit a fixed VRAM budget than the 1640 "143 in 8 GB" baseline, but
aggregate throughput is higher with far fewer sims.

**Build / test / git state:**
- Branch `migrate/linux64-wsl2`; this session's commits `a5f277d`, `adbb3be`, `4594b66` (all `rnr/gpu/`).
  Earlier this session: `e07dde1`, `db957b8`, `98d155b` (the prior O(N)-build + scale-test handoff). Fork
  `tissue-forge` (branch `feat/native-rnr-reconnection`) UNCHANGED this session, clean. **Nothing pushed.**
- **Gate: `pixi run test` = 132 passed** after the LAST code change `4594b66` (`scratchpad/test_gate_detect2.log`);
  since then only `docs/*.md` + `rnr/exports/*` changed (`git diff --name-only 4594b66`) → not re-run.
- Memory updated: `reconnect-sweep-scan-bottleneck` (compact + detect reuse + tighten + the refuted
  readback hypothesis). `docs/2026-06-25_gpu-single-run-scaling.md` got a dated UPDATE note (committed in
  this handoff).
- `git status --short` (workspace), all regenerable/ephemeral — LEAVE:
```
 M docs/2026-06-25_gpu-single-run-scaling.md   (committed with this handoff)
 M rnr/exports/{dpmax.json,fig1e_*,fig1f_*}     (prior-session regenerable blobs)
?? rnr/exports/  (many gpu_*/native_*/sort_oracle_* + the 2 committed scale PNGs already tracked)
?? scratchpad/   (this session's profilers + verify scripts; ephemeral, referenced by the memory/doc)
```

## Kickoff — next session

You are continuing the Warp/CUDA GPU port of TissueForge's 3D vertex model + RNR (RTX 5090, 32 GB) on
branch `migrate/linux64-wsl2`. Per-step throughput was just ~halved (compact + find_short_edges buffer
reuse) and GPU util lifted 34→45%. The remaining per-step cost is the I→H reconnect sweep's **host-driven
round loop**.

**Your task (deeper, BIT-IDENTITY-RISKY): make the I→H reconnect round fully DEVICE-RESIDENT** — kill the
per-round host readbacks so concurrent sims stop serializing and util rises further.

The per-round host trip in `schedule_warp.py::reconnect_sweep_warp_device` (lines ~491-538):
1. `find_short_edges_warp` (`detect_warp.py`) returns a HOST `(M,2)` array — and internally syncs **3×**
   (`count.numpy()`, `keep.numpy()`, `out_v10/v11[:k].numpy()`) + does host `np.unique` (dedup + lex sort).
2. `won.numpy().sum()` — the winner count to decide loop termination (only 4%, secondary).

Priority order:
1. **Move the candidate filter+dedup+sort ON DEVICE** so detect returns a small DEVICE array (+ a device
   count), no host `np.unique`. The hard part: reproduce `np.unique(axis=0)`'s **lex-ascending** order
   exactly (a Warp radix sort on key `v10*BIG+v11` then dedup-adjacent). The gather (`gather_warp.py`)
   then reads the device candidate array directly (it already takes packed device arrays downstream).
2. **Device-side round termination** so the host doesn't read counts every round: e.g. a device flag
   "any winner this round?" read once, or fuse the loop. Note Warp launch dims are HOST ints, so the
   candidate count `M` must still reach the host to size the gather/reserve/apply launches — unless you
   launch over a fixed cap and mask (the detect-buffer tightening already established the small-cap +
   overflow-grow pattern; reuse it).
3. **Validate bit-identicality at EVERY step** — this is the whole risk. Round 1 must stay fingerprint-
   exact; the memory warns device round-ORDER can diverge past round 1 (also-valid but different batch),
   so check the full trajectory, not just step 1.

**Validation commands / caveats:**
```
pixi run test                                                          # 132 expected (RE-RUN -- gpu/*.py changes)
pixi run gpu-stability --n 10 --steps 20000 --dt 0.01 --ic mixed      # BIT-IDENTICAL ref: recon I/H=4010/3028, het@10k=0.4638
pixi run python scratchpad/prof_perstep.py 16 150                     # per-phase forward_step breakdown
pixi run python scratchpad/prof_isweep2.py                            # I->H sweep loop breakdown (early active phase)
pixi run python scratchpad/concurrency_probe.py 5.0 4000             # util/throughput microscope (K=1,16)
```
- The I→H cost is STATE-DEPENDENT (heavy only in the early active sort; ~0.79 rounds/step once quiesced) —
  profile the EARLY phase (prof_isweep2/prof_detect warm up only ~5 steps on purpose).
- `scratchpad/` is ephemeral (not committed); the profilers/verify scripts live there — recreate from the
  memory/doc if gone, or move to `rnr/scripts/` to make durable.
- Lower-priority levers (documented in `reconnect-sweep-scan-bottleneck` memory): reserve owner-array
  reuse (~1.5%), force kernel (0.55 ms, O(valence²) active-drive dedup, bit-identical-able). compact's
  double-buffer is the remaining per-sim VRAM (inherent).

**Scope + license guardrails:** GPU-port phase only; reimplement from Okuda 2013 / our `rnr/`, NEVER copy
GPL `tvm/`. Read-only oracles (own `.git`, never commit): `tvm/ 3DVertVor/ cellGPU/ VertAX/
gpu_reference_papers/`. **`tissue-forge/` is our ACTIVE fork** — engine changes commit to ITS repo
(`feat/native-rnr-reconnection`), never staged into the workspace `rnr` repo. Don't scope-creep into
growth/morphogenesis. Commit at handoff (standing auth); push only on explicit ask.
