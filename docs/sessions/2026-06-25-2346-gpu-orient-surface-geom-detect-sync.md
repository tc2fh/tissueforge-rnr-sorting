# GPU orient surface-only geometry + detect n_s→cap_s sync-removal

## Summary 2026-06-25 23:46 EDT

Continuation of the 2310 handoff's priority-1 ("re-profile, decide the next lever from fresh numbers").
Both reconnect sweeps were already device-resident; this session re-profiled and landed two **bit-identical**
single-sim wins, then mapped where the remaining ceiling is.

**Re-profile (`scratchpad/prof_perstep.py 16 150`, recon every step), the fresh landscape:**
`5_recon_ItoH 1.37 (33%) > 3_forces 0.93 (22%) > 6_recon_HtoI 0.66 > 8_orient 0.51 > 2_geometry 0.30`.
The 2310 kickoff guessed forces would be the top phase — it isn't; the I→H sweep still is.

**Win 1 — orient surface-only geometry (`physics_warp.py:340`, `orient_warp.py:97`).** `orient_repair_warp`
called the full `compute_geometry_warp` (surface **and** body kernels, 7 allocs) but uses **only `snorm`**
(orient_warp.py:98). Added `compute_surface_geom_warp` (runs just `surface_geom_kernel`, skips the body
kernel + its 4 allocs). Measured the discarded body work via `scratchpad/prof_geom_split.py`: **body
kernel = 46% of full geometry**. orient **0.519→0.404 ms (−22%)**. Bit-identical: `snorm` comes from the
same surface kernel; the body kernel never feeds back into it.

**Win 2 — detect `n_s`→`cap_s` (`detect_warp.py:114` H-side, `:398` I-side).** Both device detect
functions read `g["n_used"].numpy()[1]` every round to set the scan dim — **1 of detect's 3 per-round host
syncs** (the other two, `k` and `M`, are inherent for launch sizing). Replaced with the **host-known
`g["cap_s"]`** (no sync): dead surfaces are masked by `surf_alive` in the scan kernel and emit nothing, so
the candidate set is **byte-identical**. At n=16 that's ~9% more (cheap, masked) threads (cap_s 62086 vs
n_s ~57000). The H-side buffer `_ensure_tri_buf` is now sized to `cap_s` (still negligible).

**Combined perf (git-stash before/after, n=16):** single-sim **natural per-step 3.97→3.55 ms (−10.6%)**
(orient −0.13, detect −0.29). The detect win is **bigger than the in-bracket sweep drop** (~0.17 ms) — the
rest is the **hidden host-sync win** the per-phase profiler (which forces syncs) structurally can't show.
Concurrency K=16: **util 46→52-55%** (GPU stays busier), throughput ~flat (240→242, within noise).

**Ceiling findings that bound future levers (the real deliverable):**
- **Buffer-reuse is DEAD.** `prof_geom_split.py` measured persistent-buffers vs `wp.zeros` = **0 win**
  (−0.0005 ms): the Warp **mempool makes allocs free** at this scale. The prior compact/detect buffer-reuse
  wins are exhausted; do NOT re-chase the pattern. Remaining cost is kernel-COMPUTE + host-SYNCS.
- **Concurrency is SYNC-BOUND.** Util only ~47% at K=16 → GPU half-idle even at max concurrency, so per-sim
  compute cuts free SM cycles other sims can't fill → throughput flat. The K=16-headline lever is removing
  per-round host syncs, but the easy one (`n_s`) is now gone; `k`/`M` are inherent (launch sizing) and the
  engine's per-step `n_used` read (engine.py:64) is **entangled** with the stability harness's every-step
  slot-exhaustion safety check (`gpu_stability.py:162,168`) — can't defer without a device-side capacity flag.
- **Force kernel is NOT bit-identically improvable.** Its real cost is `d_area_grad` (heavy fp64 cross/length
  over each ring, `nval`× per vertex), **not** the active-drive O(valence²) dedup the 2310 kickoff flagged
  (that's cheap int-compares). Reducing it needs fp-sum reordering → breaks the byte-identical trajectory gate.

**Build / test / git state:**
- Branch `migrate/linux64-wsl2`. New commit this session (see below) on top of `f5016aa`. Fork `tissue-forge`
  (`feat/native-rnr-reconnection`) UNCHANGED — no engine change. **Nothing pushed.**
- **Gate: `pixi run test` = 134 passed** (run after BOTH code changes; only memory files changed since →
  not re-run for the commit). 2k trajectory byte-identical vs `scratchpad/REF_traj.csv`; 20k stability
  recon I/H=**4010/3028** + het 0.4604 (exact reference).
- This commit stages ONLY the 3 code files. Everything else is prior-session/ephemeral, LEAVE:
```
 M rnr/gpu/{detect_warp,orient_warp,physics_warp}.py   (committed)
 M rnr/exports/*  + ?? rnr/exports/*    (prior-session regenerable blobs — LEAVE)
?? scratchpad/   (this session's prof_geom_split + prior profilers/REF_traj — ephemeral — LEAVE)
```
- Memory `reconnect-sweep-scan-bottleneck` updated (new section + description).

## Kickoff — next session

You are continuing the Warp/CUDA GPU port of TissueForge's 3D vertex model + RNR (RTX 5090, 32 GB) on branch
`migrate/linux64-wsl2`. Science (Manning Fig 1E/1F) is reproduced; this is perf-only, all changes must stay
**bit-identical** (the 134-gate + 2k/20k trajectory checks below are the definition of correct). Last session
landed orient surface-only geometry + detect `n_s`→`cap_s` sync-removal: single-sim per-step −10.6%
(3.97→3.55 ms), K=16 util 46→52%, throughput flat. **The easy bit-identical wins are now captured** — read the
ceiling findings above before picking a lever; several obvious-looking ones are already proven dead/entangled.

**Validation commands (the bit-identity gate — RE-RUN `pixi run test` on ANY `rnr/gpu/*.py` change):**
```
pixi run test                                                     # expect 134
pixi run gpu-stability --n 10 --steps 20000 --dt 0.01 --ic mixed  # REF: recon I/H=4010/3028, het@end=0.4604
# fast inner-loop bit-ident check (rebuild scratchpad/REF_traj.csv from current code if scratchpad gone):
pixi run gpu-stability --n 10 --steps 2000 --dt 0.01 --ic mixed --check-every 500 --csv /tmp/t.csv
cut -d, -f1-10 /tmp/t.csv | diff - scratchpad/REF_traj.csv     # (col 11 = wall-clock sec; STRIP it)
```

**Candidate next levers, priority order — but note diminishing returns (the cheap syncs/allocs are gone):**
1. **Concurrency ceiling (the headline metric) needs a STRUCTURAL change, not another micro-sync.** Util is
   stuck ~47-55% at K=16 because each sim's `forward_step` is a host-driven sequence of ~10 syncs/step
   (detect `k`/`M` per round ×2 sweeps, engine `n_used`, orient `counter`). The real fix is removing the host
   from the inner loop: **CUDA graphs** (capture the static per-step kernel sequence) or **multi-stream**
   (overlap independent sims' kernels instead of round-robin). Big, risky for bit-identicality — scope it
   deliberately. This is where the K=16 throughput actually lives.
2. **engine `n_used` per-step sync (engine.py:64)** — removable only by moving the slot-exhaustion safety
   check (`gpu_stability.py:168`) device-side (compare n_used to cap in a kernel, set a flag, read the flag
   every ~500 steps not every step). Then `nv_max` tracking also needs a device-side max. Medium effort,
   helps every-step (not just recon-step) concurrency.
3. **orient per-iteration `counter.numpy()` (orient_warp.py:110)** — recon-gated, 1-2 syncs/recon-step.
   Fixed-iteration (no early break) is bit-identical (extra iters are no-op flips) but adds wasted compute;
   net-positive only at concurrency. Marginal; measure with the concurrency probe, not prof_perstep.
4. **Force kernel — DO NOT attempt bit-identically** (see ceiling finding). Only if the user relaxes the
   byte-identical gate to a tolerance is `d_area_grad` precompute/share worth it.

**Measurement protocol (reuse this session's):** for before/after, `git stash push rnr/gpu/<files>` to restore
the committed code, run `prof_perstep.py 16 150` (single-sim) + `concurrency_probe.py 5.0 4000` (K=16, run 2×
for noise — it's ±5%), then `git stash pop`. **The per-phase-bracketed profilers HIDE sync-removal wins**
(brackets force syncs) — judge sync levers by "natural per-step" + the concurrency probe's util, NOT the
bracketed phase deltas. `scratchpad/prof_geom_split.py` is the template for isolating a sub-kernel's cost.

**Scope + license guardrails:** GPU-port phase only; reimplement from Okuda 2013 / our `rnr/`, NEVER copy GPL
`tvm/`. Read-only oracles (own `.git`, never commit): `tvm/ 3DVertVor/ cellGPU/ VertAX/ gpu_reference_papers/`.
**`tissue-forge/` is our ACTIVE fork** — engine changes commit to ITS repo (`feat/native-rnr-reconnection`),
never staged into the workspace `rnr` repo. Don't scope-creep into growth/morphogenesis. Commit at handoff
(standing auth); push only on explicit ask.
