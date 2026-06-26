# GPU device-resident H→I (reverse) detect — mirror of the I→H lever

## Summary 2026-06-25 23:10 EDT

Goal (continuation of the 2256 handoff's priority-1): mirror the device-resident detect to the **H→I
(reverse) sweep**, which still did `find_small_triangles_warp` (host) + `gather_h_configs_warp` (h2d
upload) every round — the same host round-trip the I→H sweep shed at 2256.

**Done — bit-identical, 134-gate green.** The reverse sweep's candidate triangle list now stays on the GPU
through the whole round; only the scalar count `M` is read back.

- **`detect_warp.find_small_triangles_device(g, threshold) → (c_tris, M)`** (`rnr/gpu/detect_warp.py:114`):
  scan → device sort. **SIMPLER than the I-side** (the key realization): `scan_small_triangles_kernel`
  emits one thread per surface, so there are NO duplicates → the host did `np.sort` (not `np.unique`), so
  the device needs only a plain `radix_sort_pairs` on the int32 surface indices — no dedup pass, no mark-
  first/scan/scatter. Reuses `scan_small_triangles_kernel` UNCHANGED. Buffer `_ensure_tri_buf` sized ≥ n_s
  (≤1 emit/surface → the scan never overflows, no bounds guard needed; ~n_s ints/sim, negligible).
- **`gather_warp.gather_h_configs_warp_device(g, c_tris, m, buf)`** (`rnr/gpu/gather_warp.py:576`): consumes
  the device triangle array directly (no h2d), returns the same packed-device dict shape.
- **`schedule_warp.reconnect_sweep_h_to_i_warp_device`** (loop at `rnr/gpu/schedule_warp.py:601`) rewired.
  Host `find_small_triangles_warp` + `gather_h_configs_warp` LEFT INTACT (hybrid path + tests).
- `reserve_h_won_device` + `apply_h_to_i_device_warp` unchanged — they already consumed the packed device
  dict (`tri_cand` is now a device slice instead of an uploaded array; reads identically on-device).

**Bit-identicality proven:** new unit test `test_h_device_detect_matches_host_sort`
(`rnr/tests/test_gpu_detect_warp.py:91`) — device == host `np.sort` byte-for-byte;
`scratchpad/verify_detect_h_device.py` — byte-identical across 120 evolving active meshes (116 nonempty);
bit-identical 2000-step trajectory vs `scratchpad/REF_traj.csv`; 20k reference recon I/H=4010/3028,
het@10k=0.4638 exact.

**Perf (n=16, git-stash before/after vs the committed I-device-only state):**
- single-sim natural per-step 4.27 → 3.96 ms (−7%); `6_recon_HtoI` phase 0.889 → 0.741 ms (−17%).
- concurrency K=16: 224 → 238 sim-steps/s (+6%), util 47% → 49%.

**Cumulative this session (I+H device-resident) vs the 2218-handoff baseline:** K=16 **202 → 238
sim-steps/s (+18%)**, util **43% → 49%**; single-sim per-step **4.59 → 3.96 ms (−14%)**.

**Build / test / git state:**
- Branch `migrate/linux64-wsl2`. Prior commit this session: `003c66e` (I→H device-resident detect). Fork
  `tissue-forge` (branch `feat/native-rnr-reconnection`) UNCHANGED, clean — no engine commit. **Nothing pushed.**
- **Gate: `pixi run test` = 134 passed** (133 + the new H equivalence test) — see `scratchpad/gate_h.log`.
- Memory `reconnect-sweep-scan-bottleneck` updated (H→I device-detect section + description).
- `git status --short` — this commit stages ONLY the 4 code/test files + this handoff. Everything else is
  prior-session/ephemeral, LEAVE:
```
 M rnr/gpu/{detect_warp,gather_warp,schedule_warp}.py  (committed)
 M rnr/tests/test_gpu_detect_warp.py                   (committed)
 M rnr/exports/*  + ?? rnr/exports/*    (prior-session regenerable blobs — LEAVE)
?? scratchpad/   (this session's profilers + REF_traj/verify scripts; ephemeral — LEAVE)
```

## Kickoff — next session

You are continuing the Warp/CUDA GPU port of TissueForge's 3D vertex model + RNR (RTX 5090, 32 GB) on
branch `migrate/linux64-wsl2`. BOTH reconnect sweeps (I→H and H→I) are now fully device-resident — candidate
lists stay on GPU; the per-round host round-trip is gone. Cumulative this 2-part session: K=16 throughput
+18% (202→238 sim-steps/s), util 43→49%, single-sim per-step −14%. All bit-identical (134-gate).

**Validation commands (the bit-identity gate):**
```
pixi run test                                                     # 134 expected (RE-RUN on any gpu/*.py change)
pixi run gpu-stability --n 10 --steps 20000 --dt 0.01 --ic mixed  # REF: recon I/H=4010/3028, het@10k=0.4638
# fast inner-loop bit-ident check (rebuild scratchpad/REF_traj.csv from current code if scratchpad gone):
pixi run gpu-stability --n 10 --steps 2000 --dt 0.01 --ic mixed --check-every 500 --csv /tmp/t.csv
cut -d, -f1-10 /tmp/t.csv | diff - scratchpad/REF_traj.csv     # (col 11 = wall-clock sec; STRIP it)
```

**Candidate next levers, priority order (all perf; science Fig 1E/1F already reproduced):**
1. **Re-profile first — the per-step landscape shifted.** Run `pixi run python scratchpad/prof_perstep.py 16
   150`. After this session both reconnect sweeps are cheaper; the top phases are now likely **`3_forces`
   (~0.93 ms — the 4 actors, REAL work not overhead)** and the two recon sweeps (~1.5 + 0.74 ms). Decide the
   next target from the fresh numbers, not these.
2. **`force` kernel (`physics_warp.compute_forces_warp`)** — ~0.93 ms/step, the single largest phase. The
   active-drive term does an O(valence²) dedup per vertex; bit-identical-able + GPU==host gated. Was flagged
   low-ROI before but it's now the top phase. Worth a profile-led look.
3. **Concurrency VRAM floor** — compact's double-buffer is ~37 MB/sim (a 2nd full mesh, inherent to the
   ping-pong), the per-sim floor that caps K. Only attack if you need more concurrent sims than VRAM allows.
4. **Deferred / low-ROI:** device-side `won.sum` round termination (marginal — M is read per round anyway);
   reserve owner-array reuse (~1.5%).

**Measurement protocol (reuse this session's):** for before/after, `git stash push rnr/gpu/<files>` to
restore the committed code (untracked exports/scratchpad untouched), run `prof_perstep.py 16 150` +
`concurrency_probe.py 5.0 4000`, then `git stash pop`. The per-phase-bracketed profilers (`prof_isweep2`)
HIDE sync-removal wins (brackets force syncs) — use "natural per-step" + the concurrency probe for sync
levers. Concurrency K=1 in the probe is sample-starved/noisy; trust `prof_perstep` for single-sim, the
probe's K=16 for concurrency.

**Scope + license guardrails:** GPU-port phase only; reimplement from Okuda 2013 / our `rnr/`, NEVER copy
GPL `tvm/`. Read-only oracles (own `.git`, never commit): `tvm/ 3DVertVor/ cellGPU/ VertAX/
gpu_reference_papers/`. **`tissue-forge/` is our ACTIVE fork** — engine changes commit to ITS repo
(`feat/native-rnr-reconnection`), never staged into the workspace `rnr` repo. Don't scope-creep into
growth/morphogenesis. Commit at handoff (standing auth); push only on explicit ask.
