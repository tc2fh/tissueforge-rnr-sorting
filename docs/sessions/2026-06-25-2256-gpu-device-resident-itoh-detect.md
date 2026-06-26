# GPU device-resident I→H detect (kill the per-round candidate d2h/h2d + host np.unique)

## Summary 2026-06-25 22:56 EDT

Goal (kickoff from the 2218 handoff): make the I→H reconnect round's DETECT device-resident — keep the
candidate short-edge list on the GPU so concurrent sims stop serializing on per-round host syncs.

**Done — handoff priority 1, bit-identical, 133-gate green.** The candidate list now never leaves the
device through a whole I→H round; only the scalar count `M` is read back.

- **`detect_warp.find_short_edges_device(g, threshold) → (c_v10, c_v11, M)`** (`rnr/gpu/detect_warp.py:350`):
  scan → interior-filter (keep stays on device) → **on-device dedup/lex-sort reproducing
  `np.unique(axis=0)`**. The crux (the whole bit-identity risk): the lowest-id-wins reservation depends on
  the candidate order being lex-ascending. Reproduced on-device by packing each kept edge into an int64 key
  `v10*(1<<32)+v11` (filtered-out → sentinel `1<<62`), `warp.utils.radix_sort_pairs` (int64 keys ARE
  supported — checked the source), then mark-first (`mark_first_kernel`) + `array_scan` (inclusive) +
  scatter (`scatter_unique_kernel`, reads the unpermuted (v10,v11) via the sort-rode-along `values` index,
  so no int64 unpack). 3 new kernels + `_ensure_detect_buf` extended with keys/values (2×cap, the radix
  2*count scratch req) + is_first/out_pos/cand_v10v11.
- **`gather_warp.gather_i_configs_warp_device(g, c_v10, c_v11, m, buf)`** (`rnr/gpu/gather_warp.py:286`):
  same as `gather_i_configs_warp`'s reuse path but takes the DEVICE candidate arrays directly — no h2d
  upload. Returns the identical packed-device dict shape.
- **`schedule_warp.reconnect_sweep_warp_device`** (loop at `rnr/gpu/schedule_warp.py:526`) rewired to the two new
  funcs. The host `find_short_edges_warp` + `gather_i_configs_warp` are LEFT INTACT (still used by the
  hybrid path `detect_short_edges_hybrid` + `gather_i_configs_to_list` + tests).

**Why the win is modest single-sim but real under concurrency:** the candidate data was already tiny
(~150 edges); the cost removed was the SYNCS (keep[:k] + v10v11[:k] readbacks + host np.unique + gather
h2d), which only serialize visibly when K sims contend for the one GPU. Measured (n=16, **git-stash
before/after**):
- single-sim natural per-step 4.59 → 4.40 ms (−4%); I→H sweep phase 1.73 → 1.43 ms (−17%).
- **concurrency K=16: 202.4 → 226.4 sim-steps/s (+12%), ms/sim-step 4.94 → 4.42, util 43% → 46%.**

**Bit-identicality proven 4 ways:** (1) new unit test `test_i_device_detect_matches_host_unique`
(`rnr/tests/test_gpu_detect_warp.py:115`) — device output byte-equal to host `np.unique`; (2) ephemeral
`scratchpad/verify_detect_device.py` — byte-identical across 120 evolving active meshes; (3) bit-identical
2000-step trajectory (het/nv/ns/recon to the last digit vs `scratchpad/REF_traj.csv`); (4) 20k reference
hits **recon I/H=4010/3028, het@10k=0.4638** exactly.

**Deferred — priority 2 (`won.numpy().sum()` round-termination readback), deliberately NOT done.** It's
~8% of I→H, but `M` is read every round anyway to size the gather/reserve/apply launches, and `won.sum`
guards against spinning on all-vetoed candidates — so device-side termination saves little and risks an
extra empty round/step. Low ROI; left as a future lever.

**Build / test / git state:**
- Branch `migrate/linux64-wsl2`. Fork `tissue-forge` (branch `feat/native-rnr-reconnection`) UNCHANGED
  this session, clean — no engine commit needed. **Nothing pushed.**
- **Gate: `pixi run test` = 133 passed** (was 132 + the new equivalence test) after the last code change
  (`scratchpad/gate_final.log`). The earlier 132-gate also passed mid-session on the gpu/*.py changes.
- Memory `reconnect-sweep-scan-bottleneck` updated (device-resident detect section + description).
  `docs/2026-06-25_reconnect-sweep-optimization.md` got a dated "Update" note (the old "next levers" are
  stale — lever 1 skip-scan is dead; this lever now done).
- `git status --short` (workspace) — this session's commit stages ONLY the 4 code/test files + the
  optimization doc + this handoff. Everything else is prior-session/ephemeral, LEAVE:
```
 M docs/2026-06-25_reconnect-sweep-optimization.md   (committed with this handoff)
 M rnr/gpu/{detect_warp,gather_warp,schedule_warp}.py (committed)
 M rnr/tests/test_gpu_detect_warp.py                  (committed)
 M rnr/exports/{dpmax.json,fig1e_*,fig1f_*}           (prior-session regenerable blobs — LEAVE)
?? rnr/exports/  (many gpu_*/native_*/sort_oracle_* — prior-session artifacts — LEAVE)
?? scratchpad/   (this session's profilers + REF_traj/verify scripts; ephemeral — LEAVE)
```

## Kickoff — next session

You are continuing the Warp/CUDA GPU port of TissueForge's 3D vertex model + RNR (RTX 5090, 32 GB) on
branch `migrate/linux64-wsl2`. The I→H detect is now fully device-resident (candidate list stays on GPU;
concurrency K=16 +12%, util 43→46%). All bit-identical.

**Validation commands (memorize these — they are the bit-identity gate):**
```
pixi run test                                                     # 133 expected (RE-RUN on any gpu/*.py change)
pixi run gpu-stability --n 10 --steps 20000 --dt 0.01 --ic mixed  # REF: recon I/H=4010/3028, het@10k=0.4638
# fast inner-loop bit-ident check (rebuild scratchpad/REF_traj.csv first from current code if scratchpad gone):
pixi run gpu-stability --n 10 --steps 2000 --dt 0.01 --ic mixed --check-every 500 --csv /tmp/t.csv
cut -d, -f1-10 /tmp/t.csv | diff - scratchpad/REF_traj.csv     # (col 11 = wall-clock sec; STRIP it)
```

**Candidate next levers, priority order (all perf; science Fig 1E/1F already reproduced):**
1. **Mirror the device-resident detect to the H→I (reverse) sweep.** `reconnect_sweep_h_to_i_warp_device`
   (`schedule_warp.py:~560`) still does `PaddedMesh.from_warp(g)` + host `find_small_triangles_csr` /
   `detect_small_triangles_hybrid` per round — the SAME host-round-trip the I→H sweep just shed. H→I is
   ~20% of the step (`prof_perstep.py 16 150`, `6_recon_HtoI` ≈ 0.92 ms). The H-trigger is simpler
   (surface-ascending int sort, not (v10,v11) lex) so the device dedup is easier. Same bit-identity gate.
2. **Profile the NEW per-step breakdown first** (`pixi run python scratchpad/prof_perstep.py 16 150`,
   `prof_isweep2.py`) — after this session the I→H sweep is cheaper, so re-confirm what dominates before
   optimizing. Likely `force` (0.93 ms, 4 actors — real work) and `6_recon_HtoI` are now the top phases.
3. **Concurrency VRAM floor:** compact's double-buffer is ~37 MB/sim (a 2nd full mesh, inherent to the
   ping-pong) — the per-sim floor that caps K. Only worth attacking if you need more concurrent sims than
   VRAM allows; revisit `compact_warp.py` if so.
4. **Lower-ROL/deferred:** the `won.numpy().sum()` device-side termination (priority-2 above — marginal).

**Measurement protocol (this session's, reuse it):** for before/after, `git stash push rnr/gpu/<files>`
to restore old code (untracked exports/scratchpad untouched), run `prof_perstep.py 16 150` +
`concurrency_probe.py 5.0 4000`, then `git stash pop`. The per-phase-bracketed profilers (`prof_isweep2`)
HIDE sync-removal wins (their brackets force syncs) — use the "natural per-step" number + the concurrency
probe for sync-removal levers. Concurrency K=1 in the probe is sample-starved/noisy; trust `prof_perstep`
for single-sim and the probe's K=16 for concurrency.

**Scope + license guardrails:** GPU-port phase only; reimplement from Okuda 2013 / our `rnr/`, NEVER copy
GPL `tvm/`. Read-only oracles (own `.git`, never commit): `tvm/ 3DVertVor/ cellGPU/ VertAX/
gpu_reference_papers/`. **`tissue-forge/` is our ACTIVE fork** — engine changes commit to ITS repo
(`feat/native-rnr-reconnection`), never staged into the workspace `rnr` repo. Don't scope-creep into
growth/morphogenesis. Commit at handoff (standing auth); push only on explicit ask.
