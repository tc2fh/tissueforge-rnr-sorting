# Session — GPU 3D RNR: paper-scale stability validation + RNR-at-scale bug (root cause pinned)

## Summary 2026-06-24 20:00 EDT

**Goal:** the post-Gate-E priority "**100k-step stability validation** (required to call a sort
*faithful*)", then per user request run it on **a very large tissue like the Manning paper**.
Outcome: built the long-run gate, found paper-scale does NOT yet complete, and **pinned the root
cause** to a specific I→H surgery defect (fix NOT yet landed — 2 attempts ruled out, engine
reverted to green).

**What happened & why (decisions / gotchas / findings):**

- **Paper scale = N=1728.** The dropped PDF (`gpu_reference_papers/Manning_journal.pcbi.1011724.pdf`
  = Lawson-Keister, Zhang, Nazari, Fagotto, Manning, *PLOS Comp Biol* 2024) §2.1: "confluent tissue
  composed of **N = 1728 cells**", segregated systems "**relax over 10⁵ time steps**", dt=0.01,
  V₀=1, S₀=5.6, kT=0.1, σ∈{0.1,0.2,0.5} (Fig 1E). = exactly our oracle regime; 1728 = 12³ =
  `3DVertVor/main.py` default. (Paper *text* says white-noise; project uses native **active** drive
  — settled, [[active-motility-not-thermal-noise]].)
- **New harness `rnr/scripts/gpu_stability.py`** (+ `pixi run gpu-stability`): drives
  `engine.forward_step` for the full Fig-1E/1F length and audits the mesh (consistency, finite
  positions, **vol_max AND vol_min bounds**, slot bounds, het demixing, topological-freeze flag).
  Reuses the validated `_setup_unit_foam` from `rnr/tests/test_gpu_engine.py` verbatim.
- **Builder ceiling (O(N²), handoff priority #3):** single-foam build times 432→6.8s, 1024→27s,
  2000→89s. So a single connected **1728–2000-cell** foam builds in ~60–90s and the GPU steps it
  at ~3.5 ms → 100k in ~7 min. Paper scale is feasible as ONE connected foam (no replication).
- **FALSE PASS caught + fixed.** The first 100k run at N=2000 printed "STABLE" — but a cell had
  ballooned to vol≈1500 (V₀=1) and reconnections had frozen. The verdict only checked `vol_min>0`,
  which a *degenerate-but-valid* mesh passes. Fixed `gpu_stability.py` to bound `vol_max`
  (`--vol-factor`, default 20) and flag a reconnection freeze.
- **Diagnosis chain (5 stability runs + 3 scratchpad catchers):**
  - **dt RULED OUT:** N=128 stable at dt=0.01 AND 5e-3; N=2000 fails at both.
  - **forces/integration/geometry RULED OUT:** N=2000 with **reconnection OFF** is rock-solid 20k
    steps (vol∈[0.93,1.04]). So the integrator is sound at scale.
  - **Flat/Convex regularizers RULED OUT:** they constrain face shape, not volume; K_V=10 would
    snap a real balloon back — a cell *sitting* at 1500×V₀ means its geometry is corrupted.
  - **It's the I→H reconnection.** Ceiling map (dt=5e-3,σ=0.5,mixed,seed7,100k budget): N=128 ✅
    completes (demixes 0.477→0.412, 612 I→H); N=432 ❌@~60k (1080 I→H); N=1024 ❌@~6k (1392);
    N=2000 ❌@~1.1k (1046). **Failure = cumulative I→H crossing ~1000–1400, independent of N.**
    Separator is forward **I→H count specifically** (H→I doesn't separate).
- **ROOT CAUSE (pinned, scratchpad `catch_closure.py`):** an I→H leaves the new **cap-cap triangle
  face with REVERSED WINDING** vs its `b1/b2` → the cell loses surface **closure**
  (Σ_faces sense·snorm ≠ 0; sense=+1 if b==b1 else −1) → its divergence-theorem **volume is wrong
  and origin-dependent** (wobbles as the cell drifts) → balloon. Cell stays a valid closed manifold
  (Euler V−E+F=2) so `check_consistency` misses it. Deterministic repro: **n=8, seed 7, closure
  breaks at step 60** on one cell; the bad face = the cap-cap triangle (3 consecutive fresh-slot
  verts), stored ring `[v0,v0+2,v0+1]`.
- **Fixes ATTEMPTED + RULED OUT (don't repeat):** (1) periodic min-image in I→H *placement*
  (`r0=0.5(p10+p11)` had no min-image — a real latent bug but NOT the balloon cause); (2) orienting
  the triangle ring at CREATION (`reconnect_warp.py` `i_to_h_batch_kernel` lines ~534-546). **#2 is a
  no-op because the stored ring `[v0,v0+2,v0+1]` ≠ what creation writes `[v0,v0+1,v0+2]`** → the
  winding is reversed **post-creation** by a downstream surgery primitive
  (`d_insert_between`/`d_ring_insert_after` re-wiring the fresh triangle as a neighbour's top/bottom
  face, or the known cascade side-face collapse). **Both edits were REVERTED** (`git checkout`),
  engine is back at the green baseline. Memory [[gpu-rnr-scale-corruption]] has the full record.

**Build / test / git state:**
- `reconnect_warp.py` reverted to its committed (green) state — `git diff` empty for it. Only NEW
  work: `rnr/scripts/gpu_stability.py` + the `gpu-stability` line in `pixi.toml` (neither imported by
  any test, so the gate is functionally unchanged). **`pixi run test` re-run at handoff — RESULT IN
  COMMIT MESSAGE** (only ran the GPU *subset* = 79 passed earlier this session, and `pixi.toml`
  changed, so a full-gate re-run was required by the skip rule).
- Branch `migrate/linux64-wsl2`. Memory updated (outside repo): `gpu-rnr-scale-corruption` (root
  cause + ceiling + ruled-out fixes) + `MEMORY.md` index.
- **Do NOT commit** `rnr/exports/*` (incl. this session's `gpu_stability_paperscale.csv` — it's the
  misleading false-pass data) or the read-only oracle repos.

Full `git status --short` (everything under `rnr/exports/` is leave-it output, this + prior sessions):
```
 M pixi.toml                                            # THIS session (gpu-stability task)
?? rnr/scripts/gpu_stability.py                         # THIS session (long-run gate harness)
 M rnr/exports/{dpmax.json, fig1e_*, fig1f_*}           # prior sessions — leave
?? rnr/exports/gpu_stability_paperscale.csv             # THIS session (false-pass run data) — leave
?? rnr/exports/{native_*_mixed*.mp4, sort_oracle_M8_*_native[_demixed].csv, vertex_motion_native.gif}  # prior — leave
```
(Scratchpad — gone next session, not in repo: `catch_closure.py` [the closure-violation catcher],
`catch_balloon.py`, `build_probe.py`.)

## Kickoff — next session

You are continuing a GPU port of TissueForge's 3D vertex model + RNR. The staged plan (Gates A–E)
is done & green (127 tests). This session validated long-run stability and **found a real bug**:
the GPU sort does NOT complete at paper scale (N=1728). Root cause is pinned; the fix is the job.
Read memory `gpu-rnr-scale-corruption` first; full context in
`docs/sessions/2026-06-24-2000-gpu-rnr-paperscale-stability-bug.md` and
`docs/2026-06-24_gpu-3d-vertex-model-exploration.md` §10.

**THE BUG (one line):** an I→H reconnection leaves the new cap-cap triangle face with reversed
winding vs its `b1/b2` → broken cell closure (Σ sense·snorm ≠ 0) → wrong origin-dependent volume →
balloon after ~1000 cumulative I→H. Count-driven: N≤128 survives a 100k sort (612 I→H, under the
zone); N≥432 fails.

**Priority-ordered next steps:**

1. **Find which surgery op reverses the fresh triangle's winding, then fix it.** The ring is set
   to `[v0,v0+2,v0+1]` *after* creation (creation writes `[v0,v0+1,v0+2]`), so the culprit is a
   primitive in `rnr/gpu/reconnect_warp.py` — most likely `d_insert_between`/`d_ring_insert_after`
   when a FRESH cap-cap triangle is re-wired as a neighbour's top/bottom face in a later/same-round
   I→H, or the cascade side-face→triangle collapse. Instrument the device sweep to recompute
   per-cell closure right after each I→H batch and report the first cell + face + op that breaks it.
   Then fix the winding upkeep so the triangle's normal stays consistent with its `b1/b2`.
   - **Deterministic repro (fast — breaks at step 60, no need to wait for the balloon):**
     build n=8 (`_setup_unit_foam(dev,n=8,headroom=4000,ic="mixed")`), params kv=10/ka=1/
     sigma=0.5/v_active=0.1, dt=5e-3/dr=1/lth=1e-3/interval=2/seed=7; each step do
     `director_update_warp → compute_geometry_warp → compute_forces_warp → integrate_warp`, then on
     even steps `reconnect_sweep_warp_device(g,lth,lth,8)` then `_h_to_i_..._device`; after each
     sweep compute `closure[b] = Σ_s (+snorm[s] if b==s2b[s,0] else −snorm[s])` from
     `compute_geometry_warp(g)["snorm"]` + `g["s2b"]`/`g["surf_alive"]` (vectorized `np.add.at`,
     `nb=g["nb"]`); first cell with ‖closure‖>1e-2 is the corrupted one (step 0 baseline ~4e-7).
   - **The flip-test** to ID the bad face: for the bad cell, the face `s` where
     `‖resid − 2·sense_s·snorm_s‖ ≈ 0` is the mis-wound one (it was the cap-cap triangle: 3
     consecutive fresh-slot verts).
2. **Propagate the fix** to the CPU mirror (`reconnect_csr.i_to_h_csr` + `reconnect.i_to_h`), the
   single-config kernel, and the **H→I** placement (`h_to_i_batch_kernel` ~line 632 also lacks
   periodic min-image — `r0=(p0+p1+p2)/3` breaks if tri verts straddle the box; a latent twin bug).
   Keep the round-trip + CPU==GPU bit-exact gates green.
3. **Validate:** `pixi run gpu-stability --n 10 --steps 100000 --dt 0.01 --sigma 0.5 --ic mixed
   --csv rnr/exports/gpu_stability_paperscale.csv` must end **STABLE** (vol bounded, no freeze,
   het demixes) → then paper-scale (N=1728/2000) faithful sort is unblocked. Also re-run `pixi run
   test` (expect 127).
4. **(Optional, also a real bug)** the I→H placement lacks periodic min-image (the reverted fix #1) —
   fold it in with the H→I min-image of step 2 for correctness, even though it isn't the balloon cause.

**Commands / caveats:**
```
pixi run gpu-stability --steps 5000                 # quick smoke (defaults N=2000 paper scale)
pixi run gpu-stability --n 8 --steps 20000          # fails ~step 6000 today (repro of the bug)
pixi run test                                        # full gate (~5.5 min, expect 127)
pixi run python -m pytest rnr/tests/test_gpu_*.py -q # GPU subset (~37s, 79)
```
- The Warp kernel recompiles on edit (~15-30s first launch). Every fp64 literal meeting a
  `wp.float64` must be `wp.float64(...)`; a cross-module `@wp.func` must be IMPORTED into the calling
  module (e.g. `from .physics_warp import d_minimg`).
- **`gpu_stability.py` is the real stability gate** — Gate-E tests (≤600 steps / n=4) are too short
  to trigger this bug; don't trust them for long-run faithfulness.
- **Scope/license:** GPU-port phase only; reimplement from Okuda 2013 / our `rnr/`; never copy GPL
  `tvm/`; `cellGPU/ VertAX/ gpu_reference_papers/` are read-only oracles (own `.git`, don't commit).
- `tf.init()` is one-per-process; standalone scripts mirror `conftest.vsolver`
  (`tf.init(windowless=True, dim=[60,60,60], cutoff=5.0, dt=0.001); tfv.init(); quality=None`).
