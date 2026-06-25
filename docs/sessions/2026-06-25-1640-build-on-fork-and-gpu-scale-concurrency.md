# Foam build O(N²)→O(N) on the fork engine + GPU single-run & concurrency scaling

## Summary 2026-06-25 16:40 EDT

Goal: continue mesh-scaling (handoff "Option A"), then — at the user's steer — **switch to proper
fork development** and **scale-test the GPU** (speed, util, max tissue size on 32 GB).

**Arc 1 — build O(N²)→O(N) (two complementary fixes, both bit-identical, gate-green):**
- **Root cause was NOT `become()`** (the prior handoff's prime suspect — measured 0.01 s, red herring).
  cProfile of the full build (`scratchpad/prof_fullbuild.py`) at n=16 vs n=20 showed the super-linearity
  is **TF's per-element C++ object creation**: handles are raw pointers into `std::vector` pools, and
  `Mesh::increment{Vertices,Surfaces,Bodies}` grows by a FIXED `TFMESHINV_INCR=100`, copying the vector +
  re-fixing every pointer (O(N)) each grow → N/100 grows → O(N²). Vertex.create exp 2.08, surface create
  exp ~1.82.
- **Fix 1 (Python, `rnr/geometry.py:296`):** batch `Vertex.create([all FVector3])` → ONE
  `incrementVertices(N)`. 12.70 s → 0.085 s at n=20. (Surfaces/bodies can't batch from the bindings — no
  nested-vector typemap, no exposed reserve.) Committed `e07dde1`.
- **Fix 2 (ENGINE, fork `tissue-forge/source/models/vertex/solver/tfMesh.cpp:259-305`):** geometric
  pool growth — `allocate{Vertex,Surface,Body}` grow by `max(TFMESHINV_INCR, pool.size())` (double) not
  +100 → all per-element creates amortized O(N). Surface create 8.5 s → 0.47 s at n=20. Committed to the
  **fork** `47f6fa4` (branch `feat/native-rnr-reconnection`). Both fixes preserve id order
  (smallest-free-first) → foams **bit-identical** (verified vs cached n=16, `scratchpad/verify_bitident.py`).
- **Result:** build is **O(N)**, constant ~0.75–0.87 ms/cell, **n=20 66 s → 12 s (5.5×)**.

**KEY correction (surfaced this session):** the pixi env's `tissue_forge` is **OUR FORK built from
source**, NOT the conda release (pixi.toml drops the channel/dep; `_tissue_forge.so` is a from-source
artifact; `set_motility` present). My earlier "conda package" statements were wrong. Dev loop works:
edit `tissue-forge/source/...` → `pixi run build-tf` (incremental relink, ~min) → `verify` → `test`.
CLAUDE.md updated: env section + commit guardrail now treat `tissue-forge/` as an **active dev target**
(engine commits go to its own `.git`). Memory: `fork-is-active-dev-engine` (new), `foam-build-scaling`
(updated → resolved).

**Arc 2 — GPU scale test + concurrency** (full numbers + 2 figures in
`docs/2026-06-25_gpu-single-run-scaling.md`; probes in `scratchpad/scale_probe.py`,
`concurrency_probe.py`, plots `plot_scale.py` / `plot_concurrency.py`):
- `n` = BCC seeds/axis → **cells = 2·n³**, **one single non-batched sim** (not concurrent). Verts ≈ 6.1·cells.
- **Single run:** VRAM linear ~11.6 KB/cell over a ~4.9 GB base → 32 GB ≈ 2.3M-cell ceiling (never the
  limit). GPU **latency-bound, 21–35% util**. Per-step ~1–1.8 µs/cell (and scales with *capacity*, so
  don't over-provision headroom). Largest confirmed **n=48 = 221,184 cells** (238 ms/step, 7.5 GB);
  **n=64 / 524k crashed in the HOST builder** (~3.2M TF objects) — the real ceiling now, not VRAM/GPU.
- **Concurrency (n=16, 8192 cells/sim):** ~38 MB/sim over a ~2.69 GB base → **~143 sims fit in 8 GB**
  (measured 140 = 7.88 GB, 152 = 8.32 GB). Concurrency lifts util 18%→34% + aggregate 90→122
  sim-steps/s, **saturating by K≈16** (degrades past ~96 from memory pressure). The **~34% util ceiling
  is the per-step host sync** (`forward_step` reads device counters to host every step) — not compute.
  Sweet spot ~16–48 concurrent.
- **Two robustness/throughput levers found:** (a) reconnection bump-allocator overflows a *fixed*
  headroom at large N → illegal memory access / context corruption (must scale headroom with N or fail
  gracefully); (b) the per-step host sync caps util at ~34% → **next session's focus**.

**Build / test / git state:**
- Workspace branch `migrate/linux64-wsl2`: commits this session `e07dde1` (vertex batch + CLAUDE.md),
  `db957b8` (scaling docs + 2 plots). Fork `feat/native-rnr-reconnection`: `47f6fa4` (engine geometric
  growth). **Neither pushed** (push needs explicit ask).
- **Gate: `pixi run test` = 132 passed** earlier this session (`scratchpad/test_gate_engine.log`, run
  AFTER the `tfMesh.cpp` rebuild, covering both code fixes). Only docs/`*.md`/plots/`scratchpad/` changed
  since → not re-run.
- `git status --short` (workspace) — all regenerable, intentionally uncommitted, LEAVE:
```
 M rnr/exports/dpmax.json
 M rnr/exports/fig1e_demixing_native.csv
 M rnr/exports/fig1e_demixing_native.png
 M rnr/exports/fig1f_stability_native.csv
 M rnr/exports/fig1f_stability_native.png
?? rnr/exports/  (many gpu_*/native_*/sort_oracle_* CSV/PNG/MP4 + fig1e1f_gpu_summary.csv,
                  gpu_dpmax.json, vertex_motion_native.gif — prior-session artifacts)
?? scratchpad/   (this session's probe + plot scripts; ephemeral, referenced by the scaling doc)
```
  Fork tree clean. (The two committed plots `rnr/exports/gpu_scale_sweep.png`,
  `gpu_concurrency_n16.png` are tracked now; the rest of `rnr/exports/` stays untracked.)

## Kickoff — next session

You are continuing a Warp/CUDA GPU port of TissueForge's 3D vertex model + RNR (RTX 5090, 32 GB) on
branch `migrate/linux64-wsl2`. The foam build is now O(N) and the engine is **our fork built from source**
(edit `tissue-forge/source/...` → `pixi run build-tf` → `verify`/`test`; engine commits go to the fork's
own `.git` on `feat/native-rnr-reconnection`). The GPU sim is **latency-bound, ~34% util**, and the scale
test pinned the cause.

**Your task (priority order): attack the per-step HOST-SYNC bottleneck capping GPU util at ~34%.**

1. **Confirm + localize the sync.** `rnr/gpu/engine.py::forward_step` returns a `rep` dict
   (`rep["i"]`, `rep["h"]`, `rep["nv"]`, `rep["ns"]`) — each is a device→host counter readback that
   forces a per-step sync (and serializes the concurrency sweep). Profile where the wall goes: is it the
   count readbacks, the compaction, or the reconnect-sweep scan? Use the concurrency probe as the
   utilization microscope: `pixi run python scratchpad/concurrency_probe.py 8.5 4000` (util should rise if
   a sync is removed). Also a single-run per-step profile (Warp's `wp.ScopedTimer` / `wp.synchronize`
   bracketing) at n=16/n=20.
2. **Batch / defer the readbacks.** The counters are only *needed* for the host-side audit/guards and the
   `cap_v/cap_s` exhaustion check. Options: (a) only read counters every K steps (audit cadence), keeping
   the device loop async between audits; (b) keep `nv/ns` on-device and let a device-side guard flag
   exhaustion (read a single small flag, or none); (c) fuse steps so the per-step Python→device round-trip
   amortizes. Target: lift util well past 34% and raise aggregate sim-steps/s.
3. **Validate:** `pixi run test` (132 expected — this touches `rnr/gpu/*.py`, so RE-RUN the gate) and a
   bit-identical / trajectory check (the audit-cadence change must not alter the physics, only when the
   host *reads*). Re-run the concurrency probe + `scratchpad/scale_probe.py` to quantify the util/throughput
   win; refresh `docs/2026-06-25_gpu-single-run-scaling.md` if the numbers move.

**Secondary levers (deprioritized, all documented):**
- **Headroom robustness:** the reconnection bump-allocator overflows a *fixed* headroom at large N →
  illegal memory access (n=32 needed headroom≈40000). Scale headroom with N or fail gracefully; note
  per-step cost scales with *capacity* so keep it "just enough".
- **Host-build ceiling (~½M cells):** n=64/524k crashed in the CPU-side TF object build. The TF-free
  direct-CSR builder ("Option B": assemble CSR straight from Voronoi in numpy + a `csr_mesh.build_tf_from_csr`
  for on-demand viewing) is the path past it.
- **`compute_geometry`** (host, ~8 s at n=20, LINEAR): vectorize or use `compute_geometry_warp`; note
  `v0 = box_vol/nb` is exact for a space-filling foam (skip the volume pass).

**Validation commands / caveats:**
```
pixi run test                                                    # 132 expected (RE-RUN — gpu/*.py changes)
pixi run python scratchpad/concurrency_probe.py 8.5 4000         # util microscope: ~143 n=16 sims to 8 GB
pixi run python scratchpad/scale_probe.py 16 200 0              # single-run per-n speed/VRAM/util (auto-headroom)
pixi run build-tf  &&  pixi run verify                          # after any tissue-forge/source edit
```
- `scratchpad/` is ephemeral (not committed); the probe/plot scripts referenced by the scaling doc live
  there — recreate from the doc if gone, or move to `rnr/scripts/` if you want them durable.
- Don't `--rebuild-foam` a cached foam expecting bit-identity vs the OLD ghost build (it relabels).

**Scope + license guardrails:** GPU-port phase only; reimplement from Okuda 2013 / our `rnr/`, NEVER copy
GPL `tvm/`. Read-only oracles (own `.git`, never commit): `tvm/ 3DVertVor/ cellGPU/ VertAX/
gpu_reference_papers/`. **`tissue-forge/` is our ACTIVE fork** — engine changes commit to ITS repo
(`feat/native-rnr-reconnection`), never staged into the workspace `rnr` repo. Don't scope-creep into
growth/morphogenesis. Commit at handoff (standing auth); push only on explicit ask.
