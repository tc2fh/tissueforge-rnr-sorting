# Mesh scaling: foam-build O(N²)→O(N), GPU headroom, and the build/viewer architecture (options A & B)

## Summary 2026-06-25 12:58 EDT

Goal this session: started as GPU-perf polish, then pivoted (user's steer) to **mesh scaling** — "how
large a tissue can we simulate on GPU", and how to keep the **TF viewer + Python bindings** while scaling.
Three commits landed on `migrate/linux64-wsl2` (NOT pushed):

- `19560e1` **gather buffer reuse** (~6.6%, bit-identical) — details in the prior handoff
  `docs/sessions/2026-06-25-1141-gather-buffer-reuse-and-skip-scan-deadend.md` (also covers the DEAD
  skip-scan lever).
- `83fa24d` **native periodic Voronoi foam build** (O(N²)→~O(N), ~500×) — this session's main win.

**What changed and why:**
- **Build was O(N²) and it was pyvoro, not TF.** cProfile of the n=8 build: `pyvoro.compute_voronoi`
  = 28.4 s of 29.5 s (**96%**). The old `build_periodic_voronoi` (`rnr/geometry.py`) 3×3×3-ghost-tiled
  every seed (27× the points) AND ran voro++ in a SINGLE brute-force block (`dispersion` = whole box) →
  O(N²). Fix: voro++'s **native periodic mode** (`periodic=[True]*3`) on the N seeds directly, proper
  spatial blocking. Verified cell-adjacency topology is **bit-for-bit identical** to the ghost build
  (n=3..8) and the foam is space-filling (Σvol/box = 1.00000), per-cell volume multiset matching the
  cached n=10 foam to **1e-8**. **Voronoi ~500× faster** (n=8: 27 s → 0.05 s).
- **REBUILD caveat (important):** a rebuilt foam is equivalent-but-RELABELED (different vertex IDs +
  ~1e-6 voro++ fp) vs the old ghost build, so the engine trajectory differs under `--rebuild-foam`. The
  cache FORMAT was intentionally NOT bumped → all existing `rnr/exports/foam_cache/*.npz` stay valid and
  the committed **`4010/3028` bit-identical reference holds for the cached foam** (just not under rebuild).
- **The new build bottleneck at large N is TF object creation** (O(N²), suspect `become()` +
  `position_changed()` — the build creates every body as type-A then `become()`s half). Full build incl.
  TF: n=10/2k cells **2 s**, n=16/8k **14 s**, n=20/16k **66 s** (Voronoi is only ~0.7 s of that at n=20).

**Scaling + GPU-usage facts measured (reference for next session):**
- 100k-step run at N=2000 (n=10): **224.9 s, 2.25 ms/step**, STABLE. (Per-step drops from the 20k avg of
  2.64 as the sort quiesces.) Pre-session was 9.09 ms/step (~15 min for 100k).
- **GPU is latency/launch-bound, hugely under-utilized:** at N=2000, **sm ~43%**, mem-BW **0%**, VRAM
  **5.24 GB / 32 GB (16%)**, power ~125 W/600 W. At N=8192 (n=16): engine **~11.8 ms/step** STABLE, sm
  **~33%**, VRAM **5.37 GB (barely more!)**. 4× the cells → +0.12 GB VRAM. **VRAM alone could hold
  ~50–100k cells**; bigger meshes will finally lift utilization off ~33%.
- So the immediate scale ceiling is the **build** (now fixed to ~O(N) Voronoi, TF-create the remainder),
  not the GPU. Per-step compute is the eventual ceiling for *long* runs at very large N.

**Build / test / git state:**
- Branch `migrate/linux64-wsl2`, **ahead of origin by 6, NOT pushed** (push needs an explicit ask).
- **Gate `pixi run test` = 132 passed** with the new builder (full run, 5:34). Foams space-filling +
  adjacency-consistent at n=3..20.
- All tracked code committed. `git status --short` = only `rnr/exports/*` (regenerable: figs/CSVs/mp4s +
  the new `foam_cache/foam_n16_mixed_*.npz` from this session's scaling run) — **intentionally
  uncommitted, LEAVE**. Nothing else outstanding.
- Memory updated: `reconnect-sweep-scan-bottleneck` (gather reuse + skip-scan dead). NEW memory worth
  adding next time: the foam-build O(N²)→O(N) finding (not yet written).

## Build/viewer architecture — options A & B (the decision to carry forward)

**Project goal (clarified by the user this session):** convenient, efficient **3D vertex modeling via
Python bindings**, with the **TF viewer available** for building/iterating models, on top of an efficient
**C++/CUDA backend of TissueForge**. The Warp/Python GPU port (`rnr/gpu/`) is the **algorithm
proving-ground** for the eventual native C++/CUDA TF backend (CLAUDE.md's "harden into a native C++
`MeshQuality` later"). In that END state TF builds the mesh natively and the CUDA backend runs it —
viewer + bindings intact, no "TF-free" anything.

**Key insight — TF-free build does NOT cost the viewer or bindings.** The **CSR is the interchange
format** and TF↔CSR is bidirectional: we already have **TF→CSR** (`rnr/gpu/csr_mesh.py::extract_csr`);
the only missing half is **CSR→TF** (materialize TF `Vertex`/`Surface`/`Body` from a CSR — the inverse,
reusing the same TF-object-creation calls the current builder makes). With that bridge, "how the mesh is
built" decouples from "how it's viewed", and you can drop ANY CSR (initial foam, GPU-evolved state, a
downsampled slab) into the TF viewer on demand. Note: even today the viewer only sees the INITIAL foam —
once the GPU sim reconnects, the evolved mesh lives only in Warp, so CSR→TF is wanted regardless.

**Option A (NEXT — keeps the viewer):** optimize the EXISTING TF build to ~O(N) while still creating TF
objects (so the viewer + full TF Python API keep working). Remaining super-linearity (Voronoi now fixed)
is TF per-element ops; prime suspect `become()` (create-as-A-then-retype-half = per-body TF work × N).
Likely gets to tens-of-k cells with the viewer intact. Cheap, low-risk, preserves the modeling workflow.

**Option B (LATER — for 50–100k+):** a TF-free direct-CSR builder (assemble the CSR straight from the
Voronoi output in numpy, no TF objects) → truly O(N) → unlocks 50–100k+ cells; pair with the **CSR→TF
bridge** so the viewer is materialized on demand. Bigger change; do only when A's ceiling is hit.

**Recommendation (agreed):** do A first (preserves viewer/bindings, enough for interactive modeling),
add B only to push past A's ceiling, always paired with the CSR→TF bridge.

## Kickoff — next session (PLAN FOR OPTION A)

You are continuing a Warp/CUDA GPU port of TissueForge's 3D vertex model + RNR (RTX 5090). The engine
runs faithful paper-scale sorts (N=2000, ~2.25 ms/step) and the foam build was just made ~O(N) in the
Voronoi step (`83fa24d`). The user wants to **scale the mesh up** (toward 50k+ cells) while keeping the
**TF viewer + Python bindings** usable for interactive modeling. Read the "options A & B" section above.

**Your task: OPTION A — make the existing TF foam build ~O(N) without losing TF objects/viewer.**

Priority order:
1. **Pin the O(N²).** cProfile the full build at n=16 and n=20 (the Voronoi is now ~O(N), so the
   super-linearity is downstream). Use the pattern from this session:
   `scratchpad/prof_fullbuild.py` (cProfile `_build_unit_foam_host`) — re-point it at n=16/20 and look at
   cumulative time in `become()`, `position_changed()`, TF `Vertex/Surface/Body` creation, `extract_csr`,
   `PaddedMesh.from_csr`, `compute_geometry`. (At n=8 with the OLD pyvoro, TF-create was only 0.14 s, but
   that was a small mesh + before the Voronoi fix exposed the next wall — re-measure at n=16/20.)
2. **Kill the suspected `become()` cost:** in `rnr/tests/test_gpu_physics_csr.py::_build_two_type_foam`,
   bodies are created as `btA` then `b.become(btB)` for half (mixed) or by z-split (demixed). If `become()`
   is O(mesh) per call → O(N²). Create each body with its FINAL type up front instead (pass the right
   `btype` to `build_periodic_voronoi`, or split the body list before creation). Check whether
   `position_changed()` (called once) is also super-linear inside TF — if so it may be unavoidable from
   Python (a point in favor of B / the eventual C++ build).
3. **Keep TF objects** so the viewer path is preserved — A's whole point is NOT to drop TF.
4. **Validate:** `pixi run test` (132 expected); re-run the scaling table
   (`scratchpad/validate_build.py`: build n=3..20, assert Σvol/box=1, volmin>0, adjacency consistent) and
   confirm the build is now ~linear in N (n=20 should drop well below 66 s). Foams must stay space-filling.

**Then (optional, only if pushing past A's ceiling): OPTION B** — TF-free direct-CSR builder + a new
`csr_mesh.py::build_tf_from_csr` (CSR→TF) for on-demand viewing. Validate the CSR→TF round-trips
(`extract_csr(build_tf_from_csr(csr)) == csr` up to relabeling).

**Secondary (deprioritized) GPU per-step perf levers** (from the 1141 handoff, still valid): force kernel
(~0.55 ms, O(valence²) active-drive dedup, bit-identical-able), reserve owner-array reuse (~1.5%), compact
throttle (faithfulness tradeoff). Lower priority than scaling now.

**Validation commands / caveats:**
```
pixi run test                                                          # 132 expected
pixi run gpu-stability --n 16 --steps 3000 --dt 0.01 --ic mixed       # large-foam smoke (builds + caches n=16)
pixi run gpu-stability --n 10 --steps 20000 --dt 0.01 --ic mixed --csv /tmp/x.csv
#   the GATHER-reuse bit-identical check: het@10k=0.46382583300146024, recon@20k=4010/3028 (CACHED foam only)
```
- Do NOT `--rebuild-foam` the cached n=10 expecting 4010/3028 — the new builder relabels (see caveat).
- The build is cached (`rnr/exports/foam_cache/*.npz`, gitignored); a new n needs one ~build then loads.

**Scope + license guardrails:** GPU-port phase only; reimplement from Okuda 2013 / our `rnr/`, **never copy
GPL `tvm/`**; `tvm/ 3DVertVor/ tissue-forge/ cellGPU/ VertAX/ gpu_reference_papers/` are read-only oracles
(own `.git`, never commit). Don't scope-creep into growth/morphogenesis. Commit at handoff (standing auth);
push only on explicit ask.
