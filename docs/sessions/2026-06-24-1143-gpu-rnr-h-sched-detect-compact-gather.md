# Session — GPU 3D RNR: H→I scheduler + on-GPU detection + Gate D compaction + device gather

## Summary 2026-06-24 11:43 EDT

**Goal:** continue the GPU port of TissueForge's 3D vertex model + RNR. Picked up from the Gate-C
handoff (Gates A,B,C + the H→I detector C0′ done; 81-test gate). Completed FOUR pieces in priority
order: (1) the H→I scheduler, (2) on-GPU detection, (3) Gate D stream-compaction, (4) the device
gather + fully-on-device sweeps. Design study + progress log: `docs/2026-06-24_gpu-3d-vertex-model-
exploration.md` §10. **Full `pixi run test` = 108 passed** (was 81); GPU subset 33 → 60.

**What changed & why (decisions / gotchas / findings):**

- **(1) H→I scheduler — C1′ (host) + C2′ (GPU)** — the reverse mirror of C1/C2a/C2b/C2c.
  - Host `rnr/gpu/schedule_csr.py:228+`: `h_footprint` (the reverse footprint is **9 verts /
    10 surfs / 5 bodies** — UNLIKE the forward one, the triangle + its 3 verts ARE existing
    elements, so they join the footprint; that is exactly what makes a cascade side-collapse
    triangle conflict with its parent cap-cap triangle and serialise correctly), `h_to_i_veto_csr`
    + index helper `faces_share_multiple_edges_csr` (an edge IS a cyclic vertex pair; ≥2 shared ⇒
    veto), `h_independent_set`, `h_reserve_*`, `h_apply_batch`, both reverse sweeps.
  - GPU `rnr/gpu/reconnect_warp.py:430+` `h_to_i_batch_kernel` (the `h_to_i_kernel` body indexed by
    `tid`; births bump `n_used[0]` by 2, no surface alloc) + `schedule_warp.py` `reserve_h_kernel`/
    `check_h_kernel` (`_HFV=9,_HFS=10,_HFB=5`) + `reconnect_sweep_h_to_i_warp`.
  - Gates: `rnr/tests/test_gpu_schedule_h_{csr,warp}.py` (10) — incl. the capstone **full GPU
    round-trip: N parallel I→H then N parallel H→I restore the fingerprint**.

- **(2) On-GPU detection** `rnr/gpu/detect_warp.py` — the O(mesh) per-round scan moved off the host
  Python loop. `scan_small_triangles_kernel` (1 thread/surface) + `scan_short_edges_kernel`
  (1 thread/vertex; each edge emitted by its SMALLER endpoint ⇒ NO cross-thread dedup, only O(k²)
  per-thread dedup + body-count inline, NO scratch arrays). `detect_*_hybrid` = GPU scan + host
  gather on the few candidates; wired into both sweeps via `gpu_scan=False|True`.
  - **Subtlety that mattered:** `find_short_edges_csr` returns sites in Python SET-iteration order,
    but the lowest-id-wins reservation is ORDER-SENSITIVE → sorted GPU scan vs unsorted host scan
    picked different (both valid) winners → divergent fingerprints. Fixed by **canonicalising the
    detection order** (sort by `(v10,v11)` / triangle idx) in ALL FOUR reservation sweeps (2 warp +
    2 host mirrors) — reproducible across host/GPU detection AND keeps the C2c bit-for-bit gate exact.
  - Gates: `rnr/tests/test_gpu_detect_warp.py` (6) — GPU trigger == host trigger exactly; hybrid ==
    `find_*_csr`; **`gpu_scan` sweep == host-scan sweep bit-for-bit**, both directions.

- **(3) Gate D — stream-compaction** (the bump allocator never reclaims, so counters only grew).
  - Host `PaddedMesh.compact()` (`rnr/gpu/device_mesh.py:159+`): in-place, same capacity; renumbers
    live elements into a contiguous prefix (ascending old-slot order); remaps s2v via vmap, v2s/b2s
    via smap, bodies unchanged.
  - Device `rnr/gpu/compact_warp.py`: `wp.utils.array_scan` exclusive prefix-sum (each live slot's
    new index) + scatter kernels into fresh arrays + on-device `n_used`; NO O(mesh) host work; swaps
    arrays into `g` in place.
  - Gates: `rnr/tests/test_gpu_compact.py` (4) — fingerprint preserved, counters drop, idempotent;
    bounded over many passes; device == host slot-for-slot.

- **(4) Device GATHER — "never returns to the host", both directions** `rnr/gpu/gather_warp.py` —
  the hardest kernel: the neighbourhood walk + fused Condition-4 veto on-device.
  - `gather_i_kernel` = device `i_neighbourhood_csr` + `i_to_h_veto_csr` FUSED; `gather_h_kernel` =
    the reverse mirror. **No per-thread scratch** — results write straight to per-candidate output
    rows; set ops are O(k²) over bounded adjacency.
  - Fully-on-device sweeps `reconnect_sweep_warp_device` / `reconnect_sweep_h_to_i_warp_device`
    (`schedule_warp.py`) — scan→gather(+veto)→reserve→apply with NO `from_warp` (only O(cands) data
    leaves the device; caps read from `g` via `reserve_*_independent_set_warp_g`).
  - Gates: `rnr/tests/test_gpu_gather_warp.py` (7) — device gather == host gather+veto per-candidate
    (both directions); **a fully-device sweep round == the host-scan round by fingerprint**, both
    directions. (Round 1 fingerprint-exact; device gather may order arms differently → permuted
    tri-vertex positions, same topology.)
  - **Warp gotcha:** a cross-module `@wp.func` (`d_vert_body_count` from `detect_warp`) MUST be
    imported into the calling module's namespace or the whole module fails to compile with
    "Referencing undefined symbol" — surfaced via `module.load(dev)`, NOT the bare pytest trace.

**Build / test / git state:**
- **`pixi run test` = 108 passed** (~5 min), confirmed green at handoff. GPU subset
  `pytest rnr/tests/test_gpu_*.py -q` → 60 passed (~8s). RTX 5090 / Warp 1.14 / sm_120 / fp64.
- Branch `migrate/linux64-wsl2`. This session's commit is the handoff commit (see below).
- **This session's tracked files:** `docs/2026-06-24_gpu-3d-vertex-model-exploration.md`,
  `rnr/gpu/{device_mesh,reconnect_warp,schedule_csr,schedule_warp}.py` (modified),
  `rnr/gpu/{compact_warp,detect_warp,gather_warp}.py` (new),
  `rnr/tests/{test_gpu_compact,test_gpu_detect_warp,test_gpu_gather_warp,test_gpu_schedule_h_csr,test_gpu_schedule_h_warp}.py` (new),
  + this handoff.
- **Do NOT commit** the `rnr/exports/*` blobs (prior-session figs/videos/CSVs) or the read-only
  oracle repos (`tvm/ 3DVertVor/ tissue-forge/ cellGPU/ VertAX/ gpu_reference_papers/`, own `.git`).

Full `git status --short` (the long `rnr/exports/*` list is ALL prior-session output — leave it):
```
 M docs/2026-06-24_gpu-3d-vertex-model-exploration.md   # THIS session (§10)
 M rnr/exports/{dpmax.json, fig1e_*, fig1f_*}           # prior sessions
 M rnr/gpu/device_mesh.py                               # THIS: PaddedMesh.compact()
 M rnr/gpu/reconnect_warp.py                            # THIS: h_to_i_batch_kernel + apply_h_to_i_batch_warp
 M rnr/gpu/schedule_csr.py                              # THIS: C1' H scheduler + canonical-order sort
 M rnr/gpu/schedule_warp.py                             # THIS: C2' + gpu_scan + device sweeps + reserve_*_g
?? rnr/gpu/compact_warp.py                              # THIS (Gate D device)
?? rnr/gpu/detect_warp.py                               # THIS (on-GPU scans)
?? rnr/gpu/gather_warp.py                               # THIS (device gather, I+H)
?? rnr/tests/test_gpu_compact.py                        # THIS (4)
?? rnr/tests/test_gpu_detect_warp.py                    # THIS (6)
?? rnr/tests/test_gpu_gather_warp.py                    # THIS (7)
?? rnr/tests/test_gpu_schedule_h_csr.py                 # THIS (5)
?? rnr/tests/test_gpu_schedule_h_warp.py                # THIS (5)
?? rnr/exports/{native_*_mixed*.mp4, sort_oracle_M8_*_native[_demixed].csv, vertex_motion_native.gif}  # prior sessions
?? cellGPU/ VertAX/ gpu_reference_papers/               # read-only oracles — DO NOT COMMIT (own .git)
```

## Kickoff — next session

You are continuing a GPU port of TissueForge's 3D vertex model + RNR (Okuda I↔H). Read
`docs/2026-06-24_gpu-3d-vertex-model-exploration.md` (plan + progress §10) first. **DONE & green:
Gates A,B,C (parallel count-changing I↔H), the H→I scheduler (both directions), on-GPU detection,
Gate D stream-compaction (host+device), and the device gather + fully-on-device sweeps (a whole
reconnection round runs with NO `from_warp`, both directions).** Full suite `pixi run test` (108
expected, ~5 min); GPU-only `pixi run python -m pytest rnr/tests/test_gpu_*.py -q` (60, ~8s).
RTX 5090, Warp 1.14 native, fp64 RNR path.

**Only Gate E remains in the GPU-port plan. Priority order:**

1. **Gate E — force / geometry / integration kernels + end-to-end sorting.** Port the Stage-1
   physics to Warp on the CSR/SoA mesh: per-cell volume + per-surface area (centroid-triangulation,
   matching `Surface` convention), the heterotypic tension/adhesion + surface-area-constraint
   gradients, the per-cell `director` active drive (`v0·⟨incident directors⟩`), and the overdamped
   integrator step `x += dt·force/μ` (μ=1). Wire ONE full forward step (forces → integrate →
   `reconnect_sweep_*_warp_device` → `compact_warp`). **Gate:** per-vertex GPU forces match the CPU
   oracle (fp32 tol); end-to-end Fig 1E/1F sorting STATISTICALLY matches the CPU oracle (determinism
   stops mattering here — validate distributions, not bit-equality). The CPU energetics live in the
   `rnr/` native-drive code + the TissueForge actors (VolumeConstraint / SurfaceAreaConstraint /
   Adhesion); re-derive against the SoA, don't transcribe.
2. **Optional perf:** port the trigger scans' atomic-append compaction to a device prefix-sum so
   candidate indices never round-trip (currently an O(cands) readback, NOT O(mesh) — low priority).

**Caveats / guardrails:**
- **License:** reimplement from Okuda 2013 / our own `rnr/` code. **Never copy GPL `tvm/`.**
  `cellGPU/`, `VertAX/`, `gpu_reference_papers/` are read-only oracles — study, don't paste, **don't
  commit** (own `.git`).
- **Round-trip / fingerprint is correctness** for the RNR path; for Gate E switch to STATISTICAL
  validation (forces/distributions), determinism no longer holds once forces move vertices.
- **Precision:** RNR placement stays **fp64** (bit-reversible); fp32 is fine for Gate-E force kernels.
- **Warp:** kernels live in real `.py` files; a literal-init accumulator mutated in a dynamic loop
  needs `wp.int32(...)`; a cross-module `@wp.func` must be IMPORTED into the calling module
  (else "Referencing undefined symbol", seen only via `module.load(dev)`).
- `tf.init()` is one-per-process; tests share the session-scoped `vsolver` fixture (`rnr/tests/conftest.py`).
- Scope: GPU-port phase only — don't scope-creep into growth or the in-engine C++/CUDA fork unless asked.
```
pixi run test                                            # full gate (~5 min)
pixi run python -m pytest rnr/tests/test_gpu_*.py -q     # GPU subset (~8s, kernels cached)
pixi run python -c "import warp as wp; wp.init(); print([d for d in wp.get_devices() if d.is_cuda])"
```
