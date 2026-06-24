# Session — GPU 3D vertex model + RNR: exploration → port kickoff (Gates A, B1)

## Summary 2026-06-24 07:15 EDT

**Goal:** explore options for running the 3D vertex model + reversible network reconnection
(RNR/I↔H) on the GPU; pick the optimal method; start building it.

**New phase.** This is orthogonal to the (done) RNR/sorting science — a GPU port effort. Full
design study + staged plan + live progress log: `docs/2026-06-24_gpu-3d-vertex-model-exploration.md`.

**Findings (4 deep-dives: cellGPU, VertAX, the Chaste/FLAME-GPU paper, TF internals):**
- The literature gap is **3D + topology-change + on-GPU**. cellGPU = 2D (topology on GPU but
  element-count-PRESERVING T1); Chaste+FLAME-GPU = 3D but topology-FREE (overlapping spheres,
  force-only offload, Amdahl-capped ~3.7×); VertAX = 2D JAX, an inverse-design optimizer, not a
  dynamics engine. → filling it is the novel contribution.
- **The make-or-break = parallel, conflict-free, element-count-CHANGING I↔H** on a GPU-resident
  ragged mesh. Reuse cellGPU's conflict scheduler (atomic-reservation → maximal-independent-set →
  iterated batch). What's missing everywhere = **parallel slot allocation/compaction** for the
  created/destroyed verts+faces.

**Decisions (user-chosen):** target the novel result (Stage 2, GPU-native RNR); vehicle =
**Warp → CUDA-in-fork** (prototype cheaply in Warp, port to hand-CUDA in the TF fork later);
**forward-sim only** (no inverse design → no autodiff constraint → mutate in place freely).

**Key design call (gotcha avoided):** births **bump** a high-water counter, deaths just mark
`alive=0`, reclaim via Gate-D compaction — **no concurrent free-list** (a lock-free stack pop is the
one genuinely hazardous GPU primitive). Justified: I↔H nets only ±1 vert/±1 surface and **bodies are
never created/destroyed**. Matches cellGPU's grow-then-compact model.

**Hardware/feasibility (surprise: there's a GPU here):** `nvidia-smi` sees an **RTX 5090, 32 GB,
sm_120 (Blackwell)** under WSL2 (GPU via `/dev/dxg`; absent `/dev/nvidia*` is normal). Warp 1.14.0
initializes it natively (CUDA 12.9), atomic-bump + fp64 kernels run, and Warp **coexists with TF** in
the pixi env (the CUDA-runtime-collision worry was moot — TF is built without CUDA). fp64 verified →
available if fp32 threatens RNR reversibility (a flagged Gate-B3 risk).

**Gotcha resolved (test fix):** `v2s`/`b2s`/the `s2b` body-pair are **unordered** adjacency; only the
`s2v` ring is ordered (winding). Round-trip equality must compare the former as sets — else a revert
that re-appends an incidence in a different slot looks like a mismatch (`test_gpu_device_mesh.py:_conn_equal`).

**Built + green this session:**
- **Gate A** — `rnr/gpu/csr_mesh.py`: TF pointer-graph → index-based CSR/SoA, round-trips exactly
  (minimal [I] config + 91-cell Kelvin block), verifier rejects corruption, bit-exact on the 5090.
  Confirmed 3D raggedness (vertex valence 3–8, face 3–6, cell 7–14 → no fixed stride, CSR mandatory).
  `pixi run gpu-csr`. Test `rnr/tests/test_gpu_csr_roundtrip.py` (4).
- **Gate B1** — `rnr/gpu/device_mesh.py`: padded mutable mesh (fixed-width rows + per-row len +
  capacity) + bump allocator + surgery primitives (`replace_v`/`insert_between`/`drop_v`/`attach_body`/
  `detach_body`, both-sides adjacency, mirror `reconnect.py`). Test `rnr/tests/test_gpu_device_mesh.py`
  (4): CSR↔padded exact, allocator bumps, primitives mutate-then-revert restore, padded SoA on GPU.

**Build/test/git state:**
- pixi: `warp-lang>=1.14` added to `[pypi-dependencies]`; new task `gpu-csr`. `pixi.toml`/`pixi.lock` modified.
- Tests: the **8 new GPU tests pass** (`pytest rnr/tests/test_gpu_*` → 8 green, ~2s). Pre-GPU gate was
  **52 green** at Gate A (`pixi run test`). **Full combined suite (56) not yet re-run** — do it at the
  end of B2/B3 (the ~5-min run wasn't worth repeating mid-increment; new files are isolated).
- Nothing committed (commit only when asked). Branch `migrate/linux64-wsl2`.
- Memory written (outside repo): `gpu-3d-vertex-direction.md`, `dev-gpu-rtx5090-wsl2.md` (+ `MEMORY.md`).
- **Note:** reference repos `cellGPU/`, `VertAX/`, `gpu_reference_papers/` are untracked **read-only
  oracles** — they have their own `.git`; do NOT commit them. The git-tracked new work is `rnr/gpu/`,
  the two `rnr/tests/test_gpu_*.py`, `rnr/scripts/gpu_csr_demo.py`, `docs/2026-06-24_gpu-*.md`, pixi files.

Full `git status --short`:
```
 M docs/BUGS.md
 M docs/renderer_notes.md
 M pixi.lock
 M pixi.toml
 M rnr/exports/dpmax.json
 M rnr/exports/fig1e_demixing_native.csv
 M rnr/exports/fig1e_demixing_native.png
 M rnr/exports/fig1f_stability_native.csv
 M rnr/exports/fig1f_stability_native.png
 M rnr/scripts/fig1e_demixing.py
 M rnr/scripts/video_native_gl.py
?? VertAX/                                    # read-only oracle (own .git) — don't commit
?? cellGPU/                                   # read-only oracle (own .git) — don't commit
?? gpu_reference_papers/                      # PDFs — cell_gpu_chaste, vertax
?? docs/2026-06-23_M8-sweep-findings.md
?? docs/2026-06-24_gpu-3d-vertex-model-exploration.md   # THE design study + plan + progress
?? rnr/clip.py
?? rnr/gpu/                                   # csr_mesh.py, device_mesh.py, __init__.py  (Gate A + B1)
?? rnr/scripts/gpu_csr_demo.py
?? rnr/scripts/shot_native_sort.py
?? rnr/scripts/video_native_cells.py
?? rnr/tests/test_gpu_csr_roundtrip.py
?? rnr/tests/test_gpu_device_mesh.py
?? rnr/exports/native_cells_frames/ , native_cells_frames_clipmz/ , native_gl_frames_clipmz/
?? rnr/exports/native_cells_sort_native_mixed*.mp4 , native_gl_sort_native_mixed*.mp4
?? rnr/exports/vertex_motion_native.gif , unstable_dt0.005/
?? rnr/exports/sort_oracle_M8_*_native[_demixed].csv   (~40 M8-sweep CSVs)
```
(The `rnr/exports/*` and `M8` CSVs + the non-GPU script/doc changes are from prior sessions, not this one.)

## Kickoff — next session

You are continuing a GPU port of TissueForge's 3D vertex model + RNR. Read
`docs/2026-06-24_gpu-3d-vertex-model-exploration.md` (the plan + progress §10) first. Gates A and B1
are DONE and green. Now do **Gate B2** — the count-changing I↔H surgery + its round-trip — which is
THE make-or-break proof. Priority order:

1. **B2 surgery.** Create `rnr/gpu/reconnect_csr.py`:
   - Copy `place_i_to_h` / `place_h_to_i` from `rnr/reconnect.py:103-142` (already pure numpy — Okuda
     Appendix-1 placement; keep as-is).
   - Write `i_to_h_csr(pm, icsr, dl_th)` and `h_to_i_csr(pm, hcsr, dl_th)` as a **direct translation
     of `rnr/reconnect.py:170-300`** onto `PaddedMesh` primitives (`pm.replace_v/insert_between/drop_v/
     attach_body/detach_body/alloc_vertex/alloc_surface/free_vertex/free_surface`). `i_to_h` must
     RETURN the post-state H-neighborhood (in indices) so the round-trip can invert without re-searching.
   - Write a translator `iconfig_to_indices(cfg, vid2i, sid2i, bid2i)` turning a `topo.IConfig` (TF
     handles) into CSR indices, using id→index maps. (Expose those maps from `extract_csr` — currently
     internal; add an optional return or a small `id_maps(bodies)` helper.)
   - Add a **body-anchored, slot-invariant fingerprint** to `csr_mesh.py`: vertex fp = frozenset of
     incident body indices; face fp = (frozenset body indices, multiset of its verts' fps); mesh fp =
     (multiset of vert fps, multiset of face fps). This is the round-trip gate (slots get relabeled, so
     array equality won't work post-surgery).
2. **B2 test** `rnr/tests/test_gpu_reconnect_roundtrip.py`: `helpers.build_minimal_i_config` →
   `topo.i_neighbourhood(v10,v11)` → `extract_csr` + id maps → `PaddedMesh.from_csr` → translate cfg →
   `i_to_h_csr` → `h_to_i_csr` → `to_csr` → assert **fingerprint == original**, the 6 outer verts are
   unmoved (byte-exact), and the recovered edge endpoints are within O(dl_th) of the originals. Mirror
   `rnr/tests/test_roundtrip.py`'s philosophy. **This passing == Gate B's hard part proven.** Also test
   a Kelvin interior edge (`helpers.build_kelvin_block` + `topo.find_short_edges`).
   Run: `pixi run python -m pytest rnr/tests/test_gpu_reconnect_roundtrip.py -v`.
3. **B3 (GPU kernel).** Port the B2 surgery into a Warp kernel (single op, `dim=1`) mutating the
   device arrays (`PaddedMesh.to_warp`); round-trip on the 5090; assert it matches the host reference.
   Settle **fp32 vs fp64**: do placement in fp64 if the fp32 round-trip drifts past tol (fp64 confirmed
   working).
4. Run the **full gate** `pixi run test` (expect 56 green) and update progress §10 in the design doc.
5. Then Gate C (independent-set scheduler — `cellGPU/src/models/vertexModelBase.cu:172` + host loop
   `.cpp:830-895` are the reference), D (compaction), E (force kernels + Fig 1E/1F sorting validation).

**Caveats / guardrails:**
- **License:** reimplement RNR from Okuda 2013 / our own `rnr/reconnect.py`; **never copy GPL `tvm/`**.
  `cellGPU/`, `VertAX/`, `gpu_reference_papers/` are **read-only oracles** (study, don't paste; don't commit).
- **Round-trip reversibility is the definition of correct** — do not move past a red round-trip (same
  discipline as the CPU RNR).
- GPU: RTX 5090 sm_120, Warp 1.14 native. `wp.init()` then check `[d for d in wp.get_devices() if d.is_cuda]`.
- `tf.init()` is a one-per-process singleton; tests share the session-scoped `vsolver` fixture
  (`rnr/tests/conftest.py`) and scope to a `bodies` list.
