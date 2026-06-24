# Session — GPU 3D RNR: Gate B (make-or-break) + Gate C (the novel result) DONE

## Summary 2026-06-24 08:59 EDT

**Goal:** continue the GPU port of TissueForge's 3D vertex model + RNR. Picked up from the
Gate-A/B1 handoff; built Gate B (single count-changing I↔H round-trip, host + GPU) and Gate C
(the cellGPU independent-set protocol in 3D, on the GPU) — **the novel contribution**: parallel,
conflict-free, element-count-CHANGING I→H on a GPU-resident ragged 3D mesh.

Design study + full progress log: `docs/2026-06-24_gpu-3d-vertex-model-exploration.md` §10.
Every brick gated by a round-trip / equivalence test, mirroring the CPU-RNR methodology.

**What changed & why (decisions / gotchas / surprises):**
- **Placement = ONE formula, no CPU/GPU drift.** Refactored `rnr/reconnect.py:103` placement into
  pure position-array cores `place_i_to_h_xyz`/`place_h_to_i_xyz` (the only edit to the tracked CPU
  oracle); the cfg-handle wrappers delegate. GPU + CPU reuse the identical Okuda math. CPU
  round-trip still green (no behaviour change).
- **The round-trip gate post-surgery is a FINGERPRINT, not array equality.** Alloc/free/compact
  relabel vertex/surface slots, so only a body-anchored slot-invariant fingerprint
  (`rnr/gpu/csr_mesh.py:fingerprint`) detects topology restoration. Bodies never change index
  (extract_csr + to_csr both order by ascending TF id) → the stable anchor. `id_maps` added to
  bridge the TF topology walk into the index world.
- **B3 precision (risk #2 SETTLED with data):** on-device **fp64** placement == numpy oracle to
  **0.0** (bit-exact → fully reversible); **fp32 drifts 1.78e-7** (fine for the dl_th budget, NOT
  bit-reversible). → fp64 is the RNR-path precision. Integer surgery is precision-independent →
  device matches host **bit-for-bit**.
- **Warp gotcha:** a literal-init accumulator mutated inside a *dynamic* loop must be declared typed
  (`idx = wp.int32(-1)`), else "Error mutating a constant". Also: Warp can't introspect source from
  `exec()`/heredoc — kernels must live in real files.
- **C1 dynamics FINDING:** a *static-mesh* sweep does NOT converge — one I→H places triangle verts
  that themselves form new short [I] edges (measured 1→3), so reconnections cascade. In production,
  force relaxation between steps grows them back. So the Gate-C gate is **"N non-conflicting I↔H in
  one parallel batch"** (order-independent), not sweep-to-exhaustion. The cascade is a dynamics
  property, not a scheduler bug.
- **Parallel-safety proof = order-independence.** An independent batch applied in ANY order yields
  the same fingerprint (proven at scale: a ~10-reconnection Kelvin batch, fwd==rev). This is the
  property the GPU's nondeterministic atomic scheduling relies on.
- **C2a reservation is lowest-id-wins** (`atomic_min`), deterministic → matches the host reference
  bit-for-bit. Conflict-free by construction; one round is NOT maximal (needs the iterate loop).
- **C2b parallel apply is race-free** because the reservation guarantees disjoint footprints
  (no shared existing vert/surf/cap) and the shared `atomic_add` bump gives each thread distinct
  fresh slots. Validated: parallel GPU apply == host sequential apply (fingerprint).
- **Bump allocator never reclaims** (+3 verts / +1 surface per I→H, deaths free-marked) → long runs
  need **Gate D** stream-compaction. Host-ref tests size headroom to the workload as a stand-in.

**Built this session (all green):**
- `rnr/gpu/reconnect_csr.py` (B2 host surgery) + `rnr/tests/test_gpu_reconnect_roundtrip.py` (3).
- `rnr/gpu/reconnect_warp.py` (B3 Warp kernels + C2b batch kernel) + `test_gpu_reconnect_warp.py` (3).
- `rnr/gpu/topology_csr.py` (C0 index detector) + `test_gpu_topology_csr.py` (3).
- `rnr/gpu/schedule_csr.py` (C1 host scheduler) + `test_gpu_schedule_csr.py` (5).
- `rnr/gpu/schedule_warp.py` (C2a GPU reservation) + `test_gpu_schedule_warp.py` (5).
- Additions to `csr_mesh.py` (`id_maps`, `fingerprint`) and `device_mesh.py`
  (`set_ring`, `ring_neighbors`, `from_warp`).

**Build / test / git state:**
- **Full gate green: `pixi run test` → 75 passed (5:02).** GPU subset: `pytest rnr/tests/test_gpu_*`
  → 27 passed (~2s, kernels cached). RTX 5090 / Warp 1.14 / sm_120 / fp64 confirmed.
- Pre-session gate was 56; +3 B2 +3 B3 +3 C0 +5 C1 +5 C2 GPU tests = 75.
- **Nothing committed** (commit only when asked). Branch `migrate/linux64-wsl2`.
- **Do NOT commit** reference repos `cellGPU/`, `VertAX/`, `gpu_reference_papers/` (own `.git`,
  read-only oracles). Git-tracked new work = `rnr/gpu/`, the 6 `test_gpu_*.py`, the modified
  `rnr/reconnect.py`, `docs/2026-06-24_gpu-*.md`, this handoff, pixi files.

Full `git status --short`:
```
 M docs/BUGS.md
 M docs/renderer_notes.md
 M pixi.lock
 M pixi.toml
 M progress.md
 M rnr/reconnect.py                                   # placement refactored to *_xyz cores (this session)
 M rnr/scripts/fig1e_demixing.py
 M rnr/scripts/video_native_gl.py
 M rnr/exports/{dpmax.json, fig1e_*, fig1f_*}         # prior sessions
?? VertAX/  cellGPU/  gpu_reference_papers/           # read-only oracles — DO NOT COMMIT (own .git)
?? docs/2026-06-23_M8-sweep-findings.md
?? docs/2026-06-24_gpu-3d-vertex-model-exploration.md # THE design study + progress §10
?? docs/sessions/2026-06-24-0715-gpu-3d-vertex-port.md
?? docs/sessions/2026-06-24-0859-gpu-rnr-gate-b-and-c.md   # this handoff
?? rnr/gpu/                                           # csr_mesh, device_mesh, reconnect_csr,
                                                      #   reconnect_warp, topology_csr, schedule_csr,
                                                      #   schedule_warp, __init__  (Gates A,B,C)
?? rnr/tests/test_gpu_{csr_roundtrip,device_mesh,reconnect_roundtrip,reconnect_warp,topology_csr,schedule_csr,schedule_warp}.py
?? rnr/clip.py  rnr/scripts/{gpu_csr_demo,shot_native_sort,video_native_cells}.py
?? rnr/exports/{native_*_frames*/, native_*_mixed*.mp4, vertex_motion_native.gif, unstable_dt0.005/, sort_oracle_M8_*_native[_demixed].csv}
```
(The `rnr/exports/*`, `rnr/clip.py`, video scripts, and non-GPU doc/script changes are from prior
sessions, not this one.)

## Kickoff — next session

You are continuing a GPU port of TissueForge's 3D vertex model + RNR (Okuda I↔H). Read
`docs/2026-06-24_gpu-3d-vertex-model-exploration.md` (plan + progress §10) first. **Gates A, B
(B1/B2/B3), and C (C0/C1/C2a/C2b) are DONE and green** — the make-or-break parallel, conflict-free,
count-changing I→H runs on the GPU (RTX 5090) and matches the host reference. Full suite:
`pixi run test` → 75 passed.

Priority order:

1. **Finish wiring Gate C into one GPU iterated sweep** (mostly glue): a `reconnect_sweep_warp(g,
   threshold, dl_th)` that loops {detect → reserve (C2a) → parallel apply (C2b) → re-detect} on a
   device flag. Detection is currently host-side (`topology_csr.find_short_edges_csr`); a first
   version can keep detection on host and only the reserve+apply on GPU. Add a test that one GPU
   round == the host `schedule_csr.reconnect_sweep_i_to_h` one-round result (fingerprint). NOTE the
   C1 cascade finding: a static-mesh sweep won't converge — bound rounds / expect dynamics.
2. **H→I reverse direction in the scheduler** (forward I→H is done): mirror C0/C2 — a
   `find_small_triangles_csr` detector (mirror `topology.find_small_triangles` / `h_neighbourhood`),
   `h_to_i` footprint + reservation, and the parallel `h_to_i_batch_kernel` (mirror
   `i_to_h_batch_kernel` in `reconnect_warp.py`). Round-trip gate as always.
3. **Gate D — stream-compaction** of dead vertex/surface slots (the bump allocator's +3 verts/op
   makes arrays grow unboundedly; compaction reclaims `alive==0` slots, like cellGPU's grow-list).
   Gate: arrays stay bounded over many reconnection passes; fingerprint preserved across a compact.
4. **On-GPU detection** (optional perf): port `find_short_edges_csr` to a parallel scan kernel so a
   full step never returns to the host.
5. **Gate E — force/geometry/integration kernels** (volume grad, surface-area grad, tension,
   overdamped Euler) + end-to-end Fig 1E/1F sorting statistically matching the CPU oracle. This is
   where determinism stops mattering (validate distributions, not bit-equality).

Commands:
- Full gate: `pixi run test` (75; ~5 min). GPU-only: `pixi run python -m pytest rnr/tests/test_gpu_*.py -q` (~2s).
- One test: `pixi run python -m pytest rnr/tests/test_gpu_schedule_warp.py -v -s`.
- Warp check: `pixi run python -c "import warp as wp; wp.init(); print([d for d in wp.get_devices() if d.is_cuda])"`.

Caveats / guardrails:
- **License:** reimplement RNR from Okuda 2013 / our own `rnr/reconnect.py`. **Never copy GPL
  `tvm/`.** `cellGPU/`, `VertAX/`, `gpu_reference_papers/` are read-only oracles — study, don't paste,
  **don't commit** (they carry their own `.git`).
- **Round-trip / fingerprint is the definition of correct** — never move past a red round-trip.
  The body-anchored `csr_mesh.fingerprint` is the post-surgery invariant (slots get relabelled).
- **Precision:** keep RNR placement in **fp64** (bit-reversible; fp32 drifts ~1e-7). fp32 is fine
  later for the Gate-E force kernels (no reversibility at stake).
- **GPU:** RTX 5090 sm_120, Warp 1.14 native. Warp kernels must live in real `.py` files (no
  heredoc). A literal-init accumulator mutated in a dynamic loop needs `wp.int32(-1)`.
- `tf.init()` is one-per-process; tests share the session-scoped `vsolver` fixture
  (`rnr/tests/conftest.py`) and scope to a `bodies` list.
- Scope: this is the GPU-port phase (orthogonal to the done RNR/sorting science). Don't scope-creep
  into growth or the in-engine C++/CUDA fork unless asked.
