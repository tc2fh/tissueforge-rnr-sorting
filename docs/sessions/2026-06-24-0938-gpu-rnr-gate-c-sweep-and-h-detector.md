# Session — GPU 3D RNR: Gate-C iterated sweep (priority 1 DONE) + H→I detector started

## Summary 2026-06-24 09:38 EDT

**Goal:** continue the GPU port of TissueForge's 3D vertex model + RNR. Picked up from the
Gate-B/C handoff (Gates A,B,C2a/b green, 75-test gate). Did **priority 1** in full (wire Gate C
into one GPU iterated sweep) and started **priority 2** (the H→I reverse direction) with its
detector brick. Design study + progress log: `docs/2026-06-24_gpu-3d-vertex-model-exploration.md` §10.

**What changed & why (decisions / gotchas / findings):**

- **C2c — the GPU iterated sweep, glued (priority 1, DONE).**
  `schedule_warp.reconnect_sweep_warp(g, threshold, dl_th)` (`rnr/gpu/schedule_warp.py:131`) runs
  the cellGPU iterated-batch loop end-to-end on the device: each round `PaddedMesh.from_warp(g)`
  (slot-preserving) → host detect (`find_short_edges_csr` + Cond-4 veto) → GPU reserve (C2a) → GPU
  parallel apply (C2b, mutates `g` in place) → re-detect, bounded by `max_rounds`. Host mirror added:
  `schedule_csr.reconnect_sweep_reserve_host` + `reserve_independent_set_host`
  (`rnr/gpu/schedule_csr.py:124,253`).
  - **The subtlety that mattered (correctness, not style):** the GPU sweep selects each round via
    the C2a **reservation** (lowest-id-wins, ONE round — conflict-free but NOT maximal). The host
    mirror must use the SAME reservation selection — **NOT** the existing `reconnect_sweep_i_to_h`,
    which uses **greedy maximal** `independent_set`. Greedy keeps a candidate if disjoint from the
    WINNERS so far; one reservation round keeps it only if disjoint from ALL lower-id candidates
    (winners *and* losers) — strictly more restrictive. Measured on the n=4 Kelvin block:
    **360 candidates → greedy 10, one reservation round 1.** Gating the GPU sweep against the greedy
    host sweep (as the prior handoff literally suggested) would have compared 1 reconnection vs 10
    and **failed**. Fixed by adding the reservation-based host mirror and gating against it.
  - **Equivalence proof structure:** round 1 starts from the host's slot layout, so detect+reserve
    are identical (C2a is bit-for-bit) and the parallel apply matches host-sequential by fingerprint
    (C2b) — so ONE round is exact. Across rounds the device's atomic-bump slot order diverges from
    the host's sequential order (same topology, different slot labels), so only round 1 is asserted
    fingerprint-equal; later rounds are validated for consistency (mirrors the C1 host sweep).
  - **Efficiency finding (NOT correctness):** deterministic lowest-id-wins is very *non-maximal* on
    a dense candidate set (1/360) — ≈one low-id "seed" per round, near-serial. This is a worst case
    of the static Kelvin block (every edge a candidate). In production only a few edges fall below
    threshold per step (sparse, mostly disjoint) so one round admits most. If dense batches ever
    matter, cellGPU's **randomised per-round priorities** give a near-maximal set in O(log n) rounds
    — but trade the bit-for-bit host match for a statistical one. Deferred.
  - Gate `rnr/tests/test_gpu_sweep.py` (3): one GPU round == one host reservation round (fingerprint);
    bounded 3-round device sweep re-detects on the mutated mesh + stays consistent; sub-threshold = 0-round no-op.

- **C0′ — the reverse-direction [H] detector (priority 2, brick 1, DONE).**
  `topology_csr.h_neighbourhood_csr` + `find_small_triangles_csr` (`rnr/gpu/topology_csr.py:165,247`)
  — index-world mirror of `topology.h_neighbourhood`/`find_small_triangles` (no TF handles), emitting
  the same `HCfgIdx` that `h_to_i_csr` consumes. Condition-2 triggers on the **MAX** triangle edge
  (not min — Honda's wrong "condition H"). Gate `rnr/tests/test_gpu_topology_h_csr.py` (3).
  - **Reverse-direction cascade FINDING:** an I→H can collapse a *quad* side-face
    `[outer_top, v10, v11, outer_bot]` into a triangle `[outer_top, tri_k, outer_bot]` — a genuine,
    immediately-reverse-reconnectable [H] site. So one I→H yields the cap-cap triangle **plus** ≥0
    side-collapse triangles (measured 1 extra per op on one Kelvin block ⇒ detector finds **2N**, not
    N — confirmed via diagnostic: surf 510 `[368,369,370]` all-new = cap-cap, surf 12 `[21,18,369]`
    2-old-1-new = side-collapse). They **share** the new tri vertex with their cap-cap triangle
    (overlapping footprints) ⇒ reversing ALL detected triangles double-touches it. The gate reverses
    only the cap-cap sites (disjoint across an independent batch); reversing a cap-cap re-expands its
    collapsed side-faces back to quads, restoring the neighbourhood. H→I analogue of the forward C1
    cascade; production force-relaxation separates the scales. This is why the H-footprint (next) MUST
    include all 9 verts / 10 surfs / 5 bodies so the reverse scheduler treats these as conflicts.

**Build / test / git state:**
- `pixi run test` ran **green at 78** mid-session (priority 1: 75 → +3 `test_gpu_sweep`). The
  `test_gpu_topology_h_csr` (+3) was added *after* that run, so a fresh full gate is **81**; those 3
  pass in the GPU subset. **GPU subset `pytest rnr/tests/test_gpu_*.py -q` → 33 passed (~2.6s).**
  RTX 5090 / Warp 1.14 / sm_120 / fp64 confirmed.
- **Nothing committed** (project practice — the prior handoff did the same; commit only when asked). Branch `migrate/linux64-wsl2`.
- **This session's tracked changes:** `rnr/gpu/{schedule_csr,schedule_warp,topology_csr}.py`,
  `docs/2026-06-24_gpu-3d-vertex-model-exploration.md`, `rnr/tests/{test_gpu_sweep,test_gpu_topology_h_csr}.py`.
- **Do NOT commit** reference repos `cellGPU/`, `VertAX/`, `gpu_reference_papers/` (own `.git`,
  read-only oracles). The many `rnr/exports/*` + `.claude/skills/handoff/SKILL.md` in `git status` are
  from PRIOR sessions, not this one — leave them.

Full `git status --short` (abbreviated — the long `rnr/exports/*` list is prior-session output):
```
 M .claude/skills/handoff/SKILL.md                    # prior session
 M docs/2026-06-24_gpu-3d-vertex-model-exploration.md # THIS session (§10 C2c + C0′)
 M rnr/exports/{dpmax.json, fig1e_*, fig1f_*}         # prior sessions
 M rnr/gpu/schedule_csr.py                            # THIS: reserve_independent_set_host + reconnect_sweep_reserve_host
 M rnr/gpu/schedule_warp.py                           # THIS: reconnect_sweep_warp + imports
 M rnr/gpu/topology_csr.py                            # THIS: h_neighbourhood_csr + find_small_triangles_csr + _tri_edges
?? rnr/tests/test_gpu_sweep.py                        # THIS (3)
?? rnr/tests/test_gpu_topology_h_csr.py               # THIS (3)
?? rnr/exports/{native_*_mixed*.mp4, sort_oracle_M8_*_native[_demixed].csv, vertex_motion_native.gif}  # prior sessions
?? cellGPU/ VertAX/ gpu_reference_papers/             # read-only oracles — DO NOT COMMIT (own .git)
```

## Kickoff — next session

You are continuing a GPU port of TissueForge's 3D vertex model + RNR (Okuda I↔H). Read
`docs/2026-06-24_gpu-3d-vertex-model-exploration.md` (plan + progress §10) first. **Gates A, B, and C
(incl. C2c — the GPU iterated I→H sweep) are DONE and green; the H→I reverse detector (C0′) is DONE.**
Full suite: `pixi run test` (81 expected); GPU-only: `pixi run python -m pytest rnr/tests/test_gpu_*.py -q`
(33, ~3s). RTX 5090, Warp 1.14 native.

Priority order:

1. **Finish the H→I scheduler** (the reverse mirror of C1/C2a/C2b — fully scoped below):
   - **Host (C1′)** in `rnr/gpu/schedule_csr.py`, mirroring the I-side functions:
     `h_footprint(HCfgIdx)` → (verts, surfs, bodies) = **9 verts** (3 tri + 6 outer), **10 surfs**
     (the triangle + 3 side + 3 top + 3 bottom), **5 bodies** (2 caps + 3 side). [This footprint makes
     the cascade side-collapse triangle conflict with its cap-cap triangle — they share the new tri
     vertex — so the reverse scheduler serialises them correctly.] Then `h_independent_set`,
     `h_reserve_won_mask_host`, `h_reserve_independent_set_host`, `h_batch_is_conflict_free`,
     `h_apply_batch` (uses `h_to_i_csr`), `reconnect_sweep_h_to_i`, and `h_to_i_veto_csr` (mirror
     `conditions.h_to_i_veto` on indices: caps share ≥2 faces; any side-cell pair shares ≥2 faces;
     any side-face pair shares ≥2 edges — the last needs a `faces_share_multiple_edges` index helper:
     two faces share an edge = share 2 cyclically-consecutive verts).
   - **GPU (C2′)** in `rnr/gpu/schedule_warp.py` + `reconnect_warp.py`: `pack_h_footprints` +
     `reserve_h_kernel`/`check_h_kernel` (copy the C2a kernels with `_FV=9,_FS=10,_FB=5`), wrappers,
     and `h_to_i_batch_kernel` (mirror `i_to_h_batch_kernel` at `reconnect_warp.py:431` — index the
     existing `h_to_i_kernel` body per-candidate by `tid`; births bump `n_used` by 2). Then optionally
     a `reconnect_sweep_h_to_i_warp` glued like C2c.
   - **Gates** (`rnr/tests/test_gpu_schedule_h_*.py`): H reservation == host bit-for-bit; parallel
     `h_to_i` apply == host sequential (body-anchored fingerprint, `csr_mesh.fingerprint`); a forward
     i_to_h batch then a reverse h_to_i batch restores the original fingerprint. **Headroom note:** the
     bump allocator never reclaims — size `v_headroom ≥ 5·N` for an I→H(+3)-then-H→I(+2) round-trip.
2. **On-GPU detection** (perf): port `find_short_edges_csr` / `find_small_triangles_csr` to parallel
   scan kernels so a sweep never returns to the host.
3. **Gate D — stream-compaction** of dead vertex/surface slots (bump allocator's +3 verts/op grows
   arrays unboundedly; reclaim `alive==0` slots, like cellGPU's grow-list). Gate: arrays stay bounded
   over many passes; fingerprint preserved across a compact.
4. **Gate E** — force/geometry/integration kernels + end-to-end Fig 1E/1F sorting statistically
   matching the CPU oracle (determinism stops mattering here — validate distributions).

Caveats / guardrails:
- **License:** reimplement RNR from Okuda 2013 / our own `rnr/reconnect.py`,`rnr/topology.py`,
  `rnr/conditions.py`. **Never copy GPL `tvm/`.** `cellGPU/`, `VertAX/`, `gpu_reference_papers/` are
  read-only oracles — study, don't paste, **don't commit** (own `.git`).
- **Round-trip / fingerprint is the definition of correct** — never move past a red round-trip. The
  body-anchored `csr_mesh.fingerprint` is the post-surgery invariant (slots get relabelled).
- **Precision:** keep RNR placement **fp64** (bit-reversible; fp32 drifts ~1e-7). fp32 is fine later
  for Gate-E force kernels.
- **GPU:** Warp kernels must live in real `.py` files (no heredoc). A literal-init accumulator mutated
  in a dynamic loop needs `wp.int32(-1)`. Per-round reservation is lowest-id-wins (deterministic →
  host-matchable) but very non-maximal on dense sets — see the C2c efficiency finding.
- `tf.init()` is one-per-process; tests share the session-scoped `vsolver` fixture (`rnr/tests/conftest.py`).
- Scope: this is the GPU-port phase (orthogonal to the done RNR/sorting science). Don't scope-creep
  into growth or the in-engine C++/CUDA fork unless asked.
```
pixi run test                                            # full gate (~5 min)
pixi run python -m pytest rnr/tests/test_gpu_*.py -q     # GPU subset (~3s, kernels cached)
pixi run python -c "import warp as wp; wp.init(); print([d for d in wp.get_devices() if d.is_cuda])"
```
