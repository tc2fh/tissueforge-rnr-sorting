---
name: periodic-geometry-verified
description: "2026-06-01 periodic vertex-mesh geometry slice (mesh.periodic_geometry flag) verified correct — energy, force, RNR edge, native I↔H round-trip all min-image, AND (P4) a periodic space-filling Kelvin packing generator (no free surface, adjacency wraps); P0–P4 done, 39 green; periodicity via geometry not coordinate wrapping; foam box MUST equal universe box"
metadata: 
  node_type: memory
  type: project
  originSessionId: 8ef79112-0e04-4319-9bca-5f565d8ffaab
---

Codex's periodic vertex-mesh geometry slice on `feat/native-rnr-reconnection` (the
`mesh.periodic_geometry` flag + `tf_mesh_metrics.{h,cpp}` minimum-image helpers routed through
Surface/Body/actor/MeshQuality) was REVIEWED and VERIFIED correct on 2026-06-01.

**Verified (gate `rnr/tests/test_periodic_geometry.py`, 3 tests, in `pixi run test` → 36 green):**
1. area/volume of a boundary-straddling unit cube = interior cube (6, 1); flag OFF = box-spanning
   (238, 59) so the flag is load-bearing.
2. `VolumeConstraint` + `SurfaceAreaConstraint` FORCES on straddling vs interior cube match to
   float32 (Δ: volume 1.65e-6 = pure `L−x` roundoff, area 0.0). So gradients are periodic-correct,
   not just energies. body-`Adhesion` shares SurfaceAreaConstraint's identical area-gradient loop →
   covered by equivalence (not separately tested; would need a 2-type straddling interface).
3. native `analyze_i_reconnection` edge-length on an [I] config straddling z=0 = 0.5 (min-image),
   not ~box-0.5; still valid+legal.
4. (P3 gate, added 2026-06-01) `test_native_roundtrip.py::test_native_periodic_roundtrip_across_boundary`
   — with the flag ON, the native MUTATE path `force_reconnect_i_to_h`→`force_reconnect_h_to_i` on a
   straddling [I] config returns counts+adjacency to the original [I], keeps bodies positive AND
   min-image-small (`<100`, box-spanning would be huge), and recovers the short edge to a min-image
   drift `<1e-4`. So the periodic placement (`rnr_placeIToH`/`rnr_placeHToI`, unwrap-local + wrap-new)
   is now proven across the wall, not just the edge-length trigger. Test-only addition (no engine
   change — mutate path was already wired in commit 13edb6f); `rnr/` not under git so nothing to commit.

**Key design facts (see PORTING_NOTES §6g, the impl of oracle stabilizer §6d#1):**
- Default BC is already PERIODIC_FULL → the particle integrator wraps `p->x` itself. The gap was
  only the GEOMETRY layer (used plain coordinate diffs). Forces are topological+geometric (not
  spatial-neighbour), so engine cell-list periodicity is irrelevant to them.
- Periodicity is achieved by minimum-image GEOMETRY, NOT coordinate wrapping. `meshWrapPosition`
  is NOT on the per-step hot path (only `Vertex::create` + explicit `setPosition`; integration goes
  `p->x += v·dt` then `positionChanged()` re-reads). This is fine because min-image makes wrapping
  unnecessary. Do NOT "fix" by wrapping in positionChanged (would desync from `p->x`/`p->p0`).
- The swap `metrics::relativePosition`→`meshRelativePosition` trades engine-BC per-axis periodicity
  for the mesh flag + full `Universe::dim()` box on all 3 axes (no per-axis control). Fine for the
  cubic sorting box.
- Float32 caveat: a straddling cell's forces differ from an interior twin by ~1.6e-6/eval; negligible
  in a regularized run but CHAOS-amplified in an unstable config (a lone cube under pure volume
  constraint diverged in ~80 steps — chaos, not a logic bug; forces match at step 0). Judge periodic
  correctness by force/energy parity, not long trajectories of ill-posed configs.

**P0–P4 DONE.** P4 (2026-06-01): periodic Kelvin packing in `rnr/geometry.py` — `periodic_bcc_seeds`
+ `build_periodic_voronoi`. Pure-Python, disk-only (no engine change → nothing to commit). Gate
`test_periodic_geometry.py::test_periodic_voronoi_pack_is_space_filling_closed_and_wraps`
(+ `…_rejects_sub_box`): a 54-cell BCC foam in `[0,60]³` is space-filling (Σvol=boxvol, every cell=
boxvol/54), closed (all 378 surfaces 2-body, zero free faces), wraps (145 wrap faces, 38 straddling
bodies each min-image-small). `pixi run test` = **39 green**. Design + footguns: **PORTING_NOTES §6h**.
KEY P4 lessons: (1) used GHOST-TILING (3×3×3 supercell, keep central, remap adjacency) — pyvoro's own
`periodic=` returns box-spanning garbage on this build; (2) brute-force `dispersion` = full enlarged box
(voro++ block aliasing wrecks regular lattices at most dispersions); (3) min-image vertex dedup
(canonicalize into [lo,hi)) + frozenset-of-vertex-ids face dedup (cell-pair key breaks on wrap-double
adjacency); (4) **THE footgun** — the foam box MUST equal the universe box `[[0,dim]]³` because the
engine min-images at `Universe::dim()`; a sub-box silently gives box-spanning straddling-cell
volumes/centroids (areas/surfaces stay correct, so it looks fine until you check volumes). The function
now asserts this. Also surfaced (and confirmed correct) the engine's never-before-exercised min-image
body centroid + `refreshBodies` b1/b2 reorder for straddling SHARED-surface cells.

**Resume at P5:** min-image `rnr/metrics.py` (demixing_index/contact_summary Python-side
centroid/distance), then run `sort_faithful_3dvertvor.py` in the periodic bulk (s0=5.6, σ sweep, kT=0.1,
μ=1, native reconnection, energy gate OFF); retune toward oracle gentle regime (decoupled placement
~1e-3, §6d#2) + §6d#3 orientation/volume repair only if a cell inverts. Gate: D beats the −0.065
finite-cluster ceiling, stable, noise ON. Kickoff: `docs/periodic_mesh_kickoff_plan.md` §9 (resumes at
P5). See [[thermal-noise-destabilizes-reconnection]], [[native-instability-is-displacement-overshoot]].
