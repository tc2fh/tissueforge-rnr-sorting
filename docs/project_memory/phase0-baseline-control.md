---
name: phase0-baseline-control
description: Phase 0 control done — 3D Voronoi finite cluster jams (heterotypic neighbor pairs frozen) without reconnection
metadata: 
  node_type: memory
  type: project
  originSessionId: 56f34c8e-7568-4bf4-8f2c-10ca977c3fb6
---

**Phase 0 (control) is complete** as of 2026-05-29. `pixi run baseline`
(`rnr/scripts/baseline_no_reconnect.py`) builds a finite 3D Voronoi cluster of 20
cells, two types with differential adhesion (homotypic λ=−1 sticky, heterotypic λ=+1
repulsive), `VolumeConstraint` + `SurfaceAreaConstraint`, MeshQuality disabled, and
integrates 200 headless steps.

Result: **heterotypic neighbor-PAIR count is exactly constant (35/91 every step) →
topology frozen → the tissue JAMS and cannot sort.** The area-weighted het fraction
and adhesion energy drift down (shape relaxation at fixed topology), which is NOT
sorting. This is the intended control: it proves the missing piece is the 3D
reconnection (Okuda I↔H / 3D T1), the target of Phase 1+.

Key supporting code (all in `rnr/`):
- `geometry.py` `build_voronoi_cluster()` — pyvoro→TF via global vertex/face dedup;
  each interior face = one Surface shared by 2 bodies; validated vs pyvoro (volumes
  match ~1e-5, space-filling). See [[pyvoro-dispersion-gotcha]].
- `metrics.py` `contact_summary()` — the **topological** het-pair count is the
  jamming signature; the area fraction alone is misleading.
- `PORTING_NOTES.md` — verified C++↔Python API seams + design departures.

Design decisions (depart from CLAUDE.md, by user direction / API reality):
finite cluster (vertex layer has **no** periodic support), Voronoi packing,
MeshQuality off for the control ([[tf-mesh-quality-default-on]]).

**Next:** Phase 1 — `rnr/topology.py` + `reconnect.py` (i_to_h/h_to_i) +
`conditions.py`, gated by `rnr/tests/test_roundtrip.py`. Do NOT start until asked.
Tests need one-init-per-process handling ([[tf-headless-init]]).
