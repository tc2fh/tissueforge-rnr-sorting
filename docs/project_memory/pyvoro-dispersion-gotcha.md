---
name: pyvoro-dispersion-gotcha
description: pyvoro-mmalahe returns overlapping full-box cells unless dispersion >= seed spacing
metadata: 
  node_type: memory
  type: reference
  originSessionId: 56f34c8e-7568-4bf4-8f2c-10ca977c3fb6
---

`pyvoro.compute_voronoi(points, limits, dispersion)` (the `pyvoro-mmalahe` fork used
for initial packing) gives **wrong, overlapping full-box cells** — each seed assigned
the entire bounding box, Σ cell-volume ≫ box volume, ~0 internal faces — when
`dispersion` is **smaller than the seed spacing**. `dispersion` is voro++'s block-grid
hint; too small and cross-block neighbors aren't linked.

Fix: pass `dispersion >= ` the largest box edge (always ≥ spacing). `rnr/geometry.py`
defaults it to `max(hi-lo for lo,hi in limits)`. Correctness check: a valid
space-filling tessellation has **Σ cell volume == bounding-box volume** (use this as
an assert).

Related: [[phase0-baseline-control]].
