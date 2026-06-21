---
name: tf-mesh-quality-default-on
description: TissueForge auto-enables MeshQuality on every mesh; disable with mesh.quality=None
metadata: 
  node_type: memory
  type: reference
  originSessionId: 56f34c8e-7568-4bf4-8f2c-10ca977c3fb6
---

TissueForge's `Mesh` constructor sets `_quality{new MeshQuality()}`, so **every mesh
has automatic MeshQuality enabled by default**, and `MeshSolver` runs
`doQuality()` every step (`tfMeshSolver.cpp`: `if(mesh->hasQuality()) doQuality()`).

Its only 3D operations are **irreversible degenerate collapses** (surface-demote,
body-demote, vertex-merge) — NOT a reconnection. On a Voronoi mesh these silently
remove small/sliver contacts during stepping (observed total cell-cell contacts drop
91→81), which falsely mimics partial sorting.

Disable it for any frozen-topology control:
`tfv.MeshSolver.get().get_mesh().quality = None`  (→ `setQuality(nullptr)`;
`mesh.has_quality` then False). The reversible RNR (Okuda I↔H) we are building will be
the *only* intended topology-change operator.

**Why this matters:** supplying the missing 3D T1 reconnection is the whole project;
the native ops are collapses only. **How to apply:** disable quality in controls;
when wiring reconnection, decide explicitly whether collapses should also be on.
Related: [[phase0-baseline-control]], [[tf-headless-init]].
