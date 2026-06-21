---
name: phase1-reconnection-done
description: Phase 1 complete — the Okuda I↔H 3D-T1 reconnection is built and provably reversible (pixi run test green)
metadata: 
  node_type: memory
  type: project
  originSessionId: 03c38b24-8fab-4238-af5e-e2edc121ddc5
---

Phase 1 (the 3D T1 / Okuda I↔H reversible network reconnection) is **DONE as of
2026-05-30**. `pixi run test` is green (6/6 in `rnr/tests/test_roundtrip.py`).

What exists now:
- `rnr/topology.py` — read-only walk: `i_neighbourhood(v10,v11)` / `h_neighbourhood(tri)`
  gather the 5-cell / 9-face / 6-outer-vertex Okuda neighbourhood from TF adjacency
  (TF has no explicit Edge: the short edge = an ordered consecutive vertex pair in the 3
  side surfaces). `find_short_edges` / `find_small_triangles` are the Condition-2 scanners.
- `rnr/conditions.py` — Condition-4 vetoes (4(ii) double edge, 4(iii) double trigonal face).
- `rnr/reconnect.py` — `i_to_h` / `h_to_i`, check-half + mutate-half, **Strategy A**
  (manual surface-list surgery), Okuda Appendix-1 placement (Eqs. 42–56).

Reversibility proven: topology restores **exactly**, geometry within O(Δl_th) (Okuda
Eqs. 5–7) — non-central verts byte-identical, edge verts exact when edge==Δl_th
(minimal config drift <1e-6; Kelvin ~0.013).

Hard-won TF surgery seams (now in `rnr/PORTING_NOTES.md §0 + §4b`): low-level
`Surface.replace/insert/remove` are **one-directional** (mirror with `vertex.add/remove`);
body↔surface attach needs **both** `surface.add(body)` AND `body.add(surface)`;
`Surface.neighbor_vertices` returns a non-iterable SWIG tuple (walk the ring instead);
destroy via `handle.destroy()` (no args), not the static `Vertex.destroy([...])`.

Next is **Phase 2** (only when asked): `rnr/operator.py` to scan edges < Δl_th and run
reconnection in the step loop, then reproduce 3DVertVor sorting. Real open risk: Phase 1
proved topology+geometry reversibility but NOT many-step integrator stability of the
post-reconnection mesh — check that first. See [[phase0-baseline-control]],
[[tf-mesh-quality-default-on]].
