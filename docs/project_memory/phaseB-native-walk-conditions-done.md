---
name: phaseb-native-walk-conditions-done
description: Native RNR Phase B done (C++ neighborhood walk + Condition-4 vetoes) + two Phase-D-critical findings (stock-pass segfault on finite blocks; serial chain build)
metadata: 
  node_type: memory
  type: project
  originSessionId: 7859e15a-d5ff-4b50-9983-8ab15159ff1f
---

Native RNR port **Phase B is done** (committed 888f07c on `feat/native-rnr-reconnection`,
after Phase A 1671227; not pushed). Ported `rnr/topology.py` + `rnr/conditions.py` into an
anonymous-namespace block in `tissue-forge/source/models/vertex/solver/tfMeshQuality.cpp`:
`rnr_iNeighbourhood`/`rnr_hNeighbourhood` (walk), `rnr_iToHVeto`/`rnr_hToIVeto` (+ primitives),
`rnr_findShortEdges`/`rnr_findSmallTriangles` (Condition-2 scanners). `ReconnectionOperation`
got real `check()`/`prep()`/`targets`; `implement()` still a no-op (Phase C). Test seam = 3
read-only JSON diagnostics on `MeshQuality` (`analyze_i_reconnection`/`analyze_h_reconnection`/
`find_reconnection_candidates`, SWIG-wrapped) read by `rnr/tests/test_native_reconnection.py`
(6 tests, cross-checked edge-for-edge vs the Python oracle). `pixi run test` = 25 green.
Detached `tfv.Quality()` works for diagnostics (they read global `Mesh::get()`).

**Two findings Phase C/D MUST heed (both verified empirically):**

1. **The stock MeshQuality passes segfault on a finite Kelvin block** over enough repeated
   `doQuality()` calls **even with `reconnect_length=0`** (our pass off) — the degenerate 3D
   collapses (Surface/Body/EdgeDemote) corrupt the finite block (dangling op in a prev/next set
   → bad deref). This is exactly why conftest/CLAUDE keep `mesh.quality=None`. Our reconnection
   pass IN ISOLATION (stock thresholds zeroed, `collision_2d=False`, no forces) is stable + inert
   40 calls × 5 procs. So **Phase D must run reconnection WITHOUT the stock collapses** (exclude
   path / on-demand), or fix the stock collapses on finite blocks, before any live sort loop.
   The earlier "segfault" scare during Phase B smoke was THIS, not our code.

2. **Build the reconnection dependency chains SERIALLY.** The stock parallel
   `MeshQuality_constructChains` races on the *dense* reconnection graph (~9 touched surfaces ×
   hundreds of candidates) → fabricated cycle → unbounded `MeshQuality_upstreams` recursion →
   stack-overflow segfault. The reconnection pass calls `MeshQualityOperation_checkChain` in a
   serial loop (valid DAG); the later parallel `doOperations` walk is safe on a valid DAG.

Relates to [[phase1-reconnection-done]], [[faithful-instability-is-winding-signflip]] (the
separate Phase-D/E volume-sign fix), [[tf-swig-subi-needs-forced-regen]] (rm the wrap cxx when
editing the .i). Next: Phase C = port `reconnect.py` surgery + Okuda Appendix-1 placement into
`implement()`, gated by a native round-trip test.
