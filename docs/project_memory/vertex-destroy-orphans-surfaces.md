---
name: vertex-destroy-orphans-surfaces
description: TF Body.destroy() leaves surfaces alive (orphaned) and the vertex renderer still draws them — hiding cells requires destroying surfaces too
metadata:
  node_type: memory
  type: project
  originSessionId: 22b6287b-dc28-4664-b155-4cc8198dac18
---

`Body.destroy()` in the TF vertex model does NOT destroy the body's surfaces. Source path:
`Body::destroy → Mesh::remove(Body*)` does `for(auto &s : b->getSurfaces()) s->remove(b);` —
it only DETACHES the body pointer from each surface, then frees the body slot. Surfaces stay
live and registered. The vertex `MeshRenderer::draw` iterates EVERY surface in the mesh
(`parallel_for(mesh->surfaces->size())`, draws any with `objectId() >= 0`) regardless of body
membership — so orphaned surfaces keep rendering.

**Why it matters / how it bit us:** "destroy boundary cells / keep N interior cells" leaves all
the boundary surfaces alive, so a screenshot shows the WHOLE block, not the kept subset.
Verified 2026-05-29: built 189 Voronoi cells (890 quad + 216 hex surfaces), destroyed 188
bodies → bodies=1 but surface tally UNCHANGED (890+216). The lone kept cell's own 14 faces are
correct, but ~1092 orphaned surfaces still render. This is what made the old
`kelvin_shape_demo.py` single-cell modes render the full all-blue block (the block's exterior
hull is all square faces).

**Fix (used in `rnr/scripts/sorting_demo_start.py`):** after destroying the unwanted bodies,
also destroy the now-orphaned surfaces:
```python
for st in (stA, stB):
    for s in list(st.instances):
        if len(s.getBodies()) == 0:   # no surviving body -> orphan
            s.destroy()
```
Verified: reduces the render to exactly the kept cluster (e.g. lone cell → 6 quad + 8 hex).
Alternative, cleanest for single cells: BUILD ONLY the surfaces you want (see
`rnr/scripts/kelvin_single_cell.py`, which builds one cell directly — no destroy needed).

This is almost certainly the real cause behind several entries previously logged as separate
"render colour" gotchas. See [[vertex-render-color-gotchas]]. Possible future native-port patch:
have `MeshRenderer::draw` skip surfaces with no live body, OR have `Body::destroy` optionally
cascade-destroy its exclusively-owned surfaces — but for now handle it in Python.
