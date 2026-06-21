---
name: vertex-render-color-gotchas
description: How to actually get per-cell render colors in the TF vertex MeshRenderer (gotchas + which earlier "gotchas" turned out to be misdiagnoses)
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 22b6287b-dc28-4664-b155-4cc8198dac18
---

Getting two visibly-colored cell populations rendered in TissueForge's vertex
MeshRenderer (headless screenshot path). The WORKING recipe is in
`rnr/scripts/sorting_demo_start.py`. Verify every result by PIXEL-COUNTING the PNG
(`PIL.Image.open` → count blue vs orange); a single eyeballed shot / stale `/tmp` file
fooled the original session repeatedly — which is how the two debunked claims below crept in.

**Re-investigated 2026-05-29 with controlled, separate-process, pixel-counted A/B tests.**
The renderer (`tfMeshRenderer.cpp::render_meshFacesEdges`) reads
`s->style ? s->style : s->type()->style` FRESH every draw; `s->style` is NULL after creation
(ctor inits `style{NULL}`), so color comes from the surface TYPE's style each frame.

CONFIRMED traps:
1. **Render colour = the SURFACE type's style, fixed at SURFACE CREATION.** Pass the colored
   `SurfaceType` to the surface constructor. `surface.become(otherType)` does NOT change the
   rendered colour in the windowless single-`screenshot` path — VERIFIED: after `become`, the
   C++ `typeId`/`type()`/`type().style.color` are all correct (e.g. 478 surfaces now type B,
   B.style=orange) and `s.style is None`, yet the render stays the creation-time colour across
   repeated screenshots AND a `position_changed()`. The data is right; only the rendered colour
   is stale. So colour MUST be set at creation. (`view_cluster.py` escapes this only because
   `tf.show()` runs a live loop.)
2. **`Body.destroy()` orphans surfaces, which keep rendering — THE big one.** See
   [[vertex-destroy-orphans-surfaces]]. To hide cells you must destroy their surfaces (or build
   only what you render), not just the bodies.

DEBUNKED (were in this note, now shown FALSE by pixel-identical A/B tests — do NOT cargo-cult):
- ❌ "A 2nd BodyType clobbers surface colours." Building bodies as two BodyTypes vs one gives
  PIXEL-IDENTICAL multi-colour renders. BodyType count has no effect on surface colour. (Using
  one BodyType is still fine/simplest, but it is NOT required for colour.)
- ❌ "Camera move after build resets colours to one colour." Camera-before-build vs
  camera-after-build are PIXEL-IDENTICAL, both multi-colour. And a camera rotate BETWEEN two
  screenshots does change the image — the renderer re-renders fresh each shot. Camera ordering
  has zero effect on colour. (The original "all-blue after tilt" was almost certainly the
  destroy-orphan bug rendering the full block, whose exterior hull is one colour.)
- ⚠️ "Index-based per-face colouring renders all-blue; use nearest-seed geometry instead." Not
  re-tested in isolation; now believed to be a symptom of the destroy-orphan bug (the whole
  ~189-cell block rendered, dominated by hull faces). nearest-seed still works; revisit if it
  recurs after the orphan fix.

Relies on renderer Patches A/B — see [[py311-source-build-migration]].
