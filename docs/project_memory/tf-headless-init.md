---
name: tf-headless-init
description: "TissueForge headless init — windowless=True required, and tf.init() is a one-per-process singleton"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 56f34c8e-7568-4bf4-8f2c-10ca977c3fb6
---

Running TissueForge headless (the SegoLab/VertexModeling project, macOS osx-arm64):

- **`tf.init(windowless=True, ...)` is REQUIRED.** Without it, `tf.init()` hangs
  forever (process alive but ~0 CPU, blocked) when launched from a non-GUI shell —
  it blocks creating a GL/window context. Symptom: a `pixi run` that never returns.
  Reference: bundled `tissue_forge/examples/windowless.py`.
- **`tf.init()` is a singleton — calling it twice in one process hangs.** One init
  per process. Implication for the Phase-1 reconnection tests: pytest cannot re-init
  between tests → use a **session-scoped fixture or a subprocess per test file**.
- Drive headless loops with `tf.step()` (one `dt`); never `tf.show()` (opens a window).

Related: [[tf-mesh-quality-default-on]], [[pyvoro-dispersion-gotcha]].
