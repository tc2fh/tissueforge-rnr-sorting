---
name: tf-swig-subi-needs-forced-regen
description: Editing a TF sub-.i or %included header does NOT trigger SWIG regen; delete the generated wrapper to force it
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 2159783a-b26d-4549-812d-8b6845222894
---

When porting C++ into the `tc2fh/tissue-forge` fork, the CMake/Ninja SWIG step only tracks the
TOP-LEVEL `wraps/python/tissue_forge.i` as a dependency — **not** the transitively-`%include`d
sub-`.i` files (e.g. `tfMeshQuality.i`) or the `%include`d C++ headers. Confirmed:
`grep -c tfMeshQuality.i tissue-forge_build/build.ninja` → 0.

So after editing a sub-`.i` (new SWIG properties) **or** a header SWIG reads (new getters/setters),
`pixi run build-tf` will recompile + relink but **silently skip regenerating the Python wrapper** —
the new methods/properties never reach Python (`hasattr(obj, 'new_method')` is False) even though
the build "succeeds."

**Why:** stale `tissue_forgePYTHON_wrap.cxx` (the generated wrapper) is reused; Ninja sees no
dependency change.

**How to apply:** before `pixi run build-tf`, force a SWIG regen by deleting the generated wrapper
output so Ninja must re-run SWIG on current sources:
`rm -f tissue-forge_build/wraps/python/CMakeFiles/TissueForge_py.dir/tissue_forgePYTHON_wrap.cxx`
(also remove its `.o` if present). Then build. Verify the wrapper picked it up:
`grep -c <NewSymbol> .../tissue_forgePYTHON_wrap.cxx` (>0), and confirm from Python with `hasattr`.
Pure C++ changes (no SWIG surface) don't need this — the normal incremental build relinks them.
Note `FloatP_t` is float32, so a Python-set 0.1 reads back as 0.10000000149 (use ~1e-6 tol in
smoke checks). Relevant to native-port Phases A–G ([[phaseG-new-surface-tension-actor]]).
