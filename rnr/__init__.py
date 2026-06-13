"""rnr — 3D reversible network reconnection (RNR / Okuda I<->H) prototype for
TissueForge, targeting 3DVertVor-style heterotypic cell sorting.

Phase 0 (current): finite-cluster 3D vertex control with cell-sorting energetics
and NO reconnection (expected to jam / fail to sort). See rnr/geometry.py for the
Voronoi->TissueForge mesh builder and rnr/scripts/baseline_no_reconnect.py for the
control run.

Keep `import rnr` light: heavy deps (tissue_forge) are imported inside submodules.
"""

__version__ = "0.0.1"
