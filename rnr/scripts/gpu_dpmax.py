"""DP_max for the GPU (cached) periodic foam -- the finite-N demixing ceiling for Fig 1E/1F.

The paper (Sahu/Schwarz/Manning ref [4]; Manning2024 Fig 1E) plots DP normalized by its maximum
attainable value at the given system size:

    DP      = <2*(N_s/N_t - 1/2)>  (Sahu Eq. 2; N_s homotypic nbrs, N_t total)  = 1 - 2*het_frac
    DP_max  = DP of a fully segregated ("minimal-surface") arrangement at that N (< 1 for finite N,
              because a segregated mixture still has heterotypic facets at the domain interface).

We RE-DERIVE DP_max (NOT copy GPL) on the SAME cached foam the GPU sort uses: load it (no TF),
assign the two types by a single planar 50/50 cut through the box centre along each axis (the
minimal-area segregation in a periodic box -> two flat interfaces), and measure het_frac with the
engine's own `het_contact_fraction`. DP_max = max over axes of (1 - 2*het_frac). This mirrors
rnr/scripts/compute_dpmax.py but for the GPU foam + the GPU het metric (no TissueForge).

Usage:  pixi run python rnr/scripts/gpu_dpmax.py [N] [IC]     (defaults n=10, ic=mixed)
Writes rnr/exports/gpu_dpmax.json keyed by n; prints the JSON record.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

from rnr.gpu import physics_csr as P  # noqa: E402
from rnr.gpu.device_mesh import PaddedMesh  # noqa: E402
from rnr.gpu.engine import het_contact_fraction  # noqa: E402
from rnr.gpu.foam_cache import cache_path, load_host  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
IC = sys.argv[2] if len(sys.argv) > 2 else "mixed"   # geometry is IC-independent; types are overridden


def main():
    path = cache_path(N, IC)
    if not path.exists():
        print(f"FATAL: no cached foam {path} -- build it first "
              f"(pixi run gpu-stability --n {N} --steps 1 --ic {IC}).")
        sys.exit(2)
    host = load_host(path)
    csr, box = host["csr"], np.asarray(host["box"], float)
    pm = PaddedMesh.from_csr(csr)                      # default headroom; geometry uses live prefix
    geom = P.compute_geometry(pm, box)
    bcent = geom.bcent[:pm.nb]

    dp_axes = {}
    for ax, nm in enumerate("xyz"):
        # planar cut through the box centre along axis `ax` -> the minimal-area 50/50 segregation
        bt = (bcent[:, ax] >= 0.5 * box[ax]).astype(np.int64)
        het, total = het_contact_fraction(pm, bt)
        dp_axes[nm] = (1.0 - 2.0 * het / total) if total else 0.0
    dp_max = max(dp_axes.values())

    rec = {"n": N, "N": int(pm.nb), "DP_max": dp_max, "DP_max_axes": dp_axes,
           "box": [float(b) for b in box]}
    print(json.dumps(rec))

    exports = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exports")
    os.makedirs(exports, exist_ok=True)
    p = os.path.join(exports, "gpu_dpmax.json")
    db = {}
    if os.path.exists(p):
        try:
            db = json.load(open(p))
        except Exception:
            db = {}
    db[str(N)] = rec
    json.dump(db, open(p, "w"), indent=2)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
