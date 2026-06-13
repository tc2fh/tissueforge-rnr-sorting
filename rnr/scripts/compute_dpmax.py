"""Compute DP_max (the finite-size maximum demixing parameter) for our periodic foam.

WHY. The paper (Sahu/Schwarz/Manning, ref [4]; Manning2024 Fig 1E) plots the demixing
parameter normalized by its maximum ATTAINABLE value at a given system size:

    DP = <2*(N_s/N_t - 1/2)>        (Sahu Eq. 2; N_s homotypic nbrs, N_t total nbrs)
    DP_max = DP of a fully segregated ("minimal-surface") arrangement at that N.

DP_max < 1 for finite N because a segregated mixture still has heterotypic facets at the
domain interface (Sahu SI S2: DP_final ~ 1 - O(N^-1/3) ; interfacial cells share ~1/3 of
their facets with the other type). We reproduce DP_max by RE-DERIVING it (NOT copying GPL):
build the SAME periodic Kelvin foam our sort runs use and assign the two types by a single
planar cut (the minimal-area 50/50 segregation in a periodic box -> two flat interfaces),
then measure DP on that config.

Our `demixing_index` D = <2*(het_frac - 1/2)> = -DP (documented sign flip, metrics.py), so
DP = -D and DP_max = -D(segregated).

Usage:  pixi run python rnr/scripts/compute_dpmax.py [M] [SEED] [AXIS]
  M default 6 (N=M^3), SEED default 7 (match the sort runs), AXIS in {x,y,z} default x.
Prints a JSON line {"M":..,"N":..,"DP_max":..,"DP_max_axes":{x,y,z}} to stdout and appends
to rnr/exports/dpmax.json so the plotter can read it without re-running TissueForge.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
import tissue_forge as tf  # noqa: E402
from tissue_forge.models.vertex import solver as tfv  # noqa: E402
from tissue_forge.models.vertex.solver.mesh_types import BodyTypeSpec, SurfaceTypeSpec  # noqa: E402

M = int(sys.argv[1]) if len(sys.argv) > 1 else 6
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 7
L = float(M)
BOX = [[0.0, L]] * 3
V0, K_V, K_A, A0 = 1.0, 10.0, 1.0, 5.6

tf.init(windowless=True, dim=[L, L, L], cutoff=1.9, dt=1e-3)
tfv.init()
mesh = tfv.MeshSolver.get().get_mesh()
mesh.quality = None
mesh.periodic_geometry = True

from rnr.geometry import build_periodic_voronoi  # noqa: E402
from rnr.metrics import demixing_index  # noqa: E402

rng = np.random.default_rng(SEED)
seeds = (rng.random((M ** 3, 3)) * L).tolist()


class Iface(SurfaceTypeSpec):
    pass


class A(BodyTypeSpec):
    volume_lam = K_V; volume_val = V0
    surface_area_lam = K_A; surface_area_val = A0
    adhesion = {"A": 0.0, "B": 0.5}


class B(BodyTypeSpec):
    volume_lam = K_V; volume_val = V0
    surface_area_lam = K_A; surface_area_val = A0
    adhesion = {"A": 0.5, "B": 0.0}


stype, btA, btB = Iface.get(), A.get(), B.get()
BodyTypeSpec.bind_adhesion([A, B])
bodies, _sd, stats = build_periodic_voronoi(seeds, BOX, btA, stype)
tfv.MeshSolver.get().position_changed()

# centroid of each body, to partition by a plane through the box centre
cents = {}
for b in bodies:
    c = b.centroid
    cents[b.id] = (c[0], c[1], c[2])


def dp_for_axis(ax):
    """DP of the segregated config that splits the box in half along axis `ax`."""
    for b in bodies:
        b.become(btA)
    half = L / 2.0
    for b in bodies:
        if cents[b.id][ax] >= half:
            b.become(btB)
    # -demixing_index == Sahu DP (sorted -> +). Segregated config => DP_max for this N.
    return -demixing_index(bodies=bodies)


dp_axes = {nm: dp_for_axis(i) for i, nm in enumerate(("x", "y", "z"))}
dp_max = max(dp_axes.values())   # best (most-segregated) planar cut

rec = {"M": M, "N": len(bodies), "seed": SEED, "DP_max": dp_max,
       "DP_max_axes": dp_axes}
print(json.dumps(rec))

EXPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)
path = os.path.join(EXPORT_DIR, "dpmax.json")
db = {}
if os.path.exists(path):
    try:
        db = json.load(open(path))
    except Exception:
        db = {}
db[str(M)] = rec
json.dump(db, open(path, "w"), indent=2)
print(f"wrote {path}")
