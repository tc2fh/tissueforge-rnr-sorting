"""Probe: is the PERIODIC Kelvin-foam SUBSTRATE stable under the dynamics WITHOUT reconnection?

The periodic sort harness blows up even at Lth=0.01 (only ~10 reconnections), so the
instability may be in the substrate (the relaxation dynamics) itself, not the reconnection.
This isolates that: build the periodic foam, assign A/B types, bind the actors + (optional)
noise, set mesh.quality=None (NO reconnection at all), and watch min/max volume.

Usage: pixi run python rnr/scripts/probe_periodic_substrate.py [N_STEPS] [SIGMA] [DT] [KT] [N] [SEED]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import tissue_forge as tf
import tissue_forge.tissue_forge as _low
from tissue_forge.models.vertex import solver as tfv
from tissue_forge.models.vertex.solver.mesh_types import BodyTypeSpec, SurfaceTypeSpec

N_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
SIGMA = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
DT = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4
KT = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
N_PER_AXIS = int(sys.argv[5]) if len(sys.argv) > 5 else 4
SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 7

EDGE_A = 2.0
L = EDGE_A * N_PER_AXIS
BOX = [[0.0, L]] * 3
VOL_VAL = 4.0
SURF_LAM = 0.1
SURF_VAL = 5.6 * VOL_VAL ** (2.0 / 3.0)
D_PER_STD2 = 1.63e-5
NOISE_STD = float(np.sqrt(KT / D_PER_STD2)) if KT > 0 else 0.0

tf.init(windowless=True, dim=[L, L, L], cutoff=min(3.0, 0.49 * L), dt=DT)
tfv.init()
mesh = tfv.MeshSolver.get().get_mesh()
mesh.quality = None          # NO reconnection at all
mesh.periodic_geometry = True

from rnr.geometry import build_periodic_voronoi, periodic_bcc_seeds
from rnr.metrics import contact_summary


class Iface(SurfaceTypeSpec):
    pass


class A(BodyTypeSpec):
    volume_lam = 1.0; volume_val = VOL_VAL
    surface_area_lam = SURF_LAM; surface_area_val = SURF_VAL
    adhesion = {"A": 0.0, "B": SIGMA}


class B(BodyTypeSpec):
    volume_lam = 1.0; volume_val = VOL_VAL
    surface_area_lam = SURF_LAM; surface_area_val = SURF_VAL
    adhesion = {"A": SIGMA, "B": 0.0}


stype, btA, btB = Iface.get(), A.get(), B.get()
BodyTypeSpec.bind_adhesion([A, B])
bodies, _s, stats = build_periodic_voronoi(periodic_bcc_seeds(N_PER_AXIS, BOX), BOX, btA, stype)
tfv.MeshSolver.get().position_changed()
rng = np.random.default_rng(SEED)
for b in bodies:
    if rng.random() < 0.5:
        b.become(btB)

if NOISE_STD > 0:
    ptype = _low._vertex_solver__MeshParticleType_get()
    tf.bind.force(tf.Force.random(std=NOISE_STD, mean=0.0, duration=DT), ptype)


def vols():
    vs = [b.volume for b in bodies]
    return min(vs), max(vs)


def lam(i, j):
    return 0.0 if i == j else SIGMA


print(f"SUBSTRATE PROBE (NO reconnection) sigma={SIGMA} dt={DT} kT={KT} n={N_PER_AXIS} N={len(bodies)} "
      f"steps={N_STEPS}", flush=True)
worst_min = float("inf"); worst_max = 0.0
for i in range(0, N_STEPS + 1):
    if i:
        tf.step()
    if i % 500 == 0:
        mn, mx = vols()
        worst_min = min(worst_min, mn); worst_max = max(worst_max, mx)
        s = contact_summary(bodies=bodies, lam=lam)
        print(f"  step {i:6d}: D={s['demixing_index']:+.4f} het_area={s['het_area_fraction']:.4f} "
              f"min_vol={mn:8.3f} max_vol={mx:8.3f}", flush=True)
        if mn <= 0 or mx > 3 * VOL_VAL:
            print("  -> SUBSTRATE UNSTABLE", flush=True); break
print(f"VERDICT sigma={SIGMA} dt={DT}: worst_min={worst_min:.3f} worst_max={worst_max:.3f} "
      f"{'STABLE' if worst_min>0 and worst_max<=3*VOL_VAL else 'UNSTABLE'}", flush=True)
