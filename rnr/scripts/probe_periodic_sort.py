"""Probe: does a PERIODIC two-type sort stay VALID under reconnection + thermal noise?

Companion to probe_periodic_substrate.py. The substrate probe proved the foam integrates
stably with NO reconnection. This one turns reconnection ON (native I<->H inside doQuality)
AND adds thermal noise -- the combination that used to blow up: a native reconnection places
two vertices Lth apart, and one thermal kick (DISP_STD ~ 14*Lth) throws one past its neighbour
=> the cell everts (signed volume < 0). The fix is a per-vertex TRUST-REGION on the noise (cap
each vertex's noise displacement at NOISE_CLAMP * min-image nearest-neighbour distance); see
sort_periodic_oracle.py and PORTING_NOTES.

Prints a VERDICT line in the SAME format as probe_periodic_substrate.py so the gate test can
reuse one parser:

    VERDICT sigma=.. dt=..: worst_min=.. worst_max=.. STABLE|UNSTABLE

Usage: pixi run python rnr/scripts/probe_periodic_sort.py [STEPS] [SIGMA] [KT] [LTH] [DT] [CUT] [CLAMP] [M] [SEED]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
import tissue_forge as tf  # noqa: E402
from tissue_forge.models.vertex import solver as tfv  # noqa: E402
from tissue_forge.models.vertex.solver.mesh_types import BodyTypeSpec, SurfaceTypeSpec  # noqa: E402

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
SIGMA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
KT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
LTH = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-3
DT = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-3
CUT = float(sys.argv[6]) if len(sys.argv) > 6 else 1.9
CLAMP = float(sys.argv[7]) if len(sys.argv) > 7 else 0.4
M = int(sys.argv[8]) if len(sys.argv) > 8 else 4
SEED = int(sys.argv[9]) if len(sys.argv) > 9 else 7
INTERVAL_ARG = int(sys.argv[10]) if len(sys.argv) > 10 else 10

L = float(M)
BOX = [[0.0, L]] * 3
V0, K_V, K_A, A0, MU = 1.0, 10.0, 1.0, 5.6, 1.0
DISP_STD = float(np.sqrt(2.0 * MU * KT * DT)) if KT > 0 else 0.0
INTERVAL = INTERVAL_ARG
MAX_VOL_FAC = 4.0

tf.init(windowless=True, dim=[L, L, L], cutoff=CUT, dt=DT)
tfv.init()
mesh = tfv.MeshSolver.get().get_mesh()
mesh.quality = None
mesh.periodic_geometry = True

from rnr.geometry import build_periodic_voronoi  # noqa: E402

rng = np.random.default_rng(SEED)
seeds = (rng.random((M ** 3, 3)) * L).tolist()


class Iface(SurfaceTypeSpec):
    pass


class A(BodyTypeSpec):
    volume_lam = K_V; volume_val = V0
    surface_area_lam = K_A; surface_area_val = A0
    adhesion = {"A": 0.0, "B": SIGMA}


class B(BodyTypeSpec):
    volume_lam = K_V; volume_val = V0
    surface_area_lam = K_A; surface_area_val = A0
    adhesion = {"A": SIGMA, "B": 0.0}


stype, btA, btB = Iface.get(), A.get(), B.get()
BodyTypeSpec.bind_adhesion([A, B])
bodies, _sd, stats = build_periodic_voronoi(seeds, BOX, btA, stype)
tfv.MeshSolver.get().position_changed()

rng2 = np.random.default_rng(SEED + 1)
for b in bodies:
    if rng2.random() < 0.5:
        b.become(btB)

q = tfv.Quality()
q.stock_quality_operations = False
q.reconnect_length = LTH
q.reconnect_hysteresis = 0.2
q.reconnect_energy_gate = False
q.reconnect_interval = INTERVAL
q.collision_2d = False
mesh.quality = q
mesh.periodic_geometry = True

_ncache = {"nv": None, "vh": [], "edges": None}


def _rebuild():
    vh, idx = [], {}
    for i in range(mesh.size_vertices):
        v = mesh.get_vertex(i)
        if v is None or v.id < 0:
            continue
        idx[v.id] = len(vh)
        vh.append(v)
    edges = set()
    for b in bodies:
        for s in b.surfaces:
            ring = [w.id for w in s.vertices if w.id in idx]
            n = len(ring)
            for k in range(n):
                a, c = idx[ring[k]], idx[ring[(k + 1) % n]]
                edges.add((a, c) if a < c else (c, a))
    _ncache["nv"] = mesh.num_vertices
    _ncache["vh"] = vh
    _ncache["edges"] = np.array(sorted(edges), dtype=np.int64) if edges else np.empty((0, 2), np.int64)


def add_noise():
    if DISP_STD <= 0:
        return
    if _ncache["nv"] != mesh.num_vertices:
        _rebuild()
    vh, edges = _ncache["vh"], _ncache["edges"]
    n = len(vh)
    P = np.array([[p[0], p[1], p[2]] for p in (v.position for v in vh)], dtype=float)
    dx = rng2.normal(0.0, DISP_STD, (n, 3))
    if CLAMP > 0 and len(edges):
        d = P[edges[:, 0]] - P[edges[:, 1]]
        d -= L * np.round(d / L)
        elen = np.linalg.norm(d, axis=1)
        nn = np.full(n, np.inf)
        np.minimum.at(nn, edges[:, 0], elen)
        np.minimum.at(nn, edges[:, 1], elen)
        cap = CLAMP * nn
        mag = np.linalg.norm(dx, axis=1)
        scale = np.where((mag > cap) & (mag > 0), cap / np.maximum(mag, 1e-30), 1.0)
        dx *= scale[:, None]
    newP = (P + dx) % L
    for v, qq in zip(vh, newP):
        v.set_position(tf.FVector3(float(qq[0]), float(qq[1]), float(qq[2])))


def vols():
    vs = [b.volume for b in bodies]
    return min(vs), max(vs)


print(f"SORT PROBE (reconnection ON) sigma={SIGMA} kT={KT} Lth={LTH} dt={DT} cut={CUT} "
      f"clamp={CLAMP} M={M} N={len(bodies)} steps={STEPS}", flush=True)
worst_min, worst_max = float("inf"), 0.0
recon = 0
nv_prev = mesh.num_vertices
for i in range(0, STEPS + 1):
    if i:
        add_noise()
        tf.step()
        nv = mesh.num_vertices
        if nv != nv_prev:
            recon += abs(nv - nv_prev); nv_prev = nv
    if i % 250 == 0:
        mn, mx = vols()
        worst_min = min(worst_min, mn); worst_max = max(worst_max, mx)
        print(f"  step {i:6d}: min_vol={mn:8.3f} max_vol={mx:8.3f} recon~{recon}", flush=True)
        if mn <= 0 or mx > MAX_VOL_FAC * V0:
            print("  -> SORT UNSTABLE", flush=True); break
verdict = "STABLE" if (worst_min > 0 and worst_max <= MAX_VOL_FAC * V0) else "UNSTABLE"
print(f"VERDICT sigma={SIGMA} dt={DT}: worst_min={worst_min:.3f} worst_max={worst_max:.3f} "
      f"recon={recon} {verdict}", flush=True)
