"""Probe: does the FAITHFUL active-motility noise restore the reconnection rate WITHOUT a clamp?

Companion to probe_periodic_sort.py. That probe injects THERMAL Brownian noise (Euler-Maruyama
kick DISP_STD = sqrt(2*mu*kT*dt), scaling as sqrt(dt)), which at Lth=1e-3 is 14-45x Lth per step
and blows collapsing edges back above the reconnect trigger -- so without the per-vertex noise
CLAMP, reconnection STARVES (~1/3000 steps).

But the 3DVertVor/Manning oracle does NOT use thermal Brownian noise. Its per-vertex thermal line
(`Run.cpp:1344`, `cR*ndist`) is COMMENTED OUT; the active line (`:1345`) instead displaces by
`dt * motility`, i.e. ACTIVE self-propulsion that scales as dt (ballistic), NOT sqrt(dt). With
v0 ~ temperature = 0.1 and dt = 1e-3 the per-step active displacement is ~1e-4 = 0.1*Lth -- below
Lth, so a collapsing edge persists below the trigger and is caught with NO clamp.

This probe re-derives that active model against the TF API (nothing copied from the GPL source):
  - each CELL carries a director n_c in S^2 (unit vector)
  - rotational diffusion (active-Brownian): n_c <- normalize(n_c + sqrt(2*Dr*dt)*(xi - n_c)),
    xi ~ uniform on S^2, Dr = 1   (the ONLY sqrt(dt) term -- on ORIENTATION, not position)
  - per-vertex motility velocity: u_v = v0 * mean_{c incident to v} n_c   (oracle Vertex.cpp:78-86)
  - ballistic translation each step: x_v += dt * u_v                       (oracle Run.cpp:1345)

Prints the SAME VERDICT/recon format as probe_periodic_sort.py for an apples-to-apples comparison.

Usage: pixi run python rnr/scripts/probe_active_motility.py [STEPS] [SIGMA] [V0] [LTH] [DT] [CUT] [DR] [M] [SEED] [INTERVAL]
  (V0 is the active self-propulsion speed; the oracle uses V0 = temperature, our KT = 0.1)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
import tissue_forge as tf  # noqa: E402
from tissue_forge.models.vertex import solver as tfv  # noqa: E402
from tissue_forge.models.vertex.solver.mesh_types import BodyTypeSpec, SurfaceTypeSpec  # noqa: E402

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
SIGMA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
V0 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1     # active speed (oracle: == temperature)
LTH = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-3
DT = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-3
CUT = float(sys.argv[6]) if len(sys.argv) > 6 else 1.9
DR = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0     # rotational diffusion of the directors
M = int(sys.argv[8]) if len(sys.argv) > 8 else 4
SEED = int(sys.argv[9]) if len(sys.argv) > 9 else 7
INTERVAL = int(sys.argv[10]) if len(sys.argv) > 10 else 10
# MODEL: "active" (default) drives motility from Python (add_noise below); "native" hands it to
# the C++ engine (MeshSolver.set_motility, PORTING_NOTES §6o) and add_noise becomes a no-op. Both
# are the SAME active model, so the reconnection rate should match.
MODEL = sys.argv[11] if len(sys.argv) > 11 else "active"

L = float(M)
BOX = [[0.0, L]] * 3
V0_VOL, K_V, K_A, A0, MU = 1.0, 10.0, 1.0, 5.6, 1.0
ROT_STD = float(np.sqrt(2.0 * DR * DT)) if V0 > 0 else 0.0
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
    volume_lam = K_V; volume_val = V0_VOL
    surface_area_lam = K_A; surface_area_val = A0
    adhesion = {"A": 0.0, "B": SIGMA}


class B(BodyTypeSpec):
    volume_lam = K_V; volume_val = V0_VOL
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

if MODEL == "native" and V0 > 0:
    tfv.MeshSolver.set_motility(V0, DR, SEED + 2)  # native C++ active drive (PORTING_NOTES §6o)

# --- active-motility state: one director per CELL (body), persistent across steps -------------
rng_dir = np.random.default_rng(SEED + 2)
_dirs = rng_dir.normal(0.0, 1.0, (len(bodies), 3))
_dirs /= np.linalg.norm(_dirs, axis=1, keepdims=True)

# vertex<->cell incidence (rebuilt when topology changes); directors are per-body (stable)
_amc = {"nv": None, "vh": [], "vrow": None, "brow": None, "n": 0}


def _rebuild_incidence():
    vh, idx = [], {}
    for i in range(mesh.size_vertices):
        v = mesh.get_vertex(i)
        if v is None or v.id < 0:
            continue
        idx[v.id] = len(vh)
        vh.append(v)
    vrow, brow = [], []
    for k, b in enumerate(bodies):
        seen = set()
        for s in b.surfaces:
            for w in s.vertices:
                if w.id in idx and w.id not in seen:
                    seen.add(w.id)
                    vrow.append(idx[w.id])
                    brow.append(k)
    _amc.update(nv=mesh.num_vertices, vh=vh,
                vrow=np.array(vrow, np.int64), brow=np.array(brow, np.int64), n=len(vh))


def add_noise():
    if V0 <= 0 or MODEL == "native":
        return  # native: the C++ engine drives motility inside tf.step()
    # Rebuild handles+incidence from the LIVE mesh every step: a single doQuality pass can do
    # several I->H (+1 vertex) and H->I (-1) that NET to zero count change, so num_vertices is an
    # unsafe staleness signal -- a cached deleted handle then segfaults on v.position. (Re-fetching
    # is cheap; the fig/oracle harness can use a coarser INTERVAL-keyed trigger.)
    _rebuild_incidence()
    # active-Brownian rotational diffusion of each cell director (re-derived; sqrt(dt) on ORIENTATION)
    xi = rng_dir.normal(0.0, 1.0, _dirs.shape)
    xi /= np.linalg.norm(xi, axis=1, keepdims=True)
    _dirs[:] = _dirs + ROT_STD * (xi - _dirs)
    _dirs[:] /= np.linalg.norm(_dirs, axis=1, keepdims=True)

    vh, vrow, brow, n = _amc["vh"], _amc["vrow"], _amc["brow"], _amc["n"]
    S = np.zeros((n, 3))
    np.add.at(S, vrow, _dirs[brow])           # sum incident-cell directors per vertex
    cnt = np.zeros(n)
    np.add.at(cnt, vrow, 1.0)
    cnt[cnt == 0] = 1.0
    u = V0 * (S / cnt[:, None])               # per-vertex motility velocity = v0 * <director>
    dx = DT * u                               # ballistic translation: scales as dt (NOT sqrt(dt))
    P = np.array([[p[0], p[1], p[2]] for p in (v.position for v in vh)], dtype=float)
    newP = (P + dx) % L
    for v, qq in zip(vh, newP):
        v.set_position(tf.FVector3(float(qq[0]), float(qq[1]), float(qq[2])))


def vols():
    vs = [b.volume for b in bodies]
    return min(vs), max(vs)


per_step_disp = DT * V0  # upper bound on per-step active displacement magnitude
print(f"ACTIVE-MOTILITY PROBE [{MODEL}] (no clamp) sigma={SIGMA} v0={V0} Lth={LTH} dt={DT} cut={CUT} "
      f"Dr={DR} M={M} N={len(bodies)} steps={STEPS} | per-step disp<= {per_step_disp:.2e} "
      f"({per_step_disp / LTH:.3f}x Lth)", flush=True)
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
        if mn <= 0 or mx > MAX_VOL_FAC * V0_VOL:
            print("  -> SORT UNSTABLE", flush=True); break
verdict = "STABLE" if (worst_min > 0 and worst_max <= MAX_VOL_FAC * V0_VOL) else "UNSTABLE"
print(f"VERDICT sigma={SIGMA} dt={DT}: worst_min={worst_min:.3f} worst_max={worst_max:.3f} "
      f"recon={recon} {verdict}", flush=True)
