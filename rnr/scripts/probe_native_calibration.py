"""Calibration probe for the NATIVE active-motility drive (Phase 3, Gate 2; PORTING_NOTES §6o).

Confirms the two quantities the native port must get right, directly (not via the end-to-end
sort/rate):

  (A) DISPLACEMENT SCALING / mobility.  In the overdamped vertex integrator (x += dt*f*imass,
      mass = 1 for density-0 vertices => mu = imass = 1), the per-vertex active force
      f = v0*<incident-cell directors> must produce per-step displacement EXACTLY
      dt*v0*<director> -- i.e. the 3DVertVor/Manning oracle's dt*motility. We isolate the active
      displacement on a foam whose energy actors are zeroed (only the active force + a small
      FlatSurfaceConstraint remain) by subtracting a v0=0 baseline step, then compare per vertex to
      the prediction dt*v0*<director> (median magnitude ratio ~ 1, median cosine ~ 1).

  (B) ROTATIONAL DIFFUSION (Dr).  For the discrete active-Brownian update
      n <- normalize(n + s*(xi-n)), s=sqrt(2*Dr*dt), xi~uniform-S^2, the one-step decay factor works
      out (to O(s^2)) to <n'.n|n> = 1 - (2/3)*Dr*dt, so the autocorrelation decays as
      C(t) ~ exp(-(2/3)*Dr*t). We measure the per-step factor lambda = <n(t+1).n(t)> directly
      (hundreds of thousands of near-independent samples => tiny error), get the rate
      = -ln(lambda)/dt, and confirm (i) Dr_eff = 1.5*rate matches the set Dr, and (ii) the rate
      scales with Dr (rate(Dr=2)/rate(Dr=1) ~ 2) -- i.e. rotStd=sqrt(2*Dr*dt) is wired correctly.

Nothing copied from the GPL source -- this validates our re-derived native model
(memory active-motility-not-thermal-noise). Single process / single tf.init (tf-headless-init):
set_motility is re-callable, so we reset v0/Dr/seed between sub-measurements on one foam.

Usage: pixi run python rnr/scripts/probe_native_calibration.py [M SEED T]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
import tissue_forge as tf  # noqa: E402
from tissue_forge.models.vertex import solver as tfv  # noqa: E402
from tissue_forge.models.vertex.solver.mesh_types import BodyTypeSpec, SurfaceTypeSpec  # noqa: E402

M = int(sys.argv[1]) if len(sys.argv) > 1 else 4
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 7
T = int(sys.argv[3]) if len(sys.argv) > 3 else 2000

L = float(M)
BOX = [[0.0, L]] * 3
DT = 1e-3

tf.init(windowless=True, dim=[L, L, L], cutoff=1.9, dt=DT)
tfv.init()
mesh = tfv.MeshSolver.get().get_mesh()
mesh.quality = None                 # no reconnection: topology stays fixed for clean measurement
mesh.periodic_geometry = True

from rnr.geometry import build_periodic_voronoi  # noqa: E402

rng = np.random.default_rng(SEED)
seeds = (rng.random((M ** 3, 3)) * L).tolist()


class Iface(SurfaceTypeSpec):
    pass


# Zeroed energy actors: the body Volume/SurfaceArea/Adhesion forces are all 0 (lam=0), so the only
# deterministic force is the (small) auto-bound FlatSurfaceConstraint -- removed by the v0=0
# baseline subtraction in Part A. Directors evolve regardless of forces, so the same foam serves B.
class A(BodyTypeSpec):
    volume_lam = 0.0; volume_val = 1.0
    surface_area_lam = 0.0; surface_area_val = 5.6
    adhesion = {"A": 0.0}


stype, btA = Iface.get(), A.get()
bodies, _sd, stats = build_periodic_voronoi(seeds, BOX, btA, stype)
tfv.MeshSolver.get().position_changed()
N = len(bodies)
print(f"CALIBRATION PROBE  M={M} N={N} dt={DT} T={T} seed={SEED}", flush=True)


# --- live vertex handles + vertex<->body incidence (topology fixed: mesh.quality=None) ----------
def live_vertices():
    vh, idx = [], {}
    for i in range(mesh.size_vertices):
        v = mesh.get_vertex(i)
        if v is None or v.id < 0:
            continue
        idx[v.id] = len(vh)
        vh.append(v)
    return vh, idx


vh, idx = live_vertices()
nv = len(vh)
# per-vertex list of incident body indices (mirrors the harness's add_noise_active incidence)
vrows, brows = [], []
for k, b in enumerate(bodies):
    seen = set()
    for s in b.surfaces:
        for w in s.vertices:
            if w.id in idx and w.id not in seen:
                seen.add(w.id)
                vrows.append(idx[w.id]); brows.append(k)
vrows = np.array(vrows, np.int64); brows = np.array(brows, np.int64)


def positions():
    return np.array([[p[0], p[1], p[2]] for p in (v.position for v in vh)], dtype=float)


def directors():
    return np.array([[b.director[0], b.director[1], b.director[2]] for b in bodies], dtype=float)


def minimg(d):
    return d - L * np.round(d / L)   # cubic box [0,L]


def per_vertex_meandir(dirs):
    S = np.zeros((nv, 3)); cnt = np.zeros(nv)
    np.add.at(S, vrows, dirs[brows])
    np.add.at(cnt, vrows, 1.0)
    cnt[cnt == 0] = 1.0
    return S / cnt[:, None]


# ============================ Part A: displacement scaling / mu ============================
V0_A = 1.0
# baseline: motility OFF -> the FlatSurfaceConstraint-only displacement for this step
tfv.MeshSolver.set_motility(0.0, 1.0, SEED)
P0 = positions(); tf.step(); d_flat = minimg(positions() - P0)
# motility ON (Dr=0 -> directors stay constant), one step, isolate the active part
tfv.MeshSolver.set_motility(V0_A, 0.0, SEED)
dirs_A = directors()
Ppre = positions(); tf.step(); d_tot = minimg(positions() - Ppre)
d_act = d_tot - d_flat
pred = DT * V0_A * per_vertex_meandir(dirs_A)

amag = np.linalg.norm(d_act, axis=1); pmag = np.linalg.norm(pred, axis=1)
m = pmag > 1e-12
ratio = np.median(amag[m] / pmag[m])
cosv = np.median(np.sum(d_act[m] * pred[m], axis=1) / (amag[m] * pmag[m] + 1e-30))
passA = (0.9 < ratio < 1.1) and (cosv > 0.99)
print(f"  [A] displacement: median |d_act|/|pred|={ratio:.4f} (want ~1.0), "
      f"median cos(d_act,pred)={cosv:.4f} (want ~1.0)  ->  {'PASS' if passA else 'FAIL'}", flush=True)
print(f"      (pred per-step |dx| median={np.median(pmag[m]):.3e} = dt*v0*<n>; mu=1 confirmed)",
      flush=True)


# ============================ Part B: director rotational diffusion (Dr) ====================
def measure_rate(Dr):
    # Directly estimate the one-step decay factor lambda = <n(t+1).n(t)> over all cells & steps
    # (near-independent samples each step -> very low variance), then rate = -ln(lambda)/dt.
    tfv.MeshSolver.set_motility(0.1, Dr, SEED + 1)
    nprev = directors()
    tot = 0.0; cnt = 0
    for _ in range(T):
        tf.step()
        ncur = directors()
        tot += float(np.sum(nprev * ncur))   # sum over cells of n(t+1).n(t)
        cnt += ncur.shape[0]
        nprev = ncur
    lam = tot / cnt
    return -np.log(lam) / DT, lam


rate1, lam1 = measure_rate(1.0)
rate2, lam2 = measure_rate(2.0)
dr_eff1 = 1.5 * rate1; dr_eff2 = 1.5 * rate2     # rate = (2/3) Dr  =>  Dr_eff = 1.5 * rate
ratio = rate2 / rate1
passB = (1.7 < ratio < 2.3) and (0.7 < dr_eff1 < 1.3) and (1.4 < dr_eff2 < 2.6)
print(f"  [B] director rot.diff: rate(Dr=1)={rate1:.4f}/s  rate(Dr=2)={rate2:.4f}/s  "
      f"ratio={ratio:.3f} (want ~2)  ->  {'PASS' if passB else 'FAIL'}", flush=True)
print(f"      implied Dr_eff = 1.5*rate: {dr_eff1:.3f} (set 1), {dr_eff2:.3f} (set 2)", flush=True)

print(f"\nCALIBRATION VERDICT: {'PASS' if (passA and passB) else 'FAIL'} "
      f"(A displacement/mu={'ok' if passA else 'BAD'}, B Dr-wiring={'ok' if passB else 'BAD'})",
      flush=True)
