"""Faithful periodic-bulk sort with the 3DVertVor ORACLE's exact parameters (derisk P5).

Goal (user-chosen path 2026-06-09): before paying for the deep engine periodic-ghost fix,
confirm whether the periodic bulk CAN demix at all, using the stable small-cutoff substrate
(memory `periodic-substrate-engine-bug`) + manual Python thermal noise as a stand-in for the
engine noise (which dies at small cutoff).

ORACLE params (read from 3DVertVor: Energy/Volume.cpp kv_=10, Energy/Interface.cpp
tension=2(A_cell-A0)+sigma & s0_=5.4 [paper 5.6], Reconnection/Reconnection.cpp Lth_=1e-3,
dtr=10*dt, main.py: N=L^3 random points in [0,L]^3 so V0=1):
  V0 = 1, L = N^(1/3)            (1728 cells in 12^3 in the paper)
  K_V = 10 (volume modulus)      -> TF volume_lam = K_V (TF force uses lam*(V-V0), oracle 2*kv -> see note)
  K_A = 1  (area modulus)        -> TF surface_area_lam = K_A
  A0 = s0 = 5.6 (paper) [5.4 oracle default]
  sigma_ij in {0.04,0.08,0.16,0.32,0.64} (Fig2) or {0.1,0.2,0.5} (Fig1); homotypic 0
  Lth = 1e-3 (gentle), reconnect every dtr=10*dt -> reconnect_interval=10
  noise = white thermal, kT=0.1, mu=1   (manual Euler-Maruyama: dx ~ N(0, sqrt(2*mu*kT*dt)))

NOTE on the TF vs oracle modulus factor of 2: oracle pressure = -2*kv*(V-V0), tension =
2*(A-A0). TF VolumeConstraint/SurfaceAreaConstraint force = lam*(V-V0)*gradV etc. (the 2 and
the gradient sign are folded into the actor). We set volume_lam=K_V, surface_area_lam=K_A and
note any factor-of-2 as a DEPARTURE (it only rescales the timescale, not the equilibrium sort).

DEPARTURES (flagged per CLAUDE.md):
  * small cutoff (engine periodic-ghost bug workaround) -> engine noise dead -> manual noise.
  * manual noise is isotropic per-axis Gaussian displacement (faithful effective temperature).
  * TF native I<->H reconnection (re-derived Okuda) vs oracle's; energy gate OFF.
  * N < 1728 (a true periodic bulk, just smaller) unless you ask for N=1728.

Usage: pixi run python rnr/scripts/sort_periodic_oracle.py \
         [MODE] [M] [SIGMA] [KT] [LTH] [DT] [CUT] [NSTEPS] [SEED]
  MODE = substrate (sigma forced 0, no noise, no recon -> stability scan) | sort
  M    = cells per axis (N = M^3, L = M). default 6 -> N=216
defaults: sort 6 0.5 0.1 1e-3 0.005 0.3 20000 7
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
import tissue_forge as tf  # noqa: E402
from tissue_forge.models.vertex import solver as tfv  # noqa: E402
from tissue_forge.models.vertex.solver.mesh_types import BodyTypeSpec, SurfaceTypeSpec  # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else "sort"
M = int(sys.argv[2]) if len(sys.argv) > 2 else 6           # N = M^3 cells, L = M (V0=1)
SIGMA = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
KT = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
LTH = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-3
DT = float(sys.argv[6]) if len(sys.argv) > 6 else 5e-3
CUT = float(sys.argv[7]) if len(sys.argv) > 7 else 0.3
NSTEPS = int(sys.argv[8]) if len(sys.argv) > 8 else 20000
SEED = int(sys.argv[9]) if len(sys.argv) > 9 else 7
# rel_frac of the min-image nearest-neighbour distance to which each vertex's THERMAL NOISE
# displacement is capped (trust-region). 0 disables (the old, UNSTABLE behaviour). See the
# NOISE-CLAMP note below and PORTING_NOTES.
NOISE_CLAMP = float(sys.argv[10]) if len(sys.argv) > 10 else 0.4
# Initial condition for MODE=sort: "mixed" (random 50/50, Fig 1E) or "demixed" (a segregated
# z-slab = the DP_max config; Fig 1F energetic-preference test -- does it STAY demixed?).
IC = sys.argv[11] if len(sys.argv) > 11 else "mixed"
# Noise model. "active" (FAITHFUL, Python comparison path): the 3DVertVor/Manning fork's ACTIVE self-propulsion
# (Run.cpp:1345 `x += dt*motility`, motility = temperature*<cell director>, directors rotate with
# rotational diffusion Dr=1). Per-step displacement ~ dt*V0 << Lth, so the instantaneous-edge
# reconnect trigger catches collapses with NO clamp. "thermal": the old Euler-Maruyama sqrt(dt)
# Brownian kick (the fork's thermal line Run.cpp:1344 is COMMENTED OUT) -- it is 14-45x Lth per
# step and starves reconnection unless propped up by NOISE_CLAMP (the departure this replaces).
# See PORTING_NOTES §6n. Under "active", KT is reused as the active speed V0 and NOISE_CLAMP is
# ignored (the model needs none). "native" (PORTING_NOTES §6o): the SAME active model, but run
# inside the C++ engine (per-cell Body director + per-vertex active force in MeshSolver) via
# MeshSolver.set_motility -- no Python per-step injection. native and active are statistically
# equivalent (same seed -> matching demixing/reconnection rate); native is the DEFAULT/production
# path (PORTING_NOTES §6o gate 6). Pass "active" for the Python-injection comparison, "thermal" legacy.
NOISE_MODEL = sys.argv[12] if len(sys.argv) > 12 else "native"

if MODE == "substrate":
    SIGMA = 0.0; KT = 0.0

L = float(M)
BOX = [[0.0, L]] * 3
V0 = 1.0
K_V = 10.0
K_A = 1.0
S0 = 5.6
A0 = S0 * V0 ** (2.0 / 3.0)        # = 5.6
MU = 1.0
DISP_STD = float(np.sqrt(2.0 * MU * KT * DT)) if (KT > 0 and NOISE_MODEL == "thermal") else 0.0
# active-motility params (faithful default): V0 == temperature, director rotational diffusion Dr=1
V0_ACT = KT if NOISE_MODEL in ("active", "native") else 0.0
DR = 1.0
ROT_STD = float(np.sqrt(2.0 * DR * DT))
MAX_VOL_FAC = 4.0
INTERVAL = 10

EXPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exports")
tag = (f"oracle_M{M}_S{SIGMA:g}_KT{KT:g}_L{LTH:g}_dt{DT:g}_cut{CUT:g}_seed{SEED}"
       + ("" if NOISE_MODEL == "thermal" else f"_{NOISE_MODEL}")  # thermal keeps the legacy name
       + ("" if IC == "mixed" else f"_{IC}"))

tf.init(windowless=True, dim=[L, L, L], cutoff=CUT, dt=DT)
tfv.init()
mesh = tfv.MeshSolver.get().get_mesh()
mesh.quality = None
mesh.periodic_geometry = True

from rnr.geometry import build_periodic_voronoi  # noqa: E402
from rnr.metrics import contact_summary  # noqa: E402

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
vinit = np.array([b.volume for b in bodies])

rng2 = np.random.default_rng(SEED + 1)
if MODE == "sort":
    if IC == "demixed":
        # Fig 1F IC: a maximally segregated slab (z-cut through the box centre == the DP_max
        # config of compute_dpmax.py). DP(0) ~ DP_max; the test is whether the het tension keeps
        # it demixed (energetic preference) rather than mixing back toward DP~0.
        half = L / 2.0
        for b in [bb for bb in bodies if bb.centroid[2] >= half]:
            b.become(btB)
    else:  # mixed (Fig 1E IC)
        for b in bodies:
            if rng2.random() < 0.5:
                b.become(btB)

if MODE != "substrate":
    q = tfv.Quality()
    q.stock_quality_operations = False
    q.reconnect_length = LTH
    q.reconnect_hysteresis = 0.2
    q.reconnect_energy_gate = False
    q.reconnect_interval = INTERVAL
    q.collision_2d = False
    mesh.quality = q
    mesh.periodic_geometry = True

# NATIVE active drive (PORTING_NOTES §6o): hand the active self-propulsion to the C++
# engine -- a per-cell director (rotational diffusion Dr) + a per-vertex active force
# v0*<incident-cell directors>, evaluated inside tf.step(). The Python add_noise_active
# injection is then NOT used (noise_step() is a no-op below). One call, before the loop:
# directors are seeded random-on-S^2 here and evolve in the engine each step.
if NOISE_MODEL == "native" and V0_ACT > 0:
    tfv.MeshSolver.set_motility(V0_ACT, DR, SEED + 2)


def lam(i, j):
    return 0.0 if i == j else SIGMA


# --- NOISE CLAMP (the fix for the periodic-sort reconnection blow-up) -----------------
# Root cause (diagnosed 2026-06-10, scripts/diag_recon_overshoot.py + diag_read_side_effect.py):
# a native I->H reconnection places two new vertices Lth=1e-3 apart, but one Euler-Maruyama
# thermal kick is DISP_STD=sqrt(2*mu*kT*dt)=0.0141 -- ~14x that gap. So a single noise step
# throws a freshly-reconnected vertex clean past its neighbour, everting the cell (signed
# volume goes negative). A clean factorial showed the blow-up needs BOTH reconnection AND
# noise (each alone is stable). FIX: a per-vertex TRUST-REGION on the noise -- cap each
# vertex's noise displacement at NOISE_CLAMP * (min-image nearest-neighbour distance). For a
# normal vertex (nn~0.5) the cap ~0.2 >> 0.014 so it never binds; it binds only on the
# near-degenerate fresh-reconnection vertices. This is the position-level analogue of the
# proven Python operator.stable_step clamp (memory winding-clamp-stabilizes-sort) and of the
# oracle's orientation-repair stabilizer; flagged as a DEPARTURE (a locally-adaptive timestep
# safeguard, not in naive Euler-Maruyama). The eventual C++ port should put this trust-region
# in the vertex integrator so it also covers ENGINE noise + post-reconnection force overshoot.
#
# Implementation: the implicit-edge topology is cached and only rebuilt when num_vertices
# changes (i.e. after a reconnection), so the per-step cost is O(V) reads + O(E) vectorised.
_ncache = {"nv": None, "vh": [], "edges": None}


def _rebuild_noise_topology():
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
        _rebuild_noise_topology()
    vh = _ncache["vh"]
    edges = _ncache["edges"]
    n = len(vh)
    P = np.array([[p[0], p[1], p[2]] for p in (v.position for v in vh)], dtype=float)
    dx = rng2.normal(0.0, DISP_STD, (n, 3))
    if NOISE_CLAMP > 0 and len(edges):
        d = P[edges[:, 0]] - P[edges[:, 1]]
        d -= L * np.round(d / L)                        # min-image (cubic box [0,L])
        elen = np.linalg.norm(d, axis=1)
        nn = np.full(n, np.inf)
        np.minimum.at(nn, edges[:, 0], elen)
        np.minimum.at(nn, edges[:, 1], elen)
        cap = NOISE_CLAMP * nn
        mag = np.linalg.norm(dx, axis=1)
        scale = np.where((mag > cap) & (mag > 0), cap / np.maximum(mag, 1e-30), 1.0)
        dx *= scale[:, None]
    newP = (P + dx) % L
    for v, q in zip(vh, newP):
        v.set_position(tf.FVector3(float(q[0]), float(q[1]), float(q[2])))


# --- ACTIVE MOTILITY (FAITHFUL default) ------------------------------------------------
# Re-derived from the 3DVertVor/Manning fork's dynamics (NOT copied -- GPL): each CELL carries a
# director n_c in S^2 that rotates with active-Brownian rotational diffusion (Dr=1); each vertex
# gets motility velocity u_v = V0 * mean_{c incident to v} n_c (Vertex.cpp:78-86); the position
# advances ballistically x_v += dt*u_v (Run.cpp:1345). dt-scaled (NOT sqrt(dt)) => per-step
# displacement <= dt*V0 << Lth, so reconnections are caught with NO clamp. See PORTING_NOTES §6n.
rng_dir = np.random.default_rng(SEED + 2)
_dirs = rng_dir.normal(0.0, 1.0, (len(bodies), 3))
_dirs /= np.linalg.norm(_dirs, axis=1, keepdims=True)
# incidence cache. A single doQuality pass can do several I->H(+1 vert)/H->I(-1) that NET to zero
# count change, so num_vertices is an unsafe staleness signal (a cached deleted handle then
# segfaults). Rebuild every step from the LIVE mesh -- cheap relative to tf.step, and bulletproof.
_amc = {"vh": [], "vrow": None, "brow": None, "n": 0}


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
    _amc.update(vh=vh, vrow=np.array(vrow, np.int64), brow=np.array(brow, np.int64), n=len(vh))


def add_noise_active():
    if V0_ACT <= 0:
        return
    _rebuild_incidence()
    # active-Brownian rotational diffusion of each cell director (sqrt(dt) on ORIENTATION only)
    xi = rng_dir.normal(0.0, 1.0, _dirs.shape)
    xi /= np.linalg.norm(xi, axis=1, keepdims=True)
    _dirs[:] = _dirs + ROT_STD * (xi - _dirs)
    _dirs[:] /= np.linalg.norm(_dirs, axis=1, keepdims=True)

    vh, vrow, brow, n = _amc["vh"], _amc["vrow"], _amc["brow"], _amc["n"]
    S = np.zeros((n, 3))
    np.add.at(S, vrow, _dirs[brow])               # sum incident-cell directors per vertex
    cnt = np.zeros(n)
    np.add.at(cnt, vrow, 1.0)
    cnt[cnt == 0] = 1.0
    u = V0_ACT * (S / cnt[:, None])               # per-vertex motility velocity
    dx = DT * u                                   # ballistic translation: scales as dt (NOT sqrt(dt))
    P = np.array([[p[0], p[1], p[2]] for p in (v.position for v in vh)], dtype=float)
    newP = (P + dx) % L
    for v, q in zip(vh, newP):
        v.set_position(tf.FVector3(float(q[0]), float(q[1]), float(q[2])))


def noise_step():
    if NOISE_MODEL == "native":
        return  # the C++ engine drives motility inside tf.step() (PORTING_NOTES §6o)
    if NOISE_MODEL == "active":
        add_noise_active()
    else:
        add_noise()


if NOISE_MODEL in ("active", "native"):
    _noise_desc = (f"{NOISE_MODEL}(V0={V0_ACT:g}, per-step<= {DT * V0_ACT:.2e}="
                   f"{DT * V0_ACT / LTH:.3f}xLth, Dr={DR:g})")
else:
    _noise_desc = f"thermal(disp_std={DISP_STD:.4g}, clamp={NOISE_CLAMP})"
print(f"=== ORACLE-FAITHFUL PERIODIC {MODE}/{IC} | M={M} N={len(bodies)} L={L} V0={V0} "
      f"K_V={K_V} K_A={K_A} A0={A0:.2f} | sigma={SIGMA} kT={KT} noise={NOISE_MODEL}:{_noise_desc} "
      f"Lth={LTH} dt={DT} cutoff={CUT} INT={INTERVAL} "
      f"steps={NSTEPS} ===", flush=True)
print(f"  foam: surf={stats['n_surfaces']} verts={stats['n_vertices']} wrap={stats['n_wrap_faces']} "
      f"| vol[min={vinit.min():.3f} max={vinit.max():.3f} mean={vinit.mean():.3f}]", flush=True)

nv_prev = mesh.num_vertices
recon = 0
worst_min = 1e9
worst_max = 0.0
rows = []


def checkpoint(step):
    global worst_min, worst_max
    s = contact_summary(bodies=bodies, lam=lam)
    vs = [b.volume for b in bodies]
    mn, mx = min(vs), max(vs)
    worst_min = min(worst_min, mn); worst_max = max(worst_max, mx)
    rows.append((step, s["demixing_index"], s["het_area_fraction"], s["het_pairs"],
                 s["total_pairs"], mn, mx, recon))
    print(f"  step {step:6d}: D={s['demixing_index']:+.4f} hetA={s['het_area_fraction']:.4f} "
          f"het_pairs={s['het_pairs']}/{s['total_pairs']} min_vol={mn:.3f} max_vol={mx:.3f} "
          f"recon~{recon}", flush=True)


checkpoint(0)
CKPT = max(500, NSTEPS // 40)
for step in range(1, NSTEPS + 1):
    noise_step()
    tf.step()
    nv = mesh.num_vertices
    if nv != nv_prev:
        recon += abs(nv - nv_prev); nv_prev = nv
    if step % CKPT == 0 or step == NSTEPS:
        checkpoint(step)
        mn = rows[-1][5]; mx = rows[-1][6]
        if mn <= 0 or mx > MAX_VOL_FAC * V0:
            print("  -> UNSTABLE; stopping.", flush=True); break

ok = worst_min > 0 and worst_max <= MAX_VOL_FAC * V0
D0, Dend = rows[0][1], rows[-1][1]
print(f"\nVERDICT [{tag}]: {'STABLE' if ok else 'UNSTABLE'} (worst_min={worst_min:.3f} "
      f"worst_max={worst_max:.3f}) | D {D0:+.4f} -> {Dend:+.4f} | "
      f"hetA {rows[0][2]:.4f} -> {rows[-1][2]:.4f} | {recon} reconnections", flush=True)

if MODE == "sort" and rows:
    os.makedirs(EXPORT_DIR, exist_ok=True)
    csv = os.path.join(EXPORT_DIR, f"sort_{tag}.csv")
    with open(csv, "w") as fh:
        fh.write("step,D,het_area,het_pairs,total_pairs,min_vol,max_vol,recon\n")
        for r in rows:
            fh.write(",".join(f"{x:.6g}" for x in r) + "\n")
    print(f"wrote {csv}", flush=True)
