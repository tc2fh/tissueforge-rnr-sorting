"""Render a GIF that tracks EVERY VERTEX (not cell centroids), each a unique persistent colour, so
you can watch the individual vertices move + reconnect during the sort.

Same physics as sort_periodic_oracle.py (periodic two-type Voronoi bulk + native I<->H reconnection +
active-motility drive). Default MODEL=native (the C++ engine drive, MeshSolver.set_motility); pass
"active" for the Python-injection model. Unlike video_periodic_active.py (which plots one dot per
CELL, coloured by type/sortedness), this plots one dot per MESH VERTEX coloured by a STABLE id->colour
map, with a FIXED camera so the only motion you see is the vertices themselves.

Colour identity: each vertex id gets a fixed bright colour the first time it is seen and keeps it for
life. Vertices are SHARED between cells, so each physical vertex is drawn once. Reconnections (I<->H)
create/destroy vertices -- a newly created vertex gets a fresh colour (so a burst of new colours marks
reconnection activity); a destroyed vertex's colour simply disappears.

Periodic note: positions are drawn raw (in the box); a vertex that crosses a periodic face will
appear to jump to the opposite side (truthful to the periodic BC, not a bug).

Usage: pixi run video-vertices [N_STEPS] [SIGMA] [V0] [M] [SEED] [CAPTURE_EVERY] [MODEL] [ROTATE] [DT] [LTH] [CUT]
       defaults: 40000 0.5 0.1 6 7 500 native 0 1e-3 1e-3 1.9   (~80 frames, fixed camera)
       ROTATE=1 slowly spins the view (default 0 = fixed, best for tracking vertex motion).
Output: rnr/exports/vertex_motion_frames/frame_#####.png  +  rnr/exports/vertex_motion_<MODEL>.gif
"""
import colorsys
import glob
import itertools
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tissue_forge as tf  # noqa: E402
from tissue_forge.models.vertex import solver as tfv  # noqa: E402
from tissue_forge.models.vertex.solver.mesh_types import BodyTypeSpec, SurfaceTypeSpec  # noqa: E402

N_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
SIGMA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
V0_ACT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
M = int(sys.argv[4]) if len(sys.argv) > 4 else 6
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 7
CAPTURE_EVERY = int(sys.argv[6]) if len(sys.argv) > 6 else 500
MODEL = sys.argv[7] if len(sys.argv) > 7 else "native"     # native (engine drive) | active (Python)
ROTATE = (sys.argv[8] if len(sys.argv) > 8 else "0") in ("1", "true", "True")
DT = float(sys.argv[9]) if len(sys.argv) > 9 else 1e-3
LTH = float(sys.argv[10]) if len(sys.argv) > 10 else 1e-3
CUT = float(sys.argv[11]) if len(sys.argv) > 11 else 1.9

L = float(M)
BOX = [[0.0, L]] * 3
V0, K_V, K_A, A0 = 1.0, 10.0, 1.0, 5.6
DR = 1.0
ROT_STD = float(np.sqrt(2.0 * DR * DT))
INTERVAL = 10

EXPORT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exports")
FRAMES = os.path.join(EXPORT, "vertex_motion_frames")
GIF = os.path.join(EXPORT, f"vertex_motion_{MODEL}.gif")

_TFTHREADS = os.environ.get("TF_THREADS")
_init_kw = {"threads": int(_TFTHREADS)} if _TFTHREADS else {}
tf.init(windowless=True, dim=[L, L, L], cutoff=CUT, dt=DT, **_init_kw)
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

if MODEL == "native":
    tfv.MeshSolver.set_motility(V0_ACT, DR, SEED + 2)   # native C++ active drive (no Python injection)

# --- active-motility injection (only used when MODEL == "active"; identical model to the oracle) ---
rng_dir = np.random.default_rng(SEED + 2)
_dirs = rng_dir.normal(0.0, 1.0, (len(bodies), 3))
_dirs /= np.linalg.norm(_dirs, axis=1, keepdims=True)
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
    if MODEL != "active" or V0_ACT <= 0:
        return
    _rebuild_incidence()
    xi = rng_dir.normal(0.0, 1.0, _dirs.shape)
    xi /= np.linalg.norm(xi, axis=1, keepdims=True)
    _dirs[:] = _dirs + ROT_STD * (xi - _dirs)
    _dirs[:] /= np.linalg.norm(_dirs, axis=1, keepdims=True)
    vh, vrow, brow, n = _amc["vh"], _amc["vrow"], _amc["brow"], _amc["n"]
    S = np.zeros((n, 3))
    np.add.at(S, vrow, _dirs[brow])
    cnt = np.zeros(n)
    np.add.at(cnt, vrow, 1.0)
    cnt[cnt == 0] = 1.0
    dx = DT * V0_ACT * (S / cnt[:, None])
    P = np.array([[p[0], p[1], p[2]] for p in (v.position for v in vh)], dtype=float)
    newP = (P + dx) % L
    for v, qq in zip(vh, newP):
        v.set_position(tf.FVector3(float(qq[0]), float(qq[1]), float(qq[2])))


# --- persistent per-vertex-id colours -------------------------------------------------------------
_color_rng = np.random.default_rng(SEED + 99)
_vcolor = {}


def color_for(vid):
    c = _vcolor.get(vid)
    if c is None:
        h = _color_rng.random()                       # random hue, high sat/val -> bright + distinct
        s = 0.60 + 0.40 * _color_rng.random()
        val = 0.80 + 0.20 * _color_rng.random()
        c = colorsys.hsv_to_rgb(float(h), float(s), float(val))
        _vcolor[vid] = c
    return c


def live_vertices():
    ids, pos = [], []
    for i in range(mesh.size_vertices):
        v = mesh.get_vertex(i)
        if v is None or v.id < 0:
            continue
        p = v.position
        ids.append(int(v.id)); pos.append([p[0], p[1], p[2]])
    return np.array(ids, dtype=np.int64), np.array(pos, dtype=float)


_BOX_PTS = np.array(list(itertools.product([0.0, L], repeat=3)))
_BOX_EDGES = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
              (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
frame_idx = 0


def capture(step, recon):
    global frame_idx
    ids, P = live_vertices()
    cols = np.array([color_for(i) for i in ids]) if len(ids) else np.zeros((0, 3))
    azim = (-60 + frame_idx * 1.2) % 360 if ROTATE else -60

    fig = plt.figure(figsize=(9.5, 9))
    ax = fig.add_subplot(111, projection="3d")
    for a, b in _BOX_EDGES:
        ax.plot(*zip(_BOX_PTS[a], _BOX_PTS[b]), color="0.75", lw=0.6, alpha=0.5)
    if len(P):
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=16, c=cols, depthshade=True,
                   edgecolors="none", alpha=0.95)
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_zlim(0, L)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=18, azim=azim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(f"per-vertex motion — {MODEL} drive, $\\sigma$={SIGMA:g}, N={len(bodies)} cells\n"
                 f"step {step}   |   {len(ids)} vertices   |   {recon} reconnections",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FRAMES, f"frame_{frame_idx:05d}.png"), dpi=100)
    plt.close(fig)
    frame_idx += 1
    print(f"  frame {frame_idx:3d} @ step {step:6d}: {len(ids)} verts, {recon} recon", flush=True)


def stitch_gif():
    paths = sorted(glob.glob(os.path.join(FRAMES, "frame_*.png")))
    if not paths:
        print("no frames to stitch", flush=True)
        return
    try:
        from PIL import Image
    except Exception as e:
        print(f"PIL unavailable ({e}); frames saved but no GIF.", flush=True)
        return
    imgs = [Image.open(p).convert("P", palette=Image.ADAPTIVE, colors=256) for p in paths]
    imgs[0].save(GIF, save_all=True, append_images=imgs[1:], duration=120, loop=0, optimize=True)
    print(f"wrote {GIF} ({len(imgs)} frames, {os.path.getsize(GIF) / 1e6:.1f} MB)", flush=True)


if os.path.isdir(FRAMES):
    shutil.rmtree(FRAMES)
os.makedirs(FRAMES, exist_ok=True)

print(f"=== VERTEX-MOTION VIDEO [{MODEL}]: N={len(bodies)} sigma={SIGMA} v0={V0_ACT} M={M} "
      f"steps={N_STEPS} | frame every {CAPTURE_EVERY} | {'rotating' if ROTATE else 'fixed'} camera ===",
      flush=True)
capture(0, 0)
recon = 0
nv_prev = mesh.num_vertices
for i in range(1, N_STEPS + 1):
    add_noise_active()          # no-op for MODEL=native
    tf.step()
    nv = mesh.num_vertices
    if nv != nv_prev:
        recon += abs(nv - nv_prev); nv_prev = nv
    if i % CAPTURE_EVERY == 0 or i == N_STEPS:
        capture(i, recon)

print(f"\nDONE: {frame_idx} frames; {recon} reconnections; {len(_vcolor)} distinct vertex ids seen",
      flush=True)
stitch_gif()
