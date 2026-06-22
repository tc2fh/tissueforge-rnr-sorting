"""Render a VIDEO of the faithful periodic active-motility sort demixing (clamp-free).

Self-contained: runs one periodic two-type sort (build_periodic_voronoi + native I<->H
reconnection + ACTIVE self-propulsion noise -- the faithful model, PORTING_NOTES §6n), captures a
matplotlib frame every CAPTURE_EVERY steps, then stitches the frames into a looping GIF (no ffmpeg
needed -- Pillow).

WHY matplotlib, not TF's renderer: TF's vertex renderer draws SOLID faces / ignores per-surface
hiding, so a native screenshot shows only the outer shell and hides the interior demixing
(sort_with_video.py header). Instead we render every cell's centroid coloured by TYPE and by
SORTEDNESS (unlike-neighbour fraction), with the demixing curves beside it -- robust + headless.

Usage: pixi run python rnr/scripts/video_periodic_active.py [N_STEPS] [SIGMA] [V0] [M] [SEED] [CAPTURE_EVERY] [DT] [LTH] [CUT]
       defaults: 40000 0.5 0.1 6 7 800 1e-3 1e-3 1.9   (~50 frames)
Output: rnr/exports/sort_active_frames/frame_#####.png  +  rnr/exports/sort_active_demixing.gif
"""
import glob
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
CAPTURE_EVERY = int(sys.argv[6]) if len(sys.argv) > 6 else 800
DT = float(sys.argv[7]) if len(sys.argv) > 7 else 1e-3
LTH = float(sys.argv[8]) if len(sys.argv) > 8 else 1e-3
CUT = float(sys.argv[9]) if len(sys.argv) > 9 else 1.9

L = float(M)
BOX = [[0.0, L]] * 3
V0, K_V, K_A, A0 = 1.0, 10.0, 1.0, 5.6
DR = 1.0
ROT_STD = float(np.sqrt(2.0 * DR * DT))
INTERVAL = 10
COLOR = {"A": "#d6336c", "B": "#1c7ed6"}

EXPORT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exports")
FRAMES = os.path.join(EXPORT, "sort_active_frames")
GIF = os.path.join(EXPORT, "sort_active_demixing.gif")

# Honour TF_THREADS (the sweep pins 1 thread/job; TF threading gives ~0 speedup at this scale).
_TFTHREADS = os.environ.get("TF_THREADS")
_init_kw = {"threads": int(_TFTHREADS)} if _TFTHREADS else {}
tf.init(windowless=True, dim=[L, L, L], cutoff=CUT, dt=DT, **_init_kw)
tfv.init()
mesh = tfv.MeshSolver.get().get_mesh()
mesh.quality = None
mesh.periodic_geometry = True

from rnr.geometry import build_periodic_voronoi  # noqa: E402
from rnr.metrics import contact_summary, type_name  # noqa: E402

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


def lam(i, j):
    return 0.0 if i == j else SIGMA


# --- active-motility noise (faithful; identical model to sort_periodic_oracle.py) -------------
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
    if V0_ACT <= 0:
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


def cell_fields():
    """Per-cell centroid, type, volume, and unlike-neighbour (het) fraction."""
    names = {b.id: type_name(b) for b in bodies}
    scope = set(names)
    cen, typ, vol, hetf = [], [], [], []
    for b in bodies:
        c = b.centroid
        cen.append([c[0], c[1], c[2]])
        typ.append(names[b.id])
        vol.append(abs(b.volume))
        nbrs = [nb for nb in b.connected_bodies if nb.id in scope]
        het = sum(1 for nb in nbrs if names[nb.id] != names[b.id])
        hetf.append(het / len(nbrs) if nbrs else 0.5)
    return np.array(cen), np.array(typ), np.array(vol), np.array(hetf)


HALF = L / 2.0
CTR = np.array([HALF, HALF, HALF])
hist = []          # (step, D, hetA)
frame_idx = 0


def _setup3d(ax, azim):
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_zlim(0, L)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=18, azim=azim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def capture(step):
    global frame_idx
    s = contact_summary(bodies=bodies, lam=lam)
    hist.append((step, s["demixing_index"], s["het_area_fraction"]))
    cen, typ, vol, hetf = cell_fields()
    size = 90.0 * (np.clip(vol, 0.2, None) / V0) ** (2.0 / 3.0)
    azim = (frame_idx * 1.4) % 360

    fig = plt.figure(figsize=(16.5, 5.6))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    for t in ("A", "B"):
        m = typ == t
        if m.any():
            ax1.scatter(cen[m, 0], cen[m, 1], cen[m, 2], s=size[m], c=COLOR[t],
                        alpha=0.85, edgecolors="white", linewidths=0.3, label=f"type {t}")
    _setup3d(ax1, azim); ax1.set_title("cell type"); ax1.legend(loc="upper right", fontsize=9)

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    sc = ax2.scatter(cen[:, 0], cen[:, 1], cen[:, 2], s=size, c=hetf, cmap="coolwarm",
                     vmin=0.0, vmax=1.0, alpha=0.9, edgecolors="white", linewidths=0.3)
    _setup3d(ax2, azim)
    ax2.set_title("sortedness: unlike-neighbour fraction\n(blue = sorted, red = mixed)")
    cb = fig.colorbar(sc, ax=ax2, fraction=0.03, pad=0.02); cb.set_label("het fraction", fontsize=8)

    ax3 = fig.add_subplot(1, 3, 3)
    st = [r[0] for r in hist]
    ax3.plot(st, [r[2] for r in hist], "-o", ms=3, color="#b3007a", label="het-area fraction")
    ax3.set_xlim(0, N_STEPS); ax3.set_ylim(0.40, 0.56)
    ax3.set_xlabel("step"); ax3.set_ylabel("heterotypic interface-area fraction")
    ax3.set_title("interface-area demixing (down = more sorted)")
    ax3.annotate(f"step {step}\nhetA = {hist[-1][2]:.3f}\nD = {hist[-1][1]:+.3f}\n"
                 f"min vol = {min(vol):.2f}",
                 xy=(0.96, 0.96), xycoords="axes fraction", fontsize=10, va="top", ha="right",
                 bbox=dict(boxstyle="round", fc="#fff3bf", ec="#e6a700"))
    ax3.grid(alpha=0.3); ax3.legend(loc="lower left", fontsize=9)

    fig.suptitle(f"Faithful active-motility periodic sort in TissueForge 3D vertex "
                 f"($\\sigma$={SIGMA:g}, N={len(bodies)}, clamp-free) — PORTING_NOTES §6n", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FRAMES, f"frame_{frame_idx:05d}.png"), dpi=96)
    plt.close(fig)
    frame_idx += 1
    print(f"  frame {frame_idx:3d} @ step {step:6d}: hetA={hist[-1][2]:.4f} D={hist[-1][1]:+.4f}",
          flush=True)


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
    imgs = [Image.open(p).convert("P", palette=Image.ADAPTIVE, colors=128) for p in paths]
    imgs[0].save(GIF, save_all=True, append_images=imgs[1:], duration=120, loop=0, optimize=True)
    mb = os.path.getsize(GIF) / 1e6
    print(f"wrote {GIF} ({len(imgs)} frames, {mb:.1f} MB)", flush=True)


if os.path.isdir(FRAMES):
    shutil.rmtree(FRAMES)
os.makedirs(FRAMES, exist_ok=True)

print(f"=== ACTIVE SORT VIDEO: N={len(bodies)} sigma={SIGMA} v0={V0_ACT} M={M} steps={N_STEPS} "
      f"| frame every {CAPTURE_EVERY} ===", flush=True)
capture(0)
recon = 0
nv_prev = mesh.num_vertices
for i in range(1, N_STEPS + 1):
    add_noise_active()
    tf.step()
    nv = mesh.num_vertices
    if nv != nv_prev:
        recon += abs(nv - nv_prev); nv_prev = nv
    if i % CAPTURE_EVERY == 0 or i == N_STEPS:
        capture(i)

print(f"\nDONE: {frame_idx} frames; hetA {hist[0][2]:.4f} -> {hist[-1][2]:.4f}; "
      f"{recon} reconnections", flush=True)
stitch_gif()
