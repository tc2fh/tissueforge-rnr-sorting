"""Render the native sort as a WHOLE-CELL-by-type video (each cell one solid colour).

WHY THIS EXISTS (the per-cell-type-colour fix):
The GL / matplotlib face videos (`video_native_gl.py`, `shot_native_sort.py`, `watch_native_sort_window.py`)
colour each cell-cell *surface* by the TYPE PAIR it separates (A|A blue, B|B red, A|B gold). That is the
right way to watch *interfaces* shrink, but it makes every individual cell look two-toned (an A cell shows
blue faces where it meets other A cells and gold faces where it meets B cells), so you cannot cleanly read
the type-A vs type-B *populations*. A cell-cell surface is shared by two cells, so in a solid-face renderer
a single A|B face physically cannot be both colours -- the interface colouring is the only honest per-face
answer.

To colour by WHOLE CELL we instead render each cell as its OWN solid polyhedron, shrunk slightly toward
its centroid so neighbours separate with a thin gap, and paint ALL of that cell's faces with that cell's
type colour. Every cell is then unambiguously one colour (A = royal blue, B = red), and you watch the two
populations demix. This is camera-independent and drawn straight from the mesh geometry with matplotlib
(Agg, no GL), so it also sidesteps the WSL2 GL driver issues (`docs/renderer_notes.md`).

PALETTE: A = royal blue (#1c4fd6), B = red (#d62828) -- the SAME A/B convention as the GL/shot/watch
face videos, so colours mean the same type across every 3D view. (`video_periodic_active.py`'s centroid
plot still uses its own pink/blue palette; unify separately if desired.)

Physics are identical to `video_native_gl.py` / `sort_periodic_oracle.py MODEL=native`: a periodic
two-type Voronoi bulk + native I<->H reconnection + the C++ active-motility drive.

Run:
    pixi run video-cells [N_STEPS] [SIGMA] [V0] [M] [SEED] [CAPTURE_EVERY] [MODEL] [IC] [DT] [LTH] [CUT]
    defaults: 20000 0.5 0.1 6 7 250 native mixed 1e-3 1e-3 1.9      (~80 frames)
  IC      = mixed (sort from random) | demixed (pre-sorted z-slab, watch it hold)
  MODEL   = native (engine drive) | active (Python injection)
  Env:
    SHRINK=0.85   per-cell shrink toward centroid (smaller = bigger gaps between cells).
    CLIP=z        FIXED clip plane: keep only the half the normal points toward, so you see a
                  cross-section. <sign?><axis>: x y z +x -z ... (sign = kept side); CLIP_AT=0.5
                  sets the plane position as a box fraction. There is no GL here (matplotlib/Agg),
                  so each cell FACE is polygon-clipped against the plane (rnr.clip) -- same
                  point+normal convention as the native-GL video, so the two cut identically.
    ELEV=.. AZIM=-60   camera angles (deg). ELEV defaults to 32 when a clip plane is set (look
                  DOWN into a top cut, e.g. CLIP=-z), else 20.
    ROTATE=1      turntable (azim advances 3 deg/frame) to read the 3D structure. Default `auto`
                  = ON whenever CLIP is set, so you orbit and see into the cut from every side.
    FPS=20        output frame rate.
Output: rnr/exports/native_cells_frames/frame_#####.png
        + rnr/exports/native_cells_sort_<MODEL>_<IC>.mp4   (falls back to .gif if ffmpeg is missing).
        When CLIP is set the basename gains a `_clip<spec>` tag.
"""
import glob
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
import tissue_forge as tf  # noqa: E402
from tissue_forge.models.vertex import solver as tfv  # noqa: E402
from tissue_forge.models.vertex.solver.mesh_types import BodyTypeSpec, SurfaceTypeSpec  # noqa: E402

from rnr.clip import clip_polygon_halfspace, parse_clip_env  # noqa: E402

N_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
SIGMA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
V0_ACT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
M = int(sys.argv[4]) if len(sys.argv) > 4 else 6
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 7
CAPTURE_EVERY = int(sys.argv[6]) if len(sys.argv) > 6 else 250
MODEL = sys.argv[7] if len(sys.argv) > 7 else "native"     # native (engine drive) | active (Python)
IC = sys.argv[8] if len(sys.argv) > 8 else "mixed"         # mixed = sort from random | demixed = slab
DT = float(sys.argv[9]) if len(sys.argv) > 9 else 1e-3
LTH = float(sys.argv[10]) if len(sys.argv) > 10 else 1e-3
CUT = float(sys.argv[11]) if len(sys.argv) > 11 else 1.9

SHRINK = float(os.environ.get("SHRINK", "0.85"))
_ROTATE_ENV = os.environ.get("ROTATE", "auto").lower()   # auto = turntable ON when a clip plane is active
AZIM = float(os.environ.get("AZIM", "-60"))
FPS = int(os.environ.get("FPS", "20"))

L = float(M)
BOX = [[0.0, L]] * 3
V0, K_V, K_A, A0 = 1.0, 10.0, 1.0, 5.6
DR = 1.0
ROT_STD = float(np.sqrt(2.0 * DR * DT))
INTERVAL = 10
COL = {"A": "#1c4fd6", "B": "#d62828"}   # royal blue / red -- same A/B meaning as the face videos

_CLIP_SPEC = os.environ.get("CLIP", "").strip().lower()
CLIP = parse_clip_env(L)                        # (point, normal) lists or None -- per-face polygon clip
_CLIP_TAG = ("_clip" + _CLIP_SPEC.replace("+", "p").replace("-", "m")) if CLIP is not None else ""
# When a clip plane is active, default to a turntable looking further DOWN so you orbit and
# see INTO the cut (cut-on-top for CLIP=-z); a fixed view shows the cross-section from one side.
ELEV = float(os.environ.get("ELEV", "32" if CLIP is not None else "20"))
ROTATE = (CLIP is not None) if _ROTATE_ENV == "auto" else (_ROTATE_ENV in ("1", "true", "yes", "on"))

EXPORT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exports")
# Tag the frames dir too (not just the mp4) so a clipped render never clobbers a
# concurrent un-clipped one -- each (clip) variant gets its own scratch dir.
FRAMES = os.path.join(EXPORT, "native_cells_frames" + _CLIP_TAG)
OUT_BASE = os.path.join(EXPORT, f"native_cells_sort_{MODEL}_{IC}{_CLIP_TAG}")

_TFTHREADS = os.environ.get("TF_THREADS")
_init_kw = {"threads": int(_TFTHREADS)} if _TFTHREADS else {}
tf.init(windowless=True, dim=[L, L, L], cutoff=CUT, dt=DT, **_init_kw)   # GL never used; Agg renders
tfv.init()
mesh = tfv.MeshSolver.get().get_mesh()
mesh.quality = None
mesh.periodic_geometry = True

from rnr.geometry import build_periodic_voronoi  # noqa: E402
from rnr.metrics import demixing_index, type_name  # noqa: E402

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
if IC == "demixed":
    for b in [bb for bb in bodies if bb.centroid[2] >= L / 2.0]:
        b.become(btB)
else:
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
    tfv.MeshSolver.set_motility(V0_ACT, DR, SEED + 2)   # native C++ active drive


def _pos(v):
    p = v.position
    return [p[0], p[1], p[2]]


def cell_polys():
    """Each cell -> its faces as shrunk polygons (periodic-unwrapped to the cell's own image),
    coloured by the cell's TYPE. Returns (polys, base_rgb_per_poly, centroids_per_poly)."""
    polys, base, cens = [], [], []
    for b in bodies:
        bv = b.vertices
        if not bv:
            continue
        P = np.array([_pos(v) for v in bv], dtype=float)
        # primary-image centroid: unwrap the cell's vertices relative to one anchor, then average
        Pun = P - L * np.round((P - P[0]) / L)
        cen = Pun.mean(axis=0)
        rgb = np.array(matplotlib.colors.to_rgb(COL[type_name(b)]))
        for s in b.surfaces:
            vs = s.vertices
            if not vs or len(vs) < 3:
                continue
            fp = np.array([_pos(v) for v in vs], dtype=float)
            fp = fp - L * np.round((fp - cen) / L)        # bring the face into the cell's image
            fp = cen + SHRINK * (fp - cen)                # shrink toward the cell centroid
            if CLIP is not None:                          # keep only the half the normal points toward
                fp = clip_polygon_halfspace(fp, CLIP[0], CLIP[1])
                if fp is None:                            # face entirely on the clipped side -> drop
                    continue
            polys.append(fp)
            base.append(rgb)
            cens.append(cen)
    return polys, base, np.array(cens) if cens else np.zeros((0, 3))


# simple lambert shading so the solid cells read as 3D (flat facecolours look like a paper cut-out)
_LIGHT = np.array([0.3, 0.5, 1.0]); _LIGHT /= np.linalg.norm(_LIGHT)


def _shade(poly, rgb):
    a, b, c = poly[0], poly[1], poly[2]
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return rgb
    lam = abs(float(np.dot(n / nn, _LIGHT)))     # abs: back faces (hidden anyway) stay lit, not black
    return np.clip(rgb * (0.55 + 0.45 * lam), 0, 1)


frame_idx = 0


def capture(step, recon):
    global frame_idx
    polys, base, cens = cell_polys()
    azim = AZIM + (frame_idx * 3.0 if ROTATE else 0.0)
    facecols = [_shade(p, rgb) for p, rgb in zip(polys, base)]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    pc = Poly3DCollection(polys, facecolors=facecols, edgecolors=(0, 0, 0, 0.35), linewidths=0.25)
    ax.add_collection3d(pc)
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_zlim(0, L)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=ELEV, azim=azim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    nA = sum(1 for b in bodies if type_name(b) == "A")
    D = demixing_index(bodies=bodies)
    ax.set_title(f"native sort [{IC}]  N={len(bodies)}  step={step}\n"
                 f"A/B={nA}/{len(bodies)-nA}   demixing D={D:+.3f}   (lower = more sorted)",
                 fontsize=10)
    ax.legend(handles=[Patch(facecolor=COL["A"], label="type A"),
                       Patch(facecolor=COL["B"], label="type B")],
              loc="upper right", fontsize=9, framealpha=0.9)
    path = os.path.join(FRAMES, f"frame_{frame_idx:05d}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    ok = os.path.exists(path) and os.path.getsize(path) > 0
    print(f"  frame {frame_idx:3d} @ step {step:6d}: verts={mesh.num_vertices:5d}  "
          f"A/B={nA}/{len(bodies)-nA}  D={D:+.3f}  recon={recon}  {'OK' if ok else 'FAIL'}", flush=True)
    frame_idx += 1


# --- active-motility Python injection (only when MODEL == "active") --------------------------------
rng_dir = np.random.default_rng(SEED + 2)
_dirs = rng_dir.normal(0.0, 1.0, (len(bodies), 3))
_dirs /= np.linalg.norm(_dirs, axis=1, keepdims=True)


def add_noise_active():
    if MODEL != "active" or V0_ACT <= 0:
        return
    idx, vh = {}, []
    for i in range(mesh.size_vertices):
        v = mesh.get_vertex(i)
        if v is None or v.id < 0:
            continue
        idx[v.id] = len(vh); vh.append(v)
    vrow, brow = [], []
    for k, b in enumerate(bodies):
        seen = set()
        for s in b.surfaces:
            for w in s.vertices:
                if w.id in idx and w.id not in seen:
                    seen.add(w.id); vrow.append(idx[w.id]); brow.append(k)
    vrow = np.array(vrow, np.int64); brow = np.array(brow, np.int64); n = len(vh)
    xi = rng_dir.normal(0.0, 1.0, _dirs.shape); xi /= np.linalg.norm(xi, axis=1, keepdims=True)
    _dirs[:] = _dirs + ROT_STD * (xi - _dirs)
    _dirs[:] /= np.linalg.norm(_dirs, axis=1, keepdims=True)
    S = np.zeros((n, 3)); np.add.at(S, vrow, _dirs[brow])
    cnt = np.zeros(n); np.add.at(cnt, vrow, 1.0); cnt[cnt == 0] = 1.0
    dx = DT * V0_ACT * (S / cnt[:, None])
    P = np.array([[p[0], p[1], p[2]] for p in (v.position for v in vh)], dtype=float)
    newP = (P + dx) % L
    for v, qq in zip(vh, newP):
        v.set_position(tf.FVector3(float(qq[0]), float(qq[1]), float(qq[2])))


def stitch():
    paths = sorted(glob.glob(os.path.join(FRAMES, "frame_*.png")))
    if not paths:
        print("no frames to stitch", flush=True)
        return
    mp4 = OUT_BASE + ".mp4"
    cmd = ["ffmpeg", "-y", "-framerate", str(FPS), "-i",
           os.path.join(FRAMES, "frame_%05d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", mp4]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"wrote {mp4} ({len(paths)} frames, {os.path.getsize(mp4) / 1e6:.1f} MB, {FPS} fps)", flush=True)
        return
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"ffmpeg failed ({e}); falling back to GIF.", flush=True)
    try:
        from PIL import Image
        imgs = [Image.open(p).convert("P", palette=Image.ADAPTIVE, colors=256) for p in paths]
        gif = OUT_BASE + ".gif"
        imgs[0].save(gif, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / FPS), loop=0, optimize=True)
        print(f"wrote {gif} ({len(imgs)} frames, {os.path.getsize(gif) / 1e6:.1f} MB)", flush=True)
    except Exception as e:
        print(f"GIF fallback failed too ({e}); frames are in {FRAMES}/", flush=True)


if os.path.isdir(FRAMES):
    shutil.rmtree(FRAMES)
os.makedirs(FRAMES, exist_ok=True)

if CLIP is not None:
    print(f"clip plane: keep half with normal {CLIP[1]} through {CLIP[0]} (per-face polygon clip)", flush=True)
print(f"=== WHOLE-CELL SORT VIDEO [{MODEL}/{IC}]: N={len(bodies)} sigma={SIGMA} v0={V0_ACT} M={M} "
      f"steps={N_STEPS} | frame every {CAPTURE_EVERY} | shrink={SHRINK} "
      f"| {'turntable' if ROTATE else f'fixed view e{ELEV:g}a{AZIM:g}'} | A=blue B=red"
      f"{' | CLIP=' + _CLIP_SPEC if CLIP is not None else ''} ===", flush=True)
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

print(f"\nDONE: {frame_idx} frames; {recon} reconnections.", flush=True)
stitch()
