"""Render the native sort as a video using TissueForge's OWN GL renderer (not matplotlib).

This is the GL twin of `video_native_vertices.py` / `shot_native_sort.py`: identical physics
(periodic two-type Voronoi bulk + native I<->H reconnection + the C++ active-motility drive), but
each frame is a real `tf.system.screenshot()` of the engine's renderer instead of a matplotlib plot.
This is possible because the headless-screenshot bug is fixed (`docs/BUGS.md` #9, `PORTING_NOTES.md`
§8) -- before that fix every non-JPEG screenshot aborted with a zero-size framebuffer.

Cells are coloured by the type pair of each cell-cell surface, exactly like the matplotlib version:
  homotypic A|A -> royal blue   homotypic B|B -> red   HETEROTYPIC A|B -> gold.
TF colours a surface by its *surface* type, not by the cell-type pair, so we set each Surface's
OWN `style` (which overrides the type style: `tfMeshRenderer.cpp` does `s->style ? s->style :
s->type()->style`) to one of three shared Style objects, refreshed every captured frame as cells
sort and reconnect.

Run:
    pixi run video-gl [N_STEPS] [SIGMA] [V0] [M] [SEED] [CAPTURE_EVERY] [MODEL] [IC] [DT] [LTH] [CUT]
    defaults: 20000 0.5 0.1 6 7 250 native mixed 1e-3 1e-3 1.9      (~80 frames)
  IC      = mixed (sort from random) | demixed (pre-sorted slab, watch it hold)
  MODEL   = native (engine drive) | active (Python injection)
  Env:
    HET_ONLY=1   draw ONLY the gold A|B interfaces (homotypic faces hidden) -- see the interior
                 interface shrink as it sorts, not just the outer shell.
    CLIP=z       FIXED clip plane: keep only the half the normal points toward, so you see a
                 cross-section through the tissue. <sign?><axis>: x y z +x -z ... (sign = kept
                 side). CLIP_AT=0.5 sets the plane position as a box fraction. This is TF's OWN
                 GL clip plane (vertex MeshRenderer Patch A), applied at tf.init -- see clip note.
    FPS=20       output frame rate.
    VIEW=front   camera preset (used only when the turntable is OFF): front | top | right | ...
    ROTATE=auto  turntable that orbits the camera so the clip cut stays in view from every side.
                 `auto` = ON whenever CLIP is set; set 1/0 to force. ELEV=-52 (camera pitch, deg;
                 NEGATIVE looks DOWN into a top cut -- pair with CLIP=-z), AZIM=20 start yaw,
                 AZIM_STEP=4 yaw advance per captured frame.
Output: rnr/exports/native_gl_frames/frame_#####.png  +  rnr/exports/native_gl_sort_<MODEL>_<IC>.mp4
        (falls back to a .gif if ffmpeg is unavailable). When CLIP is set the basename gains a
        `_clip<spec>` tag so clipped and full renders don't overwrite each other.

CLIP-PLANE NOTE (debugged 2026-06-24): TF's native clip plane DOES work in the headless
screenshot path (both `tf.init(clip_planes=...)` and the runtime `tf.rendering.ClipPlanes` API).
`tf.init`'s parser is strict and SILENTLY drops a malformed entry: each plane must be a
`(point, normal)` *tuple* whose point and normal are *lists* of 3 floats (not tuples, not numpy
arrays) -- `rnr.clip.parse_clip_env` returns exactly that shape. `ClipPlanes.len()==0` after init
means the entry was dropped. The kept half is the side the normal points toward.
"""
import glob
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
import tissue_forge as tf  # noqa: E402
from tissue_forge import rendering as R  # noqa: E402
from tissue_forge.models.vertex import solver as tfv  # noqa: E402
from tissue_forge.models.vertex.solver.mesh_types import BodyTypeSpec, SurfaceTypeSpec  # noqa: E402

from rnr.clip import parse_clip_env  # noqa: E402  (pure numpy/os; safe to import before tf.init)

N_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
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

HET_ONLY = os.environ.get("HET_ONLY", "0") in ("1", "true", "True")
FPS = int(os.environ.get("FPS", "20"))
VIEW = os.environ.get("VIEW", "front").lower()
# Turntable (orbit the camera so you can see INTO the clip cut). Defaults ON whenever a
# clip plane is active -- a fixed view of a cut only shows the cross-section from one side.
ROTATE = os.environ.get("ROTATE", "auto").lower()
ELEV = float(os.environ.get("ELEV", "-52"))      # camera pitch, deg. NEGATIVE = look DOWN (into a top cut)
AZIM = float(os.environ.get("AZIM", "20"))       # starting yaw, deg
AZIM_STEP = float(os.environ.get("AZIM_STEP", "4"))  # yaw advance per captured frame

L = float(M)
BOX = [[0.0, L]] * 3
V0, K_V, K_A, A0 = 1.0, 10.0, 1.0, 5.6
DR = 1.0
ROT_STD = float(np.sqrt(2.0 * DR * DT))
INTERVAL = 10

_CLIP_SPEC = os.environ.get("CLIP", "").strip().lower()
CLIP = parse_clip_env(L)                        # (point, normal) lists or None -- fixed TF clip plane
_CLIP_TAG = ("_clip" + _CLIP_SPEC.replace("+", "p").replace("-", "m")) if CLIP is not None else ""
DO_ROTATE = (CLIP is not None) if ROTATE == "auto" else (ROTATE in ("1", "true", "yes", "on"))

EXPORT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exports")
# Tag the frames dir too (not just the mp4) so a clipped render never clobbers a
# concurrent un-clipped one -- each (clip) variant gets its own scratch dir.
FRAMES = os.path.join(EXPORT, "native_gl_frames" + _CLIP_TAG)
OUT_BASE = os.path.join(EXPORT, f"native_gl_sort_{MODEL}_{IC}{'_het' if HET_ONLY else ''}{_CLIP_TAG}")

_TFTHREADS = os.environ.get("TF_THREADS")
_init_kw = {"threads": int(_TFTHREADS)} if _TFTHREADS else {}
if CLIP is not None:
    # Strict TF format: a list of (point, normal) TUPLES whose entries are LISTS of 3 floats.
    # parse_clip_env already returns plain lists; a malformed entry would be silently dropped.
    _init_kw["clip_planes"] = [(CLIP[0], CLIP[1])]
tf.init(windowless=True, dim=[L, L, L], cutoff=CUT, dt=DT, **_init_kw)
tfv.init()
mesh = tfv.MeshSolver.get().get_mesh()
mesh.quality = None
mesh.periodic_geometry = True

from rnr.geometry import build_periodic_voronoi  # noqa: E402
from rnr.metrics import type_name  # noqa: E402

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

# --- three shared, persistent styles; per-surface assignment is just a pointer swap ----------------
STYLE = {
    ("A", "A"): R.Style("#1c4fd6"),   # royal blue
    ("B", "B"): R.Style("#d62828"),   # red
    "het":      R.Style("#f1b211"),   # gold (A|B interface)
}
STYLE["het"].setVisible(True)
# HET_ONLY hides homotypic faces so you see the interior interface, not just the outer shell.
STYLE[("A", "A")].setVisible(not HET_ONLY)
STYLE[("B", "B")].setVisible(not HET_ONLY)


def recolor():
    """Point every cell-cell surface's style at the blue/red/gold style for its type pair."""
    seen = set()
    for b in bodies:
        for s in b.surfaces:
            sid = s.id
            if sid in seen:
                continue
            seen.add(sid)
            tys = [type_name(x) for x in s.bodies] if s.bodies else []
            if len(tys) >= 2 and tys[0] != tys[1]:
                s.style = STYLE["het"]
            elif len(tys) >= 1:
                st = STYLE.get((tys[0], tys[0]))
                if st is not None:
                    s.style = st


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


_VIEWS = {
    "front": tf.system.camera_view_front, "back": tf.system.camera_view_back,
    "top": tf.system.camera_view_top, "bottom": tf.system.camera_view_bottom,
    "left": tf.system.camera_view_left, "right": tf.system.camera_view_right,
}
frame_idx = 0


def aim_camera():
    """Point the camera. With the turntable on, orbit the yaw each captured frame at a
    fixed downward pitch so the clip cross-section stays in view from every side -- a
    fixed preset would only ever show the cut from one angle. camera_rotate_to_euler_angle
    takes (pitch, yaw, roll) in RADIANS (probed 2026-06-24); pitch<0 looks down."""
    if DO_ROTATE:
        yaw = AZIM + frame_idx * AZIM_STEP
        tf.system.camera_rotate_to_euler_angle(
            tf.FVector3(float(np.radians(ELEV)), float(np.radians(yaw)), 0.0))
    else:
        _VIEWS.get(VIEW, tf.system.camera_view_front)()


def capture(step, recon):
    global frame_idx
    recolor()
    aim_camera()
    path = os.path.join(FRAMES, f"frame_{frame_idx:05d}.png")
    rc = tf.system.screenshot(path)
    ok = rc == 0 and os.path.exists(path) and os.path.getsize(path) > 0
    nA = sum(1 for b in bodies if type_name(b) == "A")
    print(f"  frame {frame_idx:3d} @ step {step:6d}: verts={mesh.num_vertices:5d}  "
          f"A/B={nA}/{len(bodies) - nA}  recon={recon}  {'OK' if ok else 'RENDER-FAIL'}", flush=True)
    frame_idx += 1


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
    _nclip = R.ClipPlanes.len()
    if _nclip == 0:
        print("WARNING: CLIP set but TF registered 0 clip planes (entry dropped) -- "
              "no clipping will be applied.", flush=True)
    else:
        print(f"clip plane: keep half with normal {CLIP[1]} through {CLIP[0]} "
              f"(TF ClipPlanes.len()={_nclip})", flush=True)

_cam = f"turntable e{ELEV:g} a{AZIM:g}+{AZIM_STEP:g}/f" if DO_ROTATE else f"view={VIEW}"
print(f"=== NATIVE-GL SORT VIDEO [{MODEL}/{IC}]: N={len(bodies)} sigma={SIGMA} v0={V0_ACT} M={M} "
      f"steps={N_STEPS} | frame every {CAPTURE_EVERY} | {'HET-only' if HET_ONLY else 'all faces'} "
      f"| {_cam}{' | CLIP=' + _CLIP_SPEC if CLIP is not None else ''} ===", flush=True)
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
