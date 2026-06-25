"""GPU whole-cell-by-type sort video at PAPER SCALE (~2000 cells), turntable, NO clip plane.

The GPU analogue of video_native_cells.py: the physics is driven entirely by the GPU 3D-vertex
engine (rnr.gpu.engine.forward_step on the cached n=10 foam -- no TissueForge), and each cell is
rendered as its OWN shrunk polyhedron painted by its TYPE (A = royal blue, B = red, the same A/B
convention as every other sort view), with matplotlib/Agg (no GL -> no WSL2 driver issues).

Differences from the native-cells clip video (native_cells_sort_native_mixed_clipmz.mp4) requested:
  * ~2000 cells (n=10 BCC foam) instead of the M=6 native run,
  * NO clip plane (whole foam visible),
  * turntable rotation at 1/4 the native speed: 0.75 deg/frame (native used 3.0 deg/frame).

Run:  pixi run python rnr/scripts/gpu_video_cells.py [STEPS] [SIGMA] [IC] [CAPTURE_EVERY] [SEED]
      defaults: 100000 0.5 mixed 500 7
  Env: SHRINK=0.85  DEG_PER_FRAME=0.75  AZIM0=-60  ELEV=20  FPS=20
Output: rnr/exports/gpu_cells_frames/frame_#####.png + rnr/exports/gpu_cells_sort_<ic>.mp4
"""
import glob
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import warp as wp  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from rnr.gpu import engine as E  # noqa: E402
from rnr.gpu import physics_csr as P  # noqa: E402
from rnr.gpu.device_mesh import PaddedMesh  # noqa: E402
from rnr.gpu.engine import het_contact_fraction  # noqa: E402
from rnr.gpu.foam_cache import load_or_build  # noqa: E402

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
SIGMA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
IC = sys.argv[3] if len(sys.argv) > 3 else "mixed"
CAPTURE_EVERY = int(sys.argv[4]) if len(sys.argv) > 4 else 500
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 7

N = 10
DT, DR, LTH = 0.01, 1.0, 1e-3
V0_ACT = 0.1
INTERVAL = max(1, round(0.01 / DT))
SHRINK = float(os.environ.get("SHRINK", "0.85"))
DEG_PER_FRAME = float(os.environ.get("DEG_PER_FRAME", "0.75"))   # 1/4 of the native 3.0 deg/frame
AZIM0 = float(os.environ.get("AZIM0", "-60"))
ELEV = float(os.environ.get("ELEV", "20"))
FPS = int(os.environ.get("FPS", "20"))
COL = {0: np.array(mcolors.to_rgb("#1c4fd6")), 1: np.array(mcolors.to_rgb("#d62828"))}  # A blue / B red

EXPORT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exports")
FRAMES = os.path.join(EXPORT, "gpu_cells_frames")
OUT_BASE = os.path.join(EXPORT, f"gpu_cells_sort_{IC}")

_LIGHT = np.array([0.3, 0.5, 1.0]); _LIGHT /= np.linalg.norm(_LIGHT)


def _shade(poly, rgb):
    a, b, c = poly[0], poly[1], poly[2]
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return rgb
    lam = abs(float(np.dot(n / nn, _LIGHT)))
    return np.clip(rgb * (0.55 + 0.45 * lam), 0, 1)


def cell_polys(pm, body_type, box):
    """Each live cell -> its faces as shrunk polygons (periodic-unwrapped into the cell's own
    image), coloured by the cell's TYPE. Returns (polys, base_rgb_per_poly)."""
    L = np.asarray(box, float)
    s2v, s2vl = pm.s2v, pm.s2v_len
    b2s, b2sl = pm.b2s, pm.b2s_len
    vp = pm.vert_pos
    polys, base = [], []
    for b in range(pm.nb):
        ns = int(b2sl[b])
        if ns == 0:
            continue
        surfs = [int(s) for s in b2s[b, :ns] if s >= 0 and pm.surf_alive[s]]
        if not surfs:
            continue
        vid = {int(v) for s in surfs for v in s2v[s, :s2vl[s]] if v >= 0}
        Pv = vp[list(vid)]
        cen = (Pv - L * np.round((Pv - Pv[0]) / L)).mean(axis=0)     # unwrapped cell centroid
        rgb = COL[int(body_type[b])]
        for s in surfs:
            sl = int(s2vl[s])
            if sl < 3:
                continue
            fp = vp[[int(x) for x in s2v[s, :sl]]]
            fp = fp - L * np.round((fp - cen) / L)                    # face into the cell's image
            fp = cen + SHRINK * (fp - cen)                            # shrink toward the centroid
            polys.append(fp)
            base.append(rgb)
    return polys, base


frame_idx = 0


def capture(pm, body_type, box, step):
    global frame_idx
    L = float(np.asarray(box, float)[0])
    polys, base = cell_polys(pm, body_type, box)
    facecols = [_shade(p, rgb) for p, rgb in zip(polys, base)]
    azim = AZIM0 + frame_idx * DEG_PER_FRAME
    het, total = het_contact_fraction(pm, body_type)
    dp = 1.0 - 2.0 * het / total if total else 0.0
    nA = int((body_type == 0).sum())

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    pc = Poly3DCollection(polys, facecolors=facecols, edgecolors=(0, 0, 0, 0.3), linewidths=0.2)
    ax.add_collection3d(pc)
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_zlim(0, L)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=ELEV, azim=azim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(f"GPU 3D-vertex sort [{IC}]  N={pm.nb}  step={step}\n"
                 f"A/B={nA}/{pm.nb - nA}   DP={dp:+.3f}   (higher = more sorted)", fontsize=10)
    ax.legend(handles=[Patch(facecolor=COL[0], label="type A"),
                       Patch(facecolor=COL[1], label="type B")],
              loc="upper right", fontsize=9, framealpha=0.9)
    path = os.path.join(FRAMES, f"frame_{frame_idx:05d}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  frame {frame_idx:3d} @ step {step:6d}: nv={pm.n_v_used} DP={dp:+.3f} "
          f"A/B={nA}/{pm.nb - nA}  ({len(polys)} polys)", flush=True)
    frame_idx += 1


def stitch():
    paths = sorted(glob.glob(os.path.join(FRAMES, "frame_*.png")))
    if not paths:
        print("no frames", flush=True)
        return
    mp4 = OUT_BASE + ".mp4"
    cmd = ["ffmpeg", "-y", "-framerate", str(FPS), "-i", os.path.join(FRAMES, "frame_%05d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", mp4]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"wrote {mp4} ({len(paths)} frames, {os.path.getsize(mp4) / 1e6:.1f} MB, {FPS} fps)", flush=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"ffmpeg failed ({e}); frames in {FRAMES}/", flush=True)


def main():
    cuda = [d for d in wp.get_devices() if d.is_cuda]
    if not cuda:
        print("FATAL: no CUDA device.")
        sys.exit(2)
    dev = cuda[0]
    g, phys, body_type, box, v0, a0 = load_or_build(
        dev, n=N, ic=IC, headroom=4000,
        build_host_fn=lambda: (_ for _ in ()).throw(RuntimeError(
            f"n={N} {IC} foam not cached; build it first via gpu-stability")))
    params = P.PhysParams(box=box, kv=10.0, v0=v0, ka=1.0, a0=a0, sigma=SIGMA, v_active=V0_ACT)

    if os.path.isdir(FRAMES):
        shutil.rmtree(FRAMES)
    os.makedirs(FRAMES, exist_ok=True)
    print(f"=== GPU CELL SORT VIDEO [{IC}]: N={g['nb']} sigma={SIGMA} v0={V0_ACT} steps={STEPS} "
          f"| frame every {CAPTURE_EVERY} | shrink={SHRINK} | turntable {DEG_PER_FRAME} deg/frame "
          f"| NO clip | A=blue B=red ===", flush=True)
    t0 = time.perf_counter()
    capture(PaddedMesh.from_warp(g), body_type, box, 0)
    for step in range(1, STEPS + 1):
        E.forward_step(g, phys, params, DT, DR, seed=SEED, step=step, threshold=LTH, dl_th=LTH,
                       reconnect=True, interval=INTERVAL, compact=True, max_rounds=8)
        if step % CAPTURE_EVERY == 0 or step == STEPS:
            capture(PaddedMesh.from_warp(g), body_type, box, step)
    print(f"\nDONE: {frame_idx} frames in {(time.perf_counter() - t0) / 60.0:.1f} min", flush=True)
    stitch()


if __name__ == "__main__":
    main()
