"""Crawling demo video: visualize the lamellipodial-crawling EXTENSION (rnr/examples/crawl.py) on
the GPU 3D-vertex engine so the behavior can be evaluated by eye.

Same renderer as gpu_video_cells.py (matplotlib/Agg per-cell shrunk polyhedra, ffmpeg stitch), but:
  * physics is driven by the high-level `Engine` with the crawl hooks registered -- a per-cell
    `lamellipodium_force` (push each crawler's LEADING vertices along its polarity) + a
    `persistent_repolarization` behavior (slow rotational diffusion of polarity);
  * CRAWLER cells are painted red and TISSUE (non-crawler) cells faded gray, so the migrating
    population stands out;
  * each crawler gets a black POLARITY ARROW (its current heading) and a red MOTION TRAIL (its
    centroid path), the two cues that make "is it crawling?" visible despite the periodic foam's
    baseline rearrangement.

Crawlers = the type-1 cells by default (set N_CRAWLERS to mark only the first k type-1 cells).

Run:  pixi run python rnr/scripts/gpu_crawl_video.py [STEPS] [F_MAG] [DR] [SIGMA] [SEED]
      defaults: 600 2.0 0.1 0.0 7
  Env: CAPTURE_EVERY=15  N_CRAWLERS=all  SHRINK=0.8  DEG_PER_FRAME=0.6  AZIM0=-60  ELEV=20  FPS=12
Output: rnr/exports/gpu_crawl_frames/frame_#####.png + rnr/exports/gpu_crawl_demo.mp4
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

from rnr.gpu import physics_csr as P  # noqa: E402
from rnr.gpu.device_mesh import PaddedMesh  # noqa: E402
from rnr.gpu.engine import Engine  # noqa: E402
from rnr.gpu.extensions import constant_vector, random_unit_vectors  # noqa: E402
from rnr.gpu.foam_cache import load_or_build  # noqa: E402
from rnr.examples.crawl import (lamellipodium_force, migration_force,  # noqa: E402
                                persistent_repolarization)
from rnr.examples.energy_terms import (edge_length_penalty, face_area_penalty,  # noqa: E402
                                       mean_edge_length, mean_face_area)

# MODE: "migration" (default, whole-cell propulsion -> cells TRANSLATE) or "lamellipodium"
# (leading-edge protrusion -> cells ELONGATE; kept to show the failure mode).
MODE = os.environ.get("MODE", "migration")
# REG: add regularizer energy terms that resist local distortion -> "edge", "face", "edge,face",
# "none". Pair with MODE=lamellipodium to show the protrusion's spikes getting regularized away.
REG = os.environ.get("REG", "none")
REG_K = float(os.environ.get("REG_K", "5.0"))
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
_FMAG_DEFAULT = 0.3 if MODE == "migration" else 2.0       # migration ~ v0 scale; protrusion needs more
F_MAG = float(sys.argv[2]) if len(sys.argv) > 2 else _FMAG_DEFAULT
DR = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
SIGMA = float(sys.argv[4]) if len(sys.argv) > 4 else 0.4    # cohesion: lets the drive move the cell body
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 7
# UNIFORM=1: all crawlers head +x. Default 0 (random headings) -> no net foam drift, so crawler motion
# RELATIVE to the tissue is clean (coherent +x instead excites the periodic rigid-translation free mode).
UNIFORM = os.environ.get("UNIFORM", "0") == "1"

N = 3
IC = "mixed"
DT, LTH, INTERVAL = 5e-3, 0.02, 2
CRAWLER_TYPE = 1
CAPTURE_EVERY = int(os.environ.get("CAPTURE_EVERY", "10"))
N_CRAWLERS = os.environ.get("N_CRAWLERS", "5")   # 5 scattered crawler cells; "all" = every type-1 cell
SHRINK = float(os.environ.get("SHRINK", "0.8"))
DEG_PER_FRAME = float(os.environ.get("DEG_PER_FRAME", "0.6"))
AZIM0 = float(os.environ.get("AZIM0", "-60"))
ELEV = float(os.environ.get("ELEV", "20"))
FPS = int(os.environ.get("FPS", "10"))
ARROW_LEN = float(os.environ.get("ARROW_LEN", "0.7"))

COL_CRAWLER = np.array(mcolors.to_rgb("#d62828"))   # red  -> migrating cells (used for "all" mode)
COL_TISSUE = np.array(mcolors.to_rgb("#9aa6b6"))     # gray -> passive tissue
# distinct colors so a handful of scattered crawlers can each be tracked individually
PALETTE = [np.array(mcolors.to_rgb(c)) for c in
           ("#e63946", "#1d7fd6", "#2f9e44", "#f08c00", "#9c36b5", "#0ca678", "#d6336c")]

EXPORT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exports")
FRAMES = os.path.join(EXPORT, "gpu_crawl_frames")
_tag = "demo" if (MODE == "migration" and REG == "none") else MODE + ("_reg" if REG != "none" else "")
OUT_BASE = os.path.join(EXPORT, f"gpu_crawl_{_tag}")

_LIGHT = np.array([0.3, 0.5, 1.0]); _LIGHT /= np.linalg.norm(_LIGHT)


def _pdist2(cent, p, L):
    d = cent - p
    d = d - L * np.round(d / L)            # min-image (periodic)
    return np.einsum("ij,ij->i", d, d)


def _scatter_select(cent, k, box, seed):
    """k cells spread across the (periodic) mesh by farthest-point sampling on centroids: a random
    seed cell, then each next cell maximizes the min distance to those already chosen -> the crawlers
    are scattered, not clustered."""
    L = np.asarray(box, float)
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(cent)))]
    d2 = _pdist2(cent, cent[chosen[0]], L)
    while len(chosen) < min(k, len(cent)):
        nxt = int(np.argmax(d2))
        chosen.append(nxt)
        d2 = np.minimum(d2, _pdist2(cent, cent[nxt], L))
    return np.array(chosen, dtype=int)


def _shade(poly, rgb):
    a, b, c = poly[0], poly[1], poly[2]
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return rgb
    lam = abs(float(np.dot(n / nn, _LIGHT)))
    return np.clip(rgb * (0.55 + 0.45 * lam), 0, 1)


def cell_polys(pm, is_crawler, cell_rgb, box):
    """Each live cell -> its faces as shrunk polygons (periodic-unwrapped into the cell's own image),
    painted by cell_rgb[b] (a tracked color per crawler; gray tissue). Returns (polys, rgba_per_poly,
    centroid_by_body)."""
    L = np.asarray(box, float)
    s2v, s2vl = pm.s2v, pm.s2v_len
    b2s, b2sl = pm.b2s, pm.b2s_len
    vp = pm.vert_pos
    polys, cols, cen_by_b = [], [], {}
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
        cen_by_b[b] = cen
        crawler = bool(is_crawler[b])
        rgb = cell_rgb[b]
        alpha = 0.97 if crawler else 0.22                            # fade the passive tissue back
        for s in surfs:
            sl = int(s2vl[s])
            if sl < 3:
                continue
            fp = vp[[int(x) for x in s2v[s, :sl]]]
            fp = fp - L * np.round((fp - cen) / L)                    # face into the cell's image
            fp = cen + SHRINK * (fp - cen)                            # shrink toward the centroid
            polys.append(fp)
            cols.append((*_shade(fp, rgb), alpha))
    return polys, cols, cen_by_b


frame_idx = 0
TRAILS = {}           # body id -> list of centroids (for crawlers), drawn as a motion path


def capture(pm, is_crawler, cell_rgb, polarity, box, step, drift_msg):
    global frame_idx
    L = float(np.asarray(box, float)[0])
    polys, cols, cen_by_b = cell_polys(pm, is_crawler, cell_rgb, box)
    azim = AZIM0 + frame_idx * DEG_PER_FRAME

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(polys, facecolors=cols,
                                         edgecolors=(0, 0, 0, 0.18), linewidths=0.15))
    # polarity arrows (black, for contrast) + motion trails (in the crawler's own color) per crawler
    for b, cen in cen_by_b.items():
        if not is_crawler[b]:
            continue
        TRAILS.setdefault(b, []).append(cen.copy())
        col = tuple(np.clip(cell_rgb[b], 0, 1))
        p = polarity[b]
        nrm = np.linalg.norm(p)
        if nrm > 1e-9:
            p = p / nrm
            ax.quiver(cen[0], cen[1], cen[2], p[0], p[1], p[2], length=ARROW_LEN, color="k",
                      linewidth=1.7, arrow_length_ratio=0.35)
        tr = np.array(TRAILS[b])
        if len(tr) > 1:
            seg = np.linalg.norm(np.diff(tr, axis=0), axis=1)        # break the trail on a box wrap
            cut = np.where(seg > 0.5 * L)[0]
            pieces = np.split(tr, cut + 1) if len(cut) else [tr]
            for piece in pieces:
                if len(piece) > 1:
                    ax.plot(piece[:, 0], piece[:, 1], piece[:, 2], color=col, lw=1.8)

    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_zlim(0, L)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=ELEV, azim=azim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    nc = int(np.asarray(is_crawler).sum())
    ax.set_title(f"cell {MODE} demo   step={step}   {nc} tracked crawlers / {pm.nb} cells\n"
                 f"f={F_MAG}  Dr={DR}  sigma={SIGMA}   {drift_msg}", fontsize=10)
    ax.legend(handles=[Patch(facecolor=COL_TISSUE, alpha=0.4, label="tissue"),
                       Patch(facecolor=PALETTE[0], label="crawler (colored)")],
              loc="upper right", fontsize=9, framealpha=0.9)
    path = os.path.join(FRAMES, f"frame_{frame_idx:05d}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  frame {frame_idx:3d} @ step {step:5d}: nv={pm.n_v_used}  {drift_msg}", flush=True)
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


def _drift_msg(eng, box, p0, cent0, bt, crawl_mask):
    """Mean directed progress RELATIVE to the foam (Δ_cell - Δ_foam_mean)·p̂0 of crawlers vs tissue
    since frame 0 -- the evaluable number. Subtracting the foam-mean displacement removes the periodic
    rigid-translation free mode, isolating crawler motion THROUGH the tissue (crawler>0, tissue<0)."""
    cent = P.compute_geometry(PaddedMesh.from_warp(eng.g), box).bcent[:eng.cells.n]
    d = P.minimg(cent - cent0, box)
    d = d - d.mean(axis=0)                       # peculiar displacement (remove net foam drift)
    fwd = np.sum(d * p0, axis=1)
    return f"crawler fwd(rel)={fwd[crawl_mask].mean():+.3f}  tissue={fwd[~crawl_mask].mean():+.3f}"


def main():
    cuda = [d for d in wp.get_devices() if d.is_cuda]
    if not cuda:
        print("FATAL: no CUDA device.")
        sys.exit(2)
    dev = cuda[0]

    def _build_host():
        import tissue_forge as tf
        from tissue_forge.models.vertex import solver as tfv
        tf.init(windowless=True, dim=[60., 60., 60.], cutoff=5.0, dt=0.001)
        tfv.init()
        tfv.MeshSolver.get().get_mesh().quality = None
        from rnr.tests.test_gpu_engine import _build_unit_foam_host
        return _build_unit_foam_host(n=N, headroom=3000, ic=IC)

    g, phys, body_type, box, v0, a0 = load_or_build(dev, n=N, ic=IC, headroom=3000,
                                                    build_host_fn=_build_host)
    params = P.PhysParams(box=box, kv=10.0, v0=v0, ka=1.0, a0=a0, sigma=SIGMA, v_active=0.0)

    eng = Engine(g, phys, params, dt=DT, dr=DR, seed=SEED,
                 threshold=LTH, dl_th=LTH, reconnect=True, interval=INTERVAL)
    pol_init = constant_vector([1.0, 0.0, 0.0]) if UNIFORM else random_unit_vectors(seed=SEED)
    eng.cells.add_field("polarity", pol_init)
    # crawlers: N scattered cells (farthest-point sampling -> spread across the mesh) or every type-1.
    # cent0 here is also the frame-0 baseline for the directed-progress metric (positions are
    # unchanged through setup, before any step).
    cent0 = P.compute_geometry(PaddedMesh.from_warp(g), box).bcent[:eng.cells.n].copy()
    if N_CRAWLERS == "all":
        crawl_ids = np.where(body_type == CRAWLER_TYPE)[0]
    else:
        crawl_ids = _scatter_select(cent0, int(N_CRAWLERS), box, SEED)
    crawl_mask = np.zeros(eng.cells.n, dtype=bool)
    crawl_mask[crawl_ids] = True
    eng.cells.add_field("is_crawler", crawl_mask.astype(np.int32))
    cell_rgb = np.tile(COL_TISSUE, (eng.cells.n, 1))      # gray tissue + one tracked color per crawler
    for rank, b in enumerate(crawl_ids):
        cell_rgb[int(b)] = PALETTE[rank % len(PALETTE)]
    if DR > 0.0:                                 # repolarization behavior only when persistence is on
        eng.add_behavior(persistent_repolarization(dr=DR, seed=SEED))
    drive = migration_force if MODE == "migration" else lamellipodium_force
    eng.add_force(drive(f_mag=F_MAG))
    if REG != "none":                            # optional local-distortion regularizers
        pm0 = PaddedMesh.from_warp(g)
        if "edge" in REG:
            l0 = mean_edge_length(pm0, box)
            eng.add_force(edge_length_penalty(REG_K, l0))
            print(f"  + edge-length penalty k={REG_K} l0={l0:.3f}", flush=True)
        if "face" in REG:
            af = mean_face_area(pm0, box)
            eng.add_force(face_area_penalty(REG_K, af))
            print(f"  + face-area penalty k={REG_K} a0={af:.3f}", flush=True)

    p0 = eng.cells["polarity"].numpy().reshape(-1, 3).copy()
    p0 /= np.linalg.norm(p0, axis=1, keepdims=True)

    if os.path.isdir(FRAMES):
        shutil.rmtree(FRAMES)
    os.makedirs(FRAMES, exist_ok=True)
    print(f"=== CRAWL DEMO VIDEO: N={g['nb']} crawlers={int(crawl_mask.sum())} f={F_MAG} Dr={DR} "
          f"sigma={SIGMA} steps={STEPS} | frame every {CAPTURE_EVERY} | turntable {DEG_PER_FRAME} "
          f"deg/frame ===", flush=True)
    t0 = time.perf_counter()
    pol = eng.cells["polarity"].numpy().reshape(-1, 3)
    capture(PaddedMesh.from_warp(g), crawl_mask, cell_rgb, pol, box, 0,
            _drift_msg(eng, box, p0, cent0, body_type, crawl_mask))
    for step in range(1, STEPS + 1):
        eng.step()
        if step % CAPTURE_EVERY == 0 or step == STEPS:
            pol = eng.cells["polarity"].numpy().reshape(-1, 3)
            capture(PaddedMesh.from_warp(g), crawl_mask, cell_rgb, pol, box, step,
                    _drift_msg(eng, box, p0, cent0, body_type, crawl_mask))
    print(f"\nDONE: {frame_idx} frames in {(time.perf_counter() - t0) / 60.0:.1f} min", flush=True)
    stitch()


if __name__ == "__main__":
    main()
