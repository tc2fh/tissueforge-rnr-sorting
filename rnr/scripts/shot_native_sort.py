"""Headless SNAPSHOT of the NATIVE sort scene -- the `tf.run()` window's twin, rendered without GL.

WHY THIS EXISTS (read this if you came here from a broken `tf.run()` window):
On this WSL2 box TissueForge's GL renderer cannot produce an image -- neither on-screen nor
off-screen. The Mesa driver fails to get a GPU (`MESA: error: ZINK: failed to choose pdev`,
`libEGL: failed to create dri2 screen`), which is exactly why the live `tf.run()` window renders
wrong. And `tf.system.screenshot()` (the built-in "save screenshot") ALSO fails here: even after
forcing software GL (`LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe`, which DOES create a valid
llvmpipe 4.5 context), the windowless framebuffer read comes back zero-width
(`Trade::...convertToData(): can't convert image with a zero size: Vector(0, ...)`). So the
built-in screenshot is a dead end on this machine -- the bug is in TF's GL render/readback path
+ the WSL2 GL stack, NOT in the simulation or the scene.

So we render the SAME scene the window would show -- the actual 3D cell polygons, coloured by the
two cells each surface separates -- straight from the mesh geometry with matplotlib (Agg, no GL).
This both lets you LOOK at the scene and answers the question "is the scene itself fine, or is the
geometry broken too?": if these snapshots look like a clean packed cell block that demixes over
time, the scene is fine and the window problem is purely the WSL2 GL driver.

Scene/physics are identical to `watch_native_sort_window.py` (periodic two-type Voronoi bulk +
native I<->H reconnection + the C++ active-motility drive). Surface colours match that demo:
  homotypic A|A -> royal blue   homotypic B|B -> red   HETEROTYPIC A|B -> gold (shrinks as it sorts).

Run:
    pixi run python rnr/scripts/shot_native_sort.py [M] [SIGMA] [V0] [SEED] [IC] [STEPS] [FRAMES]
    defaults: 6 0.5 0.1 7 mixed 2000 4      (IC = mixed | demixed)
  Env: HET_ONLY=1 -> draw only the gold heterotypic interfaces (see the interior demixing, not just
       the outer shell). VIEWS="30,-60;90,-90" -> custom elev,azim camera angles (';'-separated).
Outputs (paths printed at the end): rnr/exports/shot_native_<IC>_step{n}_<view>.png
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
import tissue_forge as tf  # noqa: E402
from tissue_forge.models.vertex import solver as tfv  # noqa: E402
from tissue_forge.models.vertex.solver.mesh_types import BodyTypeSpec, SurfaceTypeSpec  # noqa: E402

M = int(sys.argv[1]) if len(sys.argv) > 1 else 6           # N = M^3 cells, L = M (V0=1)
SIGMA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5   # heterotypic interfacial tension
V0_ACT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1  # active self-propulsion speed
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 7
IC = sys.argv[5] if len(sys.argv) > 5 else "mixed"         # mixed = sorting from random | demixed = pre-sorted slab
STEPS = int(sys.argv[6]) if len(sys.argv) > 6 else 2000    # total native steps spread across the frames
FRAMES = int(sys.argv[7]) if len(sys.argv) > 7 else 4      # snapshots: frame 0 + (FRAMES-1) stepped frames
HET_ONLY = os.environ.get("HET_ONLY", "0") in ("1", "true", "True")
VIEWS = [tuple(float(x) for x in v.split(","))
         for v in os.environ.get("VIEWS", "25,-60").split(";")]   # (elev, azim) pairs

# Oracle params (identical to watch_native_sort_window.py / sort_periodic_oracle.py MODEL=native)
L = float(M)
BOX = [[0.0, L]] * 3
V0, K_V, K_A, A0 = 1.0, 10.0, 1.0, 5.6
DR, LTH, DT, CUT, INTERVAL = 1.0, 1e-3, 1e-3, 1.9, 10
COL = {("A", "A"): "#1c4fd6", ("B", "B"): "#d62828", "het": "#f1b211", None: "#999999"}

EXPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)

_TFTHREADS = os.environ.get("TF_THREADS")
_init_kw = {"threads": int(_TFTHREADS)} if _TFTHREADS else {}
tf.init(windowless=True, dim=[L, L, L], cutoff=CUT, dt=DT, **_init_kw)   # GL never used; Agg renders
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
tfv.MeshSolver.set_motility(V0_ACT, DR, SEED + 2)


def _pos(v):
    p = v.position
    return [p.x(), p.y(), p.z()]


def collect_polys():
    """Pull every cell-cell surface as a polygon (ordered vertex xyz), colour it by the type pair of
    its 1-2 bounding cells, and periodic-unwrap each polygon so box-wrapping faces aren't smeared
    across the domain. Returns (polys, facecolors, is_hetero[]) for a Poly3DCollection."""
    polys, cols, het = [], [], []
    seen = set()
    for b in bodies:
        for s in b.surfaces:
            sid = getattr(s, "id", None)
            if sid is None or sid in seen:
                continue
            seen.add(sid)
            try:
                vs = s.vertices
                if not vs or len(vs) < 3:
                    continue
                pts = np.array([_pos(v) for v in vs], dtype=float)
                # min-image unwrap relative to the first vertex (keep faces compact under PBC)
                anchor = pts[0]
                pts = pts - L * np.round((pts - anchor) / L)
                tys = [type_name(x) for x in s.bodies] if s.bodies else []
                if len(tys) >= 2 and tys[0] != tys[1]:
                    c, is_h = COL["het"], True
                elif len(tys) >= 1:
                    c, is_h = COL[(tys[0], tys[0])], False
                else:
                    c, is_h = COL[None], False
                if HET_ONLY and not is_h:
                    continue
                polys.append(pts)
                cols.append(c)
                het.append(is_h)
            except Exception:
                continue
    return polys, cols, het


def render(nstep):
    polys, cols, het = collect_polys()
    nhet = sum(het)
    paths = []
    for (elev, azim) in VIEWS:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
        alpha = 0.9 if HET_ONLY else 0.55
        pc = Poly3DCollection(polys, facecolors=cols, edgecolors=(0, 0, 0, 0.25),
                              linewidths=0.2, alpha=alpha)
        ax.add_collection3d(pc)
        ax.set_xlim(-0.5, L + 0.5); ax.set_ylim(-0.5, L + 0.5); ax.set_zlim(-0.5, L + 0.5)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"native sort [{IC}]  N={len(bodies)}  step={nstep}  "
                     f"{'HET faces only' if HET_ONLY else 'all faces'}  (gold = A|B)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        tag = f"{'het' if HET_ONLY else 'all'}_e{int(elev)}a{int(azim)}"
        path = os.path.join(EXPORT_DIR, f"shot_native_{IC}_step{nstep}_{tag}.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    nA = sum(1 for b in bodies if type_name(b) == "A")
    print(f"  step={nstep:>5}  verts={mesh.num_vertices:>5}  A/B={nA}/{len(bodies) - nA}  "
          f"faces={len(polys)} (het={nhet})  -> {', '.join(os.path.basename(p) for p in paths)}",
          flush=True)
    return paths


print(f"=== HEADLESS SNAPSHOT [{IC}]: N={len(bodies)} sigma={SIGMA} v0={V0_ACT} M={M} | "
      f"{FRAMES} frames over {STEPS} steps | views={VIEWS} | matplotlib (no GL) ===", flush=True)

all_paths = list(render(0))
per = max(1, STEPS // max(1, FRAMES - 1)) if FRAMES > 1 else STEPS
done = 0
for i in range(1, FRAMES):
    n = per if (done + per) <= STEPS else (STEPS - done)
    for _ in range(n):
        tf.step()
    done += n
    all_paths += render(done)

print(f"\nSaved {len(all_paths)} snapshots to {EXPORT_DIR}/", flush=True)
print("Clean packed/demixing block => scene is fine; the tf.run() problem is the WSL2 GL driver.",
      flush=True)
