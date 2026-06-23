"""Watch the NATIVE sort demix LIVE in TissueForge's own 3D window via tf.run().

Same physics as `sort_periodic_oracle.py MODEL=native`: a periodic two-type Voronoi bulk + native
I<->H reconnection + the C++ engine active-motility drive (MeshSolver.set_motility). Because BOTH
the drive and the reconnection run inside the engine step, `tf.run()` advances everything with NO
per-step Python hook -- which is exactly why the NATIVE model (not the Python "active" one) is the
right fit for an interactive window (the active model needs a per-step injection that tf.run()
wouldn't call).

Surfaces are coloured by the two cells they separate, so the sorting is visible:
  * homotypic A|A  -> royal blue      * homotypic B|B  -> red
  * HETEROTYPIC A|B -> bright gold  (the unsorted interfaces -- they shrink as the tissue sorts)
A recurring tf.event.on_time recolours every RECOLOR_EVERY steps so surfaces created by
reconnection keep the right colour.

CAVEAT (honest): TF's vertex renderer draws SOLID faces, so you mainly see the OUTER SHELL of the
216-cell block -- the deep-bulk demixing is partly occluded. Drag to rotate / scroll to zoom, and
watch the gold (heterotypic) area shrink on the surface over time. For an unobstructed interior view
of the demixing, the headless centroid/sortedness movie (`video_periodic_active.py`) is clearer.
Pass HET_ONLY=1 to additionally hide homotypic faces (the engine MAY ignore per-surface visibility;
if it does, you simply still see them coloured).

Run (needs a display -- WSLg provides one on this machine):
    pixi run watch-native
  or with args:
    pixi run python rnr/scripts/watch_native_sort_window.py [M] [SIGMA] [V0] [SEED] [IC] [HET_ONLY]
    defaults: 6 0.5 0.1 7 mixed 0      (IC = mixed | demixed)
Window controls: drag = rotate, scroll = zoom, press 'r' = run/pause, arrows = move camera.

Self-test (no window): WATCH_DRY=1 pixi run python rnr/scripts/watch_native_sort_window.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
import tissue_forge as tf  # noqa: E402
from tissue_forge.models.vertex import solver as tfv  # noqa: E402
from tissue_forge.models.vertex.solver.mesh_types import BodyTypeSpec, SurfaceTypeSpec  # noqa: E402

M = int(sys.argv[1]) if len(sys.argv) > 1 else 6           # N = M^3 cells, L = M (V0=1)
SIGMA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5   # heterotypic interfacial tension
V0_ACT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1  # active self-propulsion speed (== oracle temperature)
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 7
IC = sys.argv[5] if len(sys.argv) > 5 else "mixed"         # mixed = watch it demix | demixed = watch it stay sorted
HET_ONLY = (sys.argv[6] if len(sys.argv) > 6 else os.environ.get("HET_ONLY", "0")) in ("1", "true", "True")

DRY = os.environ.get("WATCH_DRY") == "1"                  # headless self-test: build + step + recolour, no window

# Oracle params (identical to sort_periodic_oracle.py MODEL=native)
L = float(M)
BOX = [[0.0, L]] * 3
V0, K_V, K_A, A0 = 1.0, 10.0, 1.0, 5.6
DR, LTH, DT, CUT, INTERVAL = 1.0, 1e-3, 1e-3, 1.9, 10
RECOLOR_EVERY = 500          # steps between recolours (catches reconnection-created surfaces)
COL_AA, COL_BB, COL_AB, COL_BND = "blue", "red", "gold", "gray"

if not DRY and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    sys.exit("No display found ($DISPLAY / $WAYLAND_DISPLAY unset). TissueForge's window needs one.\n"
             "On WSL2 this is provided by WSLg (Win11) -- launch from a WSLg shell. To validate the\n"
             "setup headlessly instead, run:  WATCH_DRY=1 pixi run python "
             "rnr/scripts/watch_native_sort_window.py")

# Engine thread pool (a single interactive run gets ~0 benefit from >1 thread at M=6; honour TF_THREADS)
_TFTHREADS = os.environ.get("TF_THREADS")
_init_kw = {"threads": int(_TFTHREADS)} if _TFTHREADS else {}
tf.init(windowless=DRY, dim=[L, L, L], cutoff=CUT, dt=DT, **_init_kw)
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

# initial condition: mixed = random 50/50 (watch it sort); demixed = a segregated z-slab (watch it hold)
rng2 = np.random.default_rng(SEED + 1)
if IC == "demixed":
    for b in [bb for bb in bodies if bb.centroid[2] >= L / 2.0]:
        b.become(btB)
else:
    for b in bodies:
        if rng2.random() < 0.5:
            b.become(btB)

# native I<->H reconnection (same knobs as the oracle sort)
q = tfv.Quality()
q.stock_quality_operations = False
q.reconnect_length = LTH
q.reconnect_hysteresis = 0.2
q.reconnect_energy_gate = False
q.reconnect_interval = INTERVAL
q.collision_2d = False
mesh.quality = q
mesh.periodic_geometry = True

# NATIVE active drive: per-cell director (rot. diffusion Dr) + per-vertex active force, all in C++
tfv.MeshSolver.set_motility(V0_ACT, DR, SEED + 2)


def _surface_color(s):
    """Colour a surface by the types of the (1 or 2) cells it bounds: A|A blue, B|B red, A|B gold."""
    bs = s.bodies
    tys = [type_name(b) for b in bs] if bs else []
    if len(tys) >= 2:
        return COL_AB if tys[0] != tys[1] else (COL_AA if tys[0] == "A" else COL_BB)
    if len(tys) == 1:
        return COL_AA if tys[0] == "A" else COL_BB
    return COL_BND


def recolor(event=None):
    """Recolour every surface by cell-type pair. Runs once up front + periodically (tf.event.on_time),
    so surfaces created by reconnection get coloured too. Robust to stale handles."""
    seen = set()
    for b in bodies:
        for s in b.surfaces:
            sid = getattr(s, "id", None)
            if sid is None or sid in seen:
                continue
            seen.add(sid)
            try:
                col = _surface_color(s)
                s.style = tf.rendering.Style(col)
                if HET_ONLY:
                    s.style.setVisible(col == COL_AB)   # show only heterotypic interfaces (engine may ignore)
            except Exception:
                pass
    return 0


recolor()

# ----- headless self-test: prove the whole setup runs (build + motility + reconnection + recolour) -----
if DRY:
    nv0 = mesh.num_vertices
    for _ in range(200):
        tf.step()
    recolor()
    nA = sum(1 for b in bodies if type_name(b) == "A")
    print(f"DRY OK [{IC}]: N={len(bodies)} ({nA} A / {len(bodies) - nA} B), "
          f"verts {nv0}->{mesh.num_vertices} after 200 native steps, recolour ran. "
          f"Window run would call tf.run() now.", flush=True)
    sys.exit(0)

# ----- windowed run -----
print(f"=== WATCH NATIVE SORT [{IC}]: N={len(bodies)} sigma={SIGMA} v0={V0_ACT} M={M} | "
      f"gold = heterotypic interfaces (shrink as it sorts){' | HET_ONLY' if HET_ONLY else ''} ===",
      flush=True)
print("  window: drag=rotate, scroll=zoom, 'r'=run/pause. Close the window to stop.", flush=True)

# recolour periodically as reconnections reshape the mesh (signature: on_time(period, invoke_method);
# period is in simulation-time units)
tf.event.on_time(RECOLOR_EVERY * DT, recolor)

# frame the box (user can drag/scroll to adjust)
try:
    tf.system.camera_view_front()
    tf.system.camera_zoom_to(-3.0 * L)
except Exception:
    pass

tf.run()
