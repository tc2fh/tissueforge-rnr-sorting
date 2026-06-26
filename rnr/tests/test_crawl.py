"""Gate for the extensibility seams (gpu/extensions.py, gpu/engine.Engine) + the worked
lamellipodial-crawling example (examples/crawl.py).

Two things must hold for the architecture to be both CORRECT and SAFE:

  * test_empty_hooks_byte_identical: an Engine with NO hooks reproduces the bare core forward_step
    bit-for-bit. The hook injection must be purely additive -- the validated core path is untouched
    when no extension is registered.
  * test_known_polarity_known_displacement: with the core forces zeroed and ONE crawler cell given a
    known polarity (+x), exactly the cell's LEADING vertices move by exactly dt*f_mag along +x and
    nothing else moves. This pins down the whole custom-force path: CellState fields -> the force
    registry -> the geometric leading-edge selector -> the integrator.

Run: pixi run python -m pytest rnr/tests/test_crawl.py -q
"""
import numpy as np
import warp as wp

from ..gpu import engine as E
from ..gpu import physics_csr as P
from ..gpu.device_mesh import PaddedMesh
from ..gpu.engine import Engine
from ..gpu.extensions import constant_vector
from ..examples.crawl import lamellipodium_force, migration_force
from .test_gpu_engine import _setup_unit_foam, _cuda_or_skip


def test_empty_hooks_byte_identical(vsolver):
    """Engine.step() with no behaviors/forces == bare forward_step, bit-for-bit. The same start
    state is run both ways (reset vert_pos in between; reconnect off + v_active 0 -> the ONLY mutated
    state is vert_pos, so a position reset is a full reset) and the final positions must be equal."""
    dev = _cuda_or_skip()
    g, phys, body_type, box, v0, a0 = _setup_unit_foam(dev)
    params = P.PhysParams(box=box, kv=10.0, v0=v0, ka=1.0, a0=a0, sigma=0.5, v_active=0.0)
    dt = 5e-3
    nv = int(g["n_used"].numpy()[0])
    snap = wp.clone(g["vert_pos"])

    for step in range(8):                       # bare core path (the existing hot loop)
        E.forward_step(g, phys, params, dt, dr=0.0, seed=0, step=step, reconnect=False)
    pos_bare = g["vert_pos"].numpy()[:nv].copy()

    wp.copy(g["vert_pos"], snap)                 # reset to the identical start
    eng = Engine(g, phys, params, dt=dt, dr=0.0, seed=0, reconnect=False)
    for _ in range(8):                           # same steps, through the hook-enabled path, 0 hooks
        eng.step()
    pos_eng = g["vert_pos"].numpy()[:nv].copy()

    assert np.array_equal(pos_bare, pos_eng), \
        f"empty-hook Engine diverged from the core step (max |Δ|={np.abs(pos_bare-pos_eng).max():.2e})"


def test_known_polarity_known_displacement(vsolver):
    """One crawler cell (body 0), polarity +x, core forces OFF: after one step, exactly body 0's
    leading vertices have moved by exactly dt*f_mag in +x; every other vertex is unmoved."""
    dev = _cuda_or_skip()
    g, phys, body_type, box, v0, a0 = _setup_unit_foam(dev)
    # core forces all zeroed -> the crawl hook is the ONLY force acting
    params = P.PhysParams(box=box, kv=0.0, v0=v0, ka=0.0, a0=a0, sigma=0.0, v_active=0.0)
    dt, f_mag, margin = 1e-2, 1.0, 1e-6
    pol = np.array([1.0, 0.0, 0.0])

    eng = Engine(g, phys, params, dt=dt, dr=0.0, seed=0, reconnect=False)
    eng.cells.add_field("polarity", constant_vector(pol))
    mask = np.zeros(eng.cells.n, dtype=np.int32)
    mask[0] = 1                                  # only body 0 crawls
    eng.cells.add_field("is_crawler", mask)
    eng.add_force(lamellipodium_force(f_mag=f_mag))

    # --- host-side prediction of the displacement field (before stepping) ---
    pm = PaddedMesh.from_warp(g)
    nv = pm.n_v_used
    bcent0 = P.compute_geometry(pm, box).bcent[0]
    before = g["vert_pos"].numpy()[:nv].copy()
    body0_verts = set()
    for s in range(pm.n_s_used):
        if not pm.surf_alive[s]:
            continue
        if pm.s2b[s, 0] == 0 or pm.s2b[s, 1] == 0:
            for k in range(int(pm.s2v_len[s])):
                body0_verts.add(int(pm.s2v[s, k]))
    proj = {v: float(P.minimg(before[v] - bcent0, box) @ pol) for v in body0_verts}
    leading = [v for v, p in proj.items() if p > margin]
    trailing = [v for v, p in proj.items() if p < -margin]
    assert leading and trailing, "degenerate test: body 0 has no clear leading/trailing split"

    # --- one step through the full Engine path, then compare to the prediction ---
    eng.step()
    after = g["vert_pos"].numpy()[:nv]
    delta = P.minimg(after - before, box)

    # (1) all motion is purely +x (no off-axis force)
    assert np.abs(delta[:, 1]).max() < 1e-9 and np.abs(delta[:, 2]).max() < 1e-9, "off-axis motion"
    # (2) leading vertices of body 0 moved by exactly dt*f_mag
    for v in leading:
        assert abs(delta[v, 0] - dt * f_mag) < 1e-9, f"leading vertex {v} Δx={delta[v,0]:.3e}"
    # (3) trailing vertices of body 0 did NOT move
    for v in trailing:
        assert abs(delta[v, 0]) < 1e-9, f"trailing vertex {v} moved Δx={delta[v,0]:.3e}"
    # (4) vertices not on body 0 did NOT move
    others = [v for v in range(nv) if v not in body0_verts]
    assert np.abs(delta[others, 0]).max() < 1e-9, "a non-crawler-cell vertex moved"


def test_migration_whole_cell_displacement(vsolver):
    """Whole-cell migration drive (the FIX for cell elongation): one crawler cell (body 0), polarity
    +x, core forces OFF -> after one step EVERY vertex of body 0 translates by exactly dt*f_mag in +x
    (rigid translation, no leading/trailing split -> no elongation), and nothing else moves."""
    dev = _cuda_or_skip()
    g, phys, body_type, box, v0, a0 = _setup_unit_foam(dev)
    params = P.PhysParams(box=box, kv=0.0, v0=v0, ka=0.0, a0=a0, sigma=0.0, v_active=0.0)
    dt, f_mag = 1e-2, 1.0
    pol = np.array([1.0, 0.0, 0.0])

    eng = Engine(g, phys, params, dt=dt, dr=0.0, seed=0, reconnect=False)
    eng.cells.add_field("polarity", constant_vector(pol))
    mask = np.zeros(eng.cells.n, dtype=np.int32)
    mask[0] = 1
    eng.cells.add_field("is_crawler", mask)
    eng.add_force(migration_force(f_mag=f_mag))

    pm = PaddedMesh.from_warp(g)
    nv = pm.n_v_used
    before = g["vert_pos"].numpy()[:nv].copy()
    body0_verts = set()
    for s in range(pm.n_s_used):
        if not pm.surf_alive[s]:
            continue
        if pm.s2b[s, 0] == 0 or pm.s2b[s, 1] == 0:
            for k in range(int(pm.s2v_len[s])):
                body0_verts.add(int(pm.s2v[s, k]))

    eng.step()
    delta = P.minimg(g["vert_pos"].numpy()[:nv] - before, box)
    b0 = sorted(body0_verts)
    others = [v for v in range(nv) if v not in body0_verts]
    # EVERY body-0 vertex translated by exactly dt*f_mag in +x (rigid, no leading/trailing split)
    assert np.allclose(delta[b0, 0], dt * f_mag, atol=1e-9), "a body-0 vertex did not translate by dt*f_mag"
    assert np.abs(delta[b0, 1]).max() < 1e-9 and np.abs(delta[b0, 2]).max() < 1e-9, "off-axis motion"
    # nothing not on body 0 moved
    assert np.abs(delta[others]).max() < 1e-9, "a non-crawler-cell vertex moved"
