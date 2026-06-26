"""Worked example: lamellipodial CRAWLING as a user extension of the GPU vertex engine.

This is the template a new user copies to add a custom cell behavior. It adds NOTHING to the
validated core physics (gpu/physics_warp.force_kernel is untouched, still byte-identical). Crawling
is expressed as one (behavior, force) pair on durable PER-CELL state -- the same shape as the
built-in active drive:

  * state    : a per-cell `polarity` (vec3) + an `is_crawler` mask (int)  [CellState.add_field]
  * behavior : `persistent_repolarization` evolves polarity each step (rotational diffusion)
  * force    : `lamellipodium_force` pushes each crawler cell's LEADING vertices along its polarity

"Leading vertices" are selected IN-KERNEL from geometry (offset-from-centroid . polarity > 0), not
by tagging vertex ids -- so the selection is robust to reconnection, which creates/destroys vertices
every step (see gpu/extensions.py for why per-vertex tags are fragile and per-cell state is durable).

Run a live demo on a small two-type foam:
    pixi run python rnr/examples/crawl.py
"""
import os
import sys

import numpy as np
import warp as wp

# allow `python rnr/examples/crawl.py` (repo root on sys.path), matching rnr/scripts/*; a no-op
# when imported as a library (the test path), where rnr is already importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rnr.gpu.physics_warp import d_minimg, set_director_step  # noqa: E402


# --------------------------------------------------------------------------------------
# FORCE hook: protrusive lamellipodium on each crawler cell's leading edge
# --------------------------------------------------------------------------------------
@wp.kernel
def lamellipodium_force_kernel(
        vert_pos: wp.array(dtype=wp.vec3d), vert_alive: wp.array(dtype=wp.int32),
        v2s: wp.array2d(dtype=wp.int32), v2s_len: wp.array(dtype=wp.int32),
        s2b: wp.array2d(dtype=wp.int32), bcent: wp.array(dtype=wp.vec3d),
        polarity: wp.array(dtype=wp.vec3d), is_crawler: wp.array(dtype=wp.int32),
        box: wp.vec3d, f_mag: wp.float64,
        fout: wp.array(dtype=wp.vec3d)):
    """For each CRAWLER cell incident to vertex v, if v is on the cell's LEADING side
    (offset-from-centroid . polarity > 0), add a protrusive force f_mag*polarity. The distinct-body
    dedup mirrors physics_warp.force_kernel's active drive, so a vertex that touches one crawler cell
    via two faces is pushed once per cell. ADDS into fout (the core force accumulator g['_force'])."""
    v = wp.tid()
    if vert_alive[v] == 0:
        return
    posv = vert_pos[v]
    fv = wp.vec3d(0.0, 0.0, 0.0)
    nval = v2s_len[v]
    for a in range(nval):
        s = v2s[v, a]
        for slot in range(2):
            b = s2b[s, slot]
            if b >= 0 and is_crawler[b] == 1:
                # dedup: count this body only on its FIRST (surface, slot) occurrence for v
                seen = wp.int32(0)
                for a2 in range(a + 1):
                    s2 = v2s[v, a2]
                    for slot2 in range(2):
                        if (a2 < a or slot2 < slot) and s2b[s2, slot2] == b:
                            seen = wp.int32(1)
                if seen == 0:
                    off = d_minimg(posv - bcent[b], box)      # min-image offset from cell centroid
                    pol = polarity[b]
                    if wp.dot(off, pol) > wp.float64(0.0):     # leading-edge selector
                        fv = fv + pol * f_mag
    fout[v] = fout[v] + fv


def lamellipodium_force(f_mag: float = 1.0):
    """Return a force hook `fn(g, cells, geom)` for the protrusive lamellipodial drive. `cells` must
    carry a vec3d `polarity` and an int32 `is_crawler` field. Capture-safe: one alloc-free launch,
    no host readback."""
    f_mag = float(f_mag)

    def hook(g, cells, geom):
        wp.launch(lamellipodium_force_kernel, dim=g["cap_v"], device=g["device"],
                  inputs=[g["vert_pos"], g["vert_alive"], g["v2s"], g["v2s_len"], g["s2b"],
                          geom["bcent"], cells["polarity"], cells["is_crawler"],
                          g["box"], wp.float64(f_mag)],
                  outputs=[g["_force"]])
    return hook


# --------------------------------------------------------------------------------------
# FORCE hook (variant): whole-cell propulsion -> genuine MIGRATION (no elongation)
# --------------------------------------------------------------------------------------
@wp.kernel
def migration_force_kernel(
        vert_alive: wp.array(dtype=wp.int32),
        v2s: wp.array2d(dtype=wp.int32), v2s_len: wp.array(dtype=wp.int32),
        s2b: wp.array2d(dtype=wp.int32),
        polarity: wp.array(dtype=wp.vec3d), is_crawler: wp.array(dtype=wp.int32),
        f_mag: wp.float64, fout: wp.array(dtype=wp.vec3d)):
    """Self-propulsion of crawler cells: EVERY vertex of a crawler cell is pushed along the cell's
    polarity (dedup over distinct incident crawler cells), so the cell TRANSLATES rigidly instead of
    elongating. This is the standard self-propelled-vertex (SPV) motility the engine's built-in active
    drive uses (v0*<incident directors>), here restricted to crawler cells. Contrast
    lamellipodium_force_kernel, which pushes only the LEADING half -> a protrusion that elongates the
    cell (front runs ahead of back) rather than moving it. ADDS into fout."""
    v = wp.tid()
    if vert_alive[v] == 0:
        return
    fv = wp.vec3d(0.0, 0.0, 0.0)
    nval = v2s_len[v]
    for a in range(nval):
        s = v2s[v, a]
        for slot in range(2):
            b = s2b[s, slot]
            if b >= 0 and is_crawler[b] == 1:
                seen = wp.int32(0)
                for a2 in range(a + 1):
                    s2 = v2s[v, a2]
                    for slot2 in range(2):
                        if (a2 < a or slot2 < slot) and s2b[s2, slot2] == b:
                            seen = wp.int32(1)
                if seen == 0:
                    fv = fv + polarity[b] * f_mag
    fout[v] = fout[v] + fv


def migration_force(f_mag: float = 0.2):
    """Return a force hook `fn(g, cells, geom)` for whole-cell self-propulsion of crawler cells --
    the proper MIGRATION model (cells translate, not elongate). Requires vec3d 'polarity' + int32
    'is_crawler'. Keep f_mag near the active-drive v0 scale (~0.1-0.3); much larger distorts the foam
    faster than reconnection can relax it. Capture-safe (one alloc-free launch, no host readback)."""
    f_mag = float(f_mag)

    def hook(g, cells, geom):
        wp.launch(migration_force_kernel, dim=g["cap_v"], device=g["device"],
                  inputs=[g["vert_alive"], g["v2s"], g["v2s_len"], g["s2b"],
                          cells["polarity"], cells["is_crawler"], wp.float64(f_mag)],
                  outputs=[g["_force"]])
    return hook


# --------------------------------------------------------------------------------------
# BEHAVIOR hook: persistent random-walk repolarization of crawler cells
# --------------------------------------------------------------------------------------
@wp.kernel
def repolarize_kernel(body_alive: wp.array(dtype=wp.int32), is_crawler: wp.array(dtype=wp.int32),
                      rot_std: wp.float64, seed: wp.int32, step_dev: wp.array(dtype=wp.int32),
                      nb: wp.int32, polarity: wp.array(dtype=wp.vec3d)):
    """Rotational diffusion of each crawler cell's polarity: n <- normalize(n + rot_std*(xi - n)),
    xi ~ uniform on S^2 (the active-Brownian update, mirroring physics_warp.director_update_kernel).
    An unset polarity (|n|~0) is lazily seeded random-on-S^2. The per-step RNG key is read from the
    DEVICE scalar step_dev[0] (set outside any capture region) so a captured launch varies per replay."""
    b = wp.tid()
    if body_alive[b] == 0 or is_crawler[b] == 0:
        return
    n = polarity[b]
    st = wp.rand_init(seed, step_dev[0] * nb + b)
    if wp.dot(n, n) < wp.float64(1e-12):
        n = wp.vec3d(wp.float64(wp.randn(st)), wp.float64(wp.randn(st)), wp.float64(wp.randn(st)))
        nl = wp.length(n)
        if nl > wp.float64(1e-9):
            n = n / nl
    xi = wp.vec3d(wp.float64(wp.randn(st)), wp.float64(wp.randn(st)), wp.float64(wp.randn(st)))
    xl = wp.length(xi)
    if xl > wp.float64(1e-9):
        xi = xi / xl
    n = n + (xi - n) * rot_std
    polarity[b] = wp.normalize(n)


def persistent_repolarization(dr: float = 1.0, seed: int = 12345):
    """Return a behavior hook `fn(g, cells, step, dt)` that rotationally diffuses crawler polarity
    (persistence ~ 1/Dr). Capture-safe: routes the per-step key through g['_step_dev']
    (set_director_step), the same device scalar the built-in director update uses."""
    dr = float(dr)

    def hook(g, cells, step, dt):
        set_director_step(g, step)                     # device step scalar (capture-safe)
        rot_std = float(np.sqrt(2.0 * dr * dt))
        wp.launch(repolarize_kernel, dim=cells.n, device=g["device"],
                  inputs=[g["body_alive"], cells["is_crawler"], wp.float64(rot_std),
                          wp.int32(int(seed)), g["_step_dev"], wp.int32(cells.n), cells["polarity"]])
    return hook


# --------------------------------------------------------------------------------------
# runnable demo
# --------------------------------------------------------------------------------------
def demo(g, phys, params, dev, steps: int = 300, crawler_type: int = 1, f_mag: float = 2.0,
         dr: float = 0.1, dt: float = 5e-3, seed: int = 7):
    """Drive a prebuilt foam with the crawling extension and report how far the crawler cells'
    centroids drifted vs the non-crawlers. Uses the high-level Engine API end-to-end."""
    from rnr.gpu.engine import Engine
    from rnr.gpu.extensions import random_unit_vectors
    from rnr.gpu.device_mesh import PaddedMesh
    from rnr.gpu import physics_csr as P

    eng = Engine(g, phys, params, dt=dt, dr=dr, seed=seed,
                 threshold=0.02, dl_th=0.02, reconnect=True, interval=2)
    eng.cells.add_field("polarity", random_unit_vectors(seed=seed))
    bt = eng.cells["body_type"].numpy()
    eng.cells.add_field("is_crawler", (bt == crawler_type).astype(np.int32))
    eng.add_behavior(persistent_repolarization(dr=dr, seed=seed))
    eng.add_force(lamellipodium_force(f_mag=f_mag))

    # initial heading per cell; the DIRECTED metric (Δ·p̂) averages out the random tension/
    # reconnection rearrangement (which dominates raw |Δ|) and isolates the crawl bias
    p0 = eng.cells["polarity"].numpy().reshape(-1, 3)[:eng.cells.n].copy()
    p0 /= np.linalg.norm(p0, axis=1, keepdims=True)
    cent0 = P.compute_geometry(PaddedMesh.from_warp(g), eng.params.box).bcent[:eng.cells.n].copy()
    for _ in range(steps):
        eng.step()
    cent1 = P.compute_geometry(PaddedMesh.from_warp(g), eng.params.box).bcent[:eng.cells.n].copy()
    d = P.minimg(cent1 - cent0, eng.params.box)
    forward = np.sum(d * p0, axis=1)             # displacement projected onto the cell's heading
    speed = np.linalg.norm(d, axis=1)
    crawl = bt == crawler_type
    print(f"[crawl demo] {steps} steps, f_mag={f_mag}, Dr={dr}")
    print(f"  crawler cells     (n={int(crawl.sum())}): forward (Δ·p̂)={forward[crawl].mean():+.4f}  |Δ|={speed[crawl].mean():.4f}")
    print(f"  non-crawler cells (n={int((~crawl).sum())}): forward (Δ·p̂)={forward[~crawl].mean():+.4f}  |Δ|={speed[~crawl].mean():.4f}")
    print("  -> crawlers make positive directed progress along their polarity; non-crawlers ~0 (random)")
    return forward, crawl


def _main():
    cuda = [d for d in wp.get_devices() if d.is_cuda]
    if not cuda:
        print("no CUDA device -- the crawl demo needs the GPU engine")
        return
    dev = cuda[0]
    from rnr.gpu.foam_cache import load_or_build
    from rnr.gpu import physics_csr as P

    def _build_host():
        # init TF once (cache-miss path only; tf.init is a per-process singleton -- a 2nd call
        # hangs), mirroring conftest.vsolver / gpu_stability so the foam matches the validated one
        import tissue_forge as tf
        from tissue_forge.models.vertex import solver as tfv
        tf.init(windowless=True, dim=[60., 60., 60.], cutoff=5.0, dt=0.001)
        tfv.init()
        tfv.MeshSolver.get().get_mesh().quality = None
        from rnr.tests.test_gpu_engine import _build_unit_foam_host
        return _build_unit_foam_host(n=3, headroom=3000, ic="mixed")

    g, phys, body_type, box, v0, a0 = load_or_build(dev, n=3, ic="mixed", headroom=3000,
                                                    build_host_fn=_build_host)
    # core forces still ON (cells keep their shape/volume); the crawl force is added on top
    params = P.PhysParams(box=box, kv=10.0, v0=v0, ka=1.0, a0=a0, sigma=0.4, v_active=0.0)
    demo(g, phys, params, dev)


if __name__ == "__main__":
    _main()
