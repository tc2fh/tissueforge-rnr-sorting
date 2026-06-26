"""Gate for the regularizer energy-term hooks (examples/energy_terms.py): the edge-length penalty
and the per-face area penalty. Each hook claims to be the force of a specific energy E; the right
test for a coupled energy term (no closed-form per-vertex displacement) is a FINITE-DIFFERENCE
GRADIENT check -- launch the hook to get the device force, then confirm force == -dE/dx by central-
differencing the SAME energy on the host. This catches sign flips, factor-of-2s, and gradient
mistakes that a "looks reasonable" test would miss.

Run: pixi run python -m pytest rnr/tests/test_energy_terms.py -q
"""
import numpy as np

from ..gpu import physics_csr as P
from ..gpu import physics_warp as W
from ..gpu.device_mesh import PaddedMesh
from ..examples.energy_terms import edge_length_penalty, face_area_penalty
from .test_gpu_engine import _setup_unit_foam, _cuda_or_skip


def _device_force(g, hook):
    """Launch ONLY this hook (core forces excluded) and return its per-vertex force array."""
    gw = W.compute_geometry_warp(g)          # allocates g['_force'] + gives geom (sarea/scent)
    g["_force"].zero_()
    hook(g, None, gw)                         # the regularizer hooks ignore `cells`
    return g["_force"].numpy().copy()


def _fd_gradient_check(pm, box, energy_fn, force, test_vs, eps=1e-5):
    """Assert force[v] == -dE/dx_v (central difference) for each tested vertex. Returns worst error."""
    base = pm.vert_pos.copy()
    worst = 0.0
    for v in test_vs:
        for d in range(3):
            pm.vert_pos = base.copy(); pm.vert_pos[v, d] += eps
            ep = energy_fn(pm, box)
            pm.vert_pos = base.copy(); pm.vert_pos[v, d] -= eps
            em = energy_fn(pm, box)
            grad = (ep - em) / (2.0 * eps)
            err = abs(force[v, d] + grad)        # force should equal -grad -> force + grad ~ 0
            worst = max(worst, err)
            assert err <= 1e-3 * max(1.0, abs(grad)) + 1e-5, \
                f"vertex {v} coord {d}: force {force[v, d]:.6e} != -grad {-grad:.6e} (err {err:.2e})"
    pm.vert_pos = base
    return worst


def test_edge_length_force_is_energy_gradient(vsolver):
    """edge_length_penalty force == -d/dx of E = (k/2) sum_{face edges} (|e| - l0)^2."""
    dev = _cuda_or_skip()
    g, phys, body_type, box, v0, a0 = _setup_unit_foam(dev, n=2)
    k, l0 = 3.0, 0.5
    force = _device_force(g, edge_length_penalty(k, l0))
    pm = PaddedMesh.from_warp(g)

    def energy(pm_, box_):
        vp = pm_.vert_pos
        e_tot = 0.0
        for s in range(pm_.n_s_used):
            if not pm_.surf_alive[s]:
                continue
            L = int(pm_.s2v_len[s])
            ring = [int(pm_.s2v[s, i]) for i in range(L)]
            for i in range(L):
                e = P.minimg(vp[ring[(i + 1) % L]] - vp[ring[i]], box_)
                e_tot += 0.5 * k * (np.linalg.norm(e) - l0) ** 2
        return e_tot

    test_vs = list(range(pm.n_v_used))[:6]
    worst = _fd_gradient_check(pm, box, energy, force, test_vs)
    print(f"\n[edge-length FD] worst |force + grad| over {len(test_vs)} verts = {worst:.2e}")


def test_face_area_force_is_energy_gradient(vsolver):
    """face_area_penalty force == -d/dx of E = (k/2) sum_{faces} (A_s - a0)^2 (areas from the host
    geometry that the device kernel mirrors -- centroid-fan, vertex-mean centroid)."""
    dev = _cuda_or_skip()
    g, phys, body_type, box, v0, a0 = _setup_unit_foam(dev, n=2)
    k, a0t = 2.0, 0.3
    force = _device_force(g, face_area_penalty(k, a0t))
    pm = PaddedMesh.from_warp(g)

    def energy(pm_, box_):
        sarea = P.compute_geometry(pm_, box_).sarea
        e_tot = 0.0
        for s in range(pm_.n_s_used):
            if pm_.surf_alive[s]:
                e_tot += 0.5 * k * (sarea[s] - a0t) ** 2
        return e_tot

    test_vs = list(range(pm.n_v_used))[:6]
    worst = _fd_gradient_check(pm, box, energy, force, test_vs)
    print(f"\n[face-area FD] worst |force + grad| over {len(test_vs)} verts = {worst:.2e}")
