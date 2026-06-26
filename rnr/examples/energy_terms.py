"""Worked examples: two GLOBAL regularizer energy terms added as force hooks (gpu/extensions.py) --
an edge-length penalty and a per-face area penalty. They demonstrate "add a new energy term to ALL
vertices" (no per-cell state needed), and together they cure the spiky-shard distortion a strong
LEADING-EDGE protrusion (examples/crawl.lamellipodium_force) produces: the 3D engine's only shape
springs are a per-cell VOLUME term and a per-cell TOTAL-area term, neither of which penalizes a
single long thin edge or a single distorted face, so nothing stops a localized protrusion from
drawing a needle. These two terms add exactly that missing local regularization.

Each is a force hook `fn(g, cells, geom)` that ADDS into the core force accumulator g['_force'] after
the fused core kernel (the validated core is untouched). Both are the gradient of a smooth energy, so
they are validated by a finite-difference gradient check (tests/test_energy_terms.py), not a closed
form -- the right test for any coupled energy term.

  edge_length_penalty(k, l0):  E = (k/2) * sum_over_face_edges (|e| - l0)^2
  face_area_penalty(k, a0):    E = (k/2) * sum_over_faces    (A_s - a0)^2

Targets l0/a0 are single scalars (durable; no per-edge/per-face reference state, which reconnection
would invalidate) -- typically the foam's mean edge length / mean face area, so the terms regularize
toward a UNIFORM foam. Use `mean_edge_length` / `mean_face_area` to pick them.
"""
import os
import sys

import numpy as np
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rnr.gpu.physics_warp import d_area_grad, d_minimg  # noqa: E402


# --------------------------------------------------------------------------------------
# edge-length penalty  (Hookean spring on every face-edge toward target length l0)
# --------------------------------------------------------------------------------------
@wp.kernel
def edge_length_force_kernel(
        vert_pos: wp.array(dtype=wp.vec3d), vert_alive: wp.array(dtype=wp.int32),
        v2s: wp.array2d(dtype=wp.int32), v2s_len: wp.array(dtype=wp.int32),
        s2v: wp.array2d(dtype=wp.int32), s2v_len: wp.array(dtype=wp.int32),
        box: wp.vec3d, k: wp.float64, l0: wp.float64, fout: wp.array(dtype=wp.vec3d)):
    """Per-vertex force of E = (k/2) sum_{face edges} (|e| - l0)^2. For vertex v and each incident
    face, v borders two ring-edges (to its prev and next neighbour); each contributes a spring
    k*(|e| - l0) along the unit edge direction toward the neighbour (pulls v in if the edge is too
    long -> resists the needle; pushes out if too short). Edges shared by m faces are summed m times
    (the per-face-perimeter weighting); the matching energy in the test counts each face-edge once so
    force == -dE/dx. ADDS into fout."""
    v = wp.tid()
    if vert_alive[v] == 0:
        return
    posv = vert_pos[v]
    fv = wp.vec3d(0.0, 0.0, 0.0)
    nval = v2s_len[v]
    for a in range(nval):
        s = v2s[v, a]
        L = s2v_len[s]
        iv = wp.int32(-1)
        for j in range(L):
            if s2v[s, j] == v:
                iv = j
        if iv >= 0:
            w_next = s2v[s, (iv + 1) % L]
            w_prev = s2v[s, (iv - 1 + L) % L]
            for side in range(2):
                w = w_next
                if side == 1:
                    w = w_prev
                e = d_minimg(vert_pos[w] - posv, box)
                el = wp.length(e)
                if el > wp.float64(1e-12):
                    fv = fv + (e / el) * (k * (el - l0))
    fout[v] = fout[v] + fv


def edge_length_penalty(k: float, l0: float):
    """Return a force hook `fn(g, cells, geom)` for an edge-length spring toward `l0` with modulus
    `k`. Penalizes long edges (the spikes) and short edges alike; for a one-sided 'cap long edges
    only' variant, clamp (el - l0) to >= 0 in the kernel. Capture-safe (one alloc-free launch)."""
    k, l0 = float(k), float(l0)

    def hook(g, cells, geom):
        wp.launch(edge_length_force_kernel, dim=g["cap_v"], device=g["device"],
                  inputs=[g["vert_pos"], g["vert_alive"], g["v2s"], g["v2s_len"],
                          g["s2v"], g["s2v_len"], g["box"], wp.float64(k), wp.float64(l0)],
                  outputs=[g["_force"]])
    return hook


# --------------------------------------------------------------------------------------
# per-face area penalty  (each face toward target area a0; the missing LOCAL area term)
# --------------------------------------------------------------------------------------
@wp.kernel
def face_area_force_kernel(
        vert_pos: wp.array(dtype=wp.vec3d), vert_alive: wp.array(dtype=wp.int32),
        v2s: wp.array2d(dtype=wp.int32), v2s_len: wp.array(dtype=wp.int32),
        s2v: wp.array2d(dtype=wp.int32), s2v_len: wp.array(dtype=wp.int32),
        scent: wp.array(dtype=wp.vec3d), sarea: wp.array(dtype=wp.float64),
        box: wp.vec3d, k: wp.float64, a0: wp.float64, fout: wp.array(dtype=wp.vec3d)):
    """Per-vertex force of E = (k/2) sum_{faces} (A_s - a0)^2 -- the per-cell TOTAL-area term
    (physics_warp.force_kernel) applied PER FACE instead of summed over the body, so it resists a
    single distorted face (the engine's total-area term cannot: a cell can redistribute area among
    faces at ~constant total). Reuses the engine's area gradient d_area_grad, which returns the
    gradient of the UNHALVED cross-product area sum = -2*dA_s/dx_v (the engine's K_A convention omits
    the 1/2). So to implement the standard (k/2)(...)^2 energy, force = -dE/dx = -k(A_s-a0)*dA/dx =
    0.5*k*(A_s-a0)*d_area_grad -- the 0.5 is NOT optional (a finite-difference gradient test pins it;
    without it the force is exactly 2x too stiff). ADDS into fout."""
    v = wp.tid()
    if vert_alive[v] == 0:
        return
    fv = wp.vec3d(0.0, 0.0, 0.0)
    nval = v2s_len[v]
    for a in range(nval):
        s = v2s[v, a]
        ag = d_area_grad(s2v, s2v_len, vert_pos, s, v, scent[s], box)
        fv = fv + ag * (wp.float64(0.5) * k * (sarea[s] - a0))
    fout[v] = fout[v] + fv


def face_area_penalty(k: float, a0: float):
    """Return a force hook `fn(g, cells, geom)` for a per-face area spring toward `a0` with modulus
    `k`. Requires geom from compute_geometry (uses geom['sarea'] + geom['scent']). Capture-safe."""
    k, a0 = float(k), float(a0)

    def hook(g, cells, geom):
        wp.launch(face_area_force_kernel, dim=g["cap_v"], device=g["device"],
                  inputs=[g["vert_pos"], g["vert_alive"], g["v2s"], g["v2s_len"],
                          g["s2v"], g["s2v_len"], geom["scent"], geom["sarea"],
                          g["box"], wp.float64(k), wp.float64(a0)],
                  outputs=[g["_force"]])
    return hook


# --------------------------------------------------------------------------------------
# target helpers (pick l0 / a0 from a mesh; typically the foam mean)
# --------------------------------------------------------------------------------------
def mean_edge_length(pm, box) -> float:
    """Mean length of all live face-edges of a PaddedMesh (the natural edge-spring target l0)."""
    from rnr.gpu.physics_csr import minimg
    vp = pm.vert_pos
    tot, n = 0.0, 0
    for s in range(pm.n_s_used):
        if not pm.surf_alive[s]:
            continue
        L = int(pm.s2v_len[s])
        ring = [int(pm.s2v[s, i]) for i in range(L)]
        for i in range(L):
            e = minimg(vp[ring[(i + 1) % L]] - vp[ring[i]], box)
            tot += float(np.linalg.norm(e)); n += 1
    return tot / n if n else 0.0


def mean_face_area(pm, box) -> float:
    """Mean area of all live faces (the natural per-face area target a0)."""
    from rnr.gpu import physics_csr as P
    geom = P.compute_geometry(pm, box)
    alive = np.array([bool(pm.surf_alive[s]) for s in range(pm.n_s_used)])
    return float(geom.sarea[:pm.n_s_used][alive].mean()) if alive.any() else 0.0
