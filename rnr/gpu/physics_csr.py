"""Gate E (Stage-1 physics), HOST reference: geometry + the four physics forces of the
TissueForge 3D vertex model, re-derived against the CSR / PaddedMesh SoA.

This is the host numpy oracle the Warp kernels (`physics_warp.py`) are gated against -- the
same host-reference-then-kernel methodology that `reconnect_csr.py` -> `reconnect_warp.py`
used for the RNR path. Each formula is re-derived from the Okuda/TF energetics by READING
(never copying) the LGPL TissueForge actors; the actor + file:line it mirrors is cited at
each function.

SCOPE (CLAUDE.md / GPU-port handoff): Gate E is the SORTING PHYSICS -- per-cell volume
elasticity, per-cell surface-area elasticity, heterotypic interfacial tension (adhesion),
and the active self-propulsion drive -- plus the overdamped integrator. TF additionally
auto-binds two mesh-hygiene regularizers on every SurfaceType (`FlatSurfaceConstraint` +
`ConvexPolygonConstraint`, lam=0.1; tfSurface.cpp:2348-2349). Those keep faces planar/convex
but are NOT the sorting physics, so they are intentionally OMITTED from this port; the force
gate validates against a TF oracle with those two zeroed (see test_gpu_physics_csr.py).

Geometry conventions match TF EXACTLY (so a CSR freshly extracted from a TF mesh reproduces
TF's `s.area` / `b.volume` / `b.centroid` to round-off):
  * surface centroid  = vertex average about the FIRST-vertex floating origin, min-image
                        (tfSurface::positionChanged, tfSurface.cpp:1072-1077)
  * surface area      = (1/2) sum_i |triNorm_i|,  triNorm_i = cross(p_i - c, p_{i+1} - c)
                        with p_k = posNear(vertex_k, c)         (tfSurface.cpp:1080-1091)
  * surface uNormal   = sum_i triNorm_i (unnormalized)          (tfSurface.cpp:1088)
  * body area         = sum_s area_s                            (tfBody.cpp:215)
  * body centroid     = area-weighted avg of surface centroids about the first-surface
                        floating origin                         (tfBody.cpp:212-217)
  * body volume       = sum_s sense(s,b) * dot(uNormal_s, posNear(c_s, c_b)) / 6,
                        sense = +1 if b==b1 else -1 (periodic branch; tfBody.cpp:221-224)
  * orientSign        = -1 and volume:=|volume| if signed volume < 0  (tfBody.cpp:233-237)

Periodicity is the engine's minimum-image over the universe box (tf_mesh_metrics.cpp:59-88);
pass `box=(Lx,Ly,Lz)` (a non-positive component disables wrapping on that axis, matching a
finite cluster).
"""
from dataclasses import dataclass

import numpy as np

from .device_mesh import PaddedMesh


# ======================================================================================
# physics parameters + per-body physics state (type + director live alongside the mesh)
# ======================================================================================
@dataclass
class PhysParams:
    """Force-model parameters (mirror the production sort, sort_periodic_oracle.py)."""
    box: np.ndarray          # (3,) periodic box; component <= 0 disables wrap on that axis
    kv: float = 10.0         # volume modulus  (VolumeConstraint.lam,  K_V)
    v0: float = 1.0          # target volume   (VolumeConstraint.constr, V0)
    ka: float = 1.0          # area modulus    (SurfaceAreaConstraint.lam, K_A)
    a0: float = 5.6          # target body area (SurfaceAreaConstraint.constr, A0)
    sigma: float = 0.0       # heterotypic interfacial tension (Adhesion.lam on the A-B pair)
    v_active: float = 0.0    # active self-propulsion speed v0 (0 => drive off)

    def box_arr(self) -> np.ndarray:
        return np.asarray(self.box, dtype=np.float64)


@dataclass
class PhysState:
    """Per-body physics state, aligned 1:1 with the mesh body indices (b = 0..nb-1).

    `body_type`: int label per body (e.g. 0=A, 1=B). Adhesion is active on a pair of DISTINCT
    types (the heterotypic interface), matching the production setup where only the A-B
    Adhesion(sigma) actor is nonzero (mesh_types.bind_adhesion).
    `body_director`: unit self-propulsion direction per body (the active drive); ignored when
    PhysParams.v_active == 0.
    """
    body_type: np.ndarray        # (nb,) int32
    body_director: np.ndarray    # (nb,3) float64


def phys_state_from_tf(bodies, type_label) -> PhysState:
    """Build a PhysState aligned to extract_csr's body ordering (ascending TF id).

    `type_label(body) -> int` classifies each body (e.g. by its TF type id). Directors are
    read from `body.director` (the native drive; FVector3, zero if unset).
    """
    order = sorted(bodies, key=lambda b: b.id)
    btype = np.array([int(type_label(b)) for b in order], dtype=np.int32)
    bdir = np.array([[b.director[0], b.director[1], b.director[2]] for b in order],
                    dtype=np.float64)
    return PhysState(body_type=btype, body_director=bdir)


# ======================================================================================
# minimum-image helper (engine convention, tf_mesh_metrics.cpp:59-71)
# ======================================================================================
def minimg(d, box) -> np.ndarray:
    """Minimum-image displacement(s). `d` is (...,3); `box` is (3,). A non-positive box
    component leaves that axis unwrapped. TF wraps each component into (-L/2, L/2] via a
    strict-inequality while-loop; for |d| < 1.5 L that equals d - L*round(d/L) (numpy's
    round-half-to-even reproduces TF's strict-> at the +/-L/2 endpoints)."""
    d = np.asarray(d, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    out = d.copy()
    for k in range(3):
        if box[k] > 0.0:
            out[..., k] = d[..., k] - box[k] * np.round(d[..., k] / box[k])
    return out


def _posNear(p, origin, box) -> np.ndarray:
    """origin + minImage(p - origin) -- the periodic image of p closest to origin."""
    return origin + minimg(p - origin, box)


# ======================================================================================
# GEOMETRY
# ======================================================================================
@dataclass
class Geometry:
    """Cached per-surface + per-body geometry (the force kernels read these)."""
    scent: np.ndarray    # (ns,3) surface centroid
    sarea: np.ndarray    # (ns,)  surface area
    snorm: np.ndarray    # (ns,3) surface UNNORMALIZED normal (sum of triangle normals)
    bvol: np.ndarray     # (nb,)  body volume (|signed|, >= 0)
    barea: np.ndarray    # (nb,)  body total surface area
    bcent: np.ndarray    # (nb,3) body centroid
    borient: np.ndarray  # (nb,)  +1, or -1 if the signed volume was negative


def surface_geometry(pm: PaddedMesh, box) -> tuple:
    """Per-surface centroid, area, unnormalized normal. Mirrors Surface::positionChanged
    (tfSurface.cpp:1064-1092)."""
    ns = pm.n_s_used
    scent = np.zeros((ns, 3))
    sarea = np.zeros(ns)
    snorm = np.zeros((ns, 3))
    for s in range(ns):
        if not pm.surf_alive[s]:
            continue
        L = int(pm.s2v_len[s])
        ring = pm.s2v[s, :L]
        P = pm.vert_pos[ring]                       # (L,3) raw vertex positions
        origin = P[0]                               # first-vertex floating origin
        cen = origin + minimg(P - origin, box).mean(axis=0)
        Pc = cen + minimg(P - cen, box)             # each vertex imaged near the centroid
        Pp = np.roll(Pc, -1, axis=0)                # the next vertex (cyclic)
        tn = np.cross(Pc - cen, Pp - cen)           # (L,3) per-edge triangle normal
        scent[s] = cen
        snorm[s] = tn.sum(axis=0)
        sarea[s] = 0.5 * np.linalg.norm(tn, axis=1).sum()
    return scent, sarea, snorm


def body_geometry(pm: PaddedMesh, scent, sarea, snorm, box) -> tuple:
    """Per-body volume, area, centroid, orientSign. Mirrors Body::updateInternals
    (tfBody.cpp:202-237), periodic branch."""
    nb = pm.nb
    bvol = np.zeros(nb)
    barea = np.zeros(nb)
    bcent = np.zeros((nb, 3))
    borient = np.ones(nb)
    for b in range(nb):
        if not pm.body_alive[b]:
            continue
        L = int(pm.b2s_len[b])
        surfs = pm.b2s[b, :L]
        a = sarea[surfs]
        atot = a.sum()
        barea[b] = atot
        origin = scent[surfs[0]]                    # first-surface floating origin
        cen = origin + minimg(scent[surfs] - origin, box)     # (L,3) centroids imaged near origin
        bc = (cen * a[:, None]).sum(axis=0) / atot
        bcent[b] = bc
        sc = bc + minimg(scent[surfs] - bc, box)              # surface centroids imaged near body centroid
        sense = np.where(pm.s2b[surfs, 0] == b, 1.0, -1.0)    # +1 if b==b1 else -1
        v = (np.einsum("ij,ij->i", snorm[surfs], sc) * sense / 6.0).sum()
        if v < 0.0:
            borient[b] = -1.0
            v = -v
        bvol[b] = v
    return bvol, barea, bcent, borient


def compute_geometry(pm: PaddedMesh, box) -> Geometry:
    scent, sarea, snorm = surface_geometry(pm, box)
    bvol, barea, bcent, borient = body_geometry(pm, scent, sarea, snorm, box)
    return Geometry(scent=scent, sarea=sarea, snorm=snorm,
                    bvol=bvol, barea=barea, bcent=bcent, borient=borient)


# ======================================================================================
# helpers for the force loops
# ======================================================================================
def _ring_index(pm: PaddedMesh, s: int, v: int) -> int:
    L = int(pm.s2v_len[s])
    row = pm.s2v[s, :L]
    hits = np.where(row == v)[0]
    return int(hits[0]) if len(hits) else -1


def incident_bodies(pm: PaddedMesh, v: int) -> list:
    """The distinct bodies incident to vertex v (union of s2b over its surfaces), in a
    deterministic order (ascending body index)."""
    bs = set()
    for s in pm.v2s[v, :int(pm.v2s_len[v])]:
        s = int(s)
        for k in range(2):
            b = int(pm.s2b[s, k])
            if b >= 0:
                bs.add(b)
    return sorted(bs)


def _area_gradient_of_surface(pm, s, v, scent_s, box):
    """The area gradient dA_s/dx_v of surface s w.r.t. vertex v: the common inner loop of
    SurfaceAreaConstraint::force (tfSurfaceAreaConstraint.cpp:48-62) and Adhesion_force_Body
    (tfAdhesion.cpp:87-101). Returns a (3,) vector (zero if v is not on s)."""
    L = int(pm.s2v_len[s])
    ring = pm.s2v[s, :L]
    g_tot = np.zeros(3)
    for i in range(L):
        vc = int(ring[i])
        vn = int(ring[(i + 1) % L])
        posvc = _posNear(pm.vert_pos[vc], scent_s, box)
        posvn = _posNear(pm.vert_pos[vn], scent_s, box)
        tn = np.cross(posvc - scent_s, posvn - scent_s)
        if not np.any(tn):
            continue
        g = (posvc - posvn) / L
        if vc == v:
            g = g + (posvn - scent_s)
        elif vn == v:
            g = g - (posvc - scent_s)
        g_tot += np.cross(tn / np.linalg.norm(tn), g)
    return g_tot


# ======================================================================================
# FORCES  (per-vertex; the four sorting-physics actors + active drive)
# ======================================================================================
def forces(pm: PaddedMesh, geom: Geometry, params: PhysParams, state: PhysState) -> np.ndarray:
    """Per-vertex total force (cap_v, 3); dead/unused rows are zero. The sum of:

      VolumeConstraint   (per incident body)  tfVolumeConstraint.cpp:40-63
      SurfaceAreaConstraint (body variant)     tfSurfaceAreaConstraint.cpp:39-65
      Adhesion (body variant, heterotypic)     tfAdhesion.cpp:69-105
      active self-propulsion v0*<directors>    tfMeshSolver.cpp:88-102
    """
    box = params.box_arr()
    f = np.zeros((pm.cap_v, 3))
    g = geom
    for v in range(pm.n_v_used):
        if not pm.vert_alive[v]:
            continue
        posv = pm.vert_pos[v]
        fv = np.zeros(3)
        bodies_v = incident_bodies(pm, v)

        for src in bodies_v:
            # surfaces of v that this body defines (b1 or b2)
            vsurfs = [int(s) for s in pm.v2s[v, :int(pm.v2s_len[v])]
                      if pm.s2b[int(s), 0] == src or pm.s2b[int(s), 1] == src]

            # ---- VolumeConstraint (tfVolumeConstraint.cpp:40-63) ----
            if params.kv != 0.0:
                ftot = np.zeros(3)
                for s in vsurfs:
                    L = int(pm.s2v_len[s])
                    i = _ring_index(pm, s, v)
                    vp = int(pm.s2v[s, (i + 1) % L])     # neighborVertices: vp = ring[i+1]
                    vn = int(pm.s2v[s, (i - 1) % L])     #                    vn = ring[i-1]
                    scent_near = _posNear(g.scent[s], posv, box)
                    edge = minimg(pm.vert_pos[vn] - pm.vert_pos[vp], box)  # relpos(vn, vp)
                    sense = 1.0 if pm.s2b[s, 0] == src else -1.0
                    ftot += (np.cross(scent_near, edge) + g.snorm[s] / L) * sense
                fv += ftot * g.borient[src] * (params.kv * (params.v0 - g.bvol[src]) / 3.0)

            # ---- SurfaceAreaConstraint, body variant (tfSurfaceAreaConstraint.cpp:39-65) ----
            if params.ka != 0.0:
                ftot = np.zeros(3)
                for s in vsurfs:
                    ftot += _area_gradient_of_surface(pm, s, v, g.scent[s], box)
                fv += ftot * (params.ka * (g.barea[src] - params.a0))

            # ---- Adhesion, body variant: heterotypic interfacial tension (tfAdhesion.cpp:69-105) ----
            if params.sigma != 0.0:
                ftot = np.zeros(3)
                for s in vsurfs:
                    b0, b1 = int(pm.s2b[s, 0]), int(pm.s2b[s, 1])
                    other = b1 if b0 == src else (b0 if b1 == src else -1)
                    if other < 0:
                        continue
                    if state.body_type[src] == state.body_type[other]:
                        continue                      # homotypic face: no adhesion tension
                    ftot += _area_gradient_of_surface(pm, s, v, g.scent[s], box)
                fv += ftot * (0.25 * params.sigma)

        # ---- active self-propulsion (tfMeshSolver.cpp:88-102) ----
        if params.v_active > 0.0 and bodies_v:
            dirsum = state.body_director[bodies_v].sum(axis=0)
            fv += dirsum * (params.v_active / len(bodies_v))

        f[v] = fv
    return f


def forces_live(pm: PaddedMesh, geom: Geometry, params: PhysParams, state: PhysState) -> np.ndarray:
    """Convenience: the (n_v_used, 3) prefix of `forces` (the live/used rows)."""
    return forces(pm, geom, params, state)[:pm.n_v_used]


def integrate(pm: PaddedMesh, force: np.ndarray, dt: float, box) -> None:
    """Overdamped forward Euler x += dt*f (mu=1), then wrap each live vertex into [0,L) per
    axis (the engine's PERIODIC_FULL particle BC; a non-positive box component is unwrapped).
    Mutates pm.vert_pos in place -- the host mirror of integrate_kernel."""
    box = np.asarray(box, dtype=np.float64)
    for v in range(pm.n_v_used):
        if not pm.vert_alive[v]:
            continue
        p = pm.vert_pos[v] + dt * force[v]
        for k in range(3):
            if box[k] > 0.0:
                p[k] = p[k] - box[k] * np.floor(p[k] / box[k])
        pm.vert_pos[v] = p
