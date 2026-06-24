"""Phase-1 I<->H reconnection (the 3D T1) for TissueForge -- the MUTATE half.

Strategy A (user-chosen): manual surface-list surgery, mirroring tvm's I_H/H_I pointer
walk but reimplemented against TissueForge's API (TF has no explicit Edge/Polygon
objects, and its low-level vertex/surface ops edit only ONE side of each adjacency, so
every change is mirrored on both sides -- see rnr/PORTING_NOTES.md for the verified seam
recipe). Reimplemented from the Okuda 2013 equations, NOT copied from the GPL reference.

Each direction is split into a CHECK half (predicate: structurally valid + Condition-4
clear?) and a MUTATE half (topology surgery + Okuda Appendix-1 vertex placement), so the
eventual native C++ MeshQualityOperation is a translation, mirroring tfMeshQuality.cpp.

Vertex placement (Okuda et al. 2013, Appendix 1):
  * I->H (edge 10-11 -> triangle 7-8-9): Eqs. 46-56. The three triangle vertices are
    placed in the plane normal to the edge through its midpoint, along projected averages
    of the directions to the six outer neighbours, scaled by Delta_l_th / L_max.
  * H->I (triangle 7-8-9 -> edge 10-11): Eqs. 42-45. The two edge vertices are placed at
    the triangle's centroid +/- 0.5*Delta_l_th along the triangle's unit normal.

Reversibility (the round-trip GATE) is exact in TOPOLOGY and within O(Delta_l_th) in
GEOMETRY (Okuda Eqs. 5-7): only the central edge/triangle vertices are created/destroyed,
so all OTHER vertices are preserved byte-for-byte; the recovered edge endpoints land
within O(Delta_l_th) of the originals (exactly so only when the original edge length
equals Delta_l_th). See rnr/tests/test_roundtrip.py.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import tissue_forge as tf
from tissue_forge.models.vertex import solver as tfv

from . import conditions as cond
from . import topology as topo


# --------------------------------------------------------------------------------------
# vector helpers
# --------------------------------------------------------------------------------------
def _np(p) -> np.ndarray:
    return np.array([p[0], p[1], p[2]], dtype=float)


def _fv(a) -> "tf.FVector3":
    return tf.FVector3(float(a[0]), float(a[1]), float(a[2]))


def _unit(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a)
    return a / n if n > 0 else a


# --------------------------------------------------------------------------------------
# low-level surgery primitives (maintain BOTH sides of each adjacency; verified recipe)
# --------------------------------------------------------------------------------------
def _replace_v(surface, old_v, new_v) -> None:
    """In `surface`, replace old_v with new_v (ring + both vertex back-pointers)."""
    surface.replace(new_v, old_v)
    new_v.add(surface)
    old_v.remove(surface)


def _insert_between(surface, new_v, v1, v2) -> None:
    """Insert new_v between v1 and v2 in `surface`'s ring (+ vertex back-pointer)."""
    surface.insert(new_v, v1, v2)
    new_v.add(surface)


def _drop_v(surface, v) -> None:
    """Remove v from `surface` (ring + vertex back-pointer)."""
    surface.remove(v)
    v.remove(surface)


def _attach_body(surface, body) -> None:
    """Attach a surface to a body (both directions; either alone is insufficient)."""
    surface.add(body)
    body.add(surface)


def _detach_body(surface, body) -> None:
    surface.remove(body)
    body.remove(surface)


def _refresh(surfaces=(), bodies=()) -> None:
    """Recompute cached geometry (centroid/area/normal/volume) after surgery."""
    for s in surfaces:
        try:
            s.position_changed()
        except Exception:
            pass
    for b in bodies:
        try:
            b.position_changed()
        except Exception:
            pass


# --------------------------------------------------------------------------------------
# Okuda Appendix-1 vertex placement
# --------------------------------------------------------------------------------------
# The numerical core of each placement is factored into a pure position-array function
# (`*_xyz`, no TF handles) so the GPU port (rnr/gpu/reconnect_csr.py) reuses the EXACT
# same Okuda formula -- one source of truth, no silent fp drift between the CPU oracle and
# the GPU round-trip. The cfg-based wrappers below just gather positions and delegate.
def place_i_to_h_xyz(p10: np.ndarray, p11: np.ndarray,
                     outer_tops: List[np.ndarray], outer_bots: List[np.ndarray],
                     dl_th: float) -> List[np.ndarray]:
    """Positions of the 3 triangle vertices (one per arm), Okuda Eqs. 46-56.

    `outer_tops[k]` / `outer_bots[k]` are arm k's two outer-vertex positions (the
    1<->4 / 2<->5 / 3<->6 coupling). Pure numpy; takes positions, returns positions.
    """
    r0 = 0.5 * (p10 + p11)                                  # Eq. 50: edge midpoint
    uT = _unit(p10 - p11)                                   # Eq. 49: edge axis
    vproj = []
    for ot, ob in zip(outer_tops, outer_bots):
        d_top = _unit(ot - r0)
        d_bot = _unit(ob - r0)
        w = 0.5 * (d_top + d_bot)                           # Eqs. 54-56
        vproj.append(w - np.dot(w, uT) * uT)                # Eqs. 51-53: project off edge
    # L_max = largest edge of the triangle formed by the projected v-vectors.
    l_max = max(np.linalg.norm(vproj[i] - vproj[j])
                for i in range(3) for j in range(i + 1, 3))
    if l_max == 0:
        l_max = 1.0
    return [r0 + (dl_th / l_max) * vp for vp in vproj]       # Eqs. 46-48


def place_h_to_i_xyz(tri_pts: List[np.ndarray], outer_tops: List[np.ndarray],
                     dl_th: float) -> Tuple[np.ndarray, np.ndarray]:
    """Positions of the 2 recovered edge vertices (v10, v11), Okuda Eqs. 42-45.

    `tri_pts` are the 3 triangle-vertex positions; `outer_tops` the 3 cap_top-side outer
    positions (used only to orient the normal). Pure numpy; takes positions, returns the
    (v10, v11) pair with v10 toward the cap_top side.
    """
    p = tri_pts
    r0 = (p[0] + p[1] + p[2]) / 3.0                          # Eq. 45: triangle centroid
    n = _unit(np.cross(p[1] - p[0], p[2] - p[0]))            # Eq. 44: triangle unit normal
    # orient n toward the cap_top-side outer vertices.
    top_mean = np.mean(outer_tops, axis=0)
    if np.dot(top_mean - r0, n) < 0:
        n = -n
    half = 0.5 * dl_th
    return r0 + half * n, r0 - half * n                      # Eqs. 42-43


def place_i_to_h(cfg: "topo.IConfig", dl_th: float) -> List[np.ndarray]:
    """Triangle-vertex placement for an I->H from a TF-handle IConfig (delegates to
    place_i_to_h_xyz). Arm k's triangle vertex pairs that arm's two outer vertices."""
    p10, p11 = _np(cfg.v10.position), _np(cfg.v11.position)
    outer_tops = [_np(a.outer_top.position) for a in cfg.arms]
    outer_bots = [_np(a.outer_bot.position) for a in cfg.arms]
    return place_i_to_h_xyz(p10, p11, outer_tops, outer_bots, dl_th)


def place_h_to_i(cfg: "topo.HConfig", dl_th: float) -> Tuple[np.ndarray, np.ndarray]:
    """Edge-vertex placement for an H->I from a TF-handle HConfig (delegates to
    place_h_to_i_xyz). v10 lands toward cap_top, v11 toward cap_bot."""
    tri_pts = [_np(a.tri_vertex.position) for a in cfg.arms]
    outer_tops = [_np(a.outer_top.position) for a in cfg.arms]
    return place_h_to_i_xyz(tri_pts, outer_tops, dl_th)


# --------------------------------------------------------------------------------------
# result record
# --------------------------------------------------------------------------------------
@dataclass
class ReconnectResult:
    ok: bool
    reason: str = ""
    # I->H: the new triangle surface (re-fetched) + its 3 new vertex ids;
    # H->I: the new edge's two vertices + their ids.
    new_surface: object = None
    new_vertex_ids: Tuple[int, ...] = ()


# --------------------------------------------------------------------------------------
# I -> H : short edge becomes a triangular face (the new c123<->c456 contact)
# --------------------------------------------------------------------------------------
def i_to_h_check(cfg: "topo.IConfig", check_conditions: bool = True) -> Optional[str]:
    """Predicate half: returns None if I->H is legal, else a veto reason string."""
    if cfg is None:
        return "no valid I-configuration"
    if check_conditions:
        return cond.i_to_h_veto(cfg)
    return None


def i_to_h(cfg: "topo.IConfig", dl_th: float, stype,
           check_conditions: bool = True) -> ReconnectResult:
    """Reconnect the short edge (v10, v11) of `cfg` into a triangular face.

    `stype` is the SurfaceType for the new triangle. Returns a ReconnectResult; on a
    veto the mesh is left untouched. Handles in `cfg` are consumed here and invalid
    afterwards -- callers must re-walk the mesh (by id) for further work.
    """
    veto = i_to_h_check(cfg, check_conditions)
    if veto is not None:
        return ReconnectResult(ok=False, reason=veto)

    v10, v11 = cfg.v10, cfg.v11
    positions = place_i_to_h(cfg, dl_th)
    tri = [tfv.Vertex.create(_fv(pos)) for pos in positions]   # vt[k] for arm k
    vt = {id(a): tri[k] for k, a in enumerate(cfg.arms)}       # arm -> its triangle vertex
    arm_by_outer_top = {a.outer_top.id: a for a in cfg.arms}
    arm_by_outer_bot = {a.outer_bot.id: a for a in cfg.arms}

    touched_surfs = []

    # (1) SIDE faces: each [.., outer_top, v10, v11, outer_bot, ..] -> [.., outer_top, vt_k, outer_bot, ..]
    for a in cfg.arms:
        s = a.side_surface
        _replace_v(s, v10, vt[id(a)])
        _drop_v(s, v11)
        touched_surfs.append(s)

    # (2) TOP faces: v10 (between two outer_top verts) -> the triangle edge (vt_p, vt_q).
    for sc_id, face in cfg.top_faces.items():
        prev_v, next_v = topo.ring_neighbors(face, v10)
        if prev_v.id not in arm_by_outer_top or next_v.id not in arm_by_outer_top:
            return ReconnectResult(ok=False,
                reason=f"top face {face.id}: ring-neighbours of v10 are not arm outer_top verts")
        vt_prev = vt[id(arm_by_outer_top[prev_v.id])]
        vt_next = vt[id(arm_by_outer_top[next_v.id])]
        _replace_v(face, v10, vt_prev)               # v10 -> vt_prev (adjacent to prev)
        _insert_between(face, vt_next, vt_prev, next_v)
        touched_surfs.append(face)

    # (3) BOTTOM faces: v11 -> the triangle edge, mirror of (2).
    for sc_id, face in cfg.bottom_faces.items():
        prev_v, next_v = topo.ring_neighbors(face, v11)
        if prev_v.id not in arm_by_outer_bot or next_v.id not in arm_by_outer_bot:
            return ReconnectResult(ok=False,
                reason=f"bottom face {face.id}: ring-neighbours of v11 are not arm outer_bot verts")
        vt_prev = vt[id(arm_by_outer_bot[prev_v.id])]
        vt_next = vt[id(arm_by_outer_bot[next_v.id])]
        _replace_v(face, v11, vt_prev)
        _insert_between(face, vt_next, vt_prev, next_v)
        touched_surfs.append(face)

    # (4) the new triangular face, shared by the two caps.
    T = stype(vertices=[tri[0], tri[1], tri[2]])
    _attach_body(T, cfg.cap_top)
    _attach_body(T, cfg.cap_bot)

    # (5) destroy the now-orphaned edge vertices.
    v10.destroy()
    v11.destroy()

    # (6) recompute geometry on everything touched.
    _refresh(surfaces=touched_surfs + [T], bodies=[cfg.cap_top, cfg.cap_bot])

    tri_ids = tuple(v.id for v in tri)
    # re-fetch the triangle by id (handles may shuffle after destroy).
    return ReconnectResult(ok=True, new_surface=T, new_vertex_ids=tri_ids)


# --------------------------------------------------------------------------------------
# H -> I : triangular face collapses back to a short edge
# --------------------------------------------------------------------------------------
def h_to_i_check(cfg: "topo.HConfig", check_conditions: bool = True) -> Optional[str]:
    """Predicate half: returns None if H->I is legal, else a veto reason string."""
    if cfg is None:
        return "no valid H-configuration"
    if check_conditions:
        return cond.h_to_i_veto(cfg)
    return None


def h_to_i(cfg: "topo.HConfig", dl_th: float,
           check_conditions: bool = True) -> ReconnectResult:
    """Collapse the triangular face of `cfg` back into a short edge (v10, v11)."""
    veto = h_to_i_check(cfg, check_conditions)
    if veto is not None:
        return ReconnectResult(ok=False, reason=veto)

    p10, p11 = place_h_to_i(cfg, dl_th)
    nv10 = tfv.Vertex.create(_fv(p10))     # cap_top-side recovered vertex
    nv11 = tfv.Vertex.create(_fv(p11))     # cap_bot-side recovered vertex
    tri_ids = set(cfg.tri_vertex_ids)
    touched_surfs = []

    # (1) SIDE faces: [.., outer_top, vt_k, outer_bot, ..] -> [.., outer_top, nv10, nv11, outer_bot, ..]
    for a in cfg.arms:
        s = a.side_surface
        _replace_v(s, a.tri_vertex, nv10)            # vt_k -> nv10 (between the two outers)
        _insert_between(s, nv11, nv10, a.outer_bot)  # nv11 toward outer_bot
        touched_surfs.append(s)

    # (2) TOP faces: the triangle edge (vt_p, vt_q) -> single nv10.
    for sc_id, face in cfg.top_faces.items():
        present = [v for v in face.vertices if v.id in tri_ids]
        if len(present) != 2:
            return ReconnectResult(ok=False,
                reason=f"top face {face.id}: expected 2 triangle vertices, found {len(present)}")
        _replace_v(face, present[0], nv10)
        _drop_v(face, present[1])
        touched_surfs.append(face)

    # (3) BOTTOM faces: mirror -> single nv11.
    for sc_id, face in cfg.bottom_faces.items():
        present = [v for v in face.vertices if v.id in tri_ids]
        if len(present) != 2:
            return ReconnectResult(ok=False,
                reason=f"bottom face {face.id}: expected 2 triangle vertices, found {len(present)}")
        _replace_v(face, present[0], nv11)
        _drop_v(face, present[1])
        touched_surfs.append(face)

    # (4) detach + destroy the triangle (its now-orphaned vertices go with it).
    T = cfg.triangle
    _detach_body(T, cfg.cap_top)
    _detach_body(T, cfg.cap_bot)
    T.destroy()

    # (5) recompute geometry.
    _refresh(surfaces=touched_surfs, bodies=[cfg.cap_top, cfg.cap_bot])

    return ReconnectResult(ok=True, new_surface=None, new_vertex_ids=(nv10.id, nv11.id))
