"""Gate B2 of the GPU port: the count-CHANGING I<->H reconnection (the 3D T1) on the
index-based PaddedMesh -- a direct translation of the validated CPU oracle
rnr/reconnect.py (i_to_h / h_to_i, lines ~170-300) onto PaddedMesh primitives.

This is the make-or-break piece of the whole port (docs/2026-06-24_gpu-3d-vertex-model-
exploration.md): unlike cellGPU's 2D T1, the 3D I<->H *creates and destroys* vertices and
faces. Here that is done with the bump allocator + free-marks of device_mesh.py
(births bump a high-water counter, deaths set alive=0; Gate-D compaction reclaims slots).

Faithfulness to the oracle is the point:
  * placement reuses reconnect.place_i_to_h_xyz / place_h_to_i_xyz VERBATIM (one Okuda
    formula, no fp drift between CPU and GPU);
  * the topology surgery mirrors reconnect.py step-for-step, using the same both-sides
    adjacency primitives (replace_v / insert_between / drop_v / attach_body / detach_body);
  * `i_to_h_csr` RETURNS the post-state H-neighbourhood (in indices) so `h_to_i_csr` can
    invert it without re-walking the mesh -- the round-trip is the Gate-B2 test.

These are the HOST reference semantics; the Gate-B3 Warp kernel will mutate the identical
flat arrays on the device and must match this bit-for-bit (fp64) / within-tol (fp32).

Condition-4 vetoes are NOT applied here: B2 round-trips known-legal configs; the vetoes
become reservation-time predicates in the Gate-C independent-set scheduler (the CPU
oracle's i_to_h_check/h_to_i_check are the reference for that).
"""
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from ..reconnect import place_h_to_i_xyz, place_i_to_h_xyz
from .device_mesh import PaddedMesh


# --------------------------------------------------------------------------------------
# index-world neighbourhood configs (mirror topology.IConfig / HConfig, by index)
# --------------------------------------------------------------------------------------
@dataclass
class ArmIdx:
    """One arm of the I-neighbourhood (mirror of topology.Arm), all indices."""
    side_surface: int   # surface idx bearing the short edge
    outer_top: int      # vertex idx: this arm's outer vertex on the v10 side
    outer_bot: int      # vertex idx: ... on the v11 side


@dataclass
class ICfgIdx:
    """I-neighbourhood of a short edge (v10, v11), in PaddedMesh indices."""
    v10: int
    v11: int
    cap_top: int                 # body idx (caps v10)
    cap_bot: int                 # body idx (caps v11)
    side_cells: List[int]        # 3 body idxs
    arms: List[ArmIdx]           # 3 arms (one per side surface)
    top_faces: Dict[int, int]    # side_cell body idx -> top face surface idx
    bottom_faces: Dict[int, int]  # side_cell body idx -> bottom face surface idx


@dataclass
class HArmIdx:
    """One arm of the H-neighbourhood (mirror of topology.HArm), all indices."""
    tri_vertex: int
    side_surface: int
    outer_top: int
    outer_bot: int


@dataclass
class HCfgIdx:
    """H-neighbourhood of a triangular face, in PaddedMesh indices. This is exactly what
    i_to_h_csr returns and h_to_i_csr consumes -- no mesh re-walk needed to invert."""
    triangle: int                # surface idx of the triangular face
    tri_verts: List[int]         # 3 vertex idxs, in arm order
    cap_top: int
    cap_bot: int
    side_cells: List[int]
    arms: List[HArmIdx]
    top_faces: Dict[int, int]
    bottom_faces: Dict[int, int]


# --------------------------------------------------------------------------------------
# translator: TF-handle IConfig -> index-world ICfgIdx
# --------------------------------------------------------------------------------------
def iconfig_to_indices(cfg, vid2i: Dict[int, int], sid2i: Dict[int, int],
                       bid2i: Dict[int, int]) -> ICfgIdx:
    """Turn a topology.IConfig (TF handles) into an ICfgIdx, via the csr_mesh.id_maps
    id->index maps (which index identically to the CSRMesh / PaddedMesh built from the
    same bodies). The bridge from the CPU topology walk into the GPU index world."""
    arms = [ArmIdx(side_surface=sid2i[a.side_surface_id],
                   outer_top=vid2i[a.outer_top.id],
                   outer_bot=vid2i[a.outer_bot.id]) for a in cfg.arms]
    top = {bid2i[sc]: sid2i[face.id] for sc, face in cfg.top_faces.items()}
    bot = {bid2i[sc]: sid2i[face.id] for sc, face in cfg.bottom_faces.items()}
    return ICfgIdx(
        v10=vid2i[cfg.v10_id], v11=vid2i[cfg.v11_id],
        cap_top=bid2i[cfg.cap_top_id], cap_bot=bid2i[cfg.cap_bot_id],
        side_cells=[bid2i[s] for s in cfg.side_cell_ids],
        arms=arms, top_faces=top, bottom_faces=bot,
    )


# --------------------------------------------------------------------------------------
# I -> H : short edge (v10, v11) becomes a triangular face (mirror of reconnect.i_to_h)
# --------------------------------------------------------------------------------------
def i_to_h_csr(pm: PaddedMesh, cfg: ICfgIdx, dl_th: float, box=None) -> HCfgIdx:
    """Reconnect the short edge of `cfg` into a triangular face, in place on `pm`.

    Returns the post-state HCfgIdx (the new triangle + its neighbourhood, in indices) so
    the inverse h_to_i_csr can run without re-searching. Step numbering follows
    reconnect.i_to_h exactly. `box` (per-axis lengths, or None) is forwarded to the Okuda
    placement for periodic minimum-image arithmetic; None = non-periodic (unchanged).
    """
    v10, v11 = cfg.v10, cfg.v11

    # Okuda Appendix-1 placement (same formula as the CPU oracle).
    outer_tops = [pm.vert_pos[a.outer_top] for a in cfg.arms]
    outer_bots = [pm.vert_pos[a.outer_bot] for a in cfg.arms]
    positions = place_i_to_h_xyz(pm.vert_pos[v10], pm.vert_pos[v11],
                                 outer_tops, outer_bots, dl_th, box=box)
    tri = [pm.alloc_vertex(positions[k]) for k in range(3)]    # tri[k] for arm k
    arm_by_outer_top = {a.outer_top: k for k, a in enumerate(cfg.arms)}
    arm_by_outer_bot = {a.outer_bot: k for k, a in enumerate(cfg.arms)}

    # (1) SIDE faces: [.., outer_top, v10, v11, outer_bot, ..] -> [.., outer_top, tri_k, outer_bot, ..]
    for k, a in enumerate(cfg.arms):
        pm.replace_v(a.side_surface, v10, tri[k])
        pm.drop_v(a.side_surface, v11)

    # (2) TOP faces: v10 (between two outer_top verts) -> the triangle edge (tri_prev, tri_next).
    for face in cfg.top_faces.values():
        prev_v, next_v = pm.ring_neighbors(face, v10)
        if prev_v not in arm_by_outer_top or next_v not in arm_by_outer_top:
            raise ValueError(f"top face {face}: ring-neighbours of v10 are not arm outer_top verts")
        kp, kn = arm_by_outer_top[prev_v], arm_by_outer_top[next_v]
        pm.replace_v(face, v10, tri[kp])                       # v10 -> tri_prev
        pm.insert_between(face, tri[kn], tri[kp], next_v)      # tri_next after tri_prev

    # (3) BOTTOM faces: v11 -> the triangle edge, mirror of (2).
    for face in cfg.bottom_faces.values():
        prev_v, next_v = pm.ring_neighbors(face, v11)
        if prev_v not in arm_by_outer_bot or next_v not in arm_by_outer_bot:
            raise ValueError(f"bottom face {face}: ring-neighbours of v11 are not arm outer_bot verts")
        kp, kn = arm_by_outer_bot[prev_v], arm_by_outer_bot[next_v]
        pm.replace_v(face, v11, tri[kp])
        pm.insert_between(face, tri[kn], tri[kp], next_v)

    # (4) the new triangular face, shared by the two caps (winding [tri0, tri1, tri2]).
    T = pm.alloc_surface()
    pm.set_ring(T, [tri[0], tri[1], tri[2]])
    pm.attach_body(T, cfg.cap_top)
    pm.attach_body(T, cfg.cap_bot)

    # (5) destroy the now-orphaned edge vertices (all incidences were re-pointed above).
    pm.free_vertex(v10)
    pm.free_vertex(v11)

    arms = [HArmIdx(tri_vertex=tri[k], side_surface=a.side_surface,
                    outer_top=a.outer_top, outer_bot=a.outer_bot)
            for k, a in enumerate(cfg.arms)]
    return HCfgIdx(
        triangle=T, tri_verts=[tri[0], tri[1], tri[2]],
        cap_top=cfg.cap_top, cap_bot=cfg.cap_bot, side_cells=list(cfg.side_cells),
        arms=arms, top_faces=dict(cfg.top_faces), bottom_faces=dict(cfg.bottom_faces),
    )


# --------------------------------------------------------------------------------------
# H -> I : triangular face collapses back to a short edge (mirror of reconnect.h_to_i)
# --------------------------------------------------------------------------------------
def h_to_i_csr(pm: PaddedMesh, cfg: HCfgIdx, dl_th: float, box=None):
    """Collapse the triangular face of `cfg` back into a short edge. Returns the two
    recovered edge-vertex indices (nv10 toward cap_top, nv11 toward cap_bot). `box` (per-axis
    lengths, or None) is forwarded to the Okuda placement for periodic minimum-image
    arithmetic; None = non-periodic (unchanged)."""
    tri_pts = [pm.vert_pos[a.tri_vertex] for a in cfg.arms]
    outer_tops = [pm.vert_pos[a.outer_top] for a in cfg.arms]
    p10, p11 = place_h_to_i_xyz(tri_pts, outer_tops, dl_th, box=box)
    nv10 = pm.alloc_vertex(p10)        # cap_top-side recovered vertex
    nv11 = pm.alloc_vertex(p11)        # cap_bot-side recovered vertex
    tri_ids = set(cfg.tri_verts)

    # (1) SIDE faces: [.., outer_top, tri_k, outer_bot, ..] -> [.., outer_top, nv10, nv11, outer_bot, ..]
    for a in cfg.arms:
        pm.replace_v(a.side_surface, a.tri_vertex, nv10)       # tri_k -> nv10
        pm.insert_between(a.side_surface, nv11, nv10, a.outer_bot)  # nv11 toward outer_bot

    # (2) TOP faces: the triangle edge (tri_p, tri_q) -> single nv10.
    for face in cfg.top_faces.values():
        present = [int(x) for x in pm.s2v[face, :int(pm.s2v_len[face])] if int(x) in tri_ids]
        if len(present) != 2:
            raise ValueError(f"top face {face}: expected 2 triangle vertices, found {len(present)}")
        pm.replace_v(face, present[0], nv10)
        pm.drop_v(face, present[1])

    # (3) BOTTOM faces: mirror -> single nv11.
    for face in cfg.bottom_faces.values():
        present = [int(x) for x in pm.s2v[face, :int(pm.s2v_len[face])] if int(x) in tri_ids]
        if len(present) != 2:
            raise ValueError(f"bottom face {face}: expected 2 triangle vertices, found {len(present)}")
        pm.replace_v(face, present[0], nv11)
        pm.drop_v(face, present[1])

    # (4) detach + destroy the triangle and its now-orphaned vertices.
    pm.detach_body(cfg.triangle, cfg.cap_top)
    pm.detach_body(cfg.triangle, cfg.cap_bot)
    pm.free_surface(cfg.triangle)
    for v in cfg.tri_verts:
        pm.free_vertex(v)

    return nv10, nv11
