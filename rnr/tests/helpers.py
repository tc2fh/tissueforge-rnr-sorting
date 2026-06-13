"""Test helpers: mesh builders + round-trip snapshot/compare utilities (Phase 1).

All builders place meshes at distinct centres so they coexist in the single shared
universe (tf.init is a singleton). Counting is always scoped to a passed `bodies` list.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import tissue_forge as tf
from tissue_forge.models.vertex import solver as tfv

from .. import reconnect as rc
from .. import topology as topo


# --------------------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------------------
def build_minimal_i_config(stype, btype, center=(30., 30., 30.),
                           edge=0.5, R=1.2, H=1.2):
    """Hand-built minimal 5-cell [I] neighbourhood with a known short edge.

    Two cap tetrahedra (apex v10 / v11, sharing the short edge along z) plus three wedge
    side-cells filling the angular gaps, bounded by free outer faces. The exact textbook
    Okuda [I] config: edge shared by 3 side cells, caps not yet touching. Returns a dict
    with the body list and the short-edge end-vertices.

    NB winding: each side face is wound [outer_top, v10, v11, outer_bot] so v10,v11 are
    CONSECUTIVE (the short edge) and the three side cells get positive volumes.
    """
    C = np.array(center, dtype=float)
    a = edge / 2.0
    v10 = tfv.Vertex.create(tf.FVector3(*(C + [0, 0, +a])))
    v11 = tfv.Vertex.create(tf.FVector3(*(C + [0, 0, -a])))
    ang = [0.0, 2 * math.pi / 3, 4 * math.pi / 3]
    top = [tfv.Vertex.create(tf.FVector3(*(C + [R * math.cos(t), R * math.sin(t), +H]))) for t in ang]
    bot = [tfv.Vertex.create(tf.FVector3(*(C + [R * math.cos(t), R * math.sin(t), -H]))) for t in ang]

    nb = lambda k: (k + 1) % 3
    side_s = [stype(vertices=[top[k], v10, v11, bot[k]]) for k in range(3)]
    top_s = [stype(vertices=[v10, top[k], top[nb(k)]]) for k in range(3)]
    bot_s = [stype(vertices=[v11, bot[nb(k)], bot[k]]) for k in range(3)]
    outer_wedge = [stype(vertices=[top[k], top[nb(k)], bot[nb(k)], bot[k]]) for k in range(3)]
    cap_top_base = stype(vertices=[top[0], top[1], top[2]])
    cap_bot_base = stype(vertices=[bot[0], bot[2], bot[1]])

    wedges = [btype([side_s[k], side_s[nb(k)], top_s[k], bot_s[k], outer_wedge[k]]) for k in range(3)]
    cap_top = btype([top_s[0], top_s[1], top_s[2], cap_top_base])
    cap_bot = btype([bot_s[0], bot_s[1], bot_s[2], cap_bot_base])
    bodies = wedges + [cap_top, cap_bot]
    return dict(bodies=bodies, v10=v10, v11=v11, top=top, bot=bot,
                wedges=wedges, cap_top=cap_top, cap_bot=cap_bot,
                side_s=side_s, top_s=top_s, bot_s=bot_s)


def build_kelvin_block(stype, btype, n=4, span=8.0, origin=(0., 0., 0.)):
    """A Kelvin (BCC-Voronoi) finite block; interior cells are 14-faced Kelvin cells with
    genuine interior short edges in the textbook [I] config (Okuda Fig. 7a). Returns the
    body list."""
    from ..geometry import build_voronoi_cluster, bcc_seeds
    o = np.array(origin, dtype=float)
    box = [[o[0], o[0] + span], [o[1], o[1] + span], [o[2], o[2] + span]]
    bodies, _cells, _stats = build_voronoi_cluster(bcc_seeds(n, box), box, btype, stype)
    return bodies


# --------------------------------------------------------------------------------------
# snapshot / compare
# --------------------------------------------------------------------------------------
def vertex_ids(bodies) -> set:
    return {v.id for b in bodies for v in b.getVertices()}


def surface_ids(bodies) -> set:
    return {s.id for b in bodies for s in b.getSurfaces()}


def all_vertices(bodies) -> Dict[int, object]:
    return {v.id: v for b in bodies for v in b.getVertices()}


@dataclass
class Snapshot:
    n_verts: int
    n_surfs: int
    side_cell_ids: set
    cap_ids: set
    outer_pos: Dict[int, np.ndarray]
    p10: np.ndarray
    p11: np.ndarray


def snapshot_i(cfg: "topo.IConfig", bodies) -> Snapshot:
    outer = {}
    for a in cfg.arms:
        outer[a.outer_top.id] = rc._np(a.outer_top.position)
        outer[a.outer_bot.id] = rc._np(a.outer_bot.position)
    return Snapshot(
        n_verts=len(vertex_ids(bodies)),
        n_surfs=len(surface_ids(bodies)),
        side_cell_ids=set(cfg.side_cell_ids),
        cap_ids={cfg.cap_top_id, cfg.cap_bot_id},
        outer_pos=outer,
        p10=rc._np(cfg.v10.position),
        p11=rc._np(cfg.v11.position),
    )


def max_outer_drift(snap: Snapshot, bodies) -> float:
    allv = all_vertices(bodies)
    return max(float(np.linalg.norm(rc._np(allv[vid].position) - p0))
               for vid, p0 in snap.outer_pos.items())


def edge_drift(snap: Snapshot, p10_new: np.ndarray, p11_new: np.ndarray) -> float:
    """Min over the two endpoint labellings (the recovered pair may be swapped)."""
    same = max(np.linalg.norm(p10_new - snap.p10), np.linalg.norm(p11_new - snap.p11))
    swap = max(np.linalg.norm(p10_new - snap.p11), np.linalg.norm(p11_new - snap.p10))
    return min(same, swap)
