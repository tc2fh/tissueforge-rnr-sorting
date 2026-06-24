"""Gate C, brick C0: read-only [I]-neighbourhood detection on the index-based PaddedMesh
-- the GPU-port analogue of rnr/topology.py (i_neighbourhood / find_short_edges), with NO
TissueForge handles.

Why a second detector? The CPU walk (topology.py) reads TF `Vertex`/`Surface`/`Body`
handles. Once the mesh lives on the device and has been MUTATED by I<->H surgery, those
handles are gone -- the scheduler must re-detect reconnection sites directly from the
PaddedMesh index arrays between batches. This module does exactly that, mirroring
topology.i_neighbourhood condition-for-condition so it stays a translation, not a redesign.

It produces the SAME ICfgIdx that reconnect_csr.i_to_h_csr consumes, so detection ->
surgery needs no TF round-trip. All host numpy here (the reference semantics); the Gate-C2
kernel will run the same predicate per candidate thread.
"""
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .device_mesh import PaddedMesh
from .reconnect_csr import ArmIdx, HArmIdx, HCfgIdx, ICfgIdx


# --------------------------------------------------------------------------------------
# small index-based adjacency helpers (mirror topology.py's, read-only)
# --------------------------------------------------------------------------------------
def vert_bodies(pm: PaddedMesh, v: int) -> Set[int]:
    """Body indices touching vertex v (union of s2b over v's incident surfaces)."""
    bs: Set[int] = set()
    for s in pm.v2s[v, :int(pm.v2s_len[v])]:
        for b in pm.s2b[int(s)]:
            if b >= 0:
                bs.add(int(b))
    return bs


def shared_surfaces(pm: PaddedMesh, a: int, b: int) -> Set[int]:
    sa = {int(x) for x in pm.v2s[a, :int(pm.v2s_len[a])]}
    sb = {int(x) for x in pm.v2s[b, :int(pm.v2s_len[b])]}
    return sa & sb


def is_consecutive(pm: PaddedMesh, s: int, a: int, b: int) -> bool:
    """True iff a, b appear consecutively (cyclically) in surface s's vertex ring."""
    L = int(pm.s2v_len[s])
    row = pm.s2v[s, :L]
    for i in range(L):
        x, y = int(row[i]), int(row[(i + 1) % L])
        if (x == a and y == b) or (x == b and y == a):
            return True
    return False


def other_ring_neighbor(pm: PaddedMesh, s: int, v: int, exclude: int) -> Optional[int]:
    """The ring-neighbour of v in surface s other than `exclude` (None if not unique)."""
    prev, nxt = pm.ring_neighbors(s, v)
    cand = [w for w in (prev, nxt) if w != exclude]
    return cand[0] if len(cand) == 1 else None


def find_interface(pm: PaddedMesh, ba: int, bb: int) -> List[int]:
    """Live surface indices separating exactly bodies ba and bb (Body.find_interface)."""
    pair = {ba, bb}
    out = []
    for s in pm.b2s[ba, :int(pm.b2s_len[ba])]:
        s = int(s)
        if pm.surf_alive[s] and {int(x) for x in pm.s2b[s] if x >= 0} == pair:
            out.append(s)
    return out


def edge_length(pm: PaddedMesh, a: int, b: int) -> float:
    return float(np.linalg.norm(pm.vert_pos[a] - pm.vert_pos[b]))


def connected_vertices(pm: PaddedMesh, v: int) -> Set[int]:
    """All vertices ring-adjacent to v in any of v's incident surfaces."""
    out: Set[int] = set()
    for s in pm.v2s[v, :int(pm.v2s_len[v])]:
        prev, nxt = pm.ring_neighbors(int(s), v)
        out.add(prev)
        out.add(nxt)
    return out


# --------------------------------------------------------------------------------------
# the I-neighbourhood (index world) -- mirror topology.i_neighbourhood
# --------------------------------------------------------------------------------------
def i_neighbourhood_csr(pm: PaddedMesh, v10: int, v11: int) -> Optional[ICfgIdx]:
    """Gather + validate the [I] neighbourhood of the ordered short edge (v10, v11), in
    PaddedMesh indices. Returns an ICfgIdx (ready for i_to_h_csr) or None. Purely
    structural -- the Condition-2 length trigger is applied by find_short_edges_csr, the
    Condition-4 vetoes by conditions/the scheduler."""
    b10, b11 = vert_bodies(pm, v10), vert_bodies(pm, v11)
    if len(b10) != 4 or len(b11) != 4:                 # both endpoints interior (4 cells)
        return None
    side = b10 & b11
    cap_top_ids = b10 - b11
    cap_bot_ids = b11 - b10
    if len(side) != 3 or len(cap_top_ids) != 1 or len(cap_bot_ids) != 1:
        return None

    # the short edge must be consecutive in exactly 3 surfaces, all interior (2-body).
    shared = shared_surfaces(pm, v10, v11)
    side_surfs = [s for s in shared if is_consecutive(pm, s, v10, v11)]
    if len(side_surfs) != 3:
        return None
    if any(int((pm.s2b[s] >= 0).sum()) != 2 for s in side_surfs):
        return None

    cap_top = next(iter(cap_top_ids))                  # caps v10
    cap_bot = next(iter(cap_bot_ids))                  # caps v11

    arms: List[ArmIdx] = []
    for s in side_surfs:
        ot = other_ring_neighbor(pm, s, v10, v11)      # outer vertex on the v10 side
        ob = other_ring_neighbor(pm, s, v11, v10)      # ... on the v11 side
        if ot is None or ob is None:
            return None
        arms.append(ArmIdx(side_surface=s, outer_top=ot, outer_bot=ob))

    # 3 top + 3 bottom faces: side<->cap interfaces, each a single face (Cond-4 extra rule).
    top_faces, bottom_faces = {}, {}
    for sc in side:
        it = find_interface(pm, sc, cap_top)
        ib = find_interface(pm, sc, cap_bot)
        if len(it) != 1 or len(ib) != 1:
            return None
        top_faces[sc] = it[0]
        bottom_faces[sc] = ib[0]

    return ICfgIdx(v10=v10, v11=v11, cap_top=cap_top, cap_bot=cap_bot,
                   side_cells=sorted(side), arms=arms,
                   top_faces=top_faces, bottom_faces=bottom_faces)


def find_short_edges_csr(pm: PaddedMesh, threshold: float
                         ) -> List[Tuple[int, int, ICfgIdx]]:
    """All interior short edges (length < threshold) forming a valid [I] config, as
    (v10, v11, ICfgIdx) triples deduped by edge. Mirrors topology.find_short_edges; this
    is the Condition-2 trigger scan the scheduler iterates on."""
    seen: Set[Tuple[int, int]] = set()
    out = []
    for v in range(pm.n_v_used):
        if not pm.vert_alive[v] or len(vert_bodies(pm, v)) != 4:
            continue
        for w in connected_vertices(pm, v):
            if w <= v or not pm.vert_alive[w]:
                continue
            key = (v, w)
            if key in seen:
                continue
            seen.add(key)
            if edge_length(pm, v, w) >= threshold:
                continue
            cfg = i_neighbourhood_csr(pm, v, w)
            if cfg is not None:
                out.append((v, w, cfg))
    return out


# --------------------------------------------------------------------------------------
# the H-neighbourhood (index world) -- mirror topology.h_neighbourhood (reverse direction)
# --------------------------------------------------------------------------------------
def _tri_edges(vids) -> List[Tuple[int, int]]:
    """The 3 undirected edges of a triangle, as sorted (a, b) tuples."""
    a, b, c = vids
    return [tuple(sorted(e)) for e in ((a, b), (b, c), (c, a))]


def h_neighbourhood_csr(pm: PaddedMesh, triangle: int) -> Optional[HCfgIdx]:
    """Gather + validate the [H] neighbourhood of a triangular face `triangle` (surface
    index), in PaddedMesh indices. Returns an HCfgIdx (ready for h_to_i_csr) or None.

    Mirrors topology.h_neighbourhood condition-for-condition; the Condition-2 max-edge
    trigger is applied by find_small_triangles_csr. The index-world reverse of
    i_neighbourhood_csr -- like that detector it uses NO TF handles, so the scheduler can
    re-detect [H] sites directly on the device-mutated mesh between batches."""
    if not pm.surf_alive[triangle] or int(pm.s2v_len[triangle]) != 3:
        return None
    tri_vs = [int(x) for x in pm.s2v[triangle, :3]]
    caps = [int(b) for b in pm.s2b[triangle] if b >= 0]
    if len(caps) != 2:                                 # triangle shared by exactly 2 caps
        return None
    cap_top, cap_bot = caps[0], caps[1]
    cap_ids = {cap_top, cap_bot}

    if any(len(vert_bodies(pm, v)) != 4 for v in tri_vs):   # tri verts interior (4 cells)
        return None

    # side cell per triangle edge = the common cell of the edge's endpoints minus the caps.
    edge_side_cell: Dict[Tuple[int, int], int] = {}
    side_ids: Set[int] = set()
    for e in _tri_edges(tri_vs):
        a, b = e
        rest = (vert_bodies(pm, a) & vert_bodies(pm, b)) - cap_ids
        if len(rest) != 1:
            return None
        sc = next(iter(rest))
        edge_side_cell[e] = sc
        side_ids.add(sc)
    if len(side_ids) != 3:
        return None

    # each triangle vertex sits in one SIDE face: the interface of the two side cells
    # flanking it (the side cells of its two incident triangle edges), which contains v +
    # its two outer vertices (one toward each cap).
    arms: List[HArmIdx] = []
    for v in tri_vs:
        flank = [edge_side_cell[e] for e in _tri_edges(tri_vs) if v in e]
        if len(flank) != 2:
            return None
        iface = find_interface(pm, flank[0], flank[1])
        side_surf = next((s for s in iface
                          if v in {int(x) for x in pm.s2v[s, :int(pm.s2v_len[s])]}), None)
        if side_surf is None:
            return None
        prev, nxt = pm.ring_neighbors(side_surf, v)     # v's two outer vertices
        outer_top = outer_bot = None
        for w in (prev, nxt):
            wb = vert_bodies(pm, w)
            if cap_top in wb:
                outer_top = w
            elif cap_bot in wb:
                outer_bot = w
        if outer_top is None or outer_bot is None:
            return None
        arms.append(HArmIdx(tri_vertex=v, side_surface=side_surf,
                            outer_top=outer_top, outer_bot=outer_bot))

    # 3 top + 3 bottom faces (side<->cap interfaces), each a single surface (Cond-4 extra).
    top_faces, bottom_faces = {}, {}
    for sc in side_ids:
        it = find_interface(pm, sc, cap_top)
        ib = find_interface(pm, sc, cap_bot)
        if len(it) != 1 or len(ib) != 1:
            return None
        top_faces[sc] = it[0]
        bottom_faces[sc] = ib[0]

    return HCfgIdx(triangle=triangle, tri_verts=[a.tri_vertex for a in arms],
                   cap_top=cap_top, cap_bot=cap_bot, side_cells=sorted(side_ids),
                   arms=arms, top_faces=top_faces, bottom_faces=bottom_faces)


def find_small_triangles_csr(pm: PaddedMesh, threshold: float
                             ) -> List[Tuple[int, HCfgIdx]]:
    """All triangular faces whose MAX edge < threshold that form a valid [H] config, as
    (triangle_idx, HCfgIdx) pairs. Mirrors topology.find_small_triangles: Condition 2 for
    the reverse direction triggers on the MAX of the three triangle edges (NOT the min --
    Honda's wrong 'condition H'; see CLAUDE.md). The H-analog of find_short_edges_csr; the
    Condition-2 trigger scan the reverse scheduler iterates on."""
    out = []
    for s in range(pm.n_s_used):
        if not pm.surf_alive[s] or int(pm.s2v_len[s]) != 3:
            continue
        cfg = h_neighbourhood_csr(pm, s)
        if cfg is None:
            continue
        t = cfg.tri_verts
        max_edge = max(edge_length(pm, t[0], t[1]), edge_length(pm, t[1], t[2]),
                       edge_length(pm, t[2], t[0]))
        if max_edge < threshold:
            out.append((s, cfg))
    return out
