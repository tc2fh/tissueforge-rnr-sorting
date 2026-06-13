"""Phase-1 read-only topology analysis for the 3D T1 / Okuda I<->H reconnection.

This module ONLY reads the mesh -- it gathers the neighbourhood of a candidate
reconnection site and validates that it has the textbook structure. All mutation
lives in rnr/reconnect.py. Keeping the *check/predicate* half here (and in
rnr/conditions.py) separate from the *mutate* half mirrors how TissueForge's own
`tfMeshQuality.cpp` operations are structured, so the eventual C++ port is a
translation rather than a redesign (see rnr/PORTING_NOTES.md).

The neighbourhood, after Okuda et al. 2013 (Biomech Model Mechanobiol 12:627-644,
Fig. 3): a short edge [I] sits in a 5-cell / 9-face neighbourhood --

    * 3 SIDE cells  : the cells sharing the short edge (each pair meets at one of
                      the 3 side faces that contain the edge),
    * 2 CAP  cells  : c123 (caps the v10 end) and c456 (caps the v11 end). In [I]
                      they do NOT touch; the I->H reconnection creates the triangle
                      face between them -- that new contact IS the neighbour exchange.
    * 9 faces       : 3 side faces (between side-cell pairs, all containing the edge),
                      3 top faces (side cell <-> c123), 3 bottom faces (side cell <-> c456).
    * 6 outer verts : 3 connected to v10, 3 connected to v11.

TissueForge has no explicit Edge object: an "edge" is an ordered consecutive vertex
pair appearing in a surface's vertex ring. We therefore identify the short edge as a
vertex pair (v10, v11) that is consecutive in exactly the 3 side surfaces. The whole
walk uses only TF adjacency helpers:
    Vertex.getBodies / connected_vertices / shared_surfaces, Surface.getBodies /
    vertices / neighbor_vertices, Body.find_interface.

Validated against a real Kelvin (BCC) interior edge in
rnr/scripts/experiment_neighbourhood.py.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import tissue_forge as tf
from tissue_forge.models.vertex import solver as tfv


# --------------------------------------------------------------------------------------
# small adjacency helpers (all read-only)
# --------------------------------------------------------------------------------------
def body_ids(v) -> frozenset:
    """Stable set of body ids touching a vertex (id is the only stable identity)."""
    return frozenset(b.id for b in v.getBodies())


def surf_ids(v) -> frozenset:
    return frozenset(s.id for s in v.getSurfaces())


def edge_length(v10, v11) -> float:
    return (v10.position - v11.position).length()


def is_consecutive(surface, va, vb) -> bool:
    """True iff va, vb appear consecutively (cyclically) in the surface's vertex ring."""
    ids = [x.id for x in surface.vertices]
    n = len(ids)
    for i in range(n):
        a, b = ids[i], ids[(i + 1) % n]
        if (a == va.id and b == vb.id) or (a == vb.id and b == va.id):
            return True
    return False


def ring_neighbors(surface, v) -> List[object]:
    """The two ring-neighbours of v in `surface`'s ordered vertex list.

    (Surface.neighbor_vertices returns a SWIG std::tuple that isn't iterable from
    Python, so we walk the ordered ring directly.)
    """
    vs = list(surface.vertices)
    n = len(vs)
    for i, x in enumerate(vs):
        if x.id == v.id:
            return [vs[(i - 1) % n], vs[(i + 1) % n]]
    return []


def other_neighbor(surface, v, exclude_id) -> Optional[object]:
    """The neighbour of v within `surface`'s ring other than the excluded vertex.

    A polygon vertex has exactly two ring-neighbours; one is the short-edge partner,
    the other is the surface's outer vertex on v's side.
    """
    nbrs = [w for w in ring_neighbors(surface, v) if w.id != exclude_id]
    return nbrs[0] if len(nbrs) == 1 else None


# --------------------------------------------------------------------------------------
# the I-neighbourhood (edge state)
# --------------------------------------------------------------------------------------
@dataclass
class Arm:
    """One of the 3 'arms' of the I-neighbourhood, keyed by a SIDE surface.

    Each side surface bears the short edge and meets two side cells. It maps 1:1 onto
    one triangle vertex of the [H] state, which will connect to this arm's two outer
    vertices (Okuda's 1<->4, 2<->5, 3<->6 pairing -- derived here from the mesh, not
    assumed). `side_cell_ids` are the two side cells this surface separates.
    """
    side_surface: object
    side_surface_id: int
    outer_top: object        # outer vertex on the v10 side (within this side surface)
    outer_bot: object        # outer vertex on the v11 side
    side_cell_ids: frozenset  # the 2 side-cell ids this surface separates


@dataclass
class IConfig:
    """Full I-neighbourhood of a short edge (v10, v11). Read-only snapshot of handles.

    Handles invalidate after any mutation -- reconnect.py must re-fetch by id. The ids
    captured here are the stable references.
    """
    v10: object
    v11: object
    v10_id: int
    v11_id: int
    side_cell_ids: List[int]                  # the 3 side cells
    cap_top: object                           # c123 (caps v10)
    cap_bot: object                           # c456 (caps v11)
    cap_top_id: int
    cap_bot_id: int
    arms: List[Arm]                           # 3 arms (one per side surface)
    top_faces: Dict[int, object]              # side_cell_id -> interface(side, cap_top)
    bottom_faces: Dict[int, object]           # side_cell_id -> interface(side, cap_bot)
    length: float


def i_neighbourhood(v10, v11) -> Optional[IConfig]:
    """Gather + validate the I-neighbourhood of the ordered short edge (v10, v11).

    Returns an IConfig if (v10, v11) is a clean interior 3-cell edge in the canonical
    Okuda [I] configuration, else None (caller treats None as "not a reconnection
    site"). This is purely structural; energetic/threshold triggers (Condition 2) and
    the Condition-4 topology guards live elsewhere.
    """
    # both endpoints must be interior: shared by exactly 4 cells (Okuda: each vertex
    # connects to 4 edges / 4 cells). Boundary (free-surface) vertices fail this.
    b10, b11 = body_ids(v10), body_ids(v11)
    if len(b10) != 4 or len(b11) != 4:
        return None

    # the edge is shared by exactly the 3 SIDE cells (the endpoints' common cells);
    # each endpoint additionally has one unique CAP cell.
    side = b10 & b11
    cap_top_ids = b10 - b11
    cap_bot_ids = b11 - b10
    if len(side) != 3 or len(cap_top_ids) != 1 or len(cap_bot_ids) != 1:
        return None

    # the short edge must be consecutive in exactly 3 surfaces, all interior (2-body).
    shared = v10.shared_surfaces(v11)
    side_surfs = [s for s in shared if is_consecutive(s, v10, v11)]
    if len(side_surfs) != 3 or any(len(s.getBodies()) != 2 for s in side_surfs):
        return None

    # body-id -> handle (caps + side cells), via the side surfaces and endpoint sets.
    bid = {}
    for s in side_surfs:
        for b in s.getBodies():
            bid[b.id] = b
    # caps are NOT on the side surfaces; fetch their handles from the endpoint vertices.
    cap_top_id = next(iter(cap_top_ids))
    cap_bot_id = next(iter(cap_bot_ids))
    cap_top = next(b for b in v10.getBodies() if b.id == cap_top_id)
    cap_bot = next(b for b in v11.getBodies() if b.id == cap_bot_id)
    for b in list(v10.getBodies()) + list(v11.getBodies()):
        bid[b.id] = b

    # build the 3 arms: each side surface -> its two outer vertices + the side-cell pair.
    arms: List[Arm] = []
    for s in side_surfs:
        ot = other_neighbor(s, v10, v11.id)
        ob = other_neighbor(s, v11, v10.id)
        if ot is None or ob is None:
            return None
        scells = frozenset(b.id for b in s.getBodies())
        arms.append(Arm(side_surface=s, side_surface_id=s.id,
                        outer_top=ot, outer_bot=ob, side_cell_ids=scells))

    # 3 top + 3 bottom faces: the side<->cap interfaces. Each must be a single surface
    # (the Condition-4 "extra rule": two cells share at most one face).
    top_faces, bottom_faces = {}, {}
    for sc_id in side:
        sc = bid[sc_id]
        it = sc.find_interface(cap_top)
        ib = sc.find_interface(cap_bot)
        if len(it) != 1 or len(ib) != 1:
            return None
        top_faces[sc_id] = it[0]
        bottom_faces[sc_id] = ib[0]

    return IConfig(
        v10=v10, v11=v11, v10_id=v10.id, v11_id=v11.id,
        side_cell_ids=sorted(side),
        cap_top=cap_top, cap_bot=cap_bot,
        cap_top_id=cap_top_id, cap_bot_id=cap_bot_id,
        arms=arms, top_faces=top_faces, bottom_faces=bottom_faces,
        length=edge_length(v10, v11),
    )


# --------------------------------------------------------------------------------------
# the H-neighbourhood (triangle state) -- reverse walk for h_to_i
# --------------------------------------------------------------------------------------
@dataclass
class HArm:
    """One arm of the H-neighbourhood, keyed by a triangle VERTEX (v7/v8/v9).

    The triangle vertex maps 1:1 onto a [I] side surface; it connects (besides the two
    other triangle vertices) to two outer vertices -- one on the c123 side, one on the
    c456 side -- which become the side surface's outer vertices after H->I.
    """
    tri_vertex: object
    tri_vertex_id: int
    side_surface: object     # the side face holding this triangle vertex (an arm's face)
    side_surface_id: int
    outer_top: object        # outer vertex toward cap_top (c123)
    outer_bot: object        # outer vertex toward cap_bot (c456)
    side_cell_ids: frozenset  # the 2 side cells this surface separates


@dataclass
class HConfig:
    triangle: object
    triangle_id: int
    tri_vertex_ids: List[int]
    cap_top: object
    cap_bot: object
    cap_top_id: int
    cap_bot_id: int
    side_cell_ids: List[int]
    # triangle edge (frozenset of 2 tri-vertex ids) -> the side cell sharing that edge
    edge_side_cell: Dict[frozenset, int]
    arms: List[HArm]                          # 3 arms (one per triangle vertex / side face)
    top_faces: Dict[int, object]              # side_cell_id -> interface(side, cap_top)
    bottom_faces: Dict[int, object]           # side_cell_id -> interface(side, cap_bot)
    max_edge: float


def _tri_edges(vids: List[int]) -> List[frozenset]:
    return [frozenset((vids[0], vids[1])),
            frozenset((vids[1], vids[2])),
            frozenset((vids[2], vids[0]))]


def h_neighbourhood(triangle) -> Optional[HConfig]:
    """Gather + validate the H-neighbourhood of a triangular face (the [H] state).

    A reconnection-site triangle has exactly 3 vertices, is shared by exactly 2 cells
    (the caps c123/c456), each triangle edge is additionally shared by exactly one side
    cell, and the surrounding 9 faces (3 side / 3 top / 3 bottom) resolve uniquely.
    Mirrors i_neighbourhood so the H->I collapse has the same handles the I->H build
    used. Returns an HConfig or None.
    """
    verts = list(triangle.vertices)
    if len(verts) != 3:
        return None
    caps = triangle.getBodies()
    if len(caps) != 2:
        return None
    cap_top, cap_bot = caps[0], caps[1]

    # each triangle vertex must be interior (4 cells).
    if any(len(v.getBodies()) != 4 for v in verts):
        return None

    vids = [v.id for v in verts]
    cap_ids = {cap_top.id, cap_bot.id}
    vbyid = {v.id: v for v in verts}

    # side cell per triangle edge = the common cell of the edge's endpoints minus caps.
    edge_side_cell: Dict[frozenset, int] = {}
    side_ids = set()
    for e in _tri_edges(vids):
        a, b = tuple(e)
        common = body_ids(vbyid[a]) & body_ids(vbyid[b])
        rest = common - cap_ids
        if len(rest) != 1:
            return None
        sc = next(iter(rest))
        edge_side_cell[e] = sc
        side_ids.add(sc)
    if len(side_ids) != 3:
        return None

    # body-id -> handle for caps + side cells.
    bid = {cap_top.id: cap_top, cap_bot.id: cap_bot}
    for v in verts:
        for b in v.getBodies():
            bid[b.id] = b

    # each triangle vertex sits in exactly one SIDE face: the interface between the two
    # side cells flanking it (= the side cells of its two incident triangle edges). That
    # face contains the triangle vertex plus its two outer vertices (toward each cap).
    arms: List[HArm] = []
    max_edge = max(edge_length(vbyid[tuple(e)[0]], vbyid[tuple(e)[1]]) for e in _tri_edges(vids))
    for v in verts:
        # the two triangle edges at v -> their two side cells flank v's side face.
        flank = [edge_side_cell[e] for e in _tri_edges(vids) if v.id in e]
        if len(flank) != 2:
            return None
        sc_a, sc_b = bid[flank[0]], bid[flank[1]]
        iface = sc_a.find_interface(sc_b)
        side_surf = next((s for s in iface if v.id in {x.id for x in s.vertices}), None)
        if side_surf is None:
            return None
        # the two ring-neighbours of v in that side face are its outer vertices.
        nbrs = ring_neighbors(side_surf, v)
        if len(nbrs) != 2:
            return None
        outer_top = outer_bot = None
        for w in nbrs:
            wb = body_ids(w)
            if cap_top.id in wb:
                outer_top = w
            elif cap_bot.id in wb:
                outer_bot = w
        if outer_top is None or outer_bot is None:
            return None
        arms.append(HArm(tri_vertex=v, tri_vertex_id=v.id,
                         side_surface=side_surf, side_surface_id=side_surf.id,
                         outer_top=outer_top, outer_bot=outer_bot,
                         side_cell_ids=frozenset((sc_a.id, sc_b.id))))

    # 3 top + 3 bottom faces (side<->cap), each a unique surface.
    top_faces, bottom_faces = {}, {}
    for sc_id in side_ids:
        sc = bid[sc_id]
        it = sc.find_interface(cap_top)
        ib = sc.find_interface(cap_bot)
        if len(it) != 1 or len(ib) != 1:
            return None
        top_faces[sc_id] = it[0]
        bottom_faces[sc_id] = ib[0]

    return HConfig(
        triangle=triangle, triangle_id=triangle.id, tri_vertex_ids=vids,
        cap_top=cap_top, cap_bot=cap_bot, cap_top_id=cap_top.id, cap_bot_id=cap_bot.id,
        side_cell_ids=sorted(side_ids), edge_side_cell=edge_side_cell,
        arms=arms, top_faces=top_faces, bottom_faces=bottom_faces, max_edge=max_edge,
    )


# --------------------------------------------------------------------------------------
# scanners (Condition 2 triggers -- used by the Phase-2 operator, handy for tests too)
# --------------------------------------------------------------------------------------
def find_short_edges(bodies, threshold: float) -> List[Tuple[object, object, IConfig]]:
    """All interior short edges (length < threshold) that form a valid [I] config.

    Condition 2 (Okuda Eq. for reconnection trigger): reconnect when the relevant edge
    length is below Delta_l_th. Returns (v10, v11, IConfig) triples, deduped by edge.
    """
    seen = set()
    out = []
    vbyid = {}
    for b in bodies:
        for v in b.getVertices():
            vbyid[v.id] = v
    for v in vbyid.values():
        if len(v.getBodies()) != 4:
            continue
        for w in v.connected_vertices:
            if w.id <= v.id:
                continue
            key = (v.id, w.id)
            if key in seen:
                continue
            seen.add(key)
            if edge_length(v, w) >= threshold:
                continue
            cfg = i_neighbourhood(v, w)
            if cfg is not None:
                out.append((v, w, cfg))
    return out


def find_small_triangles(surfaces, threshold: float) -> List[Tuple[object, HConfig]]:
    """All triangular faces whose max edge < threshold that form a valid [H] config.

    Condition 2 for the reverse direction: reconnect H->I when the max of the three
    triangle edges is below Delta_l_th (NOT the min -- that is Honda's wrong 'condition
    H'; see CLAUDE.md). Returns (triangle, HConfig) pairs.
    """
    out = []
    for s in surfaces:
        if len(list(s.vertices)) != 3:
            continue
        cfg = h_neighbourhood(s)
        if cfg is not None and cfg.max_edge < threshold:
            out.append((s, cfg))
    return out
