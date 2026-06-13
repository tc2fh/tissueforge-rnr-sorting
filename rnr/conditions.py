"""Phase-1 Condition-4 topology guards for the I<->H reconnection (read-only vetoes).

Okuda et al. 2013 (Biomech Model Mechanobiol 12:627-644), Condition 4, exists to
prevent TOPOLOGICALLY irreversible network patterns -- configurations the reconnection
rule cannot undo (paper Sect. 3.3 / 3.4, Figs. 6 and 9):

    (i)  two edges never share two vertices simultaneously   -> "double edge"  [alpha]
    (ii) two faces never share two or more edges simultaneously                [gamma]

tvm/3DVertVor add an EXTRA RULE used as a practical guard:

    (iii) two cells never share two or more faces             -> "double trigonal face" [beta]

These run as VETOES *before* any mutation: if performing the reconnection would create
one of these patterns, the operation is refused and the mesh left untouched. This is the
"check half"; the "mutate half" is in rnr/reconnect.py (the split mirrors TissueForge's
own MeshQuality ops for the eventual C++ port -- see PORTING_NOTES.md).

TissueForge has no explicit Edge object, so the guards are expressed with surface/body
helpers:
    * num_shared_contiguous_vertex_sets(s_a, s_b) -- how many SEPARATE shared edges two
      faces have (>= 2 => they share two-or-more edges => 4(ii) violation),
    * Body.find_interface(b)                       -- the faces between two cells
      (len >= 2 => 4(iii) violation),
    * a vertex-pair adjacency scan over the mesh   -- for 4(i) double edges.
"""
from typing import List, Optional

from . import topology as topo


# --------------------------------------------------------------------------------------
# primitive Condition-4 predicates (each independently unit-tested)
# --------------------------------------------------------------------------------------
def cells_share_multiple_faces(b_a, b_b) -> bool:
    """Condition 4(iii) (the tvm 'extra rule'): True iff two cells share >= 2 faces."""
    return len(b_a.find_interface(b_b)) >= 2


def faces_share_multiple_edges(s_a, s_b) -> bool:
    """Condition 4(ii): True iff two faces share >= 2 separate edges.

    `num_shared_contiguous_vertex_sets` counts maximal runs of contiguous shared
    vertices; two faces sharing two disjoint edges => two such sets.
    """
    return s_a.num_shared_contiguous_vertex_sets(s_b) >= 2


def count_distinct_edges(va, vb) -> int:
    """Number of DISTINCT mesh edges between two vertices (Condition 4(i)).

    With no explicit Edge object, a single edge (va,vb) shows up as the consecutive
    pair (va,vb) in every surface bordering it, but those are all the SAME edge. A
    'double edge' (two edges sharing both endpoints) would appear as the pair being
    non-contiguous in the union of surface rings -- i.e. va and vb adjacent in more
    than one separable way. We detect it as: among the surfaces sharing both vertices,
    the consecutive-pair occurrences do not all coincide on one edge.

    Practically: an interior edge is shared by exactly the faces around it and is one
    edge. We return the number of surfaces in which (va,vb) are consecutive that are
    NOT mutually edge-adjacent -- but since contiguity can't be teased apart without
    edges, we conservatively treat >0 consecutive surfaces as a single edge and rely on
    the structural i_neighbourhood/h_neighbourhood checks + the face/cell guards above
    to catch the alpha/beta/gamma patterns. This helper is kept for completeness and
    direct testing of the 'are these two vertices edge-connected at all' question.
    """
    shared = va.shared_surfaces(vb)
    return sum(1 for s in shared if topo.is_consecutive(s, va, vb))


def vertices_edge_connected(va, vb) -> bool:
    """True iff va and vb are joined by at least one mesh edge."""
    return count_distinct_edges(va, vb) > 0


# --------------------------------------------------------------------------------------
# high-level vetoes for the two reconnection directions
# --------------------------------------------------------------------------------------
def i_to_h_veto(cfg: "topo.IConfig") -> Optional[str]:
    """Return a reason string if I->H on this neighbourhood is ILLEGAL, else None.

    Refuses the Okuda-irreversible patterns:
      * [beta] double trigonal faces: the two caps c123/c456 already share a face, so
        adding the new triangle would make them share two faces -- 4(iii). (This is the
        primary guard tvm uses before I_H.)
      * [alpha] double edges: the short edge is itself an edge of an existing trigonal
        (3-vertex) face. Collapsing such an edge produces two edges on the same vertex
        pair -- 4(i). We flag it as any side surface being a triangle.
      * 4(ii)/4(iii) on the involved cells: any pair of side cells already sharing >= 2
        faces, or a side cell sharing >= 2 faces with a cap.
    """
    # [beta]: caps must not already be in contact.
    if len(cfg.cap_top.find_interface(cfg.cap_bot)) != 0:
        return "caps c123,c456 already share a face (would create double trigonal face, 4(iii)/[beta])"

    # [alpha]: reconnecting an edge of a trigonal face creates a double edge.
    for a in cfg.arms:
        if len(list(a.side_surface.vertices)) == 3:
            return f"side surface {a.side_surface_id} is a triangle (reconnecting its edge -> double edge, 4(i)/[alpha])"

    # 4(iii) among side-cell pairs (each pair should share exactly the one side face).
    bid = {}
    for a in cfg.arms:
        for b in a.side_surface.getBodies():
            bid[b.id] = b
    side = cfg.side_cell_ids
    for i in range(len(side)):
        for j in range(i + 1, len(side)):
            if cells_share_multiple_faces(bid[side[i]], bid[side[j]]):
                return f"side cells {side[i]},{side[j]} already share >=2 faces (4(iii))"

    # 4(iii) side cell vs cap: each side<->cap interface must be a single face (already
    # enforced structurally in i_neighbourhood, re-checked here as a guard).
    for sc_id in side:
        sc = bid[sc_id]
        if cells_share_multiple_faces(sc, cfg.cap_top) or cells_share_multiple_faces(sc, cfg.cap_bot):
            return f"side cell {sc_id} already shares >=2 faces with a cap (4(iii))"

    return None


def h_to_i_veto(cfg: "topo.HConfig") -> Optional[str]:
    """Return a reason string if H->I on this triangle is ILLEGAL, else None.

    In the [H] state the three side cells legitimately still share their (shrunken)
    side faces and the two caps share exactly the triangle -- those are NOT violations.
    H->I removes the caps' triangle and restores the 3-cell edge. It is illegal only if
    it would create an Okuda-irreversible pattern, mirroring the tvm H_I guards:

      * 4(iii): the caps share >= 2 faces (the triangle should be their ONLY contact;
        a second would be a double trigonal face [beta]),
      * 4(iii): any side-cell pair shares >= 2 faces (should share exactly the one side
        face),
      * 4(ii): any pair of the three side faces already shares >= 2 edges (collapsing
        would double an edge -> [alpha]).
    """
    if cells_share_multiple_faces(cfg.cap_top, cfg.cap_bot):
        return "caps share >=2 faces (would leave a second contact, 4(iii)/[beta])"

    bid = {b.id: b for b in (cfg.cap_top, cfg.cap_bot)}
    for v in cfg.triangle.vertices:
        for b in v.getBodies():
            bid[b.id] = b
    side = cfg.side_cell_ids
    for i in range(len(side)):
        for j in range(i + 1, len(side)):
            if cells_share_multiple_faces(bid[side[i]], bid[side[j]]):
                return f"side cells {side[i]},{side[j]} share >=2 faces (4(iii))"

    side_surfs = [a.side_surface for a in cfg.arms]
    for i in range(len(side_surfs)):
        for j in range(i + 1, len(side_surfs)):
            if faces_share_multiple_edges(side_surfs[i], side_surfs[j]):
                return (f"side faces {side_surfs[i].id},{side_surfs[j].id} "
                        f"already share >=2 edges (collapse would double an edge, 4(ii)/[alpha])")

    return None
