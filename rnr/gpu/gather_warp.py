"""On-GPU GATHER (docs/2026-06-24_gpu-3d-vertex-model-exploration.md, the "never returns to the
host" step): per-candidate [I]-neighbourhood gather as a Warp kernel -- the device counterpart of
topology_csr.i_neighbourhood_csr + the schedule_csr.i_to_h_veto_csr Condition-4 veto, FUSED.

After detect_warp's scan produces candidate short edges, this walks each candidate's neighbourhood
ON THE DEVICE and emits the surgery-ready packed arrays the reservation + apply kernels consume
(cap_top/cap_bot, side_cells[3], arm_side/otop/obot[3], top/bot[3]) plus a `valid` flag. Combined
with the scan (detect_warp), reservation + apply (schedule_warp/reconnect_warp) and compaction
(compact_warp), a whole reconnection round runs with NO PaddedMesh.from_warp(g) round-trip.

Implementation notes (the kernel constraints that shaped this):
  * NO per-thread scratch arrays -- every result is written straight to a per-candidate row of a
    global output array; set membership / dedup / interface counts are O(k^2) loops over the
    bounded local adjacency (valence <=8, face <=6, cell <=14), exactly like detect_warp's scan.
  * b10 (n c)/cap classification: cap_top = the one body incident to v10 but NOT v11; side cells =
    bodies incident to BOTH; cap_bot = the body incident to v11 but not v10 -- no materialised set.
  * The Condition-4 veto is FUSED into `valid` (caps must not already touch; no side face may be a
    triangle; no side-cell pair may already share >=2 faces) so `valid==1` <=> the host gather
    returns a config AND the host veto passes -- a drop-in for detect+gather+veto.
Order within arm/top/bot rows is free (the apply kernel matches outer verts to arms by value), so
the gate normalises (sorts) both sides before comparing per candidate.
"""
import numpy as np

import warp as wp

from .detect_warp import (d_ring_neighbor, d_vert_body_count, find_short_edges_warp,
                          find_small_triangles_warp)
from .reconnect_csr import ArmIdx, HArmIdx, HCfgIdx, ICfgIdx
from .reconnect_warp import d_ring_pos

wp.init()


# --------------------------------------------------------------------------------------
# small device adjacency predicates (read-only; mirror topology_csr helpers)
# --------------------------------------------------------------------------------------
@wp.func
def d_body_at_vert(s2b: wp.array2d(dtype=wp.int32), v2s: wp.array2d(dtype=wp.int32),
                   v2s_len: wp.array(dtype=wp.int32), b: int, v: int) -> int:
    """1 iff body b is incident to vertex v (some incident surface of v borders b)."""
    L = v2s_len[v]
    for a in range(L):
        s = v2s[v, a]
        if s2b[s, 0] == b or s2b[s, 1] == b:
            return 1
    return 0


@wp.func
def d_interface_count(s2b: wp.array2d(dtype=wp.int32), b2s: wp.array2d(dtype=wp.int32),
                      b2s_len: wp.array(dtype=wp.int32), surf_alive: wp.array(dtype=wp.int32),
                      ba: int, bb: int) -> int:
    """Number of live surfaces separating exactly bodies ba and bb (Body.find_interface count)."""
    cnt = wp.int32(0)
    L = b2s_len[ba]
    for x in range(L):
        s = b2s[ba, x]
        if surf_alive[s] == 1:
            b0 = s2b[s, 0]
            b1 = s2b[s, 1]
            if (b0 == ba and b1 == bb) or (b0 == bb and b1 == ba):
                cnt += 1
    return cnt


@wp.func
def d_interface_first(s2b: wp.array2d(dtype=wp.int32), b2s: wp.array2d(dtype=wp.int32),
                      b2s_len: wp.array(dtype=wp.int32), surf_alive: wp.array(dtype=wp.int32),
                      ba: int, bb: int) -> int:
    """The first live surface separating exactly ba and bb, or -1 (use with d_interface_count)."""
    L = b2s_len[ba]
    for x in range(L):
        s = b2s[ba, x]
        if surf_alive[s] == 1:
            b0 = s2b[s, 0]
            b1 = s2b[s, 1]
            if (b0 == ba and b1 == bb) or (b0 == bb and b1 == ba):
                return s
    return -1


@wp.func
def d_other_neighbor(s2v: wp.array2d(dtype=wp.int32), s2v_len: wp.array(dtype=wp.int32),
                     s: int, v: int, exclude: int) -> int:
    """v's unique ring-neighbour in s other than `exclude` (-1 if not unique / not present)."""
    n0 = d_ring_neighbor(s2v, s2v_len, s, v, 0)
    n1 = d_ring_neighbor(s2v, s2v_len, s, v, 1)
    if n0 == exclude and n1 != exclude:
        return n1
    if n1 == exclude and n0 != exclude:
        return n0
    return -1


# ======================================================================================
# the [I]-neighbourhood gather + fused Condition-4 veto
# ======================================================================================
@wp.kernel
def gather_i_kernel(
        vert_alive: wp.array(dtype=wp.int32),
        v2s: wp.array2d(dtype=wp.int32), v2s_len: wp.array(dtype=wp.int32),
        surf_alive: wp.array(dtype=wp.int32),
        s2v: wp.array2d(dtype=wp.int32), s2v_len: wp.array(dtype=wp.int32),
        s2b: wp.array2d(dtype=wp.int32),
        b2s: wp.array2d(dtype=wp.int32), b2s_len: wp.array(dtype=wp.int32),
        cand_v10: wp.array(dtype=wp.int32), cand_v11: wp.array(dtype=wp.int32),
        out_valid: wp.array(dtype=wp.int32),
        out_cap_top: wp.array(dtype=wp.int32), out_cap_bot: wp.array(dtype=wp.int32),
        out_side: wp.array2d(dtype=wp.int32),
        out_arm_side: wp.array2d(dtype=wp.int32), out_arm_otop: wp.array2d(dtype=wp.int32),
        out_arm_obot: wp.array2d(dtype=wp.int32),
        out_top: wp.array2d(dtype=wp.int32), out_bot: wp.array2d(dtype=wp.int32)):
    i = wp.tid()
    out_valid[i] = 0
    v10 = cand_v10[i]
    v11 = cand_v11[i]

    # ---- classify v10's DISTINCT bodies: side cells (also in v11) vs cap_top (not in v11) ----
    cap_top = wp.int32(-1)
    nside = wp.int32(0)
    n10 = wp.int32(0)
    Lv = v2s_len[v10]
    for a in range(Lv):
        s = v2s[v10, a]
        for slot in range(2):
            b = s2b[s, slot]
            if b >= 0:
                first = wp.int32(1)                         # dedup: first occurrence among v10's bodies?
                for a2 in range(Lv):
                    for slot2 in range(2):
                        if (a2 < a) or (a2 == a and slot2 < slot):
                            if s2b[v2s[v10, a2], slot2] == b:
                                first = wp.int32(0)
                if first == 1:
                    n10 += 1
                    if d_body_at_vert(s2b, v2s, v2s_len, b, v11) == 1:
                        if nside < 3:
                            out_side[i, nside] = b
                        nside += 1
                    else:
                        cap_top = b
    if n10 != 4 or nside != 3 or cap_top < 0:               # v10 interior, 3 side + 1 cap
        return

    # ---- cap_bot: v11's distinct body NOT incident to v10 ; also confirm v11 is interior ----
    cap_bot = wp.int32(-1)
    n11 = wp.int32(0)
    Lw = v2s_len[v11]
    for a in range(Lw):
        s = v2s[v11, a]
        for slot in range(2):
            b = s2b[s, slot]
            if b >= 0:
                first = wp.int32(1)
                for a2 in range(Lw):
                    for slot2 in range(2):
                        if (a2 < a) or (a2 == a and slot2 < slot):
                            if s2b[v2s[v11, a2], slot2] == b:
                                first = wp.int32(0)
                if first == 1:
                    n11 += 1
                    if d_body_at_vert(s2b, v2s, v2s_len, b, v10) == 0:
                        cap_bot = b
    if n11 != 4 or cap_bot < 0:
        return

    # ---- side surfaces / arms: shared, consecutive (v10,v11), 2-body; exactly 3 ----
    narm = wp.int32(0)
    bad = wp.int32(0)
    for a in range(Lv):
        s = v2s[v10, a]
        # consecutive (v10,v11) in s ?  (one of v10's ring neighbours is v11)
        if d_ring_neighbor(s2v, s2v_len, s, v10, 0) == v11 or \
           d_ring_neighbor(s2v, s2v_len, s, v10, 1) == v11:
            if s2b[s, 0] < 0 or s2b[s, 1] < 0:             # interior (2-body) side face only
                bad = wp.int32(1)
            ot = d_other_neighbor(s2v, s2v_len, s, v10, v11)
            ob = d_other_neighbor(s2v, s2v_len, s, v11, v10)
            if ot < 0 or ob < 0:
                bad = wp.int32(1)
            if narm < 3:
                out_arm_side[i, narm] = s
                out_arm_otop[i, narm] = ot
                out_arm_obot[i, narm] = ob
            narm += 1
    if narm != 3 or bad == 1:
        return

    # ---- top/bottom faces: each side cell shares exactly ONE face with each cap ----
    for k in range(3):
        sc = out_side[i, k]
        if d_interface_count(s2b, b2s, b2s_len, surf_alive, sc, cap_top) != 1:
            return
        if d_interface_count(s2b, b2s, b2s_len, surf_alive, sc, cap_bot) != 1:
            return
        out_top[i, k] = d_interface_first(s2b, b2s, b2s_len, surf_alive, sc, cap_top)
        out_bot[i, k] = d_interface_first(s2b, b2s, b2s_len, surf_alive, sc, cap_bot)

    # ---- Condition-4 veto (FUSED): caps mustn't touch; no triangular side face; no side-cell
    #      pair sharing >=2 faces (side<->cap >=2 is already excluded by the ==1 checks above) ----
    if d_interface_count(s2b, b2s, b2s_len, surf_alive, cap_top, cap_bot) != 0:
        return
    for k in range(3):
        if s2v_len[out_arm_side[i, k]] == 3:
            return
    for k in range(3):
        for j in range(k + 1, 3):
            if d_interface_count(s2b, b2s, b2s_len, surf_alive,
                                 out_side[i, k], out_side[i, j]) >= 2:
                return

    out_cap_top[i] = cap_top
    out_cap_bot[i] = cap_bot
    out_valid[i] = 1


def gather_i_configs_warp(g: dict, cand_edges: np.ndarray, device=None) -> dict:
    """Run the device [I]-gather over candidate edges (an (M,2) int32 array of (v10,v11) from
    detect_warp.find_short_edges_warp). Returns a dict of device arrays:
    valid (M,), cap_top/cap_bot (M,), side (M,3), arm_side/arm_otop/arm_obot (M,3), top/bot (M,3),
    plus v10/v11 (M,). `valid==1` <=> host i_neighbourhood_csr returns a config AND i_to_h_veto_csr
    passes. The packed arrays feed the reservation + apply kernels with no host round-trip."""
    dev = g["device"] if device is None else device
    m = int(cand_edges.shape[0])
    a1 = lambda a: wp.array(np.ascontiguousarray(a), dtype=wp.int32, device=dev)
    z1 = lambda: wp.zeros(max(m, 1), dtype=wp.int32, device=dev)
    z2 = lambda: wp.full((max(m, 1), 3), -1, dtype=wp.int32, device=dev)
    cand = np.ascontiguousarray(cand_edges.reshape(-1, 2).astype(np.int32)) if m else np.zeros((1, 2), np.int32)
    c_v10 = a1(cand[:, 0])
    c_v11 = a1(cand[:, 1])
    out = dict(v10=c_v10, v11=c_v11, valid=z1(), cap_top=z1(), cap_bot=z1(),
               side=z2(), arm_side=z2(), arm_otop=z2(), arm_obot=z2(), top=z2(), bot=z2())
    if m == 0:
        return out
    wp.launch(gather_i_kernel, dim=m, device=dev, inputs=[
        g["vert_alive"], g["v2s"], g["v2s_len"], g["surf_alive"], g["s2v"], g["s2v_len"],
        g["s2b"], g["b2s"], g["b2s_len"], c_v10, c_v11,
        out["valid"], out["cap_top"], out["cap_bot"], out["side"],
        out["arm_side"], out["arm_otop"], out["arm_obot"], out["top"], out["bot"]])
    wp.synchronize_device(dev)
    return out


def gather_i_configs_to_list(g: dict, cand_edges: np.ndarray, device=None):
    """Run the device gather and reconstruct the valid (v10, v11, ICfgIdx) list on the host --
    O(candidates) data only, NO PaddedMesh.from_warp / O(mesh) copy. The host ICfgIdx objects let
    the existing reservation + apply (schedule_warp/reconnect_warp) be reused unchanged. top[k]/
    bot[k] correspond to side cell side[k] (the gather writes them in that correspondence)."""
    if cand_edges is None or len(cand_edges) == 0:
        return []
    out = gather_i_configs_warp(g, cand_edges, device)
    valid = out["valid"].numpy()
    v10, v11 = out["v10"].numpy(), out["v11"].numpy()
    cap_top, cap_bot = out["cap_top"].numpy(), out["cap_bot"].numpy()
    side = out["side"].numpy()
    arm_side, arm_otop, arm_obot = out["arm_side"].numpy(), out["arm_otop"].numpy(), out["arm_obot"].numpy()
    top, bot = out["top"].numpy(), out["bot"].numpy()
    res = []
    for i in range(len(valid)):
        if not valid[i]:
            continue
        sc = [int(side[i, k]) for k in range(3)]
        arms = [ArmIdx(side_surface=int(arm_side[i, k]), outer_top=int(arm_otop[i, k]),
                       outer_bot=int(arm_obot[i, k])) for k in range(3)]
        cfg = ICfgIdx(v10=int(v10[i]), v11=int(v11[i]),
                      cap_top=int(cap_top[i]), cap_bot=int(cap_bot[i]), side_cells=sc, arms=arms,
                      top_faces={sc[k]: int(top[i, k]) for k in range(3)},
                      bottom_faces={sc[k]: int(bot[i, k]) for k in range(3)})
        res.append((int(v10[i]), int(v11[i]), cfg))
    return res


def detect_short_edges_device(g: dict, threshold: float, device=None):
    """Fully-on-device [I] detection: GPU scan + GPU gather (with the fused Condition-4 veto),
    NO PaddedMesh.from_warp. Returns the same (v10, v11, ICfgIdx) list as
    detect_warp.detect_short_edges_hybrid -- but no host mesh copy; only O(candidates) data leaves
    the device. (The reservation + apply still run on the device SoA `g`.)"""
    return gather_i_configs_to_list(g, find_short_edges_warp(g, threshold), device)


# ======================================================================================
# reverse direction: the [H]-neighbourhood gather + fused Condition-4 veto
# ======================================================================================
@wp.func
def d_surf_has_vert(s2v: wp.array2d(dtype=wp.int32), s2v_len: wp.array(dtype=wp.int32),
                    s: int, v: int) -> int:
    L = s2v_len[s]
    for i in range(L):
        if s2v[s, i] == v:
            return 1
    return 0


@wp.func
def d_edge_side_cell(s2b: wp.array2d(dtype=wp.int32), v2s: wp.array2d(dtype=wp.int32),
                     v2s_len: wp.array(dtype=wp.int32), a: int, b: int,
                     cap_top: int, cap_bot: int) -> int:
    """The unique body incident to BOTH tri-edge endpoints a,b and not a cap (the edge's side
    cell). -1 if not exactly one (mirrors h_neighbourhood's per-edge side-cell rule)."""
    La = v2s_len[a]
    found = wp.int32(-1)
    cnt = wp.int32(0)
    for x in range(La):
        s = v2s[a, x]
        for slot in range(2):
            bd = s2b[s, slot]
            if bd >= 0 and bd != cap_top and bd != cap_bot:
                if d_body_at_vert(s2b, v2s, v2s_len, bd, b) == 1:
                    first = wp.int32(1)                 # dedup among a's bodies
                    for x2 in range(La):
                        for slot2 in range(2):
                            if (x2 < x) or (x2 == x and slot2 < slot):
                                if s2b[v2s[a, x2], slot2] == bd:
                                    first = wp.int32(0)
                    if first == 1:
                        found = bd
                        cnt += 1
    if cnt != 1:
        return -1
    return found


@wp.func
def d_side_surf_with_vert(s2b: wp.array2d(dtype=wp.int32), b2s: wp.array2d(dtype=wp.int32),
                          b2s_len: wp.array(dtype=wp.int32), s2v: wp.array2d(dtype=wp.int32),
                          s2v_len: wp.array(dtype=wp.int32), surf_alive: wp.array(dtype=wp.int32),
                          fa: int, fb: int, v: int) -> int:
    """The interface surface between side cells fa,fb that contains vertex v in its ring."""
    L = b2s_len[fa]
    for x in range(L):
        s = b2s[fa, x]
        if surf_alive[s] == 1:
            b0 = s2b[s, 0]
            b1 = s2b[s, 1]
            if (b0 == fa and b1 == fb) or (b0 == fb and b1 == fa):
                if d_surf_has_vert(s2v, s2v_len, s, v) == 1:
                    return s
    return -1


@wp.func
def d_outer_for_cap(s2b: wp.array2d(dtype=wp.int32), v2s: wp.array2d(dtype=wp.int32),
                    v2s_len: wp.array(dtype=wp.int32), s2v: wp.array2d(dtype=wp.int32),
                    s2v_len: wp.array(dtype=wp.int32), ss: int, v: int, cap: int) -> int:
    """v's ring-neighbour in side surface ss that is incident to body `cap` (the outer vertex
    toward that cap). -1 if neither neighbour touches the cap."""
    n0 = d_ring_neighbor(s2v, s2v_len, ss, v, 0)
    n1 = d_ring_neighbor(s2v, s2v_len, ss, v, 1)
    if d_body_at_vert(s2b, v2s, v2s_len, cap, n0) == 1:
        return n0
    if d_body_at_vert(s2b, v2s, v2s_len, cap, n1) == 1:
        return n1
    return -1


@wp.func
def d_faces_shared_edges(s2v: wp.array2d(dtype=wp.int32), s2v_len: wp.array(dtype=wp.int32),
                         sa: int, sb: int) -> int:
    """Number of edges (cyclic vertex pairs) shared by faces sa and sb (Condition 4(ii) helper)."""
    La = s2v_len[sa]
    Lb = s2v_len[sb]
    cnt = wp.int32(0)
    for i in range(La):
        a0 = s2v[sa, i]
        a1 = s2v[sa, (i + 1) % La]
        for j in range(Lb):
            b0 = s2v[sb, j]
            b1 = s2v[sb, (j + 1) % Lb]
            if (a0 == b0 and a1 == b1) or (a0 == b1 and a1 == b0):
                cnt += 1
    return cnt


@wp.kernel
def gather_h_kernel(
        vert_alive: wp.array(dtype=wp.int32),
        v2s: wp.array2d(dtype=wp.int32), v2s_len: wp.array(dtype=wp.int32),
        surf_alive: wp.array(dtype=wp.int32),
        s2v: wp.array2d(dtype=wp.int32), s2v_len: wp.array(dtype=wp.int32),
        s2b: wp.array2d(dtype=wp.int32),
        b2s: wp.array2d(dtype=wp.int32), b2s_len: wp.array(dtype=wp.int32),
        cand_tri: wp.array(dtype=wp.int32),
        out_valid: wp.array(dtype=wp.int32),
        out_cap_top: wp.array(dtype=wp.int32), out_cap_bot: wp.array(dtype=wp.int32),
        out_tri: wp.array2d(dtype=wp.int32), out_side: wp.array2d(dtype=wp.int32),
        out_arm_side: wp.array2d(dtype=wp.int32), out_arm_otop: wp.array2d(dtype=wp.int32),
        out_arm_obot: wp.array2d(dtype=wp.int32),
        out_top: wp.array2d(dtype=wp.int32), out_bot: wp.array2d(dtype=wp.int32)):
    i = wp.tid()
    out_valid[i] = 0
    tri = cand_tri[i]
    if surf_alive[tri] == 0 or s2v_len[tri] != 3:
        return
    t0 = s2v[tri, 0]
    t1 = s2v[tri, 1]
    t2 = s2v[tri, 2]
    cap_top = s2b[tri, 0]
    cap_bot = s2b[tri, 1]
    if cap_top < 0 or cap_bot < 0:
        return
    if d_vert_body_count(v2s, v2s_len, s2b, t0) != 4:    # tri verts interior
        return
    if d_vert_body_count(v2s, v2s_len, s2b, t1) != 4:
        return
    if d_vert_body_count(v2s, v2s_len, s2b, t2) != 4:
        return

    # side cell per triangle edge (0,1),(1,2),(2,0); must each be unique + the three distinct
    sc0 = d_edge_side_cell(s2b, v2s, v2s_len, t0, t1, cap_top, cap_bot)
    sc1 = d_edge_side_cell(s2b, v2s, v2s_len, t1, t2, cap_top, cap_bot)
    sc2 = d_edge_side_cell(s2b, v2s, v2s_len, t2, t0, cap_top, cap_bot)
    if sc0 < 0 or sc1 < 0 or sc2 < 0:
        return
    if sc0 == sc1 or sc1 == sc2 or sc0 == sc2:
        return
    out_side[i, 0] = sc0
    out_side[i, 1] = sc1
    out_side[i, 2] = sc2

    # arm k <-> tri vert k: flank cells are the two edges meeting at that vertex
    #   t0 in edges 0,2 -> {sc0,sc2} ; t1 in 0,1 -> {sc0,sc1} ; t2 in 1,2 -> {sc1,sc2}
    ss0 = d_side_surf_with_vert(s2b, b2s, b2s_len, s2v, s2v_len, surf_alive, sc0, sc2, t0)
    ss1 = d_side_surf_with_vert(s2b, b2s, b2s_len, s2v, s2v_len, surf_alive, sc0, sc1, t1)
    ss2 = d_side_surf_with_vert(s2b, b2s, b2s_len, s2v, s2v_len, surf_alive, sc1, sc2, t2)
    if ss0 < 0 or ss1 < 0 or ss2 < 0:
        return
    out_tri[i, 0] = t0
    out_tri[i, 1] = t1
    out_tri[i, 2] = t2
    out_arm_side[i, 0] = ss0
    out_arm_side[i, 1] = ss1
    out_arm_side[i, 2] = ss2
    ot0 = d_outer_for_cap(s2b, v2s, v2s_len, s2v, s2v_len, ss0, t0, cap_top)
    ot1 = d_outer_for_cap(s2b, v2s, v2s_len, s2v, s2v_len, ss1, t1, cap_top)
    ot2 = d_outer_for_cap(s2b, v2s, v2s_len, s2v, s2v_len, ss2, t2, cap_top)
    ob0 = d_outer_for_cap(s2b, v2s, v2s_len, s2v, s2v_len, ss0, t0, cap_bot)
    ob1 = d_outer_for_cap(s2b, v2s, v2s_len, s2v, s2v_len, ss1, t1, cap_bot)
    ob2 = d_outer_for_cap(s2b, v2s, v2s_len, s2v, s2v_len, ss2, t2, cap_bot)
    if ot0 < 0 or ot1 < 0 or ot2 < 0 or ob0 < 0 or ob1 < 0 or ob2 < 0:
        return
    out_arm_otop[i, 0] = ot0
    out_arm_otop[i, 1] = ot1
    out_arm_otop[i, 2] = ot2
    out_arm_obot[i, 0] = ob0
    out_arm_obot[i, 1] = ob1
    out_arm_obot[i, 2] = ob2

    # top/bottom faces: each side cell shares exactly one face with each cap
    for k in range(3):
        sc = out_side[i, k]
        if d_interface_count(s2b, b2s, b2s_len, surf_alive, sc, cap_top) != 1:
            return
        if d_interface_count(s2b, b2s, b2s_len, surf_alive, sc, cap_bot) != 1:
            return
        out_top[i, k] = d_interface_first(s2b, b2s, b2s_len, surf_alive, sc, cap_top)
        out_bot[i, k] = d_interface_first(s2b, b2s, b2s_len, surf_alive, sc, cap_bot)

    # Condition-4 veto (FUSED): caps must share only the triangle; side-cell pairs only one face;
    # no two side faces share >=2 edges (mirror conditions.h_to_i_veto)
    if d_interface_count(s2b, b2s, b2s_len, surf_alive, cap_top, cap_bot) >= 2:
        return
    for k in range(3):
        for j in range(k + 1, 3):
            if d_interface_count(s2b, b2s, b2s_len, surf_alive,
                                 out_side[i, k], out_side[i, j]) >= 2:
                return
    if d_faces_shared_edges(s2v, s2v_len, ss0, ss1) >= 2:
        return
    if d_faces_shared_edges(s2v, s2v_len, ss0, ss2) >= 2:
        return
    if d_faces_shared_edges(s2v, s2v_len, ss1, ss2) >= 2:
        return

    out_cap_top[i] = cap_top
    out_cap_bot[i] = cap_bot
    out_valid[i] = 1


def gather_h_configs_warp(g: dict, cand_tris: np.ndarray, device=None) -> dict:
    """Run the device [H]-gather over candidate triangles (a 1-D int32 array of surface indices
    from detect_warp.find_small_triangles_warp). Returns device arrays mirroring the I-gather plus
    `tri` (M,3). `valid==1` <=> host h_neighbourhood_csr returns a config AND h_to_i_veto_csr passes."""
    dev = g["device"] if device is None else device
    m = int(len(cand_tris))
    z1 = lambda: wp.zeros(max(m, 1), dtype=wp.int32, device=dev)
    z2 = lambda: wp.full((max(m, 1), 3), -1, dtype=wp.int32, device=dev)
    tris = np.ascontiguousarray(np.asarray(cand_tris, np.int32).reshape(-1)) if m else np.zeros(1, np.int32)
    c_tri = wp.array(tris, dtype=wp.int32, device=dev)
    out = dict(tri_cand=c_tri, valid=z1(), cap_top=z1(), cap_bot=z1(), tri=z2(), side=z2(),
               arm_side=z2(), arm_otop=z2(), arm_obot=z2(), top=z2(), bot=z2())
    if m == 0:
        return out
    wp.launch(gather_h_kernel, dim=m, device=dev, inputs=[
        g["vert_alive"], g["v2s"], g["v2s_len"], g["surf_alive"], g["s2v"], g["s2v_len"],
        g["s2b"], g["b2s"], g["b2s_len"], c_tri,
        out["valid"], out["cap_top"], out["cap_bot"], out["tri"], out["side"],
        out["arm_side"], out["arm_otop"], out["arm_obot"], out["top"], out["bot"]])
    wp.synchronize_device(dev)
    return out


def gather_h_configs_to_list(g: dict, cand_tris: np.ndarray, device=None):
    """Run the device [H]-gather and reconstruct the valid (triangle, HCfgIdx) list on the host
    (O(candidates) data only -- no from_warp). top[k]/bot[k] correspond to side cell side[k];
    arm k corresponds to tri vertex tri[k]."""
    if cand_tris is None or len(cand_tris) == 0:
        return []
    out = gather_h_configs_warp(g, cand_tris, device)
    valid = out["valid"].numpy()
    cap_top, cap_bot = out["cap_top"].numpy(), out["cap_bot"].numpy()
    tri = out["tri"].numpy()
    side = out["side"].numpy()
    arm_side, arm_otop, arm_obot = out["arm_side"].numpy(), out["arm_otop"].numpy(), out["arm_obot"].numpy()
    top, bot = out["top"].numpy(), out["bot"].numpy()
    cand = np.asarray(cand_tris).reshape(-1)
    res = []
    for i in range(len(valid)):
        if not valid[i]:
            continue
        sc = [int(side[i, k]) for k in range(3)]
        arms = [HArmIdx(tri_vertex=int(tri[i, k]), side_surface=int(arm_side[i, k]),
                        outer_top=int(arm_otop[i, k]), outer_bot=int(arm_obot[i, k])) for k in range(3)]
        cfg = HCfgIdx(triangle=int(cand[i]), tri_verts=[int(tri[i, k]) for k in range(3)],
                      cap_top=int(cap_top[i]), cap_bot=int(cap_bot[i]), side_cells=sc, arms=arms,
                      top_faces={sc[k]: int(top[i, k]) for k in range(3)},
                      bottom_faces={sc[k]: int(bot[i, k]) for k in range(3)})
        res.append((int(cand[i]), cfg))
    return res


def detect_small_triangles_device(g: dict, threshold: float, device=None):
    """Fully-on-device [H] detection: GPU scan + GPU gather (with the fused Condition-4 veto),
    NO PaddedMesh.from_warp. Returns the same (triangle, HCfgIdx) list as
    detect_warp.detect_small_triangles_hybrid, veto-filtered."""
    return gather_h_configs_to_list(g, find_small_triangles_warp(g, threshold), device)
