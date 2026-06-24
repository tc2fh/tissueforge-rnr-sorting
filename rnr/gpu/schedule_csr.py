"""Gate C bricks C1 (host reference): the conflict-free parallel-I->H scheduler, on the
PaddedMesh. This is the cellGPU independent-set protocol (atomic-reservation ->
maximal-independent-set -> iterated batch), expressed first as host numpy so it is the
validated reference the Gate-C2 Warp kernel must match.

The protocol (per round):
  1. SCAN  -- find all short [I] edges (topology_csr.find_short_edges_csr) = candidates.
  2. VETO  -- drop candidates failing Condition-4 (i_to_h_veto_csr); Okuda's irreversible
              patterns become reservation-time predicates here.
  3. RESERVE -- each surviving candidate claims its full I-neighbourhood FOOTPRINT (the 2
              end-verts + 6 outer verts + 9 faces + 5 cells, per the design doc). Two
              candidates CONFLICT iff their footprints overlap. Greedy selection keeps a
              maximal independent (mutually disjoint) set.
  4. APPLY -- run i_to_h_csr on the independent set. Because footprints are disjoint, the
              surgeries touch no shared existing element, so the batch is safe to apply in
              parallel -- equivalently, in ANY order. (Births use the bump allocator, so
              slot numbers depend on order, but the body-anchored topology does NOT --
              that order-invariance is the parallel-safety guarantee, tested via the
              slot-invariant fingerprint.)
  5. ITERATE -- re-scan; previously-conflicting candidates run in later rounds, until none.

Footprint conservatism mirrors cellGPU reserving every cell a 2D flip's stencil touches:
it can reject a technically-safe pair, never admit an unsafe one.
"""
from typing import Dict, List, Optional, Set, Tuple

from .device_mesh import PaddedMesh
from .reconnect_csr import HCfgIdx, ICfgIdx, h_to_i_csr, i_to_h_csr
from .topology_csr import (edge_length, find_interface, find_short_edges_csr,
                           find_small_triangles_csr)


# --------------------------------------------------------------------------------------
# Condition-4 veto (index world) -- mirror conditions.i_to_h_veto
# --------------------------------------------------------------------------------------
def i_to_h_veto_csr(pm: PaddedMesh, cfg: ICfgIdx) -> Optional[str]:
    """Return a reason if I->H on this neighbourhood is ILLEGAL (Okuda Condition 4), else
    None. Mirrors conditions.i_to_h_veto on indices: caps must not already touch [beta],
    no side face may be a triangle [alpha], and no involved cell pair may already share
    >= 2 faces [4(iii)]."""
    if find_interface(pm, cfg.cap_top, cfg.cap_bot):
        return "caps already share a face (would create double trigonal face, 4(iii)/[beta])"
    for a in cfg.arms:
        if int(pm.s2v_len[a.side_surface]) == 3:
            return f"side surface {a.side_surface} is a triangle (reconnect -> double edge, 4(i)/[alpha])"
    side = cfg.side_cells
    for i in range(len(side)):
        for j in range(i + 1, len(side)):
            if len(find_interface(pm, side[i], side[j])) >= 2:
                return f"side cells {side[i]},{side[j]} already share >=2 faces (4(iii))"
    for sc in side:
        if len(find_interface(pm, sc, cfg.cap_top)) >= 2 or \
           len(find_interface(pm, sc, cfg.cap_bot)) >= 2:
            return f"side cell {sc} already shares >=2 faces with a cap (4(iii))"
    return None


# --------------------------------------------------------------------------------------
# footprint + maximal independent set
# --------------------------------------------------------------------------------------
def footprint(cfg: ICfgIdx) -> Tuple[Set[int], Set[int], Set[int]]:
    """The full I-neighbourhood a candidate reserves: (vertices, surfaces, bodies).

    verts  = 2 end + 6 outer ;  surfs = 3 side + 3 top + 3 bottom ;  bodies = 2 caps + 3 side.
    Newly-BORN elements (tri verts, the new triangle) are bump-allocated to fresh slots, so
    they never collide and are not part of the footprint.
    """
    verts = {cfg.v10, cfg.v11}
    for a in cfg.arms:
        verts.add(a.outer_top)
        verts.add(a.outer_bot)
    surfs = {a.side_surface for a in cfg.arms}
    surfs |= set(cfg.top_faces.values())
    surfs |= set(cfg.bottom_faces.values())
    bodies = {cfg.cap_top, cfg.cap_bot} | set(cfg.side_cells)
    return verts, surfs, bodies


def independent_set(cands: List[Tuple[int, int, ICfgIdx]]
                    ) -> List[Tuple[int, int, ICfgIdx]]:
    """Greedy maximal independent set: keep candidates whose footprints are mutually
    disjoint (the conflict-free batch). Deterministic in input order."""
    rv: Set[int] = set()
    rs: Set[int] = set()
    rb: Set[int] = set()
    chosen = []
    for cand in cands:
        fv, fs, fb = footprint(cand[2])
        if (fv & rv) or (fs & rs) or (fb & rb):
            continue
        rv |= fv
        rs |= fs
        rb |= fb
        chosen.append(cand)
    return chosen


def reserve_won_mask_host(cands: List[Tuple[int, int, ICfgIdx]]) -> List[int]:
    """Host reference for the GPU atomic reservation (schedule_warp): lowest-id-wins.
    owner[e] = min id wanting element e; candidate i wins iff it owns ALL its footprint
    elements. Returns the won-mask (list of 0/1). Deterministic -> the Warp kernel must
    reproduce it bit-for-bit (the device==host check for C2a)."""
    n = len(cands)
    vown: Dict[int, int] = {}
    sown: Dict[int, int] = {}
    bown: Dict[int, int] = {}
    fps = []
    for i, cand in enumerate(cands):
        fv, fs, fb = footprint(cand[2])
        fps.append((fv, fs, fb))
        for e in fv:
            vown[e] = min(vown.get(e, n), i)
        for e in fs:
            sown[e] = min(sown.get(e, n), i)
        for e in fb:
            bown[e] = min(bown.get(e, n), i)
    mask = []
    for i, (fv, fs, fb) in enumerate(fps):
        win = (all(vown[e] == i for e in fv) and all(sown[e] == i for e in fs)
               and all(bown[e] == i for e in fb))
        mask.append(1 if win else 0)
    return mask


def reserve_independent_set_host(cands: List[Tuple[int, int, ICfgIdx]]
                                 ) -> List[Tuple[int, int, ICfgIdx]]:
    """The winners of ONE host reservation round (lowest-id-wins) -- the host mirror of
    schedule_warp.reserve_independent_set_warp.

    NB this is ONE reservation round: conflict-free but NOT maximal. It can select strictly
    fewer than the greedy `independent_set` -- a candidate loses if any of its footprint
    elements is also wanted by a lower-id candidate, even one that itself loses elsewhere
    (the classic A-B, B-C conflicting / A-C disjoint chain: greedy keeps {A,C}, one
    reservation round keeps only {A}). The GPU reservation kernel reproduces THIS set
    bit-for-bit (C2a), so it -- not greedy `independent_set` -- is the per-round selection
    the GPU sweep matches."""
    mask = reserve_won_mask_host(cands)
    return [cands[i] for i in range(len(cands)) if mask[i]]


def batch_is_conflict_free(batch: List[Tuple[int, int, ICfgIdx]]) -> bool:
    """True iff every pair of candidates in `batch` has disjoint footprints."""
    rv: Set[int] = set()
    rs: Set[int] = set()
    rb: Set[int] = set()
    for cand in batch:
        fv, fs, fb = footprint(cand[2])
        if (fv & rv) or (fs & rs) or (fb & rb):
            return False
        rv |= fv
        rs |= fs
        rb |= fb
    return True


# --------------------------------------------------------------------------------------
# apply + iterated sweep
# --------------------------------------------------------------------------------------
def apply_batch(pm: PaddedMesh, batch: List[Tuple[int, int, ICfgIdx]],
                dl_th: float) -> int:
    """Run i_to_h_csr on each candidate of a (conflict-free) batch. Returns the count.
    Order does not affect the resulting body-anchored topology (parallel-safety)."""
    n = 0
    for v10, v11, cfg in batch:
        i_to_h_csr(pm, cfg, dl_th)
        n += 1
    return n


def reconnect_sweep_i_to_h(pm: PaddedMesh, threshold: float,
                           dl_th: Optional[float] = None, veto: bool = True,
                           max_rounds: int = 64) -> Dict[str, object]:
    """Iterated independent-set I->H sweep until no legal short [I] edge remains (or
    max_rounds). Returns a report with per-round batch sizes + total reconnections.

    `dl_th` is the Okuda placement length (defaults to `threshold`); `veto` toggles the
    Condition-4 predicate filter. Each round applies a maximal independent set, so the
    work is conflict-free within a round and resolves the rest across rounds."""
    if dl_th is None:
        dl_th = threshold
    total = 0
    round_sizes = []
    rounds = 0
    while rounds < max_rounds:
        sites = find_short_edges_csr(pm, threshold)
        if veto:
            sites = [s for s in sites if i_to_h_veto_csr(pm, s[2]) is None]
        if not sites:
            break
        batch = independent_set(sites)
        total += apply_batch(pm, batch, dl_th)
        round_sizes.append(len(batch))
        rounds += 1
    return dict(total=total, rounds=rounds, round_sizes=round_sizes,
                converged=(rounds < max_rounds))


def reconnect_sweep_reserve_host(pm: PaddedMesh, threshold: float,
                                 dl_th: Optional[float] = None, veto: bool = True,
                                 max_rounds: int = 64) -> Dict[str, object]:
    """Host reference for schedule_warp.reconnect_sweep_warp: the iterated I->H sweep whose
    per-round selection is the lowest-id-wins RESERVATION (reserve_independent_set_host),
    mirroring the GPU detect->reserve->apply loop exactly.

    This differs from reconnect_sweep_i_to_h only in the selection step: that one uses the
    greedy MAXIMAL independent set; this one uses ONE reservation round (the GPU's actual
    protocol -- see reserve_independent_set_host). So this, not reconnect_sweep_i_to_h, is
    the faithful per-round host mirror the GPU sweep is gated against."""
    if dl_th is None:
        dl_th = threshold
    total = 0
    round_sizes = []
    rounds = 0
    while rounds < max_rounds:
        sites = find_short_edges_csr(pm, threshold)
        sites.sort(key=lambda s: (s[0], s[1]))       # canonical order (matches reconnect_sweep_warp)
        if veto:
            sites = [s for s in sites if i_to_h_veto_csr(pm, s[2]) is None]
        if not sites:
            break
        batch = reserve_independent_set_host(sites)
        total += apply_batch(pm, batch, dl_th)
        round_sizes.append(len(batch))
        rounds += 1
    return dict(total=total, rounds=rounds, round_sizes=round_sizes,
                converged=(rounds < max_rounds))


# ======================================================================================
# Reverse direction -- brick C1': the H->I host-reference scheduler
# ======================================================================================
# The exact mirror of the I-side above, for the reverse reconnection (a small triangular
# face collapses back to a short edge). The cellGPU independent-set protocol is direction-
# agnostic; only the footprint shape and the Condition-4 predicate differ. Candidates here
# are (triangle_idx, HCfgIdx) pairs (from topology_csr.find_small_triangles_csr), so the
# helpers index cand[1] for the config (vs the I-side's cand[2]).
# --------------------------------------------------------------------------------------
def _face_edges(pm: PaddedMesh, s: int) -> Set[frozenset]:
    """The undirected edges of face s, as frozenset({a,b}) for each cyclically-consecutive
    ring pair. (TF has no explicit Edge; an edge IS a consecutive vertex pair in a ring.)"""
    L = int(pm.s2v_len[s])
    row = pm.s2v[s, :L]
    return {frozenset((int(row[i]), int(row[(i + 1) % L]))) for i in range(L)}


def faces_share_multiple_edges_csr(pm: PaddedMesh, sa: int, sb: int) -> bool:
    """Condition 4(ii) (index world): True iff faces sa, sb share >= 2 distinct edges.
    Mirrors conditions.faces_share_multiple_edges -- we intersect the two faces' edge sets
    (two shared edges => the patterns Okuda Fig. 6 forbids). Counting distinct shared edges
    is equivalent to (and simpler than) the TF helper's contiguous-vertex-run count for this
    veto; noted as the one modelling-convention departure here."""
    return len(_face_edges(pm, sa) & _face_edges(pm, sb)) >= 2


def h_to_i_veto_csr(pm: PaddedMesh, cfg: HCfgIdx) -> Optional[str]:
    """Return a reason if H->I on this triangle is ILLEGAL (Okuda Condition 4), else None.
    Mirrors conditions.h_to_i_veto on indices: the caps must share ONLY the triangle (not
    >= 2 faces), each side-cell pair must share exactly one side face, and no two side faces
    may already share >= 2 edges (collapsing would double an edge -> [alpha])."""
    if len(find_interface(pm, cfg.cap_top, cfg.cap_bot)) >= 2:
        return "caps share >=2 faces (would leave a second contact, 4(iii)/[beta])"
    side = cfg.side_cells
    for i in range(len(side)):
        for j in range(i + 1, len(side)):
            if len(find_interface(pm, side[i], side[j])) >= 2:
                return f"side cells {side[i]},{side[j]} share >=2 faces (4(iii))"
    side_surfs = [a.side_surface for a in cfg.arms]
    for i in range(len(side_surfs)):
        for j in range(i + 1, len(side_surfs)):
            if faces_share_multiple_edges_csr(pm, side_surfs[i], side_surfs[j]):
                return (f"side faces {side_surfs[i]},{side_surfs[j]} already share >=2 edges "
                        f"(collapse would double an edge, 4(ii)/[alpha])")
    return None


def h_footprint(cfg: HCfgIdx) -> Tuple[Set[int], Set[int], Set[int]]:
    """The full H-neighbourhood a reverse candidate reserves: (vertices, surfaces, bodies).

    verts  = 3 tri + 6 outer (9) ;  surfs = the triangle + 3 side + 3 top + 3 bottom (10) ;
    bodies = 2 caps + 3 side (5).

    UNLIKE the forward footprint, the triangle AND its 3 vertices are EXISTING elements (to
    be destroyed), so they belong to the footprint; only the 2 recovered edge vertices are
    bump-allocated (fresh) and excluded. Including the triangle's vertices is what makes a
    cascade side-collapse triangle conflict with its parent cap-cap triangle (they share the
    new tri vertex), so the reverse scheduler serialises them rather than double-touching it."""
    verts = set(cfg.tri_verts)
    for a in cfg.arms:
        verts.add(a.outer_top)
        verts.add(a.outer_bot)
    surfs = {cfg.triangle} | {a.side_surface for a in cfg.arms}
    surfs |= set(cfg.top_faces.values())
    surfs |= set(cfg.bottom_faces.values())
    bodies = {cfg.cap_top, cfg.cap_bot} | set(cfg.side_cells)
    return verts, surfs, bodies


def h_independent_set(cands: List[Tuple[int, HCfgIdx]]) -> List[Tuple[int, HCfgIdx]]:
    """Greedy maximal independent set over H-footprints (mirror independent_set)."""
    rv: Set[int] = set()
    rs: Set[int] = set()
    rb: Set[int] = set()
    chosen = []
    for cand in cands:
        fv, fs, fb = h_footprint(cand[1])
        if (fv & rv) or (fs & rs) or (fb & rb):
            continue
        rv |= fv
        rs |= fs
        rb |= fb
        chosen.append(cand)
    return chosen


def h_reserve_won_mask_host(cands: List[Tuple[int, HCfgIdx]]) -> List[int]:
    """Host reference for the GPU H-reservation: lowest-id-wins over H-footprints (mirror
    reserve_won_mask_host). Deterministic -> the Warp kernel must reproduce it bit-for-bit."""
    n = len(cands)
    vown: Dict[int, int] = {}
    sown: Dict[int, int] = {}
    bown: Dict[int, int] = {}
    fps = []
    for i, cand in enumerate(cands):
        fv, fs, fb = h_footprint(cand[1])
        fps.append((fv, fs, fb))
        for e in fv:
            vown[e] = min(vown.get(e, n), i)
        for e in fs:
            sown[e] = min(sown.get(e, n), i)
        for e in fb:
            bown[e] = min(bown.get(e, n), i)
    mask = []
    for i, (fv, fs, fb) in enumerate(fps):
        win = (all(vown[e] == i for e in fv) and all(sown[e] == i for e in fs)
               and all(bown[e] == i for e in fb))
        mask.append(1 if win else 0)
    return mask


def h_reserve_independent_set_host(cands: List[Tuple[int, HCfgIdx]]
                                   ) -> List[Tuple[int, HCfgIdx]]:
    """The winners of ONE host H-reservation round (lowest-id-wins) -- the host mirror of
    schedule_warp.reserve_h_independent_set_warp (conflict-free, NOT maximal)."""
    mask = h_reserve_won_mask_host(cands)
    return [cands[i] for i in range(len(cands)) if mask[i]]


def h_batch_is_conflict_free(batch: List[Tuple[int, HCfgIdx]]) -> bool:
    """True iff every pair of candidates in `batch` has disjoint H-footprints."""
    rv: Set[int] = set()
    rs: Set[int] = set()
    rb: Set[int] = set()
    for cand in batch:
        fv, fs, fb = h_footprint(cand[1])
        if (fv & rv) or (fs & rs) or (fb & rb):
            return False
        rv |= fv
        rs |= fs
        rb |= fb
    return True


def h_apply_batch(pm: PaddedMesh, batch: List[Tuple[int, HCfgIdx]], dl_th: float) -> int:
    """Run h_to_i_csr on each candidate of a (conflict-free) batch. Returns the count.
    Order does not affect the resulting body-anchored topology (parallel-safety)."""
    n = 0
    for _triangle, cfg in batch:
        h_to_i_csr(pm, cfg, dl_th)
        n += 1
    return n


def reconnect_sweep_h_to_i(pm: PaddedMesh, threshold: float,
                           dl_th: Optional[float] = None, veto: bool = True,
                           max_rounds: int = 64) -> Dict[str, object]:
    """Iterated independent-set H->I sweep (the reverse mirror of reconnect_sweep_i_to_h):
    each round detect small triangles, veto the illegal ones, apply a maximal independent
    set. Returns a report with per-round batch sizes + total reverse reconnections.

    NOTE (reverse cascade): collapsing a triangle moves its outer vertices, which can seed
    new small triangles among neighbours, so a static sweep need not converge -- bound
    max_rounds. (The H-analog of the forward C1 finding; production relaxes between steps.)"""
    if dl_th is None:
        dl_th = threshold
    total = 0
    round_sizes = []
    rounds = 0
    while rounds < max_rounds:
        sites = find_small_triangles_csr(pm, threshold)
        if veto:
            sites = [s for s in sites if h_to_i_veto_csr(pm, s[1]) is None]
        if not sites:
            break
        batch = h_independent_set(sites)
        if not batch:
            break
        total += h_apply_batch(pm, batch, dl_th)
        round_sizes.append(len(batch))
        rounds += 1
    return dict(total=total, rounds=rounds, round_sizes=round_sizes,
                converged=(rounds < max_rounds))


def reconnect_sweep_h_reserve_host(pm: PaddedMesh, threshold: float,
                                   dl_th: Optional[float] = None, veto: bool = True,
                                   max_rounds: int = 64) -> Dict[str, object]:
    """Host reference for schedule_warp.reconnect_sweep_h_to_i_warp: the iterated H->I sweep
    whose per-round selection is the lowest-id-wins RESERVATION (h_reserve_independent_set_host),
    mirroring the GPU detect->reserve->apply loop exactly. The reverse analogue of
    reconnect_sweep_reserve_host (it -- not the greedy reconnect_sweep_h_to_i -- is the
    faithful per-round mirror the GPU reverse sweep is gated against)."""
    if dl_th is None:
        dl_th = threshold
    total = 0
    round_sizes = []
    rounds = 0
    while rounds < max_rounds:
        sites = find_small_triangles_csr(pm, threshold)
        sites.sort(key=lambda s: s[0])               # canonical order (matches reconnect_sweep_h_to_i_warp)
        if veto:
            sites = [s for s in sites if h_to_i_veto_csr(pm, s[1]) is None]
        if not sites:
            break
        batch = h_reserve_independent_set_host(sites)
        if not batch:
            break
        total += h_apply_batch(pm, batch, dl_th)
        round_sizes.append(len(batch))
        rounds += 1
    return dict(total=total, rounds=rounds, round_sizes=round_sizes,
                converged=(rounds < max_rounds))
