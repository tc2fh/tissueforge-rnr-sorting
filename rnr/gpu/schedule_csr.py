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
from .reconnect_csr import ICfgIdx, i_to_h_csr
from .topology_csr import edge_length, find_interface, find_short_edges_csr


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
