"""Sorting / jamming readouts for the Phase-0 control.

Two complementary views of the cell-cell contact network:

  * TOPOLOGICAL -- counts of heterotypic vs total neighbour PAIRS. Without a 3D
    reconnection (T1 / Okuda I<->H) the mesh topology is frozen, so these counts
    stay CONSTANT for the whole run. A heterotypic-pair count that never changes
    is the signature of jamming: no neighbour exchange => no true sorting. This is
    the control's key readout.

  * GEOMETRIC -- area-weighted heterotypic-contact fraction, Sum(A_het)/Sum(A_all).
    This can drift as cells relax their shapes at FIXED topology, so a small change
    here is NOT sorting; only the topological metric proves rearrangement.

Plus an adhesion-energy proxy Sum(lam_ij * A_ij) over contacts -- the
sorting-relevant interfacial energy (TissueForge exposes no single global-energy
accessor, and this is the quantity differential adhesion actually minimises).

Body handles are re-fetched from `type.instances` on every call, since topology
mutations invalidate older handles (a standing rule for the reconnection phase).
"""
from typing import Callable, Dict, List, Optional


def type_name(b) -> str:
    """Type name of a body handle (the accessor is a method on the handle)."""
    t = b.type
    t = t() if callable(t) else t
    return t.name


def _all_bodies(types) -> List:
    bodies = []
    for t in types:
        bodies += list(t.instances)
    return bodies


def demixing_index(types=None, bodies: Optional[List] = None) -> float:
    """3DVertVor-style per-cell demixing index, SIGNED so more sorted = more negative.

    For each cell, het_frac = (# heterotypic neighbours) / (# neighbours), counting only
    neighbours within scope (same scoping rule as `contact_summary`: pass `types` to use
    all their instances, or an explicit `bodies` list when several sub-meshes share global
    types). The index is the cell-average

        D = mean_over_cells( 2 * (het_frac - 0.5) )   in [-1, +1].

    Fully sorted (cells surrounded by their own type) -> het_frac -> 0 -> D -> -1;
    salt-and-pepper (half the neighbours heterotypic) -> D ~ 0;
    anti-sorted (all neighbours heterotypic) -> D -> +1. Cells with no in-scope
    neighbours are skipped; returns 0.0 if no cell qualifies.

    Sign convention (a deliberate DEPARTURE-note vs 3DVertVor, which reports a positive
    "sorting" measure): we sign it so demixing drives D DOWN, matching the het-pair /
    het-area / adhesion-energy curves -- all decrease as the tissue sorts.
    """
    if bodies is None:
        bodies = _all_bodies(types)
    name = {b.id: type_name(b) for b in bodies}
    scope = set(name)
    vals = []
    for b in bodies:
        ni = name[b.id]
        nbrs = [nb for nb in b.connected_bodies if nb.id in scope]
        if not nbrs:
            continue
        het = sum(1 for nb in nbrs if name[nb.id] != ni)
        vals.append(2.0 * (het / len(nbrs) - 0.5))
    return float(sum(vals) / len(vals)) if vals else 0.0


def het_frac_from_D(D: float) -> float:
    """Inverse of the (count-based) demixing index: het_frac = D/2 + 1/2.

    `demixing_index` returns D = mean(2*(het_frac - 1/2)); this recovers the cell-average
    heterotypic neighbour fraction so it can be fed to `sorting_score` on the same footing as
    an area fraction.
    """
    return D / 2.0 + 0.5


def het_frac_from_oracle_demix(demix: float) -> float:
    """Heterotypic fraction implied by the 3DVertVor `dumpDemix` value.

    The oracle (`Run::dumpDemix`, re-derived NOT copied -- GPL) reports a per-cell
    demixing = mean(2*(hom_frac - 1/2)) with the POSITIVE convention (sorted -> +1). Since
    hom_frac = 1 - het_frac, that equals -D_ours, so het_frac = (1 - demix) / 2. Use this to
    put the oracle's count-based metric on the same het-fraction footing as ours.
    """
    return (1.0 - demix) / 2.0


def sorting_score(hf: float, hf0: float) -> float:
    """Normalized demixing score S = 1 - hf/hf0, in [0, 1] for monotone sorting.

    `hf` is a heterotypic-contact fraction (area- or count-based) at some time, `hf0` its
    value at t=0. S=0 at the start, rises toward 1 as heterotypic contact is eliminated.
    Computed IDENTICALLY for our run and the oracle so only the SHAPE/TREND is compared
    (absolute values are not comparable: ours is a finite 189-cell block, the oracle a
    periodic 1728-cell box). Returns 0.0 if hf0 == 0.
    """
    return float(1.0 - hf / hf0) if hf0 else 0.0


def contact_summary(types=None, lam: Optional[Callable[[str, str], float]] = None,
                    bodies: Optional[List] = None) -> Dict:
    """One pass over every unique cell-cell contact among the given cells.

    Pass either `types` (a list of BodyTypes; all their instances are used) OR an
    explicit `bodies` list. The explicit form is required when several independent
    sub-meshes share the same global types (e.g. a matched ON/OFF sorting comparison in
    one universe) -- `type.instances` would mix them, so each block must be summarised
    from its own body list.

    `lam(name_i, name_j)` -> adhesion coefficient for the type pair (for the energy
    proxy); if None, adhesion_energy is reported as 0.0.

    Returns a dict:
        total_pairs, het_pairs, hom_pairs          (topological neighbour counts)
        total_area,  het_area,  hom_area           (geometric, summed contact area)
        het_area_fraction                          (het_area / total_area)
        het_pair_fraction                          (het_pairs / total_pairs)
        adhesion_energy                            (Sum lam_ij * A_ij)
        demixing_index                             (cell-avg, signed: sorted = negative)
    """
    if bodies is None:
        bodies = _all_bodies(types)
    name = {b.id: type_name(b) for b in bodies}
    scope = set(name)                          # restrict to THIS block's cells

    total_pairs = het_pairs = 0
    total_area = het_area = 0.0
    energy = 0.0
    for b in bodies:
        ni = name[b.id]
        for nb in b.connected_bodies:
            if nb.id <= b.id or nb.id not in scope:
                continue                      # count each undirected pair once; stay in block
            nj = name.get(nb.id, type_name(nb))
            area = b.contact_area(nb)
            total_pairs += 1
            total_area += area
            if lam is not None:
                energy += lam(ni, nj) * area
            if ni != nj:
                het_pairs += 1
                het_area += area

    return dict(
        total_pairs=total_pairs,
        het_pairs=het_pairs,
        hom_pairs=total_pairs - het_pairs,
        total_area=total_area,
        het_area=het_area,
        hom_area=total_area - het_area,
        het_area_fraction=(het_area / total_area) if total_area > 0 else 0.0,
        het_pair_fraction=(het_pairs / total_pairs) if total_pairs > 0 else 0.0,
        adhesion_energy=energy,
        demixing_index=demixing_index(bodies=bodies),
    )
