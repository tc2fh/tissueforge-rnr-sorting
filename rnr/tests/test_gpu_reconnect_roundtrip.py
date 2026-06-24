"""Gate B2 of the GPU port (docs/2026-06-24_gpu-3d-vertex-model-exploration.md):
THE make-or-break proof -- the count-CHANGING I<->H surgery on the index-based PaddedMesh
is reversible. This is the GPU-port analogue of test_roundtrip.py (the CPU RNR gate).

Flow:  TF mesh -> extract_csr + id_maps -> PaddedMesh -> translate IConfig to indices
       -> i_to_h_csr (short edge -> triangle, +1 vert / +1 surface)
       -> h_to_i_csr (triangle -> short edge, -1 vert / -1 surface)
       -> to_csr  =>  assert the round-trip restored everything:

  * body-anchored slot-invariant FINGERPRINT == original  (topology restored; vertex and
    surface SLOTS get relabelled by the alloc/free/compact, so only a body-keyed
    invariant can detect restoration -- array equality is meaningless here);
  * the 6 OUTER vertices never moved (byte-exact -- only the central edge/triangle verts
    are ever created/destroyed, Okuda Eqs. 5-7);
  * the recovered edge endpoints land within O(dl_th) of the originals (exactly so when
    the original edge length == dl_th, as in the hand-built minimal config);
  * element counts return to the originals; the padded mesh stays consistent throughout.

Passing this == Gate B's hard part proven (the parallel scheduler + GPU kernel build on it).
"""
import numpy as np
import pytest

from .. import topology as topo
from ..gpu import csr_mesh as cm
from ..gpu import reconnect_csr as rcsr
from ..gpu.device_mesh import PaddedMesh
from . import helpers as H


def _n_live(pm: PaddedMesh):
    return (int(pm.vert_alive[:pm.n_v_used].sum()), int(pm.surf_alive[:pm.n_s_used].sum()))


def _caps_touch(pm: PaddedMesh, cap_a: int, cap_b: int) -> bool:
    """True iff some live surface separates exactly the two cap bodies (their contact)."""
    pair = {cap_a, cap_b}
    for s in range(pm.n_s_used):
        if pm.surf_alive[s] and {int(b) for b in pm.s2b[s] if b >= 0} == pair:
            return True
    return False


def _edge_drift(p10_new, p11_new, p10_0, p11_0) -> float:
    """Min over the two endpoint labellings (the recovered pair may be swapped)."""
    same = max(np.linalg.norm(p10_new - p10_0), np.linalg.norm(p11_new - p11_0))
    swap = max(np.linalg.norm(p10_new - p11_0), np.linalg.norm(p11_new - p10_0))
    return float(min(same, swap))


def _roundtrip(vsolver, bodies, v10, v11, dl_th, exact: bool):
    """Run I->H->H->I on the (v10, v11) edge of `bodies` on a PaddedMesh; assert restore."""
    cfg_tf = topo.i_neighbourhood(v10, v11)
    assert cfg_tf is not None, "edge is not a valid [I] neighbourhood"

    m0 = cm.extract_csr(bodies)
    vid2i, sid2i, bid2i = cm.id_maps(bodies)
    pm = PaddedMesh.from_csr(m0)
    assert pm.check_consistency() == [], "padded mesh inconsistent before surgery"
    fp0 = cm.fingerprint(m0)
    nv0, ns0 = _n_live(pm)

    cfg = rcsr.iconfig_to_indices(cfg_tf, vid2i, sid2i, bid2i)
    # index-space snapshot of the invariants
    v10i, v11i = cfg.v10, cfg.v11
    p10_0, p11_0 = pm.vert_pos[v10i].copy(), pm.vert_pos[v11i].copy()
    outer_idx = sorted({a.outer_top for a in cfg.arms} | {a.outer_bot for a in cfg.arms})
    assert len(outer_idx) == 6, "expected 6 distinct outer vertices"
    outer_pos0 = {i: pm.vert_pos[i].copy() for i in outer_idx}
    cap_top, cap_bot = cfg.cap_top, cfg.cap_bot
    assert not _caps_touch(pm, cap_top, cap_bot), "caps already touch in the [I] state"

    # --- I -> H -------------------------------------------------------------------------
    hcfg = rcsr.i_to_h_csr(pm, cfg, dl_th)
    assert pm.check_consistency() == [], "inconsistent after I->H"
    nvH, nsH = _n_live(pm)
    assert (nvH, nsH) == (nv0 + 1, ns0 + 1), "I->H must net +1 vertex / +1 surface"
    assert _caps_touch(pm, cap_top, cap_bot), "I->H must create the caps' triangular contact"
    assert {int(b) for b in pm.s2b[hcfg.triangle] if b >= 0} == {cap_top, cap_bot}
    assert int(pm.s2v_len[hcfg.triangle]) == 3, "the new contact must be a triangle"

    # --- H -> I -------------------------------------------------------------------------
    nv10, nv11 = rcsr.h_to_i_csr(pm, hcfg, dl_th)
    assert pm.check_consistency() == [], "inconsistent after H->I"
    nv1, ns1 = _n_live(pm)
    assert (nv1, ns1) == (nv0, ns0), "round-trip must restore element counts"
    assert not _caps_touch(pm, cap_top, cap_bot), "H->I must separate the caps again"

    # --- restoration gates --------------------------------------------------------------
    m1 = pm.to_csr()
    assert cm.fingerprint(m1) == fp0, "round-trip did not restore the body-anchored topology"

    drift_outer = max(float(np.linalg.norm(pm.vert_pos[i] - p0))
                      for i, p0 in outer_pos0.items())
    assert drift_outer < 1e-9, f"outer vertices moved (drift {drift_outer})"

    drift_edge = _edge_drift(pm.vert_pos[nv10], pm.vert_pos[nv11], p10_0, p11_0)
    tol = 1e-6 if exact else dl_th
    assert drift_edge < tol, f"recovered edge drift {drift_edge} exceeds tol {tol}"
    return drift_edge


# ======================================================================================
# minimal hand-built config: the strict gate (edge == dl_th -> near-exact restoration)
# ======================================================================================
def test_minimal_roundtrip_csr(vsolver):
    _tf, _tfv, stype, btype = vsolver
    cfg_in = H.build_minimal_i_config(stype, btype, center=(12., 36., 12.), edge=0.5)
    _roundtrip(vsolver, cfg_in["bodies"], cfg_in["v10"], cfg_in["v11"],
               dl_th=0.5, exact=True)


# ======================================================================================
# realistic Kelvin interior edge (variable valence; the case that matters)
# ======================================================================================
def test_kelvin_roundtrip_csr(vsolver):
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(8., 52., 8.))
    sites = topo.find_short_edges(bodies, threshold=1.0)
    assert sites, "no interior [I] short-edge sites in the Kelvin block"
    sites.sort(key=lambda t: (min(t[2].v10_id, t[2].v11_id), max(t[2].v10_id, t[2].v11_id)))
    v10, v11, cfg = sites[0]
    _roundtrip(vsolver, bodies, v10, v11, dl_th=cfg.length, exact=False)


# ======================================================================================
# the fingerprint has TEETH: it must DISTINGUISH the [I] state from the [H] state
# (an always-equal fingerprint would make the round-trip gate vacuous)
# ======================================================================================
def test_fingerprint_distinguishes_i_and_h(vsolver):
    _tf, _tfv, stype, btype = vsolver
    cfg_in = H.build_minimal_i_config(stype, btype, center=(36., 36., 12.), edge=0.5)
    bodies = cfg_in["bodies"]
    cfg_tf = topo.i_neighbourhood(cfg_in["v10"], cfg_in["v11"])
    assert cfg_tf is not None

    m0 = cm.extract_csr(bodies)
    vid2i, sid2i, bid2i = cm.id_maps(bodies)
    pm = PaddedMesh.from_csr(m0)
    fp_I = cm.fingerprint(m0)

    # I -> H only: topology genuinely changed (caps now in contact) -> fingerprint MUST differ
    cfg = rcsr.iconfig_to_indices(cfg_tf, vid2i, sid2i, bid2i)
    rcsr.i_to_h_csr(pm, cfg, dl_th=0.5)
    fp_H = cm.fingerprint(pm.to_csr())
    assert fp_H != fp_I, "fingerprint failed to distinguish the [I] and [H] topologies"
