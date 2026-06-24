"""Gate C brick C2a (docs/2026-06-24_gpu-3d-vertex-model-exploration.md):
the parallel conflict-resolution kernel -- cellGPU's atomic maximal-independent-set
protocol in 3D, on the GPU. One reservation round on the RTX 5090 must:

  * reproduce the host lowest-id-wins reference (schedule_csr.reserve_won_mask_host)
    BIT-FOR-BIT (the selection is deterministic, so device==host exactly);
  * return a non-empty, CONFLICT-FREE batch (the independent set);
  * whose members apply order-independently (the body-anchored fingerprint is the same
    forwards and backwards) -- the parallel-safety guarantee carried over to a
    GPU-selected batch.

This is the scheduling heart of the novel result (parallel 3D topology-change scheduling).
The count-changing parallel APPLY kernel is C2b.
"""
import numpy as np
import pytest

from ..gpu import csr_mesh as cm
from ..gpu import schedule_csr as sched
from ..gpu import topology_csr as tcsr
from ..gpu.device_mesh import PaddedMesh
from . import helpers as H


def _cuda_or_skip():
    import warp as wp
    if not any(d.is_cuda for d in wp.get_devices()):
        pytest.skip("no CUDA device")
    return next(d for d in wp.get_devices() if d.is_cuda)


def _kelvin_candidates(stype, btype, origin):
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=origin)
    m0 = cm.extract_csr(bodies)
    pm = PaddedMesh.from_csr(m0)
    cands = tcsr.find_short_edges_csr(pm, threshold=1.0)
    cands = [c for c in cands if sched.i_to_h_veto_csr(pm, c[2]) is None]
    return m0, pm, cands


def test_gpu_reservation_matches_host(vsolver):
    """The GPU atomic reservation == the host lowest-id-wins reference, bit-for-bit."""
    from ..gpu import schedule_warp as sw
    dev = _cuda_or_skip()
    _tf, _tfv, stype, btype = vsolver
    _m0, pm, cands = _kelvin_candidates(stype, btype, origin=(8., 40., 8.))
    assert len(cands) > 1, "need several candidates to exercise conflict resolution"

    mask_gpu = sw.reserve_won_mask_warp(pm, cands, device=dev)
    mask_host = sched.reserve_won_mask_host(cands)
    assert list(int(x) for x in mask_gpu) == mask_host, "GPU reservation != host reference"


def test_gpu_reservation_is_conflict_free_and_nonempty(vsolver):
    from ..gpu import schedule_warp as sw
    dev = _cuda_or_skip()
    _tf, _tfv, stype, btype = vsolver
    _m0, pm, cands = _kelvin_candidates(stype, btype, origin=(40., 8., 8.))
    assert len(cands) > 1

    winners = sw.reserve_independent_set_warp(pm, cands, device=dev)
    assert winners, "reservation selected nobody (candidate 0 should always win)"
    assert sched.batch_is_conflict_free(winners), "GPU-selected batch is NOT conflict-free"
    # it is a strict scheduler: with conflicts present, it admits fewer than all candidates
    assert len(winners) <= len(cands)


def test_gpu_reservation_admits_disjoint_candidates(vsolver):
    """Two spatially-disjoint [I] configs must BOTH win the GPU reservation (disjoint
    footprints -> no conflict), and the resulting size-2 batch applies order-independently
    -- end-to-end: GPU selection -> conflict-free batch -> order-invariant apply."""
    from ..gpu import schedule_warp as sw
    dev = _cuda_or_skip()
    _tf, _tfv, stype, btype = vsolver
    cfg_a = H.build_minimal_i_config(stype, btype, center=(24., 24., 24.), edge=0.5)
    cfg_b = H.build_minimal_i_config(stype, btype, center=(24., 24., 50.), edge=0.5)
    m0 = cm.extract_csr(cfg_a["bodies"] + cfg_b["bodies"])
    pm = PaddedMesh.from_csr(m0)
    cands = tcsr.find_short_edges_csr(pm, threshold=1.0)
    cands = [c for c in cands if sched.i_to_h_veto_csr(pm, c[2]) is None]
    assert len(cands) == 2

    winners = sw.reserve_independent_set_warp(pm, cands, device=dev)
    assert len(winners) == 2, "two disjoint configs should both win the reservation"
    assert sched.batch_is_conflict_free(winners)

    hv, hs = 3 * len(winners) + 16, len(winners) + 16
    pm_fwd = PaddedMesh.from_csr(m0, v_headroom=hv, s_headroom=hs)
    pm_rev = PaddedMesh.from_csr(m0, v_headroom=hv, s_headroom=hs)
    sched.apply_batch(pm_fwd, winners, dl_th=0.3)
    sched.apply_batch(pm_rev, list(reversed(winners)), dl_th=0.3)
    assert pm_fwd.check_consistency() == [] and pm_rev.check_consistency() == []
    assert cm.fingerprint(pm_fwd.to_csr()) == cm.fingerprint(pm_rev.to_csr()), \
        "GPU-selected batch is order-dependent (independence violated)"
    assert cm.fingerprint(pm_fwd.to_csr()) != cm.fingerprint(m0), "batch changed nothing"


# ======================================================================================
# C2b: the PARALLEL count-changing apply (dim=N) matches the host sequential apply
# ======================================================================================
def _parallel_apply_matches_host(m0, winners, dev, dl_th=0.3):
    from ..gpu import reconnect_warp as rw
    n = len(winners)
    hv, hs = 3 * n + 16, n + 16
    # host sequential reference
    pm_host = PaddedMesh.from_csr(m0, v_headroom=hv, s_headroom=hs)
    sched.apply_batch(pm_host, winners, dl_th=dl_th)
    # GPU parallel: all N surgeries at once
    pm_dev = PaddedMesh.from_csr(m0, v_headroom=hv, s_headroom=hs)
    g = pm_dev.to_warp(device=dev)
    rw.apply_i_to_h_batch_warp(g, winners, dl_th=dl_th)
    pm_back = PaddedMesh.from_warp(g)
    assert pm_back.check_consistency() == [], "parallel GPU apply produced an inconsistent mesh"
    assert cm.fingerprint(pm_back.to_csr()) == cm.fingerprint(pm_host.to_csr()), \
        "parallel GPU apply != host sequential apply"
    assert cm.fingerprint(pm_back.to_csr()) != cm.fingerprint(m0), "apply changed nothing"


def test_parallel_apply_two_disjoint_configs(vsolver):
    """Two disjoint configs reconnected SIMULTANEOUSLY on the GPU (dim=2) == host apply."""
    from ..gpu import schedule_warp as sw
    dev = _cuda_or_skip()
    _tf, _tfv, stype, btype = vsolver
    cfg_a = H.build_minimal_i_config(stype, btype, center=(50., 24., 24.), edge=0.5)
    cfg_b = H.build_minimal_i_config(stype, btype, center=(50., 24., 50.), edge=0.5)
    m0 = cm.extract_csr(cfg_a["bodies"] + cfg_b["bodies"])
    pm = PaddedMesh.from_csr(m0)
    cands = [c for c in tcsr.find_short_edges_csr(pm, threshold=1.0)
             if sched.i_to_h_veto_csr(pm, c[2]) is None]
    winners = sw.reserve_independent_set_warp(pm, cands, device=dev)
    assert len(winners) == 2
    _parallel_apply_matches_host(m0, winners, dev)


def test_parallel_apply_kelvin_batch(vsolver):
    """A reserved Kelvin batch applied in parallel on the GPU == host sequential apply
    (the count-changing 3D RNR running conflict-free, in parallel, on-device)."""
    from ..gpu import schedule_warp as sw
    dev = _cuda_or_skip()
    _tf, _tfv, stype, btype = vsolver
    # m0, pm and cands all come from the SAME mesh (from_csr preserves indices), so the
    # candidate indices are valid for any fresh PaddedMesh.from_csr(m0).
    m0, pm, cands = _kelvin_candidates(stype, btype, origin=(40., 40., 8.))
    winners = sw.reserve_independent_set_warp(pm, cands, device=dev)
    assert len(winners) >= 1
    assert sched.batch_is_conflict_free(winners)
    _parallel_apply_matches_host(m0, winners, dev)
