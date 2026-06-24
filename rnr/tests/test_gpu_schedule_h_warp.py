"""Gate C reverse direction -- brick C2' (docs/2026-06-24_gpu-3d-vertex-model-exploration.md):
the parallel conflict-resolution + count-changing reverse apply for H->I, on the GPU. The
reverse mirror of test_gpu_schedule_warp.py (C2a/C2b/C2c). On the RTX 5090:

  * the GPU H-reservation reproduces the host lowest-id-wins reference
    (schedule_csr.h_reserve_won_mask_host) BIT-FOR-BIT, on a conflicting candidate set;
  * a conflict-free reverse batch applied IN PARALLEL (h_to_i_batch_kernel, dim=N) equals the
    host sequential apply (body-anchored fingerprint) -- and both restore the [I] topology;
  * a full GPU round-trip -- N parallel I->H then N parallel H->I -- restores the fingerprint;
  * the glued GPU reverse sweep's first round == the host reservation round (fingerprint).

[H] sites only exist AFTER a forward I->H, so each test seeds them with a forward batch first.
"""
import pytest

from ..gpu import csr_mesh as cm
from ..gpu import reconnect_csr as rcsr
from ..gpu import schedule_csr as sched
from ..gpu import topology_csr as tcsr
from ..gpu.device_mesh import PaddedMesh
from . import helpers as H


def _cuda_or_skip():
    import warp as wp
    if not any(d.is_cuda for d in wp.get_devices()):
        pytest.skip("no CUDA device")
    return next(d for d in wp.get_devices() if d.is_cuda)


def _seed_triangles(m0, dl_th=0.3, **pm_kwargs):
    """Apply a conflict-free forward I->H batch to a fresh PaddedMesh(m0); return
    (pm, batch, capcap) -- the post-forward mesh, the forward [I] candidates, and the cap-cap
    HCfgIdx (the canonical mutually-disjoint reverse sites). Deterministic; default headroom
    covers fwd(+3) + a full rev(+2) = 5 verts/op (local mirror of the C1' test's helper)."""
    sites = tcsr.find_short_edges_csr(PaddedMesh.from_csr(m0), threshold=1.0)
    sites = [s for s in sites if sched.i_to_h_veto_csr(PaddedMesh.from_csr(m0), s[2]) is None]
    batch = sched.independent_set(sites)
    pm_kwargs.setdefault("v_headroom", 5 * len(batch) + 16)
    pm_kwargs.setdefault("s_headroom", len(batch) + 16)
    pm = PaddedMesh.from_csr(m0, **pm_kwargs)
    capcap = [rcsr.i_to_h_csr(pm, cfg, dl_th) for (_v10, _v11, cfg) in batch]
    return pm, batch, capcap


def test_gpu_h_reservation_matches_host(vsolver):
    """The GPU reverse reservation == the host lowest-id-wins reference, bit-for-bit, on the
    post-forward triangle set (cap-cap + cascade side-collapse triangles, which conflict)."""
    from ..gpu import schedule_warp as sw
    dev = _cuda_or_skip()
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(8., 40., 40.))
    pm, _batch, _capcap = _seed_triangles(cm.extract_csr(bodies))
    sites = tcsr.find_small_triangles_csr(pm, threshold=1.0)
    sites = [s for s in sites if sched.h_to_i_veto_csr(pm, s[1]) is None]
    assert len(sites) > 1, "need several [H] candidates to exercise conflict resolution"

    mask_gpu = sw.reserve_h_won_mask_warp(pm, sites, device=dev)
    mask_host = sched.h_reserve_won_mask_host(sites)
    assert list(int(x) for x in mask_gpu) == mask_host, "GPU H-reservation != host reference"


def test_gpu_h_reservation_is_conflict_free_and_nonempty(vsolver):
    from ..gpu import schedule_warp as sw
    dev = _cuda_or_skip()
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(40., 40., 8.))
    pm, _batch, _capcap = _seed_triangles(cm.extract_csr(bodies))
    sites = [s for s in tcsr.find_small_triangles_csr(pm, threshold=1.0)
             if sched.h_to_i_veto_csr(pm, s[1]) is None]
    assert len(sites) > 1

    winners = sw.reserve_h_independent_set_warp(pm, sites, device=dev)
    assert winners, "reverse reservation selected nobody (candidate 0 should always win)"
    assert sched.h_batch_is_conflict_free(winners), "GPU-selected reverse batch is NOT conflict-free"
    assert len(winners) <= len(sites)


def test_parallel_h_apply_matches_host(vsolver):
    """A conflict-free reverse (cap-cap) batch reconnected SIMULTANEOUSLY on the GPU (dim=N)
    == the host sequential apply, and both restore the original [I] fingerprint."""
    from ..gpu import reconnect_warp as rw
    dev = _cuda_or_skip()
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(40., 8., 40.))
    m0 = cm.extract_csr(bodies)
    fp0 = cm.fingerprint(m0)
    pm_post, batch, capcap = _seed_triangles(m0)
    rev_batch = [(h.triangle, h) for h in capcap]
    assert sched.h_batch_is_conflict_free(rev_batch) and len(rev_batch) >= 2

    # snapshot the post-forward state to the device BEFORE the host reverse mutates pm_post.
    g = pm_post.to_warp(device=dev)
    rw.apply_h_to_i_batch_warp(g, rev_batch, dl_th=0.3)        # GPU parallel reverse
    pm_gpu = PaddedMesh.from_warp(g)
    sched.h_apply_batch(pm_post, rev_batch, dl_th=0.3)         # host sequential reverse

    assert pm_gpu.check_consistency() == [], "parallel GPU H->I produced an inconsistent mesh"
    fp_gpu, fp_host = cm.fingerprint(pm_gpu.to_csr()), cm.fingerprint(pm_post.to_csr())
    assert fp_gpu == fp_host, "parallel GPU H->I apply != host sequential apply"
    assert fp_gpu == fp0, "forward batch then reverse batch did not restore [I]"


def test_gpu_forward_then_reverse_restores_fingerprint(vsolver):
    """The capstone round-trip, fully on the GPU: N parallel I->H then N parallel H->I restore
    the body-anchored fingerprint -- the count-changing 3D RNR and its inverse, both running
    conflict-free in parallel on-device."""
    from ..gpu import reconnect_warp as rw
    dev = _cuda_or_skip()
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(40., 40., 40.))
    m0 = cm.extract_csr(bodies)
    fp0 = cm.fingerprint(m0)

    pm0 = PaddedMesh.from_csr(m0)
    fwd = [s for s in tcsr.find_short_edges_csr(pm0, threshold=1.0)
           if sched.i_to_h_veto_csr(pm0, s[2]) is None]
    fwd_batch = sched.independent_set(fwd)
    n = len(fwd_batch)
    assert n >= 2, f"need a multi-reconnection forward batch, got {n}"

    g = PaddedMesh.from_csr(m0, v_headroom=5 * n + 16, s_headroom=n + 16).to_warp(device=dev)
    hcfgs = rw.apply_i_to_h_batch_warp(g, fwd_batch, dl_th=0.3)    # N forward I->H in parallel
    rev_batch = [(h.triangle, h) for h in hcfgs]
    rw.apply_h_to_i_batch_warp(g, rev_batch, dl_th=0.3)            # N reverse H->I in parallel
    pm_back = PaddedMesh.from_warp(g)

    assert pm_back.check_consistency() == [], "GPU forward+reverse produced an inconsistent mesh"
    assert cm.fingerprint(pm_back.to_csr()) == fp0, "GPU I->H then H->I did not restore [I]"


def test_gpu_reverse_sweep_matches_host_round1(vsolver):
    """The glued GPU reverse sweep (reconnect_sweep_h_to_i_warp): its FIRST round equals one
    host reservation round (reconnect_sweep_h_reserve_host) -- same batch size and the same
    body-anchored fingerprint (round 1 shares the host's slot layout, so it is exact)."""
    from ..gpu import schedule_warp as sw
    dev = _cuda_or_skip()
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(8., 8., 8.))
    m0 = cm.extract_csr(bodies)
    pm_post, _batch, _capcap = _seed_triangles(m0, v_headroom=4000, s_headroom=4000,
                                               ring_pad=64, vs_pad=64, bs_pad=64)
    g = pm_post.to_warp(device=dev)
    rep_gpu = sw.reconnect_sweep_h_to_i_warp(g, threshold=1.0, dl_th=0.3, max_rounds=1)
    pm_gpu = PaddedMesh.from_warp(g)
    rep_host = sched.reconnect_sweep_h_reserve_host(pm_post, threshold=1.0, dl_th=0.3, max_rounds=1)

    assert rep_gpu["round_sizes"] == rep_host["round_sizes"], \
        f"GPU reverse sweep round != host reservation round: {rep_gpu} vs {rep_host}"
    assert pm_gpu.check_consistency() == []
    assert cm.fingerprint(pm_gpu.to_csr()) == cm.fingerprint(pm_post.to_csr()), \
        "GPU reverse sweep round 1 != host reservation round 1 (fingerprint)"
