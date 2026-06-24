"""Gate C glue (docs/2026-06-24_gpu-3d-vertex-model-exploration.md):
the iterated independent-set I->H sweep assembled end-to-end on the GPU
(schedule_warp.reconnect_sweep_warp = detect -> reserve [C2a] -> parallel apply [C2b] ->
re-detect). This is the wiring step over the already-gated bricks, so its gates target the
GLUE -- the host<->device sync, the re-detection on the mutated device mesh, and the loop
control -- not the surgery (B3) or the scheduler (C2) again.

  * ONE GPU round == ONE host RESERVATION round, by body-anchored fingerprint. Round 1
    starts from the same slot layout as the host, so detection + the lowest-id-wins
    reservation are identical (C2a is bit-for-bit) and the parallel apply matches the host
    sequential apply (C2b). The host reference is reconnect_sweep_reserve_host, which uses
    the SAME reservation selection -- NOT reconnect_sweep_i_to_h (greedy maximal set), a
    different per-round selection (see reserve_independent_set_host).
  * a bounded multi-round GPU sweep re-detects on the mutated device mesh and stays
    consistent every round (mirrors the C1 host sweep-mechanics test; a static-mesh sweep
    cascades rather than converging, so we bound the rounds and check consistency, not
    convergence).
  * no short edge below threshold -> zero rounds, mesh unchanged (loop-control no-op).
"""
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


def _padded(m0):
    """Generous fixed capacity so the bump allocator never overflows over a bounded sweep
    (Gate D will replace this with stream-compaction). Identical params host + device so the
    initial live slots line up for the round-1 equivalence check."""
    return PaddedMesh.from_csr(m0, v_headroom=6000, s_headroom=6000,
                               ring_pad=64, vs_pad=64, bs_pad=64)


def test_gpu_sweep_one_round_matches_host(vsolver):
    """THE Gate-C glue gate: one GPU sweep round (detect -> GPU reserve -> GPU parallel
    apply) reproduces one host reservation round, body-anchored fingerprint exact."""
    from ..gpu import schedule_warp as sw
    dev = _cuda_or_skip()
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(8., 8., 40.))
    m0 = cm.extract_csr(bodies)
    fp0 = cm.fingerprint(m0)

    # host: one reservation round (the faithful per-round mirror of the GPU loop)
    pm_host = _padded(m0)
    rep_host = sched.reconnect_sweep_reserve_host(pm_host, threshold=1.0, dl_th=0.3,
                                                  veto=True, max_rounds=1)
    fp_host = cm.fingerprint(pm_host.to_csr())

    # GPU: one round of the device sweep over the SAME starting mesh
    pm_dev = _padded(m0)
    g = pm_dev.to_warp(device=dev)
    rep_gpu = sw.reconnect_sweep_warp(g, threshold=1.0, dl_th=0.3, veto=True, max_rounds=1)
    pm_back = PaddedMesh.from_warp(g)

    assert rep_gpu["rounds"] == 1 and rep_host["rounds"] == 1, (rep_gpu, rep_host)
    assert rep_gpu["round_sizes"] == rep_host["round_sizes"], \
        f"GPU/host selected different-size batches: {rep_gpu} vs {rep_host}"
    assert rep_gpu["total"] > 0, "the sweep round reconnected nothing"
    assert pm_back.check_consistency() == [], "GPU sweep left an inconsistent mesh"
    assert cm.fingerprint(pm_back.to_csr()) == fp_host, \
        "one GPU sweep round != one host reservation round (fingerprint)"
    assert cm.fingerprint(pm_back.to_csr()) != fp0, "the sweep round changed nothing"


def test_gpu_sweep_bounded_multiround_is_consistent(vsolver):
    """The device loop re-detects on the MUTATED device mesh each round and stays consistent.
    NOTE (C1 cascade): a static-mesh sweep does not converge (an I->H seeds new short edges),
    so we bound the rounds and verify loop mechanics + per-round consistency, not convergence.
    Across rounds the device slot order (atomic bump) diverges from any host order, so this
    checks the device path alone -- the round-1 fingerprint equality is the host cross-check."""
    from ..gpu import schedule_warp as sw
    dev = _cuda_or_skip()
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(40., 8., 8.))
    m0 = cm.extract_csr(bodies)
    fp0 = cm.fingerprint(m0)

    pm_dev = _padded(m0)
    g = pm_dev.to_warp(device=dev)
    rep = sw.reconnect_sweep_warp(g, threshold=1.0, dl_th=0.3, veto=True, max_rounds=3)

    assert rep["rounds"] == 3, f"expected the loop to run the bounded rounds: {rep}"
    assert all(s > 0 for s in rep["round_sizes"]), "a round applied an empty batch"
    assert rep["total"] == sum(rep["round_sizes"]) > 0
    pm_back = PaddedMesh.from_warp(g)
    assert pm_back.check_consistency() == [], "mesh inconsistent during the device sweep loop"
    assert cm.fingerprint(pm_back.to_csr()) != fp0, "the sweep changed nothing"


def test_gpu_sweep_no_short_edges_is_noop(vsolver):
    """Threshold below every edge -> detection finds nothing -> zero rounds, mesh untouched
    (the loop-control / termination path)."""
    from ..gpu import schedule_warp as sw
    dev = _cuda_or_skip()
    _tf, _tfv, stype, btype = vsolver
    cfg = H.build_minimal_i_config(stype, btype, center=(24., 24., 24.), edge=0.5)
    m0 = cm.extract_csr(cfg["bodies"])
    # sanity: at threshold 1.0 there IS a short edge; at 0.1 there is not
    assert tcsr.find_short_edges_csr(PaddedMesh.from_csr(m0), threshold=1.0)
    assert not tcsr.find_short_edges_csr(PaddedMesh.from_csr(m0), threshold=0.1)
    fp0 = cm.fingerprint(m0)

    g = _padded(m0).to_warp(device=dev)
    rep = sw.reconnect_sweep_warp(g, threshold=0.1, dl_th=0.05, veto=True, max_rounds=8)

    assert rep["rounds"] == 0 and rep["total"] == 0, f"no-op sweep did work: {rep}"
    assert rep["converged"], "a 0-round sweep should report converged"
    assert cm.fingerprint(PaddedMesh.from_warp(g).to_csr()) == fp0, "no-op sweep mutated the mesh"
