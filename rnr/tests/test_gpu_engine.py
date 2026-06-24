"""Gate E, end-to-end: the composed GPU forward step (director -> geometry -> force ->
integrate -> RNR -> compact) runs a stable, valid, bounded trajectory, and the heterotypic
interfacial tension actually SORTS (demixing order parameter falls).

  * test_forward_step_is_stable_and_bounded: ~60 steps of the full step on a periodic
    two-type foam (scaled to unit cells so production params apply). The mesh stays
    consistent, finite, with positive volumes, and the bump-allocated slots stay bounded by
    compaction.
  * test_sorting_demixes: with heterotypic tension on, the het-contact fraction trends DOWN
    over a longer run -- the GPU engine reproduces 3DVertVor-style cell sorting.

Run: pixi run python -m pytest rnr/tests/test_gpu_engine.py -q
"""
import numpy as np
import pytest
import warp as wp

from tissue_forge.models.vertex import solver as tfv

from ..gpu.csr_mesh import extract_csr
from ..gpu.device_mesh import PaddedMesh
from ..gpu import physics_csr as P
from ..gpu import physics_warp as W
from ..gpu import engine as E
from .test_gpu_physics_csr import _build_two_type_foam, _periodic


def _cuda_or_skip():
    cuda = [d for d in wp.get_devices() if d.is_cuda]
    if not cuda:
        pytest.skip("no CUDA device for the GPU engine gate")
    return cuda[0]


def _setup_unit_foam(dev, n=3, headroom=3000, ic="demixed"):
    """Build the TF two-type periodic foam, extract it, and SCALE to unit cells so the
    production force params (V0~1, A0~5.6) apply directly. Returns g, phys, body_type, box,
    v0, a0 (the measured post-scale means)."""
    mesh = tfv.MeshSolver.get().get_mesh()
    with _periodic(mesh, True):
        bodies, box, btA, btB, b_is_B, adh = _build_two_type_foam(n=n, ic=ic)
        csr = extract_csr(bodies)
        state = P.phys_state_from_tf(bodies, lambda b: 1 if b.id in b_is_B else 0)
    pm = PaddedMesh.from_csr(csr, v_headroom=headroom, s_headroom=headroom,
                             ring_pad=64, vs_pad=64, bs_pad=64)
    # scale positions + box so the mean cell volume is 1 (production regime)
    mean_vol = float(np.mean([b.volume for b in bodies]))
    s = mean_vol ** (1.0 / 3.0)
    pm.vert_pos[:pm.n_v_used] /= s
    box = np.asarray(box, float) / s
    geom = P.compute_geometry(pm, box)
    v0 = float(geom.bvol[:pm.nb].mean())
    a0 = float(geom.barea[:pm.nb].mean())
    g = W.attach_box(pm.to_warp(device=dev), box)
    phys = W.upload_phys(state, dev)
    return g, phys, state.body_type, box, v0, a0


def test_forward_step_is_stable_and_bounded(vsolver):
    """~60 composed steps with the four forces + active drive + RNR + compaction: the mesh
    stays consistent, finite, positive-volume, and slot-bounded."""
    dev = _cuda_or_skip()
    g, phys, body_type, box, v0, a0 = _setup_unit_foam(dev)
    params = P.PhysParams(box=box, kv=10.0, v0=v0, ka=1.0, a0=a0, sigma=0.4, v_active=0.1)
    dt, dr = 5e-3, 1.0
    cap_v = g["cap_v"]
    recon_i = recon_h = 0
    nv_max = 0
    for step in range(60):
        rep = E.forward_step(g, phys, params, dt, dr, seed=7, step=step,
                             threshold=0.02, dl_th=0.02, reconnect=True, interval=2,
                             compact=True, max_rounds=8)
        recon_i += rep["i"]
        recon_h += rep["h"]
        nv_max = max(nv_max, rep["nv"])

    pm = PaddedMesh.from_warp(g)
    problems = pm.check_consistency()
    geom = P.compute_geometry(pm, box)
    pos = g["vert_pos"].numpy()[:pm.n_v_used]
    print(f"\n[E3 stability] reconnections I->H={recon_i} H->I={recon_h} | nv_max={nv_max}/"
          f"{cap_v} | live nv={pm.n_v_used} ns={pm.n_s_used} | vol[min,max]="
          f"[{geom.bvol[:pm.nb].min():.3f},{geom.bvol[:pm.nb].max():.3f}]")
    assert not problems, f"mesh inconsistent after the run: {problems[:5]}"
    assert np.all(np.isfinite(pos)), "non-finite vertex positions"
    assert geom.bvol[:pm.nb].min() > 0.0, "a cell inverted (non-positive volume)"
    assert nv_max < cap_v, "vertex capacity exhausted (compaction not bounding slots)"


def _host_step(pm, params, state, dt, box):
    """One host-reference forward step (deterministic part: geometry -> force -> integrate).
    Mutates pm.vert_pos. No director evolution (Dr=0) and no reconnection -- the deterministic
    path where the GPU engine must reproduce the TF-validated host bit-for-bit (to fp64)."""
    geom = P.compute_geometry(pm, box)
    f = P.forces(pm, geom, params, state)
    P.integrate(pm, f, dt, box)


def test_gpu_matches_host_trajectory(vsolver):
    """THE 'matches the CPU oracle' gate: with directors fixed (Dr=0) and reconnection off, the
    composed GPU step is fully deterministic and must reproduce the host reference (itself
    TF-validated) over a multi-step trajectory, to fp64. Once forces/integration are confirmed
    identical step-after-step, the only remaining source of GPU/CPU divergence in a real sort is
    the (intentionally statistical) director RNG + atomic reconnection ordering."""
    dev = _cuda_or_skip()
    g, phys, body_type, box, v0, a0 = _setup_unit_foam(dev)
    params = P.PhysParams(box=box, kv=10.0, v0=v0, ka=1.0, a0=a0, sigma=0.5, v_active=0.1)
    dt = 5e-3
    # host mesh mirrors the device mesh's starting positions exactly
    pm_h = PaddedMesh.from_warp(g)
    state = P.PhysState(body_type=phys["body_type"].numpy().copy(),
                        body_director=phys["body_director"].numpy().reshape(-1, 3).copy())

    worst = 0.0
    nsteps = 12   # host step is pure-numpy (slow); 12 steps is ample -- a bug would compound
    for step in range(nsteps):
        # GPU: Dr=0 -> directors frozen; reconnect off -> deterministic
        E.forward_step(g, phys, params, dt, dr=0.0, seed=0, step=step, reconnect=False)
        _host_step(pm_h, params, state, dt, box)
        gpu_pos = g["vert_pos"].numpy()[:pm_h.n_v_used]
        d = P.minimg(gpu_pos - pm_h.vert_pos[:pm_h.n_v_used], box)
        worst = max(worst, float(np.max(np.linalg.norm(d, axis=1))))
    print(f"\n[E4 trajectory] GPU vs host worst |Δx| over {nsteps} steps = {worst:.2e}")
    assert worst < 1e-8, f"GPU trajectory diverged from host: {worst}"


def test_sorting_demixes(vsolver):
    """The demixing order parameter (het-contact fraction) FALLS from a MIXED initial
    condition under heterotypic tension -- the GPU engine reproduces 3DVertVor-style cell
    sorting. Statistical: the trend over a long run, not an exact value."""
    dev = _cuda_or_skip()
    g, phys, body_type, box, v0, a0 = _setup_unit_foam(dev, n=4, ic="mixed")
    # strong heterotypic tension + active drive -> clear demixing within a test-feasible run
    params = P.PhysParams(box=box, kv=10.0, v0=v0, ka=1.0, a0=a0, sigma=3.0, v_active=0.2)
    dt, dr = 5e-3, 1.0

    het0 = E.het_fraction_device(g, body_type)
    curve = [het0]
    for step in range(600):
        E.forward_step(g, phys, params, dt, dr, seed=11, step=step,
                       threshold=0.02, dl_th=0.02, reconnect=True, interval=2,
                       compact=True, max_rounds=8)
        if step % 100 == 99:
            curve.append(E.het_fraction_device(g, body_type))
    het_end = curve[-1]
    print(f"\n[E4 sorting] het-contact: {het0:.3f} -> {het_end:.3f}  curve={['%.3f' % c for c in curve]}")
    # demixing signal: a clear, monotone-ish drop in the het-contact order parameter
    assert het_end < het0 - 0.03, f"no demixing from a mixed IC: het {het0:.3f} -> {het_end:.3f}"
    assert curve[-1] < curve[1], "het-contact did not trend down (no sorting)"
