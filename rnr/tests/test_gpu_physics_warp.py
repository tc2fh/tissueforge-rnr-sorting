"""Gate E, GPU kernels: `rnr/gpu/physics_warp.py` reproduces the host reference
`physics_csr.py` (geometry + the four forces) on the RTX 5090, to fp64 round-off.

The host reference is validated against TF in test_gpu_physics_csr.py; this gate validates
the kernel PORT against the host. Both run the identical fp64 formula, so agreement is ~1e-10
(the only difference is wp.round vs np.round at exact half-images + FP op ordering) -- far
below the float32 floor of the TF comparison, so a kernel bug is unambiguous.

Run: pixi run python -m pytest rnr/tests/test_gpu_physics_warp.py -q
"""
import copy

import numpy as np
import pytest
import warp as wp

from tissue_forge.models.vertex import solver as tfv

from ..gpu.csr_mesh import extract_csr
from ..gpu.device_mesh import PaddedMesh
from ..gpu import physics_csr as P
from ..gpu import physics_warp as W
# reuse the TF-validated periodic two-type foam builder (DRY with the host-reference gate)
from .test_gpu_physics_csr import _build_two_type_foam, _periodic


def _cuda_or_skip():
    cuda = [d for d in wp.get_devices() if d.is_cuda]
    if not cuda:
        pytest.skip("no CUDA device for the GPU physics gate")
    return cuda[0]


def _relerr_rows(gpu, host, floor=1e-9):
    """max over rows of ||gpu-host|| / max(||host||, floor)."""
    gpu = np.atleast_2d(gpu)
    host = np.atleast_2d(host)
    num = np.linalg.norm(gpu - host, axis=-1) if gpu.ndim > 1 else np.abs(gpu - host)
    den = np.maximum(np.linalg.norm(host, axis=-1) if host.ndim > 1 else np.abs(host), floor)
    return float(np.max(num / den))


def _setup(dev, n=3):
    mesh = tfv.MeshSolver.get().get_mesh()
    with _periodic(mesh, True):
        bodies, box, btA, btB, b_is_B, adh = _build_two_type_foam(n=n)
        csr = extract_csr(bodies)
        pm = PaddedMesh.from_csr(csr)
        v0 = float(np.mean([b.volume for b in bodies]))
        a0 = float(np.mean([b.area for b in bodies]))
        state = P.phys_state_from_tf(bodies, lambda b: 1 if b.id in b_is_B else 0)
    g = W.attach_box(pm.to_warp(device=dev), box)
    phys = W.upload_phys(state, dev)
    return pm, box, v0, a0, state, g, phys


def test_geometry_gpu_matches_host(vsolver):
    """Surface (centroid/area/normal) + body (volume/area/centroid/orientSign) geometry: GPU == host."""
    dev = _cuda_or_skip()
    pm, box, v0, a0, state, g, phys = _setup(dev)
    gh = P.compute_geometry(pm, box)
    gw = W.compute_geometry_warp(g)

    ns, nb = pm.n_s_used, pm.nb
    se = _relerr_rows(gw["scent"].numpy()[:ns], gh.scent[:ns])
    ae = _relerr_rows(gw["sarea"].numpy()[:ns], gh.sarea[:ns])
    ne = _relerr_rows(gw["snorm"].numpy()[:ns], gh.snorm[:ns])
    ve = _relerr_rows(gw["bvol"].numpy()[:nb], gh.bvol[:nb])
    bae = _relerr_rows(gw["barea"].numpy()[:nb], gh.barea[:nb])
    bce = _relerr_rows(gw["bcent"].numpy()[:nb], gh.bcent[:nb])
    oe = float(np.max(np.abs(gw["borient"].numpy()[:nb] - gh.borient[:nb])))
    print(f"\n[E gpu-geom] s.cent={se:.2e} s.area={ae:.2e} s.norm={ne:.2e} "
          f"vol={ve:.2e} area={bae:.2e} cent={bce:.2e} orient={oe:.2e}")
    for name, e in [("s.cent", se), ("s.area", ae), ("s.norm", ne),
                    ("vol", ve), ("b.area", bae), ("b.cent", bce)]:
        assert e < 1e-9, f"GPU geometry '{name}' != host: {e}"
    assert oe == 0.0, "orientSign differs GPU vs host"


def _force_case(dev, kv, ka, sigma, v_active, label):
    pm, box, v0, a0, state, g, phys = _setup(dev)
    params = P.PhysParams(box=box, kv=kv, v0=v0, ka=ka, a0=a0, sigma=sigma, v_active=v_active)
    host_f = P.forces(pm, P.compute_geometry(pm, box), params, state)[:pm.n_v_used]
    gw = W.compute_geometry_warp(g)
    gpu_f = W.compute_forces_warp(g, gw, params, phys).numpy()[:pm.n_v_used]
    e = _relerr_rows(gpu_f, host_f)
    print(f"\n[E gpu-force {label}] rel err GPU vs host = {e:.2e}")
    return e


def test_force_volume_gpu_matches_host(vsolver):
    assert _force_case(_cuda_or_skip(), 10.0, 0.0, 0.0, 0.0, "volume") < 1e-9


def test_force_area_gpu_matches_host(vsolver):
    assert _force_case(_cuda_or_skip(), 0.0, 1.0, 0.0, 0.0, "area") < 1e-9


def test_force_adhesion_gpu_matches_host(vsolver):
    assert _force_case(_cuda_or_skip(), 0.0, 0.0, 0.5, 0.0, "adhesion") < 1e-9


def test_force_active_gpu_matches_host(vsolver):
    assert _force_case(_cuda_or_skip(), 0.0, 0.0, 0.0, 0.1, "active") < 1e-9


def test_force_combined_gpu_matches_host(vsolver):
    """All four forces together -- the full per-vertex force the integrator will use."""
    assert _force_case(_cuda_or_skip(), 10.0, 1.0, 0.5, 0.1, "combined") < 1e-9


# ======================================================================================
# E2: director rotational diffusion + overdamped integrator
# ======================================================================================
def test_integrator_gpu_matches_host(vsolver):
    """x += dt*f with periodic wrap: GPU == host given the same force. A large force is used
    so displacements cross box walls and the wrap path is exercised."""
    dev = _cuda_or_skip()
    pm, box, v0, a0, state, g, phys = _setup(dev)
    rng = np.random.default_rng(0)
    force = rng.normal(scale=float(box[0]), size=(pm.cap_v, 3))   # big -> wraps
    dt = 0.1
    pm_h = copy.deepcopy(pm)
    P.integrate(pm_h, force, dt, box)
    fw = wp.array(np.ascontiguousarray(force), dtype=wp.vec3d, device=dev)
    W.integrate_warp(g, fw, dt)
    gpu_pos = g["vert_pos"].numpy()
    err = float(np.max(np.abs(gpu_pos[:pm.n_v_used] - pm_h.vert_pos[:pm.n_v_used])))
    live = pm.vert_alive[:pm.n_v_used] == 1
    lp = gpu_pos[:pm.n_v_used][live]
    print(f"\n[E2 integrate] GPU vs host max|Δx|={err:.2e}  pos range [{lp.min():.3f},{lp.max():.3f}]")
    assert err < 1e-9, f"integrator GPU != host: {err}"
    assert lp.min() >= 0.0 and lp.max() < float(box[0]), "positions not wrapped into [0,L)"


def test_active_displacement_calibration(vsolver):
    """The native-drive calibration (probe_native_calibration Part A) on the GPU: the full
    chain director -> active force -> integrate yields per-vertex displacement EXACTLY
    dt*v0*<incident directors> (overdamped mu=1)."""
    dev = _cuda_or_skip()
    pm, box, v0, a0, state, g, phys = _setup(dev)
    V0_ACT, dt = 0.1, 1e-3
    gw = W.compute_geometry_warp(g)
    params = P.PhysParams(box=box, kv=0.0, v0=v0, ka=0.0, a0=a0, sigma=0.0, v_active=V0_ACT)
    fw = W.compute_forces_warp(g, gw, params, phys)
    pos0 = g["vert_pos"].numpy().copy()
    W.integrate_warp(g, fw, dt)
    disp = g["vert_pos"].numpy() - pos0
    bdir = phys["body_director"].numpy()
    worst = 0.0
    for v in range(pm.n_v_used):
        if not pm.vert_alive[v]:
            continue
        bs = P.incident_bodies(pm, v)
        pred = dt * V0_ACT * bdir[bs].mean(axis=0) if bs else np.zeros(3)
        worst = max(worst, np.linalg.norm(disp[v] - pred))
    print(f"\n[E2 active-disp] worst |Δx - dt*v0*<n>| = {worst:.2e}")
    assert worst < 1e-9, f"active displacement != dt*v0*<n>: {worst}"


def test_director_rotational_diffusion_rate(vsolver):
    """The director update's decorrelation rate scales as (2/3)*Dr (probe_native_calibration
    Part B): measure lambda = <n(t+1).n(t)> over many bodies*steps, rate = -ln(lambda)/dt,
    and confirm Dr_eff = 1.5*rate ~ Dr. Statistical (TF uses mt19937, we use Warp's RNG)."""
    dev = _cuda_or_skip()
    pm, box, v0, a0, state, g, phys = _setup(dev)
    Dr, dt, T = 1.0, 1e-3, 1500
    nb = g["nb"]
    prev = phys["body_director"].numpy().copy()
    tot, cnt = 0.0, 0
    for step in range(T):
        W.director_update_warp(g, phys, Dr, dt, seed=7, step=step)
        cur = phys["body_director"].numpy()
        tot += float(np.sum(prev * cur))
        cnt += nb
        prev = cur.copy()
    lam = tot / cnt
    rate = -np.log(lam) / dt
    dr_eff = 1.5 * rate
    # directors must stay unit length (normalize each step)
    norms = np.linalg.norm(phys["body_director"].numpy(), axis=1)
    print(f"\n[E2 director] lambda={lam:.5f} rate={rate:.4f} Dr_eff=1.5*rate={dr_eff:.3f} (set Dr={Dr})")
    assert np.allclose(norms, 1.0, atol=1e-6), "directors drifted off the unit sphere"
    assert 0.75 < dr_eff < 1.3, f"director rotational-diffusion rate off: Dr_eff={dr_eff} (want ~{Dr})"
