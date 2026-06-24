"""Gate B3 of the GPU port (docs/2026-06-24_gpu-3d-vertex-model-exploration.md):
the count-changing I<->H surgery runs as a Warp kernel on the GPU and matches the host
reference (reconnect_csr.py / Gate B2). This is the on-device proof of the make-or-break
primitive -- the parallel slot allocator + Okuda placement + ragged-ring surgery as
device code -- before the Gate-C scheduler runs many of them at once.

The device and host start from the IDENTICAL initial PaddedMesh and run the SAME single
I->H then H->I. Because the integer surgery is deterministic and precision-independent,
the device connectivity must match the host BIT-FOR-BIT; fp64 placement must match the
numpy oracle to round-off. The round-trip must restore the original body-anchored
fingerprint on the device path too.

Also settles risk #2 (fp32 vs reversibility): probe_placement_precision measures fp32 vs
fp64 placement on-device against the numpy fp64 oracle -- fp64 reproduces it to ~1e-12
(bit-reversible), fp32 drifts ~1e-6 (fine for the gate tolerance, not bit-reversible),
so fp64 is the chosen precision for the RNR path.
"""
import numpy as np
import pytest

from .. import topology as topo
from ..gpu import csr_mesh as cm
from ..gpu import reconnect_csr as rcsr
from ..gpu.device_mesh import PaddedMesh
from . import helpers as H

_INT_FIELDS = ["vert_alive", "v2s", "v2s_len", "surf_alive",
               "s2v", "s2v_len", "s2b", "b2s", "b2s_len"]


def _cuda_or_skip():
    import warp as wp
    if not any(d.is_cuda for d in wp.get_devices()):
        pytest.skip("no CUDA device")
    return next(d for d in wp.get_devices() if d.is_cuda)


def _cfg(bodies, v10h, v11h):
    cfg_tf = topo.i_neighbourhood(v10h, v11h)
    assert cfg_tf is not None
    m0 = cm.extract_csr(bodies)
    vid2i, sid2i, bid2i = cm.id_maps(bodies)
    return m0, rcsr.iconfig_to_indices(cfg_tf, vid2i, sid2i, bid2i)


def _device_matches_host(bodies, v10h, v11h, dl_th):
    from ..gpu import reconnect_warp as rw
    dev = _cuda_or_skip()
    m0, cfg = _cfg(bodies, v10h, v11h)
    fp0 = cm.fingerprint(m0)

    pm = PaddedMesh.from_csr(m0)          # host reference
    g = pm.to_warp(device=dev)            # device copy of the IDENTICAL initial state

    # --- host round-trip ---------------------------------------------------------------
    hcfg_h = rcsr.i_to_h_csr(pm, cfg, dl_th)
    nv_h = rcsr.h_to_i_csr(pm, hcfg_h, dl_th)

    # --- device round-trip (same single ops) -------------------------------------------
    hcfg_d = rw.i_to_h_warp(g, cfg, dl_th)
    nv_d = rw.h_to_i_warp(g, hcfg_d, dl_th)

    # bump allocator awarded the SAME slots on both paths
    assert hcfg_d.tri_verts == hcfg_h.tri_verts, "device/host allocated different tri slots"
    assert hcfg_d.triangle == hcfg_h.triangle
    assert tuple(nv_d) == tuple(nv_h), "device/host allocated different edge-vertex slots"

    pm_d = PaddedMesh.from_warp(g)
    assert pm_d.check_consistency() == [], "device result inconsistent"
    assert (pm_d.n_v_used, pm_d.n_s_used) == (pm.n_v_used, pm.n_s_used)

    # integer connectivity: device == host, bit-for-bit (precision-independent surgery)
    for name in _INT_FIELDS:
        assert np.array_equal(getattr(pm, name), getattr(pm_d, name)), \
            f"{name}: device kernel != host reference"

    # positions of the LIVE vertices: fp64 device placement matches numpy fp64 to round-off
    live = pm.vert_alive.astype(bool)
    dpos = float(np.abs(pm.vert_pos[live] - pm_d.vert_pos[live]).max())
    assert dpos < 1e-9, f"live vertex position drift device-vs-host {dpos}"

    # the device round-trip restored the original body-anchored topology
    assert cm.fingerprint(pm_d.to_csr()) == fp0, "device round-trip did not restore topology"


# ======================================================================================
# device == host: minimal hand-built config + realistic Kelvin interior edge
# ======================================================================================
def test_warp_roundtrip_matches_host_minimal(vsolver):
    _tf, _tfv, stype, btype = vsolver
    cfg_in = H.build_minimal_i_config(stype, btype, center=(24., 12., 36.), edge=0.5)
    _device_matches_host(cfg_in["bodies"], cfg_in["v10"], cfg_in["v11"], dl_th=0.5)


def test_warp_roundtrip_matches_host_kelvin(vsolver):
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(40., 8., 40.))
    sites = topo.find_short_edges(bodies, threshold=1.0)
    assert sites, "no interior [I] short-edge sites in the Kelvin block"
    sites.sort(key=lambda t: (min(t[2].v10_id, t[2].v11_id), max(t[2].v10_id, t[2].v11_id)))
    v10, v11, cfg = sites[0]
    _device_matches_host(bodies, v10, v11, dl_th=cfg.length)


# ======================================================================================
# precision probe: fp32 vs fp64 placement on-device, vs the numpy fp64 oracle (risk #2)
# ======================================================================================
def test_warp_placement_fp64_matches_oracle_fp32_coarser(vsolver, capsys):
    from ..gpu import reconnect_warp as rw
    dev = _cuda_or_skip()
    cfg_in = H.build_minimal_i_config(stype=vsolver[2], btype=vsolver[3],
                                      center=(8., 24., 40.), edge=0.5)
    m0, cfg = _cfg(cfg_in["bodies"], cfg_in["v10"], cfg_in["v11"])
    p10, p11 = m0.vert_pos[cfg.v10], m0.vert_pos[cfg.v11]
    otops = [m0.vert_pos[a.outer_top] for a in cfg.arms]
    obots = [m0.vert_pos[a.outer_bot] for a in cfg.arms]

    res = rw.probe_placement_precision(p10, p11, otops, obots, dl_th=0.5, device=dev)
    d_f64 = float(np.abs(res["gpu_f64"] - res["oracle"]).max())
    d_f32 = float(np.abs(res["gpu_f32"] - res["oracle"]).max())
    with capsys.disabled():
        print(f"\n[placement precision] max |gpu - numpy-fp64-oracle|: "
              f"fp64={d_f64:.2e}  fp32={d_f32:.2e}")

    # fp64 on-device reproduces the CPU oracle's Okuda placement to round-off => reversible
    assert d_f64 < 1e-9, f"GPU fp64 placement diverges from oracle ({d_f64})"
    # fp32 is far coarser (the reversibility risk) but still within the gate's dl_th budget
    assert d_f32 > d_f64, "fp32 unexpectedly as accurate as fp64"
    assert d_f32 < 0.5, f"fp32 placement drift {d_f32} exceeds dl_th"
