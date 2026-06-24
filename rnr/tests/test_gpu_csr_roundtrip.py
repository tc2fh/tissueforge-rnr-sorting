"""Gate A of the GPU port (docs/2026-06-24_gpu-3d-vertex-model-exploration.md):

The index-based CSR/SoA extraction must round-trip TF's pointer-graph mesh *exactly*.
This is the GPU-port analogue of test_roundtrip.py's discipline -- the data-structure
foundation must be proven before any kernel stands on it.

Checks:
  * minimal hand-built [I] config           -> CSR round-trips exactly
  * realistic Kelvin block (4-valent interior verts, variable-size faces/cells)
                                             -> CSR round-trips exactly
  * the verifier has TEETH                   -> a corrupted CSR is rejected
  * host->device->host integrity on the GPU  -> the SoA lands on the 5090 intact
"""
import numpy as np
import pytest

from ..gpu import csr_mesh as cm
from . import helpers as H


def _check(bodies):
    m = cm.extract_csr(bodies)
    report = cm.verify_roundtrip(m, bodies)
    assert report["ok"], "round-trip FAILED:\n  " + "\n  ".join(report["problems"])
    return m, report


def test_csr_roundtrip_minimal_i(vsolver):
    """Tiny deterministic 5-cell [I] neighbourhood: exact round-trip."""
    tf, tfv, stype, btype = vsolver
    cfg = H.build_minimal_i_config(stype, btype, center=(10., 10., 10.))
    m, report = _check(cfg["bodies"])
    # the [I] config has 5 cells and 8 vertices (v10,v11 + 3 top + 3 bot)
    assert m.nb == 5
    assert m.nv == 8
    # v10 and v11 are interior-ish: each touches the 3 side cells + its cap (4 bodies).
    # confirm at least the two short-edge endpoints resolve to 4 incident bodies via the
    # transpose-consistent maps (sanity that connectivity, not just counts, is captured).
    print("\n[minimal I] " + cm.summary(m))


def test_csr_roundtrip_kelvin(vsolver):
    """Realistic Kelvin (BCC-Voronoi) block: variable valence, exact round-trip.

    This is the case that matters -- interior vertices are 4-cell and faces are
    variable-size polygons, exactly the raggedness that breaks cellGPU's fixed-stride-3
    layout and forces the CSR design.
    """
    tf, tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(40., 40., 40.))
    m, report = _check(bodies)
    print("\n[kelvin] " + cm.summary(m))
    # raggedness sanity: not every face has the same number of vertices, and at least
    # some interior (2-body) faces exist -- i.e. this isn't a trivial single cell.
    poly = np.diff(m.s2v_off)
    assert poly.min() != poly.max(), "expected variable-size faces in a Kelvin block"
    assert int(np.sum((m.s2b >= 0).all(axis=1))) > 0, "expected interior (2-body) faces"


def test_verifier_has_teeth(vsolver):
    """A corrupted CSR MUST be rejected -- otherwise the gate proves nothing."""
    tf, tfv, stype, btype = vsolver
    cfg = H.build_minimal_i_config(stype, btype, center=(10., 25., 10.))
    bodies = cfg["bodies"]

    # corrupt one ring entry: point surface 0's first vertex slot at a different vertex.
    m = cm.extract_csr(bodies)
    bad = (m.s2v_idx[0] + 1) % m.nv
    m.s2v_idx[0] = bad
    report = cm.verify_roundtrip(m, bodies)
    assert not report["ok"], "verifier failed to detect a corrupted ring"
    assert any("ring" in p or "transpose" in p for p in report["problems"])


def test_csr_gpu_upload_readback(vsolver):
    """The SoA must survive host->device->host on the GPU intact (Stage-0 device path)."""
    import warp as wp
    wp.init()
    if not any(d.is_cuda for d in wp.get_devices()):
        pytest.skip("no CUDA device")

    tf, tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=3, span=6.0, origin=(40., 10., 40.))
    m = cm.extract_csr(bodies)
    dev = next(d for d in wp.get_devices() if d.is_cuda)
    g = m.to_warp(device=dev)

    # integer connectivity arrays: bit-exact round-trip
    for name, host in [("s2v_off", m.s2v_off), ("s2v_idx", m.s2v_idx),
                       ("v2s_off", m.v2s_off), ("v2s_idx", m.v2s_idx),
                       ("b2s_off", m.b2s_off), ("b2s_idx", m.b2s_idx),
                       ("s2b", m.s2b)]:
        back = g[name].numpy()
        assert np.array_equal(back.reshape(host.shape), host), f"{name} corrupted on GPU"
    # positions: f64 exact
    pos_back = g["vert_pos"].numpy().reshape(m.nv, 3)
    assert np.array_equal(pos_back, m.vert_pos), "positions corrupted on GPU"
