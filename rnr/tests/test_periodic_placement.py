"""Periodic minimum-image in the Okuda I<->H vertex placement.

A reconnection site -- a SHORT edge (I->H) or a SMALL triangle (H->I) -- can sit ON a periodic
box face, with its vertices on opposite sides (close in min-image distance, ~L apart in raw
coordinates). Naive coordinate arithmetic (`0.5*(p10+p11)`, `(p0+p1+p2)/3`, `ot-r0`) then splits
the feature through the BOX CENTRE and teleports the new vertices to garbage. The fix differences
all positions under the minimum-image convention and wraps results into [0, L).

These tests pin that behaviour: `box=None` is the exact non-periodic placement (unchanged); a
positive `box` places straddling sites near the face, not the centre; and the GPU I->H kernel
matches the CPU oracle for a straddling site. See `rnr/reconnect.py` place_*_xyz +
`rnr/gpu/reconnect_warp.py`.
"""
import numpy as np
import pytest

from rnr.reconnect import _minimg, _wrapbox, place_h_to_i_xyz, place_i_to_h_xyz

L = 10.0
BOX = np.array([L, L, L])


def _pdist(a, b, box=BOX):
    """Periodic (minimum-image) Euclidean distance."""
    return float(np.linalg.norm(_minimg(np.asarray(a, float) - np.asarray(b, float), box)))


def test_minimg_wrapbox_helpers():
    # a near-L displacement folds to the short side; wrap lands in [0, L)
    assert np.allclose(_minimg(np.array([L - 0.1, 0.0, 0.0]), BOX), [-0.1, 0.0, 0.0])
    assert np.allclose(_wrapbox(np.array([-0.1, L + 0.2, 0.3]), BOX), [L - 0.1, 0.2, 0.3])
    # a non-positive box axis is left untouched (finite-cluster fallback)
    assert np.allclose(_minimg(np.array([99.0, 0.0, 0.0]), np.array([0.0, L, L])), [99.0, 0.0, 0.0])


def _straddle_i_config():
    """A short edge across the x=0/L face, with its 6 outer neighbours straddling it too."""
    dl = 0.02
    p10 = np.array([0.01, 5.0, 5.0])
    p11 = np.array([L - 0.01, 5.0, 5.0])          # min-image edge length 0.02 (a short edge)
    tops, bots = [], []
    for k in range(3):                            # 3 arms ~120deg apart in the y-z plane
        th = 2.0 * np.pi * k / 3.0
        off = np.array([0.0, 0.3 * np.cos(th), 0.3 * np.sin(th)])
        tops.append((p10 + off) % L)
        bots.append((p11 + off) % L)
    return p10, p11, tops, bots, dl


def test_i_to_h_straddle_places_near_face_not_centre():
    p10, p11, tops, bots, dl = _straddle_i_config()
    r0_true = _wrapbox(p10 + 0.5 * _minimg(p11 - p10, BOX), BOX)      # ~x = 0/L, not L/2
    tri_box = place_i_to_h_xyz(p10, p11, tops, bots, dl, box=BOX)
    for v in tri_box:
        assert _pdist(v, r0_true) <= 5 * dl                          # within O(dl) of the face
        assert abs((v[0] % L) - L / 2) > 1.0                         # NOT dragged to the centre
    # the buggy non-periodic placement is dragged FAR from the face (what we are fixing)
    tri_naive = place_i_to_h_xyz(p10, p11, tops, bots, dl)
    assert all(_pdist(v, r0_true) > 1.0 for v in tri_naive)


def test_h_to_i_straddle_places_near_face_not_centre():
    dl = 0.02
    tri = [np.array([0.01, 5.00, 5.00]),
           np.array([L - 0.01, 5.02, 5.00]),
           np.array([0.00, 4.98, 5.02])]                             # small triangle on the face
    tops = [np.array([0.005, 5.0, 5.4]) % L,
            np.array([L - 0.005, 5.0, 5.4]) % L,
            np.array([0.0, 4.97, 5.4]) % L]                          # cap_top side (+z), straddling
    d1 = _minimg(tri[1] - tri[0], BOX)
    d2 = _minimg(tri[2] - tri[0], BOX)
    r0_true = _wrapbox(tri[0] + (d1 + d2) / 3.0, BOX)
    a, b = place_h_to_i_xyz(tri, tops, dl, box=BOX)
    for v in (a, b):
        assert _pdist(v, r0_true) <= 5 * dl
        assert abs((v[0] % L) - L / 2) > 1.0
    # naive places the recovered verts FAR from the true (periodic) triangle centroid
    na, nb = place_h_to_i_xyz(tri, tops, dl)
    assert _pdist(na, r0_true) > 1.0


def test_box_aware_equals_non_periodic_for_interior_site():
    """For an INTERIOR site (nothing straddles), min-image is the identity -> placement
    must be bit-for-bit the non-periodic result (no silent drift from the box plumbing)."""
    p10 = np.array([5.00, 5.0, 5.0])
    p11 = np.array([5.02, 5.0, 5.0])
    tops = [p10 + np.array([0.0, 0.3 * np.cos(t), 0.3 * np.sin(t)]) for t in (0.0, 2.1, 4.2)]
    bots = [p11 + np.array([0.0, 0.3 * np.cos(t), 0.3 * np.sin(t)]) for t in (0.0, 2.1, 4.2)]
    a = place_i_to_h_xyz(p10, p11, tops, bots, 0.02)
    b = place_i_to_h_xyz(p10, p11, tops, bots, 0.02, box=BOX)
    assert np.allclose(np.array(a), np.array(b), atol=1e-12)

    tri = [np.array([5.0, 5.0, 5.0]), np.array([5.02, 5.01, 5.0]), np.array([5.0, 4.99, 5.01])]
    otops = [np.array([5.0, 5.0, 5.4]), np.array([5.02, 5.0, 5.4]), np.array([5.0, 4.99, 5.4])]
    c = place_h_to_i_xyz(tri, otops, 0.02)
    d = place_h_to_i_xyz(tri, otops, 0.02, box=BOX)
    assert np.allclose(np.array(c), np.array(d), atol=1e-12)


def test_gpu_i_to_h_matches_cpu_periodic():
    """The GPU I->H placement kernel (min-image) == the CPU oracle for a straddling site
    (the production kernels use this exact arithmetic via d_minimg / d_wrapbox)."""
    import warp as wp
    cuda = [d for d in wp.get_devices() if d.is_cuda]
    if not cuda:
        pytest.skip("no CUDA device for the GPU placement parity check")
    from rnr.gpu import reconnect_warp as rw
    p10, p11, tops, bots, dl = _straddle_i_config()
    res = rw.probe_placement_precision(p10, p11, tops, bots, dl_th=dl, device=cuda[0], box=BOX)
    assert np.allclose(res["gpu_f64_box"], res["oracle_box"], atol=1e-12)
