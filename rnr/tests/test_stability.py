"""Unit tests for the winding-clamp displacement limiter (operator._clamp_to).

The clamp is the stability guard that prevents the dt-overshoot WINDING SIGN-FLIP
(negative signed volume -> reversed VolumeConstraint force -> cells "shoot off"); see
operator.stable_step and the `faithful-instability-is-winding-signflip` memory. The
limiter's MATH is pinned here, deterministically and TF-free (the pure geometry is the
unit). The END-TO-END behavioural gate -- that volumes stay positive across a real
tensioned run -- is the re-rendered `pixi run sort-video` (min_vol must remain > 0),
not a unit test: the blow-up only emerges under tension over thousands of steps, and
stable_step steps the whole shared session universe (would perturb sibling tests).
"""
import numpy as np

from ..operator import _clamp_to


def test_overshoot_pulled_back_to_cap():
    """A move longer than the cap is shortened to EXACTLY cap, same direction."""
    old = np.array([0.0, 0.0, 0.0])
    new = np.array([1.0, 0.0, 0.0])          # length 1.0
    res = _clamp_to(old, new, 0.3)
    assert np.isclose(np.linalg.norm(res - old), 0.3)
    assert np.allclose(res, [0.3, 0.0, 0.0])  # direction preserved


def test_overshoot_preserves_direction_diagonal():
    old = np.array([1.0, 1.0, 1.0])
    new = old + np.array([3.0, 4.0, 0.0])    # length 5.0 along (3,4,0)/5
    res = _clamp_to(old, new, 1.0)
    assert np.isclose(np.linalg.norm(res - old), 1.0)
    assert np.allclose(res, old + np.array([0.6, 0.8, 0.0]))


def test_small_move_untouched():
    """A move within the cap is returned unchanged (no spurious clamping)."""
    old = np.array([0.0, 0.0, 0.0])
    new = np.array([0.01, 0.0, 0.0])
    assert np.allclose(_clamp_to(old, new, 0.3), new)


def test_at_cap_untouched():
    old = np.array([0.0, 0.0, 0.0])
    new = np.array([0.3, 0.0, 0.0])          # exactly at cap
    assert np.allclose(_clamp_to(old, new, 0.3), new)


def test_zero_move_no_divide_by_zero():
    """A vertex that did not move must be left in place (cap math must not divide by 0)."""
    old = np.array([2.0, -1.0, 0.5])
    assert np.allclose(_clamp_to(old, old.copy(), 0.3), old)
