"""Deterministic tests for the demixing index (Phase-2 metric).

`demixing_index` is the headline "how sorted is it" number: D = mean_over_cells of
2*(het_frac - 0.5), SIGNED so more sorted = more negative (see metrics.py). These tests
pin it on configurations with KNOWN adjacency -- no integration, so they are exact /
deterministic and form the metric's spec.

  * The hand-built minimal [I] neighbourhood (helpers.build_minimal_i_config) has an
    adjacency we can enumerate by hand, giving D to the digit for two type assignments.
  * A real Kelvin block checks the realistic extremes: a clean planar A|B split is
    strongly sorted (D very negative); a salt-and-pepper random assignment is ~0.
"""
import numpy as np
import pytest

from tissue_forge.models.vertex.solver.mesh_types import BodyTypeSpec

from ..metrics import (contact_summary, demixing_index, het_frac_from_D,
                       het_frac_from_oracle_demix, sorting_score)
from . import helpers as H


# --- Pure-math comparison helpers (no TF; used by the oracle-comparison engine) ---

def test_het_frac_from_D_roundtrips():
    """het_frac_from_D inverts the demixing-index definition D = 2*(het_frac - 1/2)."""
    for hf in (0.0, 0.25, 0.4784, 0.5, 1.0):
        D = 2.0 * (hf - 0.5)
        assert het_frac_from_D(D) == pytest.approx(hf, abs=1e-12)


def test_oracle_demix_is_negative_of_ours():
    """The key Step-0 identity: 3DVertVor's positive-convention demix = -D_ours, so both
    map to the SAME het fraction. (Re-derived from Run::dumpDemix; GPL -- never copied.)"""
    for D in (-1.0, -0.098, -0.043, 0.0, 0.4):
        demix_oracle = -D
        assert het_frac_from_oracle_demix(demix_oracle) == pytest.approx(
            het_frac_from_D(D), abs=1e-12)


def test_sorting_score_endpoints():
    """S = 1 - hf/hf0: zero at the start, rises as het contact is eliminated, 0 if hf0=0."""
    assert sorting_score(0.48, 0.48) == pytest.approx(0.0, abs=1e-12)   # t=0
    assert sorting_score(0.24, 0.48) == pytest.approx(0.5, abs=1e-12)   # halved
    assert sorting_score(0.0, 0.48) == pytest.approx(1.0, abs=1e-12)    # fully sorted
    assert sorting_score(0.5, 0.0) == 0.0                               # guard hf0=0


def _two_celltypes():
    """Two body types (A/B) for het/hom contacts. Geometry is never integrated here, so
    the energetic params are irrelevant -- only the type NAME drives the metric."""
    class DA(BodyTypeSpec):
        volume_lam = 1.0
        volume_val = 1.0

    class DB(BodyTypeSpec):
        volume_lam = 1.0
        volume_val = 1.0
    return DA.get(), DB.get()


def test_demixing_index_handbuilt_exact(vsolver):
    """The minimal [I] config: 3 wedges in a ring (each neighbours its 2 ring-neighbours +
    both caps -> 4 neighbours) + cap_top + cap_bot (each neighbours all 3 wedges).

    All one type      -> every het_frac=0 -> D = -1 exactly.
    wedges=A, caps=B  -> wedge het_frac = 2 caps / 4 = 0.5 (val 0); each cap het_frac =
                         3 wedges / 3 = 1 (val +1). D = (0+0+0+1+1)/5 = 0.4 exactly.
    """
    _tf, _tfv, stype, _btype = vsolver
    btA, btB = _two_celltypes()
    cfg = H.build_minimal_i_config(stype, btA, center=(2., 46., 2.))
    bodies = cfg["bodies"]

    assert demixing_index(bodies=bodies) == pytest.approx(-1.0, abs=1e-9), \
        "uniform type must give the fully-sorted extreme D=-1"

    cfg["cap_top"].become(btB)
    cfg["cap_bot"].become(btB)
    assert demixing_index(bodies=bodies) == pytest.approx(0.4, abs=1e-9), \
        "wedges=A/caps=B has a hand-enumerated D=0.4"

    # contact_summary must surface the SAME number (it delegates to demixing_index).
    assert contact_summary(bodies=bodies)["demixing_index"] == pytest.approx(0.4, abs=1e-9)


def _centroid_x(b):
    return float(np.mean([v.position[0] for v in b.getVertices()]))


def test_demixing_index_kelvin_sorted_vs_random(vsolver):
    """A real Kelvin block: a clean planar A|B split is strongly sorted (D very negative);
    a salt-and-pepper random split is ~0. Sorted must be clearly below random."""
    _tf, _tfv, stype, _btype = vsolver
    btA, btB = _two_celltypes()
    bodies = H.build_kelvin_block(stype, btA, n=4, span=8.0, origin=(2., 30., 50.))

    # --- clean planar split at the block mid-plane -> sorted ---
    mid = 2.0 + 8.0 / 2.0
    for b in bodies:
        b.become(btB if _centroid_x(b) >= mid else btA)
    d_sorted = demixing_index(bodies=bodies)

    # --- salt-and-pepper (deterministic rng) -> ~0 ---
    rng = np.random.default_rng(0)
    for b in bodies:
        b.become(btB if rng.random() < 0.5 else btA)
    d_random = demixing_index(bodies=bodies)

    assert d_sorted < -0.4, f"planar split should be strongly sorted, got D={d_sorted:.3f}"
    assert abs(d_random) < 0.25, f"salt-and-pepper should be ~0, got D={d_random:.3f}"
    assert d_sorted < d_random - 0.3, \
        f"sorted ({d_sorted:.3f}) must be well below random ({d_random:.3f})"
