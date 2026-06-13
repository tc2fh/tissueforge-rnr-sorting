"""Pytest fixtures for the Phase-1 reconnection tests.

tf.init() is a ONE-PER-PROCESS singleton (a second call hangs -- see PORTING_NOTES.md),
so TissueForge + the vertex solver are initialised exactly once in a session-scoped
fixture. All test meshes coexist in this single universe; each test scopes its analysis
to its own `bodies` list, and counts vertices/surfaces from those bodies (never the
global `*.instances`, which would mix meshes across tests).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.fixture(scope="session")
def vsolver():
    """Initialise TF + vertex solver once; return (tf, tfv, stype, btype).

    MeshQuality is disabled: its only 3D ops are irreversible degenerate collapses, and
    the reversible Okuda I<->H reconnection must be the ONLY topology-change operator
    under test (CLAUDE.md / PORTING_NOTES.md).
    """
    import tissue_forge as tf
    from tissue_forge.models.vertex import solver as tfv
    from tissue_forge.models.vertex.solver.mesh_types import BodyTypeSpec, SurfaceTypeSpec

    # large box: every test mesh lives here together. dt is kept SMALL (adequate): the
    # only tests that integrate force BIG-feature reconnections on an untensioned block,
    # and the faithful experiments proved post-reconnection
    # relaxation overshoots into transient winding sign-flips at coarse dt (0.01) but
    # settles cleanly at adequate dt -- so a small dt makes the clean-integration gate
    # deterministic rather than marginal.
    tf.init(windowless=True, dim=[60., 60., 60.], cutoff=5.0, dt=0.001)
    tfv.init()
    tfv.MeshSolver.get().get_mesh().quality = None

    class Iface(SurfaceTypeSpec):
        pass

    class Cell(BodyTypeSpec):
        volume_lam = 1.0
        volume_val = 1.0

    stype = Iface.get()
    btype = Cell.get()
    return tf, tfv, stype, btype
