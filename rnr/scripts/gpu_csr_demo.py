"""Gate-A demo: extract a TissueForge vertex mesh into the GPU CSR/SoA layout, verify it
round-trips TF exactly, print its raggedness, and upload it to the GPU.

    pixi run gpu-csr            # Kelvin block (default n=4, span=8)
    pixi run gpu-csr 5 10       # n=5, span=10

This is the host-side Stage-0 of docs/2026-06-24_gpu-3d-vertex-model-exploration.md. It
needs nothing on the GPU to BUILD the CSR; the final step uploads to the 5090 if present.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import tissue_forge as tf
from tissue_forge.models.vertex import solver as tfv
from tissue_forge.models.vertex.solver.mesh_types import BodyTypeSpec, SurfaceTypeSpec

from rnr.gpu import csr_mesh as cm
from rnr.tests.helpers import build_kelvin_block


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    span = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0

    tf.init(windowless=True, dim=[60., 60., 60.], cutoff=5.0, dt=0.001)
    tfv.init()
    tfv.MeshSolver.get().get_mesh().quality = None

    class Iface(SurfaceTypeSpec):
        pass

    class Cell(BodyTypeSpec):
        volume_lam = 1.0
        volume_val = 1.0

    stype, btype = Iface.get(), Cell.get()
    bodies = build_kelvin_block(stype, btype, n=n, span=span)

    m = cm.extract_csr(bodies)
    print(cm.summary(m))
    rep = cm.verify_roundtrip(m, bodies)
    print("round-trip vs TF:", "OK" if rep["ok"] else "FAIL")
    if not rep["ok"]:
        for p in rep["problems"][:5]:
            print("   !", p)
        sys.exit(1)

    g = m.to_warp()
    print(f"uploaded SoA to Warp device: {g['device']}  "
          f"({rep['counts']['n_incid_sv']} surf-vertex + "
          f"{rep['counts']['n_incid_bs']} body-surface incidences)")


if __name__ == "__main__":
    main()
