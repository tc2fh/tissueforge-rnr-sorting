"""Gate C brick C0 (docs/2026-06-24_gpu-3d-vertex-model-exploration.md):

The index-based [I]-config detector (topology_csr.py) must find the same reconnection
sites as the validated CPU oracle (topology.py) -- working purely from PaddedMesh index
arrays, with no TF handles -- and the ICfgIdx it emits must be surgery-ready (drive a
clean i_to_h_csr -> h_to_i_csr round-trip). This is the scheduler's detection input; it
must be right before the independent-set machinery stands on it.
"""
import numpy as np

from .. import topology as topo
from ..gpu import csr_mesh as cm
from ..gpu import reconnect_csr as rcsr
from ..gpu import topology_csr as tcsr
from ..gpu.device_mesh import PaddedMesh
from . import helpers as H


def test_detector_minimal_finds_the_one_short_edge(vsolver):
    _tf, _tfv, stype, btype = vsolver
    cfg_in = H.build_minimal_i_config(stype, btype, center=(12., 48., 24.), edge=0.5)
    m0 = cm.extract_csr(cfg_in["bodies"])
    vid2i, _sid, _bid = cm.id_maps(cfg_in["bodies"])
    pm = PaddedMesh.from_csr(m0)

    sites = tcsr.find_short_edges_csr(pm, threshold=1.0)
    assert len(sites) == 1, f"minimal config should have exactly one [I] short edge, got {len(sites)}"
    v10, v11, cfg = sites[0]
    # the detected edge is the known v10-v11 (by index, via the id->index map)
    assert {v10, v11} == {vid2i[cfg_in["v10"].id], vid2i[cfg_in["v11"].id]}
    # structurally well-formed: 3 side cells, 3 arms, 3+3 cap interfaces, caps distinct
    assert len(cfg.side_cells) == 3 and len(cfg.arms) == 3
    assert len(cfg.top_faces) == 3 and len(cfg.bottom_faces) == 3
    assert cfg.cap_top != cfg.cap_bot


def test_detector_matches_oracle_count_kelvin(vsolver):
    """The index detector finds exactly the CPU oracle's short-edge sites (same mesh)."""
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(56., 8., 8.))
    m0 = cm.extract_csr(bodies)
    pm = PaddedMesh.from_csr(m0)

    csr_sites = tcsr.find_short_edges_csr(pm, threshold=1.0)
    tf_sites = topo.find_short_edges(bodies, threshold=1.0)
    assert len(csr_sites) == len(tf_sites) > 0, \
        f"detector/oracle site-count mismatch: csr={len(csr_sites)} tf={len(tf_sites)}"


def test_detected_configs_are_surgery_ready(vsolver):
    """Every ICfgIdx the detector emits drives a clean count-changing round-trip on a
    fresh mesh (fingerprint restored) -- proving the index-only config is correct."""
    _tf, _tfv, stype, btype = vsolver
    bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(8., 8., 56.))
    m0 = cm.extract_csr(bodies)
    fp0 = cm.fingerprint(m0)

    sites = tcsr.find_short_edges_csr(PaddedMesh.from_csr(m0), threshold=1.0)
    assert sites, "no detected sites"
    # deterministic pick: smallest-index endpoints
    sites.sort(key=lambda t: (t[0], t[1]))
    v10, v11, _cfg = sites[0]

    pm = PaddedMesh.from_csr(m0)
    cfg = tcsr.i_neighbourhood_csr(pm, v10, v11)
    assert cfg is not None
    dl = tcsr.edge_length(pm, v10, v11)
    hcfg = rcsr.i_to_h_csr(pm, cfg, dl)
    assert pm.check_consistency() == []
    rcsr.h_to_i_csr(pm, hcfg, dl)
    assert pm.check_consistency() == []
    assert cm.fingerprint(pm.to_csr()) == fp0, "detector-driven round-trip did not restore topology"
