"""Gate B substrate (docs/2026-06-24_gpu-3d-vertex-model-exploration.md):

The padded *mutable* mesh + bump allocator + local surgery primitives must be correct
before the I<->H surgery composes them. Checks:
  * CSR -> padded -> CSR round-trips exactly (no surgery)
  * consistency checker is clean on a real mesh
  * bump allocator: births advance the high-water mark, deaths mark slots
  * surgery primitives maintain BOTH sides of every adjacency (mutate-then-revert restores
    the mesh exactly)
  * the padded SoA uploads to the GPU intact
"""
import numpy as np
import pytest

from ..gpu import csr_mesh as cm
from ..gpu.device_mesh import PaddedMesh
from . import helpers as H


def _rows(off, idx):
    return [idx[off[i]:off[i + 1]] for i in range(len(off) - 1)]


def _conn_equal(a: cm.CSRMesh, b: cm.CSRMesh) -> bool:
    """Connectivity + geometry equal (ignores the id/handle arrays).

    The surface ring s2v is ORDERED (winding defines the normal) -> compared as ordered
    lists. v2s, b2s and the s2b body pair are UNORDERED adjacency -> compared as sets, so a
    revert that merely re-appends an incidence in a different slot still counts as restored.
    """
    if not (a.nv == b.nv and a.ns == b.ns and a.nb == b.nb):
        return False
    if not np.array_equal(a.vert_pos, b.vert_pos):
        return False
    if ([list(r) for r in _rows(a.s2v_off, a.s2v_idx)]
            != [list(r) for r in _rows(b.s2v_off, b.s2v_idx)]):
        return False
    if ([frozenset(int(x) for x in r if x >= 0) for r in a.s2b]
            != [frozenset(int(x) for x in r if x >= 0) for r in b.s2b]):
        return False
    for off_a, idx_a, off_b, idx_b in [
            (a.v2s_off, a.v2s_idx, b.v2s_off, b.v2s_idx),
            (a.b2s_off, a.b2s_idx, b.b2s_off, b.b2s_idx)]:
        if ([set(map(int, r)) for r in _rows(off_a, idx_a)]
                != [set(map(int, r)) for r in _rows(off_b, idx_b)]):
            return False
    return True


def _csr(vsolver, which="kelvin"):
    tf, tfv, stype, btype = vsolver
    if which == "kelvin":
        bodies = H.build_kelvin_block(stype, btype, n=4, span=8.0, origin=(20., 20., 20.))
    else:
        bodies = H.build_minimal_i_config(stype, btype, center=(50., 50., 50.))["bodies"]
    return cm.extract_csr(bodies)


def test_padded_roundtrip_exact(vsolver):
    """from_csr -> to_csr reproduces the source CSR exactly (no surgery)."""
    for which in ("minimal", "kelvin"):
        m0 = _csr(vsolver, which)
        pm = PaddedMesh.from_csr(m0)
        assert pm.check_consistency() == [], f"{which}: padded mesh inconsistent"
        m1 = pm.to_csr()
        assert _conn_equal(m0, m1), f"{which}: padded round-trip changed the mesh"


def test_bump_allocator(vsolver):
    m0 = _csr(vsolver, "minimal")
    pm = PaddedMesh.from_csr(m0)
    v_used0, s_used0 = pm.n_v_used, pm.n_s_used
    newv = [pm.alloc_vertex(np.array([float(i), 0., 0.])) for i in range(10)]
    news = [pm.alloc_surface() for _ in range(5)]
    assert newv == list(range(v_used0, v_used0 + 10)), "vertex bump not monotonic/unique"
    assert news == list(range(s_used0, s_used0 + 5)), "surface bump not monotonic/unique"
    assert pm.vert_alive[newv].all() and pm.surf_alive[news].all()
    pm.free_vertex(newv[0]); pm.free_surface(news[0])
    assert pm.vert_alive[newv[0]] == 0 and pm.surf_alive[news[0]] == 0
    # freed-but-uncompacted slots stay below the high-water mark (Gate D reclaims them)
    assert pm.n_v_used == v_used0 + 10 and pm.n_s_used == s_used0 + 5


def test_primitives_maintain_both_sides(vsolver):
    """Insert a fresh vertex into a ring then drop it; replace a vertex then put it back.
    Each pair must restore the mesh exactly -- proving both-sides adjacency upkeep."""
    m0 = _csr(vsolver, "kelvin")
    pm = PaddedMesh.from_csr(m0)

    # pick an interior (2-body) surface and two ring-adjacent vertices in it
    s = next(s for s in range(pm.n_s_used)
             if pm.surf_alive[s] and (pm.s2b[s] >= 0).all() and pm.s2v_len[s] >= 3)
    a = int(pm.s2v[s, 0]); b = int(pm.s2v[s, 1])

    # (1) insert a brand-new vertex between a,b then drop it -> restored
    nv = pm.alloc_vertex(0.5 * (pm.vert_pos[a] + pm.vert_pos[b]))
    pm.insert_between(s, nv, a, b)
    assert pm.s2v_len[s] == np.diff(m0.s2v_off)[s] + 1
    assert pm.check_consistency() == []
    pm.drop_v(s, nv); pm.free_vertex(nv)
    assert _conn_equal(m0, pm.to_csr()), "insert/drop did not restore the mesh"

    # (2) replace a with a fresh vertex, then replace back -> restored
    nv2 = pm.alloc_vertex(pm.vert_pos[a].copy())
    pm.replace_v(s, a, nv2)
    assert pm.check_consistency() == []
    assert s not in pm.v2s[a, :pm.v2s_len[a]] and s in pm.v2s[nv2, :pm.v2s_len[nv2]]
    pm.replace_v(s, nv2, a); pm.free_vertex(nv2)
    assert _conn_equal(m0, pm.to_csr()), "replace/replace-back did not restore the mesh"


def test_padded_gpu_upload(vsolver):
    import warp as wp
    wp.init()
    if not any(d.is_cuda for d in wp.get_devices()):
        pytest.skip("no CUDA device")
    m0 = _csr(vsolver, "kelvin")
    pm = PaddedMesh.from_csr(m0)
    dev = next(d for d in wp.get_devices() if d.is_cuda)
    g = pm.to_warp(device=dev)
    for name, host in [("s2v", pm.s2v), ("s2v_len", pm.s2v_len), ("v2s", pm.v2s),
                       ("v2s_len", pm.v2s_len), ("s2b", pm.s2b), ("b2s", pm.b2s)]:
        assert np.array_equal(g[name].numpy().reshape(host.shape), host), f"{name} corrupt on GPU"
    assert np.array_equal(g["vert_pos"].numpy().reshape(pm.cap_v, 3), pm.vert_pos)
