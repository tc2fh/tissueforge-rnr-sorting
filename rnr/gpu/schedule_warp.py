"""Gate C brick C2a: the parallel conflict-resolution kernel -- cellGPU's atomic
maximal-independent-set protocol, lifted to the 3D I-neighbourhood, in NVIDIA Warp.

This is the scheduling heart of the novel result: many candidate I->H reconnections
resolve to a conflict-free batch ON THE GPU, with no host serialization. The 3D
I-neighbourhood footprint is a FIXED 8 verts / 9 surfaces / 5 bodies (vs cellGPU's 4 cells
in 2D), so the reservation arrays are regular -- no raggedness here.

Protocol (one round):
  1. RESERVE  -- dim=N: each candidate i does `atomic_min(owner[e], i)` over every element
                 e of its footprint. After the launch, owner[e] = the lowest-id candidate
                 wanting e.
  2. CHECK    -- dim=N: candidate i WINS iff owner[e]==i for ALL its footprint elements
                 (it is the lowest id wanting each). Winners are mutually footprint-disjoint
                 => conflict-free by construction (if i,j>i share e, owner[e]<=i<j so j loses).

One round is conflict-free but not maximal (a candidate can lose an element to a lower-id
candidate that itself loses elsewhere); the scheduler ITERATES (reset owners, re-run over
the losers) until the set is maximal -- exactly cellGPU's iterated-batch loop. schedule_csr
holds the validated host reference (reserve_independent_set_host) this kernel must match.

The selection is deterministic (lowest-id-wins, not atomic-order-dependent), so it matches
the host bit-for-bit -- unlike the eventual nondeterministic apply, which we validate by
the order-invariant fingerprint instead.
"""
from typing import List, Tuple

import numpy as np

import warp as wp

from .device_mesh import PaddedMesh
from .reconnect_csr import ICfgIdx

wp.init()

# the 3D I-neighbourhood footprint is fixed-size
_FV = 8   # 2 end verts + 6 outer verts
_FS = 9   # 3 side + 3 top + 3 bottom faces
_FB = 5   # 2 caps + 3 side cells


# ======================================================================================
# reservation + check kernels
# ======================================================================================
@wp.kernel
def reserve_kernel(fp_v: wp.array2d(dtype=wp.int32), fp_s: wp.array2d(dtype=wp.int32),
                   fp_b: wp.array2d(dtype=wp.int32),
                   vown: wp.array(dtype=wp.int32), sown: wp.array(dtype=wp.int32),
                   bown: wp.array(dtype=wp.int32)):
    i = wp.tid()
    for k in range(_FV):
        wp.atomic_min(vown, fp_v[i, k], i)
    for k in range(_FS):
        wp.atomic_min(sown, fp_s[i, k], i)
    for k in range(_FB):
        wp.atomic_min(bown, fp_b[i, k], i)


@wp.kernel
def check_kernel(fp_v: wp.array2d(dtype=wp.int32), fp_s: wp.array2d(dtype=wp.int32),
                 fp_b: wp.array2d(dtype=wp.int32),
                 vown: wp.array(dtype=wp.int32), sown: wp.array(dtype=wp.int32),
                 bown: wp.array(dtype=wp.int32), won: wp.array(dtype=wp.int32)):
    i = wp.tid()
    w = wp.int32(1)
    for k in range(_FV):
        if vown[fp_v[i, k]] != i:
            w = wp.int32(0)
    for k in range(_FS):
        if sown[fp_s[i, k]] != i:
            w = wp.int32(0)
    for k in range(_FB):
        if bown[fp_b[i, k]] != i:
            w = wp.int32(0)
    won[i] = w


# ======================================================================================
# footprint packing + host wrappers
# ======================================================================================
def pack_footprints(cands: List[Tuple[int, int, ICfgIdx]]):
    """Pack candidate footprints into fixed-width (N,8)/(N,9)/(N,5) int32 arrays (verts,
    surfaces, bodies). Canonical order; every [I] config fills them exactly."""
    n = len(cands)
    fpv = np.empty((n, _FV), np.int32)
    fps = np.empty((n, _FS), np.int32)
    fpb = np.empty((n, _FB), np.int32)
    for i, (_v10, _v11, cfg) in enumerate(cands):
        fpv[i] = [cfg.v10, cfg.v11] + [x for a in cfg.arms for x in (a.outer_top, a.outer_bot)]
        fps[i] = ([a.side_surface for a in cfg.arms]
                  + list(cfg.top_faces.values()) + list(cfg.bottom_faces.values()))
        fpb[i] = [cfg.cap_top, cfg.cap_bot] + list(cfg.side_cells)
    return fpv, fps, fpb


def reserve_won_mask_warp(pm: PaddedMesh, cands: List[Tuple[int, int, ICfgIdx]],
                          device=None) -> np.ndarray:
    """Run one GPU reservation round; return the (N,) int32 won-mask (1 = in the
    independent set). Owners are sized to the mesh capacity and seeded to N (a sentinel
    above every candidate id, so atomic_min always lands on a real id)."""
    n = len(cands)
    if n == 0:
        return np.zeros(0, np.int32)
    if device is None:
        cuda = [d for d in wp.get_devices() if d.is_cuda]
        device = cuda[0] if cuda else "cpu"
    fpv, fps, fpb = pack_footprints(cands)
    a2 = lambda a: wp.array(np.ascontiguousarray(a), dtype=wp.int32, device=device)
    g_fpv, g_fps, g_fpb = a2(fpv), a2(fps), a2(fpb)
    vown = wp.array(np.full(pm.cap_v, n, np.int32), dtype=wp.int32, device=device)
    sown = wp.array(np.full(pm.cap_s, n, np.int32), dtype=wp.int32, device=device)
    bown = wp.array(np.full(pm.nb, n, np.int32), dtype=wp.int32, device=device)
    won = wp.zeros(n, dtype=wp.int32, device=device)
    wp.launch(reserve_kernel, dim=n, device=device,
              inputs=[g_fpv, g_fps, g_fpb, vown, sown, bown])
    wp.launch(check_kernel, dim=n, device=device,
              inputs=[g_fpv, g_fps, g_fpb, vown, sown, bown, won])
    wp.synchronize_device(device)
    return won.numpy()


def reserve_independent_set_warp(pm: PaddedMesh, cands: List[Tuple[int, int, ICfgIdx]],
                                 device=None) -> List[Tuple[int, int, ICfgIdx]]:
    """The candidates that won one GPU reservation round (a conflict-free batch)."""
    mask = reserve_won_mask_warp(pm, cands, device)
    return [cands[i] for i in range(len(cands)) if mask[i]]
