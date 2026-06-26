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
import types
from typing import Dict, List, Optional, Tuple

import numpy as np

import warp as wp

from .detect_warp import (detect_short_edges_hybrid, detect_small_triangles_hybrid,
                          find_short_edges_device, find_short_edges_warp,
                          find_small_triangles_warp)
from .device_mesh import PaddedMesh
from .gather_warp import (_ensure_gather_buf, gather_h_configs_warp, gather_i_configs_warp,
                          gather_i_configs_warp_device)
from .reconnect_csr import HCfgIdx, ICfgIdx
from .reconnect_warp import (apply_h_to_i_batch_warp, apply_h_to_i_device_warp,
                            apply_i_to_h_batch_warp, apply_i_to_h_device_warp)
from .schedule_csr import h_to_i_veto_csr, i_to_h_veto_csr
from .topology_csr import find_short_edges_csr, find_small_triangles_csr

wp.init()

# the 3D I-neighbourhood footprint is fixed-size
_FV = 8   # 2 end verts + 6 outer verts
_FS = 9   # 3 side + 3 top + 3 bottom faces
_FB = 5   # 2 caps + 3 side cells

# the reverse [H]-neighbourhood footprint (the triangle + its 3 verts are EXISTING, so they
# join the footprint; only the 2 recovered edge verts are fresh -- see schedule_csr.h_footprint)
_HFV = 9   # 3 tri verts + 6 outer verts
_HFS = 10  # the triangle + 3 side + 3 top + 3 bottom faces
_HFB = 5   # 2 caps + 3 side cells


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


def reserve_independent_set_warp_g(g: dict, cands: List[Tuple[int, int, ICfgIdx]]
                                   ) -> List[Tuple[int, int, ICfgIdx]]:
    """reserve_independent_set_warp reading capacities (cap_v/cap_s/nb) straight from the device
    SoA `g` instead of a host PaddedMesh -- for the fully-on-device sweep (no from_warp)."""
    caps = types.SimpleNamespace(cap_v=g["cap_v"], cap_s=g["cap_s"], nb=g["nb"])
    return reserve_independent_set_warp(caps, cands, device=g["device"])


# ======================================================================================
# C2': the reverse (H->I) reservation -- the same protocol with the H footprint sizes
# ======================================================================================
@wp.kernel
def reserve_h_kernel(fp_v: wp.array2d(dtype=wp.int32), fp_s: wp.array2d(dtype=wp.int32),
                     fp_b: wp.array2d(dtype=wp.int32),
                     vown: wp.array(dtype=wp.int32), sown: wp.array(dtype=wp.int32),
                     bown: wp.array(dtype=wp.int32)):
    i = wp.tid()
    for k in range(_HFV):
        wp.atomic_min(vown, fp_v[i, k], i)
    for k in range(_HFS):
        wp.atomic_min(sown, fp_s[i, k], i)
    for k in range(_HFB):
        wp.atomic_min(bown, fp_b[i, k], i)


@wp.kernel
def check_h_kernel(fp_v: wp.array2d(dtype=wp.int32), fp_s: wp.array2d(dtype=wp.int32),
                   fp_b: wp.array2d(dtype=wp.int32),
                   vown: wp.array(dtype=wp.int32), sown: wp.array(dtype=wp.int32),
                   bown: wp.array(dtype=wp.int32), won: wp.array(dtype=wp.int32)):
    i = wp.tid()
    w = wp.int32(1)
    for k in range(_HFV):
        if vown[fp_v[i, k]] != i:
            w = wp.int32(0)
    for k in range(_HFS):
        if sown[fp_s[i, k]] != i:
            w = wp.int32(0)
    for k in range(_HFB):
        if bown[fp_b[i, k]] != i:
            w = wp.int32(0)
    won[i] = w


def pack_h_footprints(cands: List[Tuple[int, HCfgIdx]]):
    """Pack H-candidate footprints into fixed-width (N,9)/(N,10)/(N,5) int32 arrays. Canonical
    order: verts = 3 tri + (outer_top, outer_bot)*3 ; surfs = triangle + 3 side + 3 top + 3
    bottom ; bodies = 2 caps + 3 side. Matches schedule_csr.h_footprint element-for-element."""
    n = len(cands)
    fpv = np.empty((n, _HFV), np.int32)
    fps = np.empty((n, _HFS), np.int32)
    fpb = np.empty((n, _HFB), np.int32)
    for i, (_tri, cfg) in enumerate(cands):
        fpv[i] = list(cfg.tri_verts) + [x for a in cfg.arms for x in (a.outer_top, a.outer_bot)]
        fps[i] = ([cfg.triangle] + [a.side_surface for a in cfg.arms]
                  + list(cfg.top_faces.values()) + list(cfg.bottom_faces.values()))
        fpb[i] = [cfg.cap_top, cfg.cap_bot] + list(cfg.side_cells)
    return fpv, fps, fpb


def reserve_h_won_mask_warp(pm: PaddedMesh, cands: List[Tuple[int, HCfgIdx]],
                            device=None) -> np.ndarray:
    """One GPU reverse-reservation round; return the (N,) int32 won-mask. Mirror of
    reserve_won_mask_warp with the H footprint sizes (must equal h_reserve_won_mask_host)."""
    n = len(cands)
    if n == 0:
        return np.zeros(0, np.int32)
    if device is None:
        cuda = [d for d in wp.get_devices() if d.is_cuda]
        device = cuda[0] if cuda else "cpu"
    fpv, fps, fpb = pack_h_footprints(cands)
    a2 = lambda a: wp.array(np.ascontiguousarray(a), dtype=wp.int32, device=device)
    g_fpv, g_fps, g_fpb = a2(fpv), a2(fps), a2(fpb)
    vown = wp.array(np.full(pm.cap_v, n, np.int32), dtype=wp.int32, device=device)
    sown = wp.array(np.full(pm.cap_s, n, np.int32), dtype=wp.int32, device=device)
    bown = wp.array(np.full(pm.nb, n, np.int32), dtype=wp.int32, device=device)
    won = wp.zeros(n, dtype=wp.int32, device=device)
    wp.launch(reserve_h_kernel, dim=n, device=device,
              inputs=[g_fpv, g_fps, g_fpb, vown, sown, bown])
    wp.launch(check_h_kernel, dim=n, device=device,
              inputs=[g_fpv, g_fps, g_fpb, vown, sown, bown, won])
    wp.synchronize_device(device)
    return won.numpy()


def reserve_h_independent_set_warp(pm: PaddedMesh, cands: List[Tuple[int, HCfgIdx]],
                                   device=None) -> List[Tuple[int, HCfgIdx]]:
    """The reverse candidates that won one GPU reservation round (a conflict-free batch)."""
    mask = reserve_h_won_mask_warp(pm, cands, device)
    return [cands[i] for i in range(len(cands)) if mask[i]]


def reserve_h_independent_set_warp_g(g: dict, cands: List[Tuple[int, HCfgIdx]]
                                     ) -> List[Tuple[int, HCfgIdx]]:
    """reserve_h_independent_set_warp reading capacities from the device SoA `g` -- for the
    fully-on-device reverse sweep (no from_warp)."""
    caps = types.SimpleNamespace(cap_v=g["cap_v"], cap_s=g["cap_s"], nb=g["nb"])
    return reserve_h_independent_set_warp(caps, cands, device=g["device"])


# ======================================================================================
# Option A: DEVICE-RESIDENT reservation -- read the gather's packed device arrays directly
# (no host pack_footprints / ICfgIdx round-trip). The footprint is the SAME element set as
# pack_footprints (8 v / 9 s / 5 b for [I]; 9 v / 10 s / 5 b for [H]); the candidate id is the
# row index, which (because find_*_warp emits the canonical (v10,v11)/triangle order and the
# gather preserves it) equals the host candidate id -- so the lowest-id-wins winner SET matches
# the host reference, exactly as the gated round-1 fingerprint test requires. Invalid candidates
# (valid==0) neither reserve nor win, so launching over ALL M rows == the host's veto-filtered
# reservation (relative order among valid rows is preserved, and lowest-id-wins depends only on
# relative order -- no on-device compaction or sort needed).
# ======================================================================================
@wp.kernel
def reserve_i_device_kernel(
        valid: wp.array(dtype=wp.int32),
        v10: wp.array(dtype=wp.int32), v11: wp.array(dtype=wp.int32),
        cap_top: wp.array(dtype=wp.int32), cap_bot: wp.array(dtype=wp.int32),
        side: wp.array2d(dtype=wp.int32), arm_side: wp.array2d(dtype=wp.int32),
        arm_otop: wp.array2d(dtype=wp.int32), arm_obot: wp.array2d(dtype=wp.int32),
        top: wp.array2d(dtype=wp.int32), bot: wp.array2d(dtype=wp.int32),
        vown: wp.array(dtype=wp.int32), sown: wp.array(dtype=wp.int32),
        bown: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if valid[i] == 0:
        return
    wp.atomic_min(vown, v10[i], i)
    wp.atomic_min(vown, v11[i], i)
    for k in range(3):
        wp.atomic_min(vown, arm_otop[i, k], i)
        wp.atomic_min(vown, arm_obot[i, k], i)
        wp.atomic_min(sown, arm_side[i, k], i)
        wp.atomic_min(sown, top[i, k], i)
        wp.atomic_min(sown, bot[i, k], i)
        wp.atomic_min(bown, side[i, k], i)
    wp.atomic_min(bown, cap_top[i], i)
    wp.atomic_min(bown, cap_bot[i], i)


@wp.kernel
def check_i_device_kernel(
        valid: wp.array(dtype=wp.int32),
        v10: wp.array(dtype=wp.int32), v11: wp.array(dtype=wp.int32),
        cap_top: wp.array(dtype=wp.int32), cap_bot: wp.array(dtype=wp.int32),
        side: wp.array2d(dtype=wp.int32), arm_side: wp.array2d(dtype=wp.int32),
        arm_otop: wp.array2d(dtype=wp.int32), arm_obot: wp.array2d(dtype=wp.int32),
        top: wp.array2d(dtype=wp.int32), bot: wp.array2d(dtype=wp.int32),
        vown: wp.array(dtype=wp.int32), sown: wp.array(dtype=wp.int32),
        bown: wp.array(dtype=wp.int32), won: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if valid[i] == 0:
        won[i] = wp.int32(0)
        return
    w = wp.int32(1)
    if vown[v10[i]] != i:
        w = wp.int32(0)
    if vown[v11[i]] != i:
        w = wp.int32(0)
    for k in range(3):
        if vown[arm_otop[i, k]] != i:
            w = wp.int32(0)
        if vown[arm_obot[i, k]] != i:
            w = wp.int32(0)
        if sown[arm_side[i, k]] != i:
            w = wp.int32(0)
        if sown[top[i, k]] != i:
            w = wp.int32(0)
        if sown[bot[i, k]] != i:
            w = wp.int32(0)
        if bown[side[i, k]] != i:
            w = wp.int32(0)
    if bown[cap_top[i]] != i:
        w = wp.int32(0)
    if bown[cap_bot[i]] != i:
        w = wp.int32(0)
    won[i] = w


def reserve_i_won_device(g: dict, gathered: dict):
    """One GPU reservation round over the gather's packed device arrays; return the (M,) int32
    `won` device mask (1 = a conflict-free winner). Owners are sized to the mesh capacity and
    seeded to M (a sentinel above every candidate id). No host data leaves the device."""
    dev = g["device"]
    m = int(gathered["valid"].shape[0])
    if m == 0:
        return wp.zeros(0, dtype=wp.int32, device=dev)
    vown = wp.full(int(g["cap_v"]), m, dtype=wp.int32, device=dev)
    sown = wp.full(int(g["cap_s"]), m, dtype=wp.int32, device=dev)
    bown = wp.full(int(g["nb"]), m, dtype=wp.int32, device=dev)
    won = wp.zeros(m, dtype=wp.int32, device=dev)
    args = [gathered["valid"], gathered["v10"], gathered["v11"], gathered["cap_top"],
            gathered["cap_bot"], gathered["side"], gathered["arm_side"], gathered["arm_otop"],
            gathered["arm_obot"], gathered["top"], gathered["bot"], vown, sown, bown]
    wp.launch(reserve_i_device_kernel, dim=m, device=dev, inputs=args)
    wp.launch(check_i_device_kernel, dim=m, device=dev, inputs=args + [won])
    return won


@wp.kernel
def reserve_h_device_kernel(
        valid: wp.array(dtype=wp.int32), tri_cand: wp.array(dtype=wp.int32),
        cap_top: wp.array(dtype=wp.int32), cap_bot: wp.array(dtype=wp.int32),
        tri: wp.array2d(dtype=wp.int32), side: wp.array2d(dtype=wp.int32),
        arm_side: wp.array2d(dtype=wp.int32), arm_otop: wp.array2d(dtype=wp.int32),
        arm_obot: wp.array2d(dtype=wp.int32),
        top: wp.array2d(dtype=wp.int32), bot: wp.array2d(dtype=wp.int32),
        vown: wp.array(dtype=wp.int32), sown: wp.array(dtype=wp.int32),
        bown: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if valid[i] == 0:
        return
    wp.atomic_min(sown, tri_cand[i], i)                  # the triangle itself joins the footprint
    for k in range(3):
        wp.atomic_min(vown, tri[i, k], i)
        wp.atomic_min(vown, arm_otop[i, k], i)
        wp.atomic_min(vown, arm_obot[i, k], i)
        wp.atomic_min(sown, arm_side[i, k], i)
        wp.atomic_min(sown, top[i, k], i)
        wp.atomic_min(sown, bot[i, k], i)
        wp.atomic_min(bown, side[i, k], i)
    wp.atomic_min(bown, cap_top[i], i)
    wp.atomic_min(bown, cap_bot[i], i)


@wp.kernel
def check_h_device_kernel(
        valid: wp.array(dtype=wp.int32), tri_cand: wp.array(dtype=wp.int32),
        cap_top: wp.array(dtype=wp.int32), cap_bot: wp.array(dtype=wp.int32),
        tri: wp.array2d(dtype=wp.int32), side: wp.array2d(dtype=wp.int32),
        arm_side: wp.array2d(dtype=wp.int32), arm_otop: wp.array2d(dtype=wp.int32),
        arm_obot: wp.array2d(dtype=wp.int32),
        top: wp.array2d(dtype=wp.int32), bot: wp.array2d(dtype=wp.int32),
        vown: wp.array(dtype=wp.int32), sown: wp.array(dtype=wp.int32),
        bown: wp.array(dtype=wp.int32), won: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if valid[i] == 0:
        won[i] = wp.int32(0)
        return
    w = wp.int32(1)
    if sown[tri_cand[i]] != i:
        w = wp.int32(0)
    for k in range(3):
        if vown[tri[i, k]] != i:
            w = wp.int32(0)
        if vown[arm_otop[i, k]] != i:
            w = wp.int32(0)
        if vown[arm_obot[i, k]] != i:
            w = wp.int32(0)
        if sown[arm_side[i, k]] != i:
            w = wp.int32(0)
        if sown[top[i, k]] != i:
            w = wp.int32(0)
        if sown[bot[i, k]] != i:
            w = wp.int32(0)
        if bown[side[i, k]] != i:
            w = wp.int32(0)
    if bown[cap_top[i]] != i:
        w = wp.int32(0)
    if bown[cap_bot[i]] != i:
        w = wp.int32(0)
    won[i] = w


def reserve_h_won_device(g: dict, gathered: dict):
    """One GPU reverse-reservation round over the [H]-gather's packed device arrays; return the
    (M,) int32 `won` device mask. Mirror of reserve_i_won_device with the H footprint (the
    triangle itself joins the surface footprint)."""
    dev = g["device"]
    m = int(gathered["valid"].shape[0])
    if m == 0:
        return wp.zeros(0, dtype=wp.int32, device=dev)
    vown = wp.full(int(g["cap_v"]), m, dtype=wp.int32, device=dev)
    sown = wp.full(int(g["cap_s"]), m, dtype=wp.int32, device=dev)
    bown = wp.full(int(g["nb"]), m, dtype=wp.int32, device=dev)
    won = wp.zeros(m, dtype=wp.int32, device=dev)
    args = [gathered["valid"], gathered["tri_cand"], gathered["cap_top"], gathered["cap_bot"],
            gathered["tri"], gathered["side"], gathered["arm_side"], gathered["arm_otop"],
            gathered["arm_obot"], gathered["top"], gathered["bot"], vown, sown, bown]
    wp.launch(reserve_h_device_kernel, dim=m, device=dev, inputs=args)
    wp.launch(check_h_device_kernel, dim=m, device=dev, inputs=args + [won])
    return won


# ======================================================================================
# C2 glue: the iterated independent-set I->H sweep, run on the GPU
# ======================================================================================
def reconnect_sweep_warp(g: dict, threshold: float, dl_th: Optional[float] = None,
                         veto: bool = True, max_rounds: int = 64,
                         gpu_scan: bool = False) -> Dict[str, object]:
    """The cellGPU iterated-batch loop assembled end-to-end on the device. Each round:

      1. DETECT  -- sync the device SoA back to a host PaddedMesh (PaddedMesh.from_warp,
                    slot-preserving) and scan it for short [I] edges + Condition-4 veto.
                    With `gpu_scan=True` the O(mesh) scan runs on the GPU
                    (detect_warp.detect_short_edges_hybrid: parallel trigger kernel + host
                    gather on the few candidates) instead of the host Python scan
                    find_short_edges_csr; the two produce the SAME sites in the SAME order, so
                    it is a drop-in. (The mesh always comes from `g`, so it reflects the
                    surgery of all prior rounds.)
      2. RESERVE -- the GPU atomic lowest-id-wins reservation (C2a) selects a conflict-free
                    batch from the candidates, on the device.
      3. APPLY   -- the GPU parallel count-changing surgery (C2b, apply_i_to_h_batch_warp)
                    runs every winner SIMULTANEOUSLY, mutating `g` in place.
      4. ITERATE -- re-detect on the mutated mesh; stop when no legal short edge remains
                    (or max_rounds).

    `g` is a PaddedMesh.to_warp() dict, mutated in place. `dl_th` defaults to `threshold`;
    `veto` toggles the Condition-4 filter. Returns a report (rounds, per-round batch sizes,
    total reconnections), matching reconnect_sweep_reserve_host's shape.

    CAPACITY: the bump allocator never reclaims (+3 verts / +1 surf per I->H), so `g` must
    be sized with enough headroom for the WHOLE sweep -- Gate D (stream-compaction) lifts
    this. CASCADE (C1 finding): a static-mesh sweep does NOT converge (an I->H seeds new
    short edges among the triangle verts); production relaxes the mesh between steps, so
    bound `max_rounds`.

    EQUIVALENCE TO THE HOST (reconnect_sweep_reserve_host): round 1 starts from the same
    slot layout as the host reference, so detection + reservation are identical (C2a is
    bit-for-bit) and the parallel apply matches the host sequential apply by body-anchored
    fingerprint (C2b). ACROSS rounds the device's atomic-bump slot order diverges from the
    host's sequential order (same topology, different slot labels), so only the FIRST round
    is bit/fingerprint-equal; later rounds are validated for consistency, like the C1 sweep.
    """
    if dl_th is None:
        dl_th = threshold
    dev = g["device"]
    total = 0
    round_sizes: List[int] = []
    rounds = 0
    while rounds < max_rounds:
        pm = PaddedMesh.from_warp(g)                 # device -> host, slot-preserving
        sites = (detect_short_edges_hybrid(g, pm, threshold) if gpu_scan
                 else find_short_edges_csr(pm, threshold))
        sites.sort(key=lambda s: (s[0], s[1]))       # canonical order -> deterministic reservation
        if veto:
            sites = [s for s in sites if i_to_h_veto_csr(pm, s[2]) is None]
        if not sites:
            break
        winners = reserve_independent_set_warp(pm, sites, device=dev)
        if not winners:                              # defensive: candidate 0 always wins
            break
        apply_i_to_h_batch_warp(g, winners, dl_th)
        total += len(winners)
        round_sizes.append(len(winners))
        rounds += 1
    return dict(total=total, rounds=rounds, round_sizes=round_sizes,
                converged=(rounds < max_rounds))


def reconnect_sweep_warp_device(g: dict, threshold: float, dl_th: Optional[float] = None,
                                max_rounds: int = 64) -> Dict[str, object]:
    """FULLY-ON-DEVICE forward I->H sweep (Option A: device-resident). Each round:

      1. DETECT  -- find_short_edges_device: GPU per-surface scan + interior filter + an ON-DEVICE
                    dedup/lex-sort (int64-key radix_sort + array_scan, reproducing np.unique(axis=0));
                    the candidate edges stay on the device (canonical (v10,v11)-ascending so the
                    lowest-id-wins reservation is deterministic) and only the scalar count M is read
                    back. M==0 -> EMPTY-STEP FAST PATH: skip the gather/reserve/apply entirely.
      2. GATHER  -- gather_i_configs_warp: per-candidate [I]-neighbourhood + fused Condition-4 veto,
                    emitting PACKED DEVICE ARRAYS (valid/caps/side/arms/top/bot). These stay on the
                    device -- they are NOT round-tripped to host ICfgIdx objects.
      3. RESERVE -- reserve_i_won_device: the C2a atomic lowest-id-wins reservation reads those
                    device arrays directly (invalid rows neither reserve nor win). The ONLY host
                    readback in the loop is the 1-value winner COUNT (won.sum()).
      4. APPLY   -- apply_i_to_h_device_warp: every winner's count-changing surgery runs in
                    parallel from the same device arrays, with no output readback.
      5. ITERATE -- re-detect on the mutated device mesh; stop on no legal short edge / max_rounds.

    Replaces the earlier per-candidate host round-trip (gather->ICfgIdx list->pack_footprints->
    col/mat apply packing) -- ~12 device<->host syncs/round collapsed to ~2 -- with NO change to
    the math: the footprint element set, the reservation selection, and the Okuda placement +
    surgery are identical to the host-packed path, so round 1 stays fingerprint-equal to the
    host-scan sweep (reconnect_sweep_warp) and the gated equivalence tests hold. Same
    cascade/headroom caveats as reconnect_sweep_warp; later rounds may pick a different (also
    valid) batch as the device slot order diverges."""
    if dl_th is None:
        dl_th = threshold
    dev = g["device"]
    total = 0
    round_sizes: List[int] = []
    rounds = 0
    while rounds < max_rounds:
        c_v10, c_v11, m = find_short_edges_device(g, threshold)   # DEVICE (v10,v11)-ascending + count M
        if m == 0:                                        # empty-step fast path (no host candidate copy)
            break
        buf = _ensure_gather_buf(g, "_i_gather_buf", m)   # reused scratch (no per-round alloc/fill)
        gathered = gather_i_configs_warp_device(g, c_v10, c_v11, m, device=dev, buf=buf)  # device-in: no h2d
        won = reserve_i_won_device(g, gathered)           # device won mask (skips invalid rows)
        wp.synchronize_device(dev)
        n_win = int(won.numpy().sum())                    # winner count (+ M in detect = the 2 round readbacks)
        if n_win == 0:                                    # all candidates vetoed / lost -> done
            break
        apply_i_to_h_device_warp(g, gathered, won, dl_th)  # parallel surgery from device arrays
        total += n_win
        round_sizes.append(n_win)
        rounds += 1
    return dict(total=total, rounds=rounds, round_sizes=round_sizes,
                converged=(rounds < max_rounds))


def reconnect_sweep_h_to_i_warp(g: dict, threshold: float, dl_th: Optional[float] = None,
                                veto: bool = True, max_rounds: int = 64,
                                gpu_scan: bool = False) -> Dict[str, object]:
    """The reverse (H->I) mirror of reconnect_sweep_warp, assembled end-to-end on the device.
    Each round: DETECT small triangles + Condition-4 veto -> RESERVE a conflict-free batch via
    the GPU lowest-id-wins H-reservation (C2') -> APPLY every winner's H->I SIMULTANEOUSLY
    (apply_h_to_i_batch_warp, mutating `g`) -> ITERATE. With `gpu_scan=True` the O(mesh) scan
    runs on the GPU (detect_warp.detect_small_triangles_hybrid) instead of the host
    find_small_triangles_csr -- a drop-in (same sites, same order). Same capacity/headroom +
    per-round host-equivalence caveats as reconnect_sweep_warp (round 1 matches the host
    reservation mirror reconnect_sweep_h_reserve_host bit/fingerprint; later for consistency)."""
    if dl_th is None:
        dl_th = threshold
    dev = g["device"]
    total = 0
    round_sizes: List[int] = []
    rounds = 0
    while rounds < max_rounds:
        pm = PaddedMesh.from_warp(g)                 # device -> host, slot-preserving
        sites = (detect_small_triangles_hybrid(g, pm, threshold) if gpu_scan
                 else find_small_triangles_csr(pm, threshold))
        sites.sort(key=lambda s: s[0])               # canonical order -> deterministic reservation
        if veto:
            sites = [s for s in sites if h_to_i_veto_csr(pm, s[1]) is None]
        if not sites:
            break
        winners = reserve_h_independent_set_warp(pm, sites, device=dev)
        if not winners:                              # defensive: candidate 0 always wins
            break
        apply_h_to_i_batch_warp(g, winners, dl_th)
        total += len(winners)
        round_sizes.append(len(winners))
        rounds += 1
    return dict(total=total, rounds=rounds, round_sizes=round_sizes,
                converged=(rounds < max_rounds))


def reconnect_sweep_h_to_i_warp_device(g: dict, threshold: float, dl_th: Optional[float] = None,
                                       max_rounds: int = 64) -> Dict[str, object]:
    """FULLY-ON-DEVICE reverse H->I sweep (Option A: device-resident; the mirror of
    reconnect_sweep_warp_device). Each round: find_small_triangles_warp emits the small candidate
    list in canonical (triangle-ascending) order; gather_h_configs_warp emits packed DEVICE arrays
    (Condition-4 veto fused); reserve_h_won_device runs the C2' lowest-id-wins reservation on those
    arrays (the only host readback is the winner count); apply_h_to_i_device_warp runs every
    winner's H->I in parallel from the device arrays. No PaddedMesh.from_warp(g), no per-candidate
    host round-trip. Math unchanged vs the host-packed path -> round 1 stays fingerprint-equal to
    reconnect_sweep_h_to_i_warp. Same cascade/headroom caveats."""
    if dl_th is None:
        dl_th = threshold
    dev = g["device"]
    total = 0
    round_sizes: List[int] = []
    rounds = 0
    while rounds < max_rounds:
        tris = find_small_triangles_warp(g, threshold)    # triangle-ascending; only host data
        if len(tris) == 0:                                # empty-step fast path
            break
        buf = _ensure_gather_buf(g, "_h_gather_buf", len(tris), with_tri=True)  # reused scratch
        gathered = gather_h_configs_warp(g, tris, device=dev, buf=buf)    # packed DEVICE arrays + fused veto
        won = reserve_h_won_device(g, gathered)           # device won mask (skips invalid rows)
        wp.synchronize_device(dev)
        n_win = int(won.numpy().sum())                    # the loop's only readback: winner count
        if n_win == 0:
            break
        apply_h_to_i_device_warp(g, gathered, won, dl_th)  # parallel surgery from device arrays
        total += n_win
        round_sizes.append(n_win)
        rounds += 1
    return dict(total=total, rounds=rounds, round_sizes=round_sizes,
                converged=(rounds < max_rounds))
