"""Fixed-dim, device-`M`-masked reconnect path + a CUDA-graph-captured forward step.

The production reconnect sweeps (schedule_warp.reconnect_sweep_*_warp_device) are variable-round
host-driven loops: each round reads the candidate count `M` and the winner count back to the host
to decide whether to continue. CUDA-graph capture forbids host syncs (and allocations) inside the
captured region, so this module re-expresses ONE reconnect round over FIXED launch dims with a
DEVICE-scalar `M` masking the unused tail -- no host readback. A captured graph then runs a FIXED
`max_rounds` of these (empty rounds are all-threads-early-return no-ops), reaching the SAME converged
state as the variable loop (extra rounds mutate nothing) -> byte-identical, capturable with a plain
`wp.ScopedCapture`.

Provenance + the byte-identicality proof: docs/2026-06-26_cuda-graph-experiment-scope.md (P3 SIMPLI-
FICATION + Bit-identicality). Validated against production by the fixed-dim winner-equality protos
(scratchpad/proto_fixeddim_detect.py, proto_capture_round.py: I+H winners byte-identical over a real
reconnecting trajectory) and, end-to-end, by the 2k/20k byte-identical trajectory gate (a fixed-dim
forward step reproduces gpu_stability's reference timeline).

This module ONLY ADDS to rnr/gpu: it imports the production kernels/helpers and assembles them; the
detect/gather/reserve/apply/compact kernels themselves are UNMODIFIED, so the existing variable-round
path and the 134-test gate are untouched by anything here. The only new kernels are 5 tiny device-side
helpers (3 device-`M` guards -- a guarded interior filter + two tail masks the fixed dims need -- plus
a MAX_CAND-overflow flag and an under-convergence flag); everything else is reuse.

Key seams vs the variable path (each a host sync the fixed path removes):
  * detect's `M` -> a device scalar (out_pos[CAP-1] for I, count for H), never read to host;
  * the round loop -> a FIXED `max_rounds` unrolled (no `break` on M==0 / n_win==0);
  * reserve owners (vown/sown/bown/won) -> PRE-ALLOCATED on `g` (the last per-round allocs);
  * compact/orient -> UNCONDITIONAL on a reconnect step (compact is a no-op on a gap-free mesh,
    orient is idempotent on a clean one -> byte-identical to production's `if (ni+nh)>0` gate).
"""
from typing import Optional

import warp as wp
from warp.utils import radix_sort_pairs

from . import physics_warp as W
from .compact_warp import compact_warp
from .detect_warp import (_ensure_detect_buf, _ensure_tri_buf, build_short_edge_keys_kernel,
                          d_vert_body_count, mark_first_kernel, scan_short_edges_kernel,
                          scan_small_triangles_kernel, scatter_unique_kernel)
from .gather_warp import _ensure_gather_buf, gather_h_kernel, gather_i_kernel
from .orient_warp import (_body_closure_kernel, _flip_apply_kernel, _flip_mark_kernel,
                          orient_repair_warp)
from .reconnect_warp import _box_of, apply_h_to_i_won_kernel, apply_i_to_h_won_kernel
from .schedule_warp import (check_h_device_kernel, check_i_device_kernel, reserve_h_device_kernel,
                            reserve_i_device_kernel)

# Upper bound on DISTINCT candidates/round the fixed launches cover. The gather/reserve/apply run
# over MAX_CAND rows and self-mask the [M, MAX_CAND) tail; raw deduped M is ~27 at n=10 and grows
# with cell count, so this must upper-bound M at the run's scale (verify at n>=16 -- the overflow
# flag below trips the gate on an exceedance rather than silently dropping candidates).
DEFAULT_MAX_CAND = 512
_H_SENTINEL = 1 << 30   # > any surface index, < int32 max -> stale H tail sorts LAST, then clamped


# --------------------------------------------------------------------------------------
# device-M guard kernels (the only NEW kernels; everything else is reused unmodified)
# --------------------------------------------------------------------------------------
@wp.kernel
def filter_interior_guarded_kernel(
        cand_v10: wp.array(dtype=wp.int32),
        v2s: wp.array2d(dtype=wp.int32), v2s_len: wp.array(dtype=wp.int32),
        s2b: wp.array2d(dtype=wp.int32), keep: wp.array(dtype=wp.int32),
        count: wp.array(dtype=wp.int32)):
    """filter_interior_short_edges_kernel + a DEVICE-count guard: rows tid >= count[0] are the stale
    tail of the FIXED-dim launch -> keep=0 (build_short_edge_keys then sentinels them so the lex sort
    drops them and mark_first never emits them). Rows < count[0] get the exact production interior
    test (4 incident bodies). No host read of count -- the guard replaces the host `dim=k` launch."""
    i = wp.tid()
    if i >= count[0]:
        keep[i] = wp.int32(0)
        return
    if d_vert_body_count(v2s, v2s_len, s2b, cand_v10[i]) == 4:
        keep[i] = wp.int32(1)
    else:
        keep[i] = wp.int32(0)


@wp.kernel
def mask_tail_valid_kernel(valid: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32)):
    """Zero valid[tid] for the stale tail rows tid >= count[0] of a fixed-dim gather (device-M
    scalar; no host read). Rows < count keep the gather's own valid verdict -> reserve/apply, which
    already self-skip valid==0, then ignore the tail. The fixed-dim analogue of launching gather over
    the live M."""
    i = wp.tid()
    if i >= count[0]:
        valid[i] = wp.int32(0)


@wp.kernel
def clamp_tail_kernel(c_tris: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32)):
    """Set c_tris[tid]=0 for the stale/sentinel H tail tid >= count[0] so the fixed-dim gather_h
    reads a SAFE surface index (0) there instead of the sort sentinel (1<<30 -> OOB). mask_tail_valid
    then drops those rows' valid. Device-M scalar; no host read."""
    i = wp.tid()
    if i >= count[0]:
        c_tris[i] = wp.int32(0)


@wp.kernel
def flag_overflow_kernel(count: wp.array(dtype=wp.int32), max_cand: wp.int32,
                         ovf: wp.array(dtype=wp.int32)):
    """If the round's candidate count exceeds MAX_CAND, raise the device overflow flag (no host
    read). The fixed launches cover only [0, MAX_CAND); an exceedance would silently drop the
    overflow candidates and break bit-identicality, so check_overflow() reads this flag (one sync,
    off the hot loop / post-replay) and asserts it stayed 0."""
    if count[0] > max_cand:
        ovf[0] = wp.int32(1)


@wp.kernel
def flag_if_any_won_kernel(won: wp.array(dtype=wp.int32), flag: wp.array(dtype=wp.int32)):
    """Raise `flag` if the LAST fixed round still applied a winner (any won[i]!=0). A non-empty last
    round means the variable sweep would NOT yet have broken (it breaks iff m==0 or n_win==0, both =>
    won.sum()==0), so `max_rounds` was too small and the fixed-R result may DIVERGE from the
    variable-round reference. Conversely an UNSET flag is EXACTLY the variable break condition => the
    fixed sweep converged within max_rounds => byte-identical. The under-convergence guard for a
    perf-tuned (small) max_rounds; check_underconverged() reads it (one sync, post-replay). Also reused
    to drive the capture_while loop condition (cond = this round applied a winner)."""
    i = wp.tid()
    if won[i] != 0:
        wp.atomic_max(flag, 0, wp.int32(1))


@wp.kernel
def _serial_inclusive_scan_kernel(src: wp.array(dtype=wp.int32), n: wp.int32,
                                  out: wp.array(dtype=wp.int32)):
    """Single-thread inclusive prefix sum over src[0:n] -> out[0:n]. Replaces warp.utils.array_scan,
    whose CUB call allocates a workspace that a `capture_while` conditional graph node REJECTS
    ("unsupported operation (memory allocation)") -- the ONLY such op in the reconnect round (radix_
    sort_pairs/copy/slicing all capture fine; isolated in scratchpad/proto_while_isolate.py). O(n)
    serial in one thread, n=CAP≈8192 small (~µs, off the round's critical path), and byte-identical to
    array_scan (same inclusive prefix sum)."""
    if wp.tid() != 0:
        return
    acc = wp.int32(0)
    for i in range(n):
        acc = acc + src[i]
        out[i] = acc


@wp.kernel
def _while_cap_kernel(counter: wp.array(dtype=wp.int32), max_rounds: wp.int32,
                      cond: wp.array(dtype=wp.int32)):
    """Round-count cap for the capture_while sweep: count this completed round and force cond->0 once
    `max_rounds` rounds are done. Matches the variable sweep's `while rounds < max_rounds` cap exactly
    AND is a hard infinite-loop backstop (the loop otherwise stops naturally when a round applies
    nothing). At <=2 real rounds this never fires, so capture_while == variable-round byte-for-byte."""
    r = counter[0] + wp.int32(1)
    counter[0] = r
    if r >= max_rounds:
        cond[0] = wp.int32(0)


# --------------------------------------------------------------------------------------
# pre-allocated capture-region buffers on `g` (no allocation may happen inside a capture)
# --------------------------------------------------------------------------------------
def _ensure_reserve_owners(g: dict, max_cand: int) -> dict:
    """Pre-allocate the reservation owner arrays (vown/sown/bown sized to the mesh; won_i/won_h sized
    MAX_CAND) ONCE on g['_res_own'] -- they are the last per-round allocs in the production reserve
    (reserve_i_won_device does wp.full/wp.zeros each call). Filled/zeroed in place each round."""
    own = g.get("_res_own")
    if own is not None and own["max_cand"] >= max_cand:
        return own
    dev = g["device"]
    own = dict(max_cand=max_cand,
               vown=wp.zeros(int(g["cap_v"]), dtype=wp.int32, device=dev),
               sown=wp.zeros(int(g["cap_s"]), dtype=wp.int32, device=dev),
               bown=wp.zeros(int(g["nb"]), dtype=wp.int32, device=dev),
               won_i=wp.zeros(max_cand, dtype=wp.int32, device=dev),
               won_h=wp.zeros(max_cand, dtype=wp.int32, device=dev))
    g["_res_own"] = own
    return own


def _ensure_overflow_flag(g: dict) -> wp.array:
    if "_ovf" not in g:
        g["_ovf"] = wp.zeros(1, dtype=wp.int32, device=g["device"])
    return g["_ovf"]


def _ensure_underconv_flag(g: dict) -> wp.array:
    if "_underconv" not in g:
        g["_underconv"] = wp.zeros(1, dtype=wp.int32, device=g["device"])
    return g["_underconv"]


def check_overflow(g: dict, reset: bool = True) -> bool:
    """Read the device overflow flag to host (ONE sync; call post-replay or every ~N steps, NOT in
    the captured region). True => some round's candidate count exceeded MAX_CAND -> the fixed path
    silently dropped candidates and is NOT bit-identical -> raise MAX_CAND and re-run."""
    if "_ovf" not in g:
        return False
    over = bool(int(g["_ovf"].numpy()[0]) != 0)
    if reset:
        g["_ovf"].zero_()
    return over


def check_underconverged(g: dict, reset: bool = True) -> bool:
    """Read the device under-convergence flag (ONE sync; post-replay / periodic). True => some
    sweep's LAST fixed round still applied a winner -> max_rounds was too small -> the fixed-R
    trajectory may DIVERGE from the variable-round reference -> raise max_rounds + re-capture.
    A False return GUARANTEES every sweep hit the variable path's break condition within max_rounds
    (byte-identical). This is the safety net that lets a perf run pick a small max_rounds."""
    if "_underconv" not in g:
        return False
    under = bool(int(g["_underconv"].numpy()[0]) != 0)
    if reset:
        g["_underconv"].zero_()
    return under


def _M_dev_i(g: dict) -> wp.array:
    """Copy the I-detect device-scalar M (out_pos[CAP-1], the inclusive-scan total) into a stable
    1-int array on g (capture-safe: a fixed device address the round's mask kernels read). NB: do not
    use g.setdefault(k, wp.zeros(...)) -- Python evaluates the default EVERY call, allocating inside
    the capture region; check existence first."""
    if "_M_dev_i" not in g:
        g["_M_dev_i"] = wp.zeros(1, dtype=wp.int32, device=g["device"])
    md = g["_M_dev_i"]
    b = g["_detect_buf"]
    wp.copy(md, b["out_pos"][b["cap"] - 1:b["cap"]])
    return md


# --------------------------------------------------------------------------------------
# fixed-dim, device-M detect (the body of find_short_edges_device / find_small_triangles_device
# with every post-scan launch over the FIXED buffer cap + the host k/M read removed)
# --------------------------------------------------------------------------------------
def _detect_i_fixed(g: dict, threshold: float, max_cand: int):
    """Fixed-dim I-detect: scan over cap_s, then filter/build_keys/sort/mark/scan/scatter over the
    FIXED detect-buffer cap with the guarded interior filter masking the [count, CAP) tail. M lands
    in out_pos[CAP-1] (device scalar). Byte-identical candidate set+order to find_short_edges_device
    (the sentinel tail is never emitted; see scatter/mark_first). Returns (cand_v10, cand_v11)."""
    dev = g["device"]
    cap_s = g["cap_s"]
    buf = _ensure_detect_buf(g, max_cand)    # CAP >= max_cand so cand_v10[:max_cand] (gather) is in-bounds
    CAP = buf["cap"]
    buf["count"].zero_()
    wp.launch(scan_short_edges_kernel, dim=cap_s, device=dev, inputs=[
        g["vert_pos"], g["vert_alive"], g["surf_alive"], g["s2v"], g["s2v_len"],
        wp.float64(threshold), buf["out_v10"], buf["out_v11"], buf["count"]])
    # --- fixed-dim from here (no host read of k) ---
    wp.launch(filter_interior_guarded_kernel, dim=CAP, device=dev, inputs=[
        buf["out_v10"], g["v2s"], g["v2s_len"], g["s2b"], buf["keep"], buf["count"]])
    wp.launch(build_short_edge_keys_kernel, dim=CAP, device=dev, inputs=[
        buf["out_v10"], buf["out_v11"], buf["keep"], buf["keys"], buf["values"]])
    radix_sort_pairs(buf["keys"], buf["values"], CAP)          # FIXED count = CAP (capture-safe)
    wp.launch(mark_first_kernel, dim=CAP, device=dev, inputs=[buf["keys"], buf["is_first"]])
    wp.launch(_serial_inclusive_scan_kernel, dim=1, device=dev,    # capture_while-safe scan (no alloc)
              inputs=[buf["is_first"], wp.int32(CAP), buf["out_pos"]])
    wp.launch(scatter_unique_kernel, dim=CAP, device=dev, inputs=[
        buf["values"], buf["is_first"], buf["out_pos"], buf["out_v10"], buf["out_v11"],
        buf["cand_v10"], buf["cand_v11"]])
    return buf["cand_v10"], buf["cand_v11"]


def _detect_h_fixed(g: dict, threshold: float, max_cand: int):
    """Fixed-dim H-detect: sentinel-pad keys (stale tail sorts last), scan over cap_s, sort over
    cap_s, then clamp the [count, MAX_CAND) tail to surface 0 (gather-safe). count is the device-M
    scalar. Byte-identical first-M order to find_small_triangles_device. Returns (c_tris, count_dev)."""
    dev = g["device"]
    cap_s = g["cap_s"]
    buf = _ensure_tri_buf(g, cap_s)
    buf["keys"].fill_(_H_SENTINEL)                            # stale tail -> sentinel (sorts last)
    buf["count"].zero_()
    wp.launch(scan_small_triangles_kernel, dim=cap_s, device=dev, inputs=[
        g["vert_pos"], g["surf_alive"], g["s2v"], g["s2v_len"], g["s2b"],
        wp.float64(threshold), buf["keys"], buf["count"]])
    radix_sort_pairs(buf["keys"], buf["values"], cap_s)      # FIXED count = cap_s (real keys first)
    wp.launch(clamp_tail_kernel, dim=max_cand, device=dev, inputs=[buf["keys"], buf["count"]])
    return buf["keys"], buf["count"]


# --------------------------------------------------------------------------------------
# fixed-dim, device-M reconnect rounds (detect -> gather -> mask -> reserve -> check -> apply)
# all launched over MAX_CAND; reserve/apply self-skip the masked tail. No host reads -> capturable.
# --------------------------------------------------------------------------------------
def _round_i_fixed(g: dict, owners: dict, threshold: float, dl_th: float, max_cand: int) -> None:
    """One fixed-dim I->H reconnect round. Mutates g (positions + topology of the won candidates)."""
    dev = g["device"]
    cand_v10, cand_v11 = _detect_i_fixed(g, threshold, max_cand)
    md = _M_dev_i(g)
    wp.launch(flag_overflow_kernel, dim=1, device=dev,
              inputs=[md, wp.int32(max_cand), _ensure_overflow_flag(g)])
    gbuf = _ensure_gather_buf(g, "_fx_gather_buf_i", max_cand)
    wp.launch(gather_i_kernel, dim=max_cand, device=dev, inputs=[
        g["vert_alive"], g["v2s"], g["v2s_len"], g["surf_alive"], g["s2v"], g["s2v_len"],
        g["s2b"], g["b2s"], g["b2s_len"], cand_v10, cand_v11,
        gbuf["valid"], gbuf["cap_top"], gbuf["cap_bot"], gbuf["side"],
        gbuf["arm_side"], gbuf["arm_otop"], gbuf["arm_obot"], gbuf["top"], gbuf["bot"]])
    wp.launch(mask_tail_valid_kernel, dim=max_cand, device=dev, inputs=[gbuf["valid"], md])
    vown, sown, bown, won = owners["vown"], owners["sown"], owners["bown"], owners["won_i"]
    vown.fill_(max_cand); sown.fill_(max_cand); bown.fill_(max_cand); won.zero_()
    args = [gbuf["valid"], cand_v10[:max_cand], cand_v11[:max_cand], gbuf["cap_top"], gbuf["cap_bot"],
            gbuf["side"], gbuf["arm_side"], gbuf["arm_otop"], gbuf["arm_obot"], gbuf["top"], gbuf["bot"],
            vown, sown, bown]
    wp.launch(reserve_i_device_kernel, dim=max_cand, device=dev, inputs=args)
    wp.launch(check_i_device_kernel, dim=max_cand, device=dev, inputs=args + [won])
    wp.launch(apply_i_to_h_won_kernel, dim=max_cand, device=dev, inputs=[
        g["vert_pos"], g["vert_alive"], g["v2s"], g["v2s_len"], g["surf_alive"],
        g["s2v"], g["s2v_len"], g["s2b"], g["b2s"], g["b2s_len"], g["n_used"], float(dl_th), _box_of(g),
        won, cand_v10[:max_cand], cand_v11[:max_cand], gbuf["cap_top"], gbuf["cap_bot"],
        gbuf["arm_side"], gbuf["arm_otop"], gbuf["arm_obot"], gbuf["top"], gbuf["bot"]])


def _round_h_fixed(g: dict, owners: dict, threshold: float, dl_th: float, max_cand: int) -> None:
    """One fixed-dim H->I reconnect round (mirror of _round_i_fixed; no dedup, M==raw count)."""
    dev = g["device"]
    c_tris, count_dev = _detect_h_fixed(g, threshold, max_cand)
    wp.launch(flag_overflow_kernel, dim=1, device=dev,
              inputs=[count_dev, wp.int32(max_cand), _ensure_overflow_flag(g)])
    gbuf = _ensure_gather_buf(g, "_fx_gather_buf_h", max_cand, with_tri=True)
    wp.launch(gather_h_kernel, dim=max_cand, device=dev, inputs=[
        g["vert_alive"], g["v2s"], g["v2s_len"], g["surf_alive"], g["s2v"], g["s2v_len"],
        g["s2b"], g["b2s"], g["b2s_len"], c_tris[:max_cand],
        gbuf["valid"], gbuf["cap_top"], gbuf["cap_bot"], gbuf["tri"], gbuf["side"],
        gbuf["arm_side"], gbuf["arm_otop"], gbuf["arm_obot"], gbuf["top"], gbuf["bot"]])
    wp.launch(mask_tail_valid_kernel, dim=max_cand, device=dev, inputs=[gbuf["valid"], count_dev])
    vown, sown, bown, won = owners["vown"], owners["sown"], owners["bown"], owners["won_h"]
    vown.fill_(max_cand); sown.fill_(max_cand); bown.fill_(max_cand); won.zero_()
    args = [gbuf["valid"][:max_cand], c_tris[:max_cand], gbuf["cap_top"][:max_cand],
            gbuf["cap_bot"][:max_cand], gbuf["tri"][:max_cand], gbuf["side"][:max_cand],
            gbuf["arm_side"][:max_cand], gbuf["arm_otop"][:max_cand], gbuf["arm_obot"][:max_cand],
            gbuf["top"][:max_cand], gbuf["bot"][:max_cand], vown, sown, bown]
    wp.launch(reserve_h_device_kernel, dim=max_cand, device=dev, inputs=args)
    wp.launch(check_h_device_kernel, dim=max_cand, device=dev, inputs=args + [won])
    wp.launch(apply_h_to_i_won_kernel, dim=max_cand, device=dev, inputs=[
        g["vert_pos"], g["vert_alive"], g["v2s"], g["v2s_len"], g["surf_alive"],
        g["s2v"], g["s2v_len"], g["s2b"], g["b2s"], g["b2s_len"], g["n_used"], float(dl_th), _box_of(g),
        won, c_tris[:max_cand], gbuf["cap_top"][:max_cand], gbuf["cap_bot"][:max_cand],
        gbuf["tri"][:max_cand], gbuf["arm_side"][:max_cand], gbuf["arm_otop"][:max_cand],
        gbuf["arm_obot"][:max_cand], gbuf["top"][:max_cand], gbuf["bot"][:max_cand]])


# --------------------------------------------------------------------------------------
# fixed-R sweeps + an eager fixed-dim forward step (the byte-identical validation vehicle)
# --------------------------------------------------------------------------------------
def reconnect_sweep_i_fixed(g: dict, threshold: float, dl_th: Optional[float] = None,
                            max_rounds: int = 8, max_cand: int = DEFAULT_MAX_CAND) -> None:
    """Fixed-R I->H sweep: exactly `max_rounds` fixed-dim rounds, NO host break. Reaches the SAME
    converged state as reconnect_sweep_warp_device run with the same max_rounds (empty/exhausted
    rounds are all-threads-early-return no-ops). No host reads -> capturable."""
    if dl_th is None:
        dl_th = threshold
    owners = _ensure_reserve_owners(g, max_cand)
    for _ in range(max_rounds):
        _round_i_fixed(g, owners, threshold, dl_th, max_cand)
    # under-convergence guard: the last round's won mask is non-empty iff max_rounds was too small
    wp.launch(flag_if_any_won_kernel, dim=max_cand, device=g["device"],
              inputs=[owners["won_i"], _ensure_underconv_flag(g)])


def reconnect_sweep_h_fixed(g: dict, threshold: float, dl_th: Optional[float] = None,
                            max_rounds: int = 8, max_cand: int = DEFAULT_MAX_CAND) -> None:
    """Fixed-R H->I sweep (mirror of reconnect_sweep_i_fixed)."""
    if dl_th is None:
        dl_th = threshold
    owners = _ensure_reserve_owners(g, max_cand)
    for _ in range(max_rounds):
        _round_h_fixed(g, owners, threshold, dl_th, max_cand)
    wp.launch(flag_if_any_won_kernel, dim=max_cand, device=g["device"],
              inputs=[owners["won_h"], _ensure_underconv_flag(g)])


# --------------------------------------------------------------------------------------
# capture_while sweeps -- a DEVICE-SIDE round loop (CUDA conditional graph node) that does
# EXACTLY the rounds the variable sweep would (break when a round applies nothing), no fixed
# max_rounds + no under-convergence guard needed -> always byte-identical AND no wasted no-op
# rounds. The reconnect-path throughput win without max_rounds tuning (the user-chosen path).
# --------------------------------------------------------------------------------------
def _ensure_while_scratch(g: dict):
    """Device scalars for the capture_while loop: cond (1 => run another round) + a round counter
    (the max_rounds cap). Lazily allocated (warm-up does it -> alloc-free inside capture)."""
    if "_while_cond" not in g:
        g["_while_cond"] = wp.zeros(1, dtype=wp.int32, device=g["device"])
        g["_while_count"] = wp.zeros(1, dtype=wp.int32, device=g["device"])
    return g["_while_cond"], g["_while_count"]


def reconnect_sweep_i_while(g: dict, threshold: float, dl_th: Optional[float] = None,
                            max_rounds: int = 8, max_cand: int = DEFAULT_MAX_CAND) -> None:
    """Device-side I->H sweep: capture_while runs one fixed-dim round per iteration and re-reads the
    device cond each iteration, looping until a round applies NO winner (cond = `any won this round`,
    which is EXACTLY the variable sweep's m==0 / n_win==0 break) or the max_rounds cap. Byte-identical
    to reconnect_sweep_warp_device, no fixed-R no-op waste. Capturable with force_module_load=True."""
    if dl_th is None:
        dl_th = threshold
    dev = g["device"]
    owners = _ensure_reserve_owners(g, max_cand)
    cond, count = _ensure_while_scratch(g)
    cond.fill_(1)            # force the first iteration (capture_while checks cond BEFORE the body)
    count.zero_()

    def body():
        _round_i_fixed(g, owners, threshold, dl_th, max_cand)
        cond.zero_()
        wp.launch(flag_if_any_won_kernel, dim=max_cand, device=dev, inputs=[owners["won_i"], cond])
        wp.launch(_while_cap_kernel, dim=1, device=dev, inputs=[count, wp.int32(max_rounds), cond])

    wp.capture_while(cond, body)


def reconnect_sweep_h_while(g: dict, threshold: float, dl_th: Optional[float] = None,
                            max_rounds: int = 8, max_cand: int = DEFAULT_MAX_CAND) -> None:
    """Device-side H->I sweep (mirror of reconnect_sweep_i_while)."""
    if dl_th is None:
        dl_th = threshold
    dev = g["device"]
    owners = _ensure_reserve_owners(g, max_cand)
    cond, count = _ensure_while_scratch(g)
    cond.fill_(1)
    count.zero_()

    def body():
        _round_h_fixed(g, owners, threshold, dl_th, max_cand)
        cond.zero_()
        wp.launch(flag_if_any_won_kernel, dim=max_cand, device=dev, inputs=[owners["won_h"], cond])
        wp.launch(_while_cap_kernel, dim=1, device=dev, inputs=[count, wp.int32(max_rounds), cond])

    wp.capture_while(cond, body)


def forward_step_fixed(g: dict, phys: dict, params, dt: float, dr: float, seed: int, step: int,
                       threshold: Optional[float] = None, dl_th: Optional[float] = None,
                       reconnect: bool = False, interval: int = 1, compact: bool = True,
                       max_rounds: int = 8, max_cand: int = DEFAULT_MAX_CAND) -> None:
    """EAGER fixed-dim forward step -- the byte-identical mirror of engine.forward_step, used to
    validate the fixed-dim path against the production variable-round trajectory (and the template
    for the captured wrapper). Same prefix (director->geom->force->integrate); the reconnect block
    uses the FIXED sweeps; compact + orient run UNCONDITIONALLY on a reconnect step (compact is a
    no-op on a gap-free mesh and orient is idempotent on a clean one -- so this is byte-identical to
    engine.forward_step's `if (ni+nh)>0` gate, while needing no host reconnection-count readback).

    NB this still runs eagerly (the production orient's counter readback + the per-launch host
    overhead are present); the captured wrapper (capture_forward_step) swaps in a capture-safe orient
    and records the launches into a graph."""
    if params.v_active > 0.0 and dr > 0.0:
        W.director_update_warp(g, phys, dr, dt, seed, step)
    gw = W.compute_geometry_warp(g)
    f = W.compute_forces_warp(g, gw, params, phys)
    W.integrate_warp(g, f, dt)
    if reconnect and threshold is not None and (step % interval == 0):
        reconnect_sweep_i_fixed(g, threshold, dl_th, max_rounds=max_rounds, max_cand=max_cand)
        reconnect_sweep_h_fixed(g, threshold, dl_th, max_rounds=max_rounds, max_cand=max_cand)
        if compact:
            compact_warp(g)
        orient_repair_warp(g)
        g["_healed_initial"] = True


# ======================================================================================
# CUDA-graph capture: a captured forward step replayed per integration step (sequential)
# ======================================================================================
def _orient_repair_fixed(g: dict, max_iter: int = 4) -> None:
    """Capture-safe orient_repair_warp: the SAME closure->mark->apply descent, but with the per-iter
    `counter.numpy()` readback + early `break` removed -- it runs a FIXED `max_iter` iterations. Once
    the winding is consistent the remaining iterations are no-ops (mark flags nothing, apply early-
    returns), so the result is byte-identical to orient_repair_warp(max_iter=max_iter) on the same
    mesh (the no-op-tail argument, same as the fixed reconnect rounds). The counter is still written
    (harmless) but never read -> no host sync -> capturable. Scratch is the SAME persistent buffers
    orient_repair_warp uses (alloc-free)."""
    dev = g["device"]
    cap_s = g["cap_s"]
    nb = g["nb"]
    if "_orient_snw" not in g:
        g["_orient_snw"] = wp.zeros(cap_s, dtype=wp.vec3d, device=dev)
        g["_orient_clo"] = wp.zeros(nb, dtype=wp.vec3d, device=dev)
        g["_orient_flip"] = wp.zeros(cap_s, dtype=wp.int32, device=dev)
        g["_orient_counter"] = wp.zeros(1, dtype=wp.int32, device=dev)
    gw = W.compute_surface_geom_warp(g)              # only snorm needed (skip the body kernel)
    snw = g["_orient_snw"]
    wp.copy(snw, gw["snorm"])                        # working snorm (negated in place on flips)
    clo, flip, counter = g["_orient_clo"], g["_orient_flip"], g["_orient_counter"]
    for _ in range(max_iter):
        clo.zero_()
        counter.zero_()
        wp.launch(_body_closure_kernel, dim=cap_s, device=dev,
                  inputs=[snw, g["s2b"], g["surf_alive"], clo])
        wp.launch(_flip_mark_kernel, dim=cap_s, device=dev,
                  inputs=[snw, g["s2b"], g["surf_alive"], clo, flip, counter])
        wp.launch(_flip_apply_kernel, dim=cap_s, device=dev,
                  inputs=[g["s2v"], g["s2v_len"], snw, flip])   # no-op when flip is all-zero


def _fixed_step_body(g: dict, phys: dict, params, dt: float, dr: float, seed: int,
                     threshold, dl_th, max_rounds: int, max_cand: int, max_iter: int,
                     reconnect: bool, use_while: bool = True) -> None:
    """The capture-region body: prefix (director launch + geom + force + integrate) and, on a
    reconnect step, the I/H sweeps + compact + capture-safe orient. NO host reads, NO allocs (after
    warm-up). The director launch reads g['_step_dev'] -- the CALLER sets it (set_director_step)
    OUTSIDE the captured region, so the per-step RNG varies across replays. `reconnect` is a host bool
    fixed at capture time (the full graph captures it True, the prefix graph False). `use_while` picks
    the device-side capture_while sweeps (exact rounds, no waste) over the fixed-R sweeps."""
    if params.v_active > 0.0 and dr > 0.0:
        W._launch_director_update(g, phys, dr, dt, seed)
    gw = W.compute_geometry_warp(g)
    f = W.compute_forces_warp(g, gw, params, phys)
    W.integrate_warp(g, f, dt)
    if reconnect:
        if use_while:
            reconnect_sweep_i_while(g, threshold, dl_th, max_rounds=max_rounds, max_cand=max_cand)
            reconnect_sweep_h_while(g, threshold, dl_th, max_rounds=max_rounds, max_cand=max_cand)
        else:
            reconnect_sweep_i_fixed(g, threshold, dl_th, max_rounds=max_rounds, max_cand=max_cand)
            reconnect_sweep_h_fixed(g, threshold, dl_th, max_rounds=max_rounds, max_cand=max_cand)
        compact_warp(g)
        _orient_repair_fixed(g, max_iter)


class CapturedStep:
    """Capture engine.forward_step into a CUDA graph and replay it per integration step.

    Replays SEQUENTIALLY (one captured graph per sim, replayed each step). At production cell scale
    (n>=16) this alone saturates the GPU (~90% util) -- per-sim occupancy dominates, so cross-sim
    multi-stream overlap is unnecessary (and blocked by a shared-CUB-workspace race anyway). See
    docs/2026-06-26_cuda-graph-experiment-scope.md (P3 RESOLVED).

    Use: build once (warms up + captures), then `for step: cs.step(step)`. The captured launches
    operate on g's POINTER-STABLE canonical arrays (P1 alloc-free + P2 pointer-stable compact), so
    one capture replays correctly as the mesh's CONTENT (positions + topology) evolves -- launch dims
    are all capacity constants. Bit-identical to engine.forward_step (validated by the captured-vs-
    production trajectory gate); the only non-faithful seam the device-step-seed closed is the
    director RNG, now varied per replay via set_director_step.

    NB warm-up + capture ADVANCE g by `warmup` steps (capture records launches without executing, so
    only warm-up advances). `warmup_steps`/`next_step` expose that for callers needing step alignment."""

    def __init__(self, g: dict, phys: dict, params, dt: float, dr: float, seed: int,
                 threshold=None, dl_th=None, reconnect: bool = True, interval: int = 1,
                 max_rounds: int = 8, max_cand: int = DEFAULT_MAX_CAND, max_iter: int = 4,
                 warmup: int = 3, start_step: int = 0, use_capture_while: bool = True):
        if dl_th is None:
            dl_th = threshold
        self.g = g
        self.phys = phys
        self.interval = interval
        self.reconnect = reconnect
        self.use_capture_while = use_capture_while
        dev = g["device"]
        body_kw = dict(params=params, dt=dt, dr=dr, seed=seed, threshold=threshold, dl_th=dl_th,
                       max_rounds=max_rounds, max_cand=max_cand, max_iter=max_iter,
                       use_while=use_capture_while)
        # WARM UP eagerly so EVERY lazy alloc (geom/force/detect/gather/owners/M/overflow/while-cond/
        # orient/_step_dev) happens OUTSIDE capture. Start at step 0 so a reconnect step (0%interval
        # ==0) allocates the reconnect-path buffers. This advances g `warmup` steps.
        s = start_step
        for _ in range(max(1, warmup)):
            W.set_director_step(g, s)
            _fixed_step_body(g, phys, reconnect=(reconnect and (s % interval == 0)), **body_kw)
            s += 1
        self.warmup_steps = max(1, warmup)
        self.next_step = s
        wp.synchronize_device(dev)
        # CAPTURE on the DEFAULT stream (array_scan's scan_device fails to capture on a custom
        # stream). force_module_load=True is required for the capture_while conditional-graph node.
        # Capture RECORDS launches without executing -> g is not advanced here.
        with wp.ScopedCapture(device=dev, force_module_load=True) as cap:
            _fixed_step_body(g, phys, reconnect=reconnect, **body_kw)
        self.full_graph = cap.graph
        # A prefix-only graph for non-reconnect steps (only needed when interval throttles recon).
        self.prefix_graph = None
        if reconnect and interval > 1:
            with wp.ScopedCapture(device=dev, force_module_load=True) as capp:
                _fixed_step_body(g, phys, reconnect=False, **body_kw)
            self.prefix_graph = capp.graph

    def step(self, step_number: int) -> None:
        """Replay one captured step (mutates g). Bumps the director's per-step device seed OUTSIDE
        the captured region first. On a throttled run (interval>1) replays the prefix-only graph on
        non-reconnect steps."""
        W.set_director_step(self.g, step_number)
        if self.prefix_graph is not None and (step_number % self.interval != 0):
            wp.capture_launch(self.prefix_graph)
        else:
            wp.capture_launch(self.full_graph)

    def read_stats(self) -> dict:
        """Host-read the live-slot high-water marks + the correctness flags (ONE sync -> call at
        audit checkpoints, NOT every step). overflow True => a round exceeded MAX_CAND; underconverged
        True => max_rounds too small. EITHER True => the captured trajectory is NOT bit-identical."""
        nu = self.g["n_used"].numpy()
        return dict(nv=int(nu[0]), ns=int(nu[1]), overflow=check_overflow(self.g),
                    underconverged=check_underconverged(self.g))
