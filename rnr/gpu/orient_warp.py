"""Closure-consistency repair: reverse any 2-body face whose winding is inconsistent with its
b1/b2 (= s2b[:,0]/s2b[:,1]), so every cell stays CLOSED (sum_faces sense*snorm == 0, sense=+1 iff
b==b1) and its divergence-theorem volume is correct + origin-independent.

WHY (the RNR-at-scale balloon fix, PORTING_NOTES §6p / memory gpu-rnr-scale-corruption):
the volume kernel (physics_warp.body_geom_kernel) computes vol = sum sense*dot(snorm,scent)/6,
which is only origin-independent when the cell is closed. Two sources break closure:
  1. the INITIAL foam has near-degenerate faces (zero area => undefined normal => the builder
     stores an arbitrary, often b1/b2-inconsistent winding) that sit JUST ABOVE the reconnection
     threshold, so H->I never collapses them; as dynamics grow them they stay mis-wound;
  2. the parallel I<->H surgery occasionally emits a face whose winding is b1/b2-inconsistent
     (the gather's arm order is handedness-arbitrary).
A mis-wound face contributes the WRONG sign to its cap cell's volume -> the cell balloons once the
face grows (count-driven, ~1000 I->H). No GEOMETRIC per-face test reliably detects mis-winding
(irregular foam cells have correctly-closed faces whose snorm points "into" the cell by any
centroid test). The robust signal is the EXACT closure residual: a face is mis-wound iff reversing
its ring strictly reduces BOTH incident cells' closure ||sum sense*snorm||. We iterate a few times
(a flip changes 2 cells, so a second pass re-checks) -- a descent that lands at machine round-off.

This runs on the device after each reconnection sweep (engine.forward_step); it is O(faces),
parallel, and reverses only the handful of inconsistent faces (typ. 0-3 per step)."""
import warp as wp

from .physics_warp import compute_geometry_warp

wp.init()


@wp.kernel
def _body_closure_kernel(snorm: wp.array(dtype=wp.vec3d), s2b: wp.array2d(dtype=wp.int32),
                         surf_alive: wp.array(dtype=wp.int32), clo: wp.array(dtype=wp.vec3d)):
    """clo[b] = sum over b's live faces of (sense * snorm), sense=+1 iff b==s2b[:,0]."""
    s = wp.tid()
    if surf_alive[s] == 0:
        return
    b1 = s2b[s, 0]
    b2 = s2b[s, 1]
    if b1 < 0 or b2 < 0:
        return
    sn = snorm[s]
    wp.atomic_add(clo, b1, sn)
    wp.atomic_sub(clo, b2, sn)


@wp.kernel
def _flip_mark_kernel(snorm: wp.array(dtype=wp.vec3d), s2b: wp.array2d(dtype=wp.int32),
                      surf_alive: wp.array(dtype=wp.int32), clo: wp.array(dtype=wp.vec3d),
                      flip: wp.array(dtype=wp.int32), counter: wp.array(dtype=wp.int32)):
    """Mark face s for reversal iff flipping it (negating its snorm) strictly reduces the closure
    of BOTH its cells: b1 contributes +snorm (-> delta -2snorm), b2 contributes -snorm (-> +2snorm)."""
    s = wp.tid()
    flip[s] = 0
    if surf_alive[s] == 0:
        return
    b1 = s2b[s, 0]
    b2 = s2b[s, 1]
    if b1 < 0 or b2 < 0:
        return
    two = wp.float64(2.0)
    eps = wp.float64(1.0e-12)
    sn = snorm[s]
    c1 = clo[b1]
    c2 = clo[b2]
    if wp.length(c1 - two * sn) < wp.length(c1) - eps:
        if wp.length(c2 + two * sn) < wp.length(c2) - eps:
            flip[s] = 1
            wp.atomic_add(counter, 0, 1)


@wp.kernel
def _flip_apply_kernel(s2v: wp.array2d(dtype=wp.int32), s2v_len: wp.array(dtype=wp.int32),
                       snorm: wp.array(dtype=wp.vec3d), flip: wp.array(dtype=wp.int32)):
    """Reverse the winding of marked faces (reverse ring positions [1, L) -- keep ring[0] so the
    vertex set is unchanged) and negate their snorm (a reversed ring has the opposite area vector),
    so the next iteration's closure uses the updated normal without a full geometry recompute."""
    s = wp.tid()
    if flip[s] == 0:
        return
    i = wp.int32(1)
    j = s2v_len[s] - 1
    while i < j:
        tmp = s2v[s, i]
        s2v[s, i] = s2v[s, j]
        s2v[s, j] = tmp
        i += 1
        j -= 1
    snorm[s] = -snorm[s]


def orient_repair_warp(g: dict, max_iter: int = 4) -> int:
    """Repair winding/closure consistency on the device SoA `g` IN PLACE (only s2v is mutated:
    mis-wound faces' rings are reversed). Returns the number of faces reversed. Idempotent: a
    consistently-oriented mesh yields 0 flips. Run after each reconnection sweep."""
    dev = g["device"]
    cap_s = g["cap_s"]
    nb = g["nb"]
    gw = compute_geometry_warp(g)
    snw = wp.clone(gw["snorm"])                      # working snorm (negated in place on flips)
    clo = wp.zeros(nb, dtype=wp.vec3d, device=dev)
    flip = wp.zeros(cap_s, dtype=wp.int32, device=dev)
    counter = wp.zeros(1, dtype=wp.int32, device=dev)
    total = 0
    for _ in range(max_iter):
        clo.zero_()
        counter.zero_()
        wp.launch(_body_closure_kernel, dim=cap_s, device=dev,
                  inputs=[snw, g["s2b"], g["surf_alive"], clo])
        wp.launch(_flip_mark_kernel, dim=cap_s, device=dev,
                  inputs=[snw, g["s2b"], g["surf_alive"], clo, flip, counter])
        n = int(counter.numpy()[0])
        if n == 0:
            break
        total += n
        wp.launch(_flip_apply_kernel, dim=cap_s, device=dev,
                  inputs=[g["s2v"], g["s2v_len"], snw, flip])
    return total
