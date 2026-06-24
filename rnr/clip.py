"""Clip-plane helpers shared by the half-tissue videos.

ONE convention, two renderers. A clip plane is a point + outward normal; the half
that is KEPT is the side the **normal points toward** -- identical to TissueForge's
GL `gl_ClipDistance = dot(pos - point, normal) >= 0` (see `tfFlat3D.vert`). So both
the native-GL video (`video_native_gl.py`, real TF clip plane) and the matplotlib
whole-cell video (`video_native_cells.py`, polygon clip) cut the tissue the same way.

CLIP env interface (both scripts):
  CLIP    = which half to keep, as <sign?><axis>: one of  x y z  +x +y +z  -x -y -z
            (also accepts `x+`/`x-` ...). The sign is the normal direction = the kept
            side; e.g. CLIP=z keeps the upper (z>=cut) half, CLIP=-z keeps the lower.
            Unset / "" / "0" / "none" / "off"  => no clipping.
  CLIP_AT = where the plane sits along that axis, as a fraction of the box in [0,1].
            Default 0.5 (box centre).

`parse_clip_env(L)` returns either None or `(point, normal)` where BOTH are plain
Python `list[float]` of length 3 -- already in the exact shape TF's strict parser
wants for `tf.init(clip_planes=[(point, normal)])` (point/normal MUST be lists, the
outer entry MUST be a tuple; a malformed entry is silently dropped, see PORTING_NOTES
§"clip planes"). For matplotlib just pass them on to `clip_polygon_halfspace`.
"""
import os

import numpy as np

_AXIS = {"x": 0, "y": 1, "z": 2}


def parse_clip_spec(spec, at, L):
    """Return (point, normal) lists for a clip spec, or None if `spec` is empty/off.

    `spec`  -- e.g. "z", "+z", "-z", "x-" (sign = kept side / normal direction).
    `at`    -- fraction in [0, 1] along the axis where the plane sits.
    `L`     -- box edge length (the simulation spans [0, L] on each axis).
    """
    if spec is None:
        return None
    s = str(spec).strip().lower()
    if s in ("", "0", "none", "off", "false", "no"):
        return None

    sign = 1.0
    if s[0] in "+-":
        sign = -1.0 if s[0] == "-" else 1.0
        s = s[1:]
    elif s[-1] in "+-":
        sign = -1.0 if s[-1] == "-" else 1.0
        s = s[:-1]
    if s not in _AXIS:
        raise ValueError(f"CLIP axis must be one of x/y/z (got {spec!r})")

    ax = _AXIS[s]
    point = [L / 2.0, L / 2.0, L / 2.0]
    point[ax] = float(at) * L
    normal = [0.0, 0.0, 0.0]
    normal[ax] = sign
    return point, normal


def parse_clip_env(L, clip_env="CLIP", at_env="CLIP_AT"):
    """Read CLIP / CLIP_AT from the environment -> (point, normal) lists or None."""
    spec = os.environ.get(clip_env)
    at = float(os.environ.get(at_env, "0.5"))
    return parse_clip_spec(spec, at, L)


def clip_polygon_halfspace(poly, point, normal, eps=1e-9):
    """Sutherland-Hodgman clip of a single polygon against one half-space.

    Keeps the part of `poly` (an (N,3) array of ordered vertices) on the side the
    `normal` points toward, i.e. where `dot(v - point, normal) >= 0` -- the SAME
    half the GL renderer keeps. Vertices straddling the plane get a fresh
    intersection vertex inserted, so a cell crossing the plane is cut flat rather
    than dropped whole. Returns the clipped (M,3) array, or None if nothing remains
    (polygon fully on the clipped side, or degenerate < 3 vertices).
    """
    P = np.asarray(poly, dtype=float)
    if len(P) < 3:
        return None
    n = np.asarray(normal, dtype=float)
    p0 = np.asarray(point, dtype=float)
    d = (P - p0) @ n                      # signed distance along the normal; keep d >= 0

    out = []
    N = len(P)
    for i in range(N):
        cur, di = P[i], d[i]
        nxt, dj = P[(i + 1) % N], d[(i + 1) % N]
        cur_in = di >= 0.0
        if cur_in:
            out.append(cur)
        if cur_in != (dj >= 0.0):         # edge crosses the plane -> insert intersection
            t = di / (di - dj)            # di - dj != 0 because the signs differ
            out.append(cur + t * (nxt - cur))

    if len(out) < 3:
        return None
    return np.asarray(out, dtype=float)
