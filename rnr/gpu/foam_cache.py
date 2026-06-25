"""Disk cache for the unit-cell two-type foam.

The O(N^2) TissueForge foam builder dominates wall-clock at paper scale (N=2000 setup ~10 min;
the 100k stepping itself is fast). Rather than optimize the builder, we BUILD ONCE and cache the
result to disk: future runs LOAD consistent geometry in ~ms, with a DE-NOVO fallback if the cache
file is absent (or unreadable, or a format/key mismatch).

What is cached (the host artifacts `upload_unit_foam` needs, with NO TissueForge dependency on
load): the unit-cell-SCALED COMPACT CSR + per-body phys state (type + director) + box + (v0, a0).
Headroom is NOT baked in -- the padded mesh is rebuilt from the CSR at the requested headroom on
load -- so a single cache file serves any headroom / step count. Keyed on the foam-determining
params (n, ic, jitter, rng_seed) + a FORMAT tag.

Split of responsibility:
  * the TF-dependent build (TF foam -> CSR -> unit scale -> v0/a0) lives in the caller
    (`rnr/tests/test_gpu_engine.py::_build_unit_foam_host`), passed in as `build_host_fn`;
  * this module owns the npz I/O AND the device-upload half (`upload_unit_foam`), so the cached
    path and the direct `_setup_unit_foam` path share ONE upload (no drift).
"""
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .csr_mesh import CSRMesh
from .device_mesh import PaddedMesh
from .physics_csr import PhysState
from . import physics_warp as W

# bump if the cached layout / scaling convention changes (invalidates old files by key)
FORMAT = 1

# padding mirrors _build_unit_foam_host / the validated _setup_unit_foam (keep in sync)
_RING_PAD = _VS_PAD = _BS_PAD = 64

_CSR_ARRAYS = ("vert_pos", "vert_id", "surf_id", "body_id", "s2v_off", "s2v_idx", "s2b",
               "v2s_off", "v2s_idx", "b2s_off", "b2s_idx", "vert_alive", "surf_alive",
               "body_alive")


def default_cache_dir() -> Path:
    """<repo>/rnr/exports/foam_cache (gitignored)."""
    return Path(__file__).resolve().parents[1] / "exports" / "foam_cache"


def cache_key(n: int, ic: str, jitter: float, rng_seed: int) -> str:
    return f"foam_n{n}_{ic}_j{jitter:g}_s{rng_seed}_v{FORMAT}"


def cache_path(n: int, ic: str, jitter: float = 0.10, rng_seed: int = 3,
               cache_dir: Optional[Path] = None) -> Path:
    d = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    return d / (cache_key(n, ic, jitter, rng_seed) + ".npz")


# ----------------------------------------------------------------------------- I/O ----
def save_host(path: Path, host: dict, *, n: int, ic: str, jitter: float, rng_seed: int) -> None:
    """Serialize the host artifacts (csr + state + box + v0 + a0) to a single npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    csr: CSRMesh = host["csr"]
    state: PhysState = host["state"]
    blob = {f"csr_{name}": getattr(csr, name) for name in _CSR_ARRAYS}
    blob.update(csr_nv=np.int64(csr.nv), csr_ns=np.int64(csr.ns), csr_nb=np.int64(csr.nb),
                state_body_type=state.body_type, state_body_director=state.body_director,
                box=np.asarray(host["box"], np.float64),
                v0=np.float64(host["v0"]), a0=np.float64(host["a0"]),
                meta_format=np.int64(FORMAT), meta_n=np.int64(n), meta_ic=np.str_(ic),
                meta_jitter=np.float64(jitter), meta_rng_seed=np.int64(rng_seed))
    # atomic-ish write: temp then replace, so an interrupted save never leaves a half file.
    # The temp name must END in .npz, else np.savez_compressed appends .npz to it (and the
    # rename target would not exist).
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **blob)
    tmp.replace(path)


def load_host(path: Path) -> dict:
    """Inverse of save_host. Raises (ValueError / OSError / KeyError) on a bad/old file so the
    caller can fall back to a de-novo build."""
    with np.load(Path(path), allow_pickle=False) as z:
        fmt = int(z["meta_format"])
        if fmt != FORMAT:
            raise ValueError(f"foam cache format {fmt} != expected {FORMAT}")
        csr = CSRMesh(
            nv=int(z["csr_nv"]), ns=int(z["csr_ns"]), nb=int(z["csr_nb"]),
            vert_pos=z["csr_vert_pos"], vert_id=z["csr_vert_id"],
            surf_id=z["csr_surf_id"], body_id=z["csr_body_id"],
            s2v_off=z["csr_s2v_off"], s2v_idx=z["csr_s2v_idx"], s2b=z["csr_s2b"],
            v2s_off=z["csr_v2s_off"], v2s_idx=z["csr_v2s_idx"],
            b2s_off=z["csr_b2s_off"], b2s_idx=z["csr_b2s_idx"],
            vert_alive=z["csr_vert_alive"], surf_alive=z["csr_surf_alive"],
            body_alive=z["csr_body_alive"],
        )
        state = PhysState(body_type=z["state_body_type"], body_director=z["state_body_director"])
        return dict(csr=csr, state=state, box=z["box"], v0=float(z["v0"]), a0=float(z["a0"]))


# ------------------------------------------------------------- device upload (shared) ----
def upload_unit_foam(host: dict, dev, headroom: int = 3000):
    """The device half of foam setup (NO TissueForge): host artifacts -> device SoA `g` + phys.
    Returns (g, phys, body_type, box, v0, a0) -- the same tuple as `_setup_unit_foam`.

    The padded mesh is rebuilt from the cached compact CSR at the requested `headroom`, so the
    cache is headroom-independent. Padding constants mirror _build_unit_foam_host."""
    csr: CSRMesh = host["csr"]
    state: PhysState = host["state"]
    box = host["box"]
    pm = PaddedMesh.from_csr(csr, v_headroom=headroom, s_headroom=headroom,
                             ring_pad=_RING_PAD, vs_pad=_VS_PAD, bs_pad=_BS_PAD)
    g = W.attach_box(pm.to_warp(device=dev), box)
    phys = W.upload_phys(state, dev)
    return g, phys, state.body_type, box, host["v0"], host["a0"]


# ------------------------------------------------------------------- build-or-load ----
def load_or_build(dev, n: int, ic: str, headroom: int, build_host_fn: Callable[[], dict],
                  jitter: float = 0.10, rng_seed: int = 3, rebuild: bool = False,
                  cache_dir: Optional[Path] = None):
    """Load the cached unit foam if present; otherwise build it (de novo), save it, and use it.

    `build_host_fn()` must return the host artifacts dict {csr, state, box, v0, a0} (the
    TF-dependent build; only invoked on a cache miss). Returns (g, phys, body_type, box, v0, a0)
    via `upload_unit_foam`. `rebuild=True` forces a fresh build + overwrite.
    """
    path = cache_path(n, ic, jitter, rng_seed, cache_dir)
    host = None
    if path.exists() and not rebuild:
        try:
            host = load_host(path)
            print(f"[foam-cache] loaded {path.name} (no TF build): "
                  f"nb={host['csr'].nb} nv={host['csr'].nv} ns={host['csr'].ns}")
        except Exception as e:                       # corrupt / old format -> rebuild
            print(f"[foam-cache] {path.name} unreadable ({e}); rebuilding de novo")
            host = None
    if host is None:
        why = "rebuild forced" if rebuild else ("absent" if not path.exists() else "fallback")
        print(f"[foam-cache] building de novo ({why}; this runs the TF foam builder)...")
        host = build_host_fn()
        try:
            save_host(path, host, n=n, ic=ic, jitter=jitter, rng_seed=rng_seed)
            print(f"[foam-cache] saved {path}")
        except Exception as e:                       # non-fatal: run anyway, just no cache
            print(f"[foam-cache] WARNING could not save cache ({e}); continuing without it")
    return upload_unit_foam(host, dev, headroom=headroom)
