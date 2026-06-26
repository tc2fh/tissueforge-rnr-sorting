"""Extensibility seams for the GPU 3D-vertex engine: durable per-cell state + the hook
contracts that let a user add custom forces/behaviors WITHOUT touching the validated core.

WHY THIS LAYER EXISTS
---------------------
The fused `physics_warp.force_kernel` (volume + area + tension + active drive) is the frozen,
byte-identical reproduction of the paper physics -- we do NOT edit it to add a model. Instead
the engine runs two user-supplied hook lists around it (see `engine.Engine` / `engine.forward_step`):

    behaviors :  fn(g, cells, step, dt) -> None     # evolve per-cell state (top of the step)
    forces    :  fn(g, cells, geom) -> None         # ADD into g['_force'] (after the core kernel)

A "behavior" mutates cell state (e.g. repolarize a crawling direction); a "force" reads geometry +
state and adds a force contribution. The built-in active drive (a per-cell director evolved by
`director_update_kernel` and read as a per-vertex force) is exactly one (behavior, force) pair -- so
lamellipodial crawling, chemotaxis, type-specific drives, etc. are all the same shape (see
examples/crawl.py).

THE KEY INVARIANT -- per-cell state is DURABLE, per-vertex state is FRAGILE
-------------------------------------------------------------------------
Bodies never change slots under reconnection or compaction (`compact_warp`: bodies are stable under
I<->H; only their surface refs remap). So a per-body array sized `nb` stays valid for the WHOLE
simulation with no remap and no inheritance rule -- this is the safe place for custom cell state.
Vertices, by contrast, are renumbered by compaction and BORN BLANK by reconnection, so persistent
per-vertex attributes would need extra plumbing (a compaction scatter + a creation-seeding rule) and
are intentionally NOT supported here. To act on "specific vertices" (a leading edge, a basal side),
derive them IN-KERNEL from cell membership + geometry -- e.g. offset-from-centroid . polarity > 0 --
rather than tagging vertex ids that the next reconnection may destroy.

CAPTURE SAFETY
--------------
Hooks that want to live inside a CUDA-graph `CapturedStep` must be alloc-free and do no per-step host
readback; route any per-step-varying scalar (an RNG step key) through a device scalar like
`physics_warp.set_director_step` (g['_step_dev']) so a captured launch varies it per replay.
"""
import numpy as np
import warp as wp


class CellState:
    """Durable per-cell (per-body) state, dict-like over named device arrays.

    A CellState is a SUPERSET of the engine's `phys` dict -- it always carries `body_type` and
    `body_director`, so it can be passed anywhere the engine expects `phys` (the core reads
    cells['body_type'] / cells['body_director']). `add_field()` adds your own durable fields, which
    custom hooks read as cells['name'] or cells.name.

    Sized by `nb` (the fixed body count); every field is shape (nb,) [scalar] or (nb, 3) [vec3d].
    """

    def __init__(self, n: int, device, fields: dict):
        # use object.__setattr__-free path: real attrs go in __dict__, fields in _fields
        self.__dict__["n"] = int(n)
        self.__dict__["device"] = device
        self.__dict__["_fields"] = dict(fields)

    @classmethod
    def from_phys(cls, phys: dict, device) -> "CellState":
        """Wrap an existing engine `phys` dict (body_type + body_director wp.arrays already at the
        kernels' dtypes). No copy -- the wrapped arrays ARE the engine's, so hooks and the core see
        the same per-cell state."""
        n = int(phys["body_type"].shape[0])
        return cls(n, device, dict(phys))

    def add_field(self, name: str, init, dtype=None) -> wp.array:
        """Add a durable per-cell field and return its device array.

        `init` is an ndarray (shape (n,) or (n, 3)) or a callable(n) -> ndarray. `dtype` is inferred
        when omitted: a 2-D (n, 3) array -> wp.vec3d, an integer array -> wp.int32, else wp.float64.
        Re-adding a name overwrites it. The field is created on this CellState's device."""
        if name in ("n", "device"):
            raise ValueError(f"field name {name!r} shadows a CellState attribute")
        arr = init(self.n) if callable(init) else np.asarray(init)
        arr = np.ascontiguousarray(arr)
        if arr.shape[0] != self.n:
            raise ValueError(f"field {name!r} has {arr.shape[0]} rows, expected nb={self.n}")
        if dtype is None:
            if arr.ndim == 2 and arr.shape[1] == 3:
                dtype, arr = wp.vec3d, arr.astype(np.float64)
            elif arr.dtype == bool or np.issubdtype(arr.dtype, np.integer):
                # a bool mask (e.g. body_type == MIGRATORY) is the natural way to write a 0/1
                # per-cell flag -> int32, so the advertised add_field("is_crawler", mask) just works
                dtype, arr = wp.int32, arr.astype(np.int32)
            else:
                dtype, arr = wp.float64, arr.astype(np.float64)
        elif dtype == wp.vec3d:
            arr = np.ascontiguousarray(arr.reshape(self.n, 3), dtype=np.float64)
        self._fields[name] = wp.array(arr, dtype=dtype, device=self.device)
        return self._fields[name]

    # --- dict-like so a CellState stands in for the `phys` dict ---
    def __getitem__(self, k):
        return self._fields[k]

    def __setitem__(self, k, v):
        self._fields[k] = v

    def __contains__(self, k):
        return k in self._fields

    def keys(self):
        return self._fields.keys()

    def fields(self) -> dict:
        """Introspection: return {field_name -> Warp dtype name} for every per-cell field currently
        held (e.g. {'body_type': 'int32', 'body_director': 'vec3d', 'polarity': 'vec3d',
        'is_crawler': 'int32'}). Lets you (at a REPL) or generic code DISCOVER what state a CellState
        carries without knowing it in advance -- e.g. a logger/checkpointer that saves every field,
        or a sanity check that a hook's required field is present."""
        return {name: arr.dtype.__name__ for name, arr in self._fields.items()}

    def __repr__(self):
        return f"CellState(n={self.n}, fields={self.fields()})"

    # --- attribute access: cells.polarity == cells['polarity'] ---
    def __getattr__(self, name):
        # only reached when normal lookup fails (so n/device/_fields resolve before this)
        try:
            return self.__dict__["_fields"][name]
        except KeyError:
            raise AttributeError(name)


# --------------------------------------------------------------------------------------
# field-init helpers (pass as `init=` to add_field)
# --------------------------------------------------------------------------------------
def random_unit_vectors(seed: int = 0):
    """Return a callable(n) -> (n, 3) of random unit vectors (e.g. an initial polarity field).
    Seeded for reproducibility; uses its own Generator so it does not perturb global RNG state."""
    def _init(n: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        v = rng.standard_normal((n, 3))
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    return _init


def constant_vector(vec):
    """Return a callable(n) -> (n, 3) repeating `vec` (e.g. a uniform polarity field)."""
    vec = np.asarray(vec, dtype=np.float64).reshape(3)
    return lambda n: np.tile(vec, (n, 1))
