"""Gate E: the composed forward step of the GPU 3D vertex engine + the sorting metric.

One forward step ties together everything built for Gate E and the RNR path:

    director rotational diffusion  (physics_warp.director_update_warp)
 -> geometry  (surface + body)     (physics_warp.compute_geometry_warp)
 -> per-vertex force (4 actors)     (physics_warp.compute_forces_warp)
 -> overdamped integrate x+=dt*f    (physics_warp.integrate_warp)
 -> RNR reconnection both ways      (schedule_warp.reconnect_sweep_*_warp_device)   [throttled]
 -> stream-compaction               (compact_warp.compact_warp)                     [bounds slots]
 -> winding/closure repair          (orient_warp.orient_repair_warp)                [keeps cells closed]

mirrors a TissueForge vertex step (director update in preStepStart, then force/integrate, then
doQuality reconnection). Reconnection is THROTTLED (every `interval` steps, as in the engine's
reconnectInterval) and followed by compaction so the bump-allocated births stay bounded.

The sorting readout `het_contact_fraction` is the Fig-1E demixing order parameter: the fraction
of interior faces whose two cells are DIFFERENT types. Heterotypic interfacial tension drives it
DOWN over time (like cells coalesce); it is scale-invariant, so it is the same number whether the
foam is at unit-cell or universe scale.
"""
from typing import Optional

import numpy as np

from . import physics_warp as W
from .compact_warp import compact_warp
from .device_mesh import PaddedMesh
from .extensions import CellState
from .orient_warp import orient_repair_warp
from .schedule_warp import (reconnect_sweep_h_to_i_warp_device,
                            reconnect_sweep_warp_device)


def forward_step(g: dict, phys: dict, params, dt: float, dr: float, seed: int, step: int,
                 threshold: Optional[float] = None, dl_th: Optional[float] = None,
                 reconnect: bool = False, interval: int = 1, compact: bool = True,
                 max_rounds: int = 8, behaviors=(), forces=()) -> dict:
    """Advance the device mesh `g` (+ per-body `phys`) by one overdamped step. Mutates `g`
    (positions, and topology if reconnection fires) and `phys['body_director']` in place.
    Returns {'i','h','nv','ns'}: reconnections applied each way + the live-slot high-water marks.

    `behaviors` / `forces` are optional user EXTENSION HOOKS (see gpu/extensions.py, gpu/engine.Engine):
      * each behavior `b(g, phys, step, dt)` runs at the TOP of the step (with the built-in director
        update) and may evolve per-cell state in `phys` (a CellState);
      * each force `fh(g, phys, geom)` runs AFTER the fused core force kernel and ADDS into the force
        accumulator `g['_force']`, before integration.
    Both default to empty -> this is byte-identical to the core step (the hot path the test gate and
    every existing caller exercise); the hooks are purely additive."""
    if params.v_active > 0.0 and dr > 0.0:
        W.director_update_warp(g, phys, dr, dt, seed, step)
    for b in behaviors:
        b(g, phys, step, dt)
    gw = W.compute_geometry_warp(g)
    f = W.compute_forces_warp(g, gw, params, phys)
    for fh in forces:
        fh(g, phys, gw)          # adds into g['_force'] == f, the same accumulator integrate reads
    W.integrate_warp(g, f, dt)

    ni = nh = 0
    if reconnect and threshold is not None and (step % interval == 0):
        ri = reconnect_sweep_warp_device(g, threshold, dl_th, max_rounds=max_rounds)
        rh = reconnect_sweep_h_to_i_warp_device(g, threshold, dl_th, max_rounds=max_rounds)
        ni, nh = ri["total"], rh["total"]
        if compact and (ni + nh) > 0:
            compact_warp(g)
        # Heal any b1/b2-inconsistent windings so every cell stays CLOSED and its divergence-
        # theorem volume is correct -- without this the mesh balloons at scale (orient_warp / §6p).
        # Windings only go inconsistent from (a) the initial foam's near-degenerate faces and
        # (b) reconnection surgery -- NOT the per-step integrate -- and orient_repair is idempotent
        # (a no-op, but it pays a full geometry recompute, on a clean mesh). So run it after any
        # reconnection, plus ONCE to heal the initial foam (`_healed_initial`). Skipping the
        # per-step no-op orient on no-reconnection steps is bit-identical and saves ~0.38 ms/step.
        if (ni + nh) > 0 or not g.get("_healed_initial", False):
            orient_repair_warp(g)
            g["_healed_initial"] = True
    nu = g["n_used"].numpy()
    return dict(i=ni, h=nh, nv=int(nu[0]), ns=int(nu[1]))


def het_contact_fraction(pm: PaddedMesh, body_type: np.ndarray) -> tuple:
    """(het, total) interior-face counts: `het` faces separate two DIFFERENT cell types.
    het/total is the demixing order parameter (Fig 1E); -> 0 as the two types fully sort."""
    het = 0
    total = 0
    for s in range(pm.n_s_used):
        if not pm.surf_alive[s]:
            continue
        b0 = int(pm.s2b[s, 0])
        b1 = int(pm.s2b[s, 1])
        if b0 < 0 or b1 < 0:
            continue                       # boundary face (none in a periodic foam)
        total += 1
        if body_type[b0] != body_type[b1]:
            het += 1
    return het, total


def het_fraction_device(g: dict, body_type: np.ndarray) -> float:
    """het_contact_fraction read straight off the device mesh (O(mesh) host copy; for metrics
    only, not the hot loop)."""
    pm = PaddedMesh.from_warp(g)
    het, total = het_contact_fraction(pm, body_type)
    return het / total if total else 0.0


class Engine:
    """Ergonomic front door to the GPU vertex engine WITH user extension hooks.

    Bundles a device mesh `g`, a `CellState` (durable per-cell state), the force `params`, and the
    per-step config; lets you register custom behavior/force hooks; and drives the step loop. The
    fused core physics is untouched -- hooks run around it (see forward_step / gpu/extensions.py).

        eng = Engine(g, phys, params, dt=5e-3, dr=1.0, seed=7,
                     threshold=0.02, dl_th=0.02, reconnect=True, interval=2)
        eng.cells.add_field("polarity", random_unit_vectors())
        eng.cells.add_field("is_crawler", lambda n: (eng.cells['body_type'].numpy() == 1).astype(int))
        eng.add_force(lamellipodium_force(f_mag=2.0))     # or @eng.force
        eng.add_behavior(persistent_repolarization())      # or @eng.behavior
        for _ in range(steps):
            info = eng.step()

    `phys` may be a plain engine `phys` dict (wrapped into a CellState) or a CellState already.
    """

    def __init__(self, g, phys, params, dt, dr=0.0, seed=0, threshold=None, dl_th=None,
                 reconnect=False, interval=1, compact=True, max_rounds=8):
        self.g = g
        self.cells = phys if isinstance(phys, CellState) else CellState.from_phys(phys, g["device"])
        self.params = params
        self.dt, self.dr, self.seed = dt, dr, seed
        self.threshold, self.dl_th = threshold, dl_th
        self.reconnect, self.interval = reconnect, interval
        self.compact, self.max_rounds = compact, max_rounds
        self._behaviors = []
        self._forces = []
        self.step_index = 0

    def add_behavior(self, fn):
        """Register a behavior hook `fn(g, cells, step, dt)` (runs at the top of each step, evolves
        per-cell state). Usable as a decorator (@eng.behavior) or a call. Returns `fn`."""
        self._behaviors.append(fn)
        return fn
    behavior = add_behavior

    def add_force(self, fn):
        """Register a force hook `fn(g, cells, geom)` (runs after the core force kernel, ADDS into
        g['_force']). Usable as a decorator (@eng.force) or a call. Returns `fn`."""
        self._forces.append(fn)
        return fn
    force = add_force

    def step(self) -> dict:
        """Advance one step: built-in director update + user behaviors -> geometry -> core force +
        user forces -> integrate -> reconnection. Returns forward_step's {'i','h','nv','ns'}."""
        info = forward_step(self.g, self.cells, self.params, self.dt, self.dr, self.seed,
                            self.step_index, threshold=self.threshold, dl_th=self.dl_th,
                            reconnect=self.reconnect, interval=self.interval, compact=self.compact,
                            max_rounds=self.max_rounds, behaviors=self._behaviors, forces=self._forces)
        self.step_index += 1
        return info

    def het_fraction(self) -> float:
        """Demixing order parameter off the live device mesh (metrics only; O(mesh) host copy)."""
        return het_fraction_device(self.g, self.cells["body_type"].numpy())
