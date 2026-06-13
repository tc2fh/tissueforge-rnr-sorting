"""Phase-2 per-step reconnection operator: wire the validated I<->H 3D-T1 into the
dynamics.

In the eventual C++ port the reconnection runs INSIDE `MeshQuality::doQuality()` during
each integrator step. From Python we cannot hook there, so we run this operator BETWEEN
`tf.step()` calls -- a valid prototype (noted in PORTING_NOTES.md). One `apply()` call =
one "quality pass" over the mesh.

Every design choice below is a real Phase-2 fork; each is justified inline.

TRIGGER (Okuda Condition 2).
    I->H fires when a short interior edge is shorter than `dl_th`; H->I fires when a
    triangle's MAX edge is shorter than `dl_th` (the *max*, NOT Honda's wrong "condition
    H" min -- CLAUDE.md). `dl_th` must sit BELOW the mesh's equilibrium edge length, so
    reconnection fires only when the dynamics genuinely collapses an edge -- never on the
    pristine foam.

VOLUME GUARD (optional, OFF by default -- superseded; see scripts/faithful_run.py).
    Originally added to catch a reconnection whose Appendix-1 placement inverts a
    neighbour cell: apply tentatively, check the 5 cells' volumes, REVERSE (I<->H is
    exactly reversible) if any is non-positive. The reference-faithful experiment showed
    this is a NO-OP and the WRONG TOOL: at faithful Lth it never fires (`cum.reverted=0`
    -- the mutate-half never leaves an immediately-negative neighbourhood, so the surgery
    winding is sound), because the negative volumes are NOT caused by the reconnection.
    They are a TF signed-volume WINDING SIGN-FLIP that appears DYNAMICALLY several steps
    after a reconnection when the integrator overshoots a near-collapsed face at too-large
    dt; that flip reverses the VolumeConstraint force sign and inflates the cell without
    bound. Reversing a reconnection cannot un-invert such a runaway cell. The faithful
    levers are ADEQUATE dt + SMALL features (place ~= dl_th), and -- natively -- a
    3DVertVor-style abs(volume)/orientation repair that restores the force sign (TF
    computes volume internally, so it can't be injected from Python; it is THE key
    VolumeConstraint/MeshQuality finding for the C++ port). `volume_guard=True` keeps the
    old reverse-on-inversion behaviour available as an optional knob.

ANTI-THRASH (hysteresis gap + cooldown).
    A fresh I->H triangle has edges ~`place_scale`; if that were ~`dl_th` it would sit at
    the H->I trigger and could collapse straight back. We open a forward/reverse gap two
    ways: (1) created features are sized at `place_scale = dl_th*(1+hysteresis) > dl_th`,
    and (2) a just-reconnected (or reversed) 5-cell neighbourhood is put on a `cooldown`
    of N steps during which no further reconnection among those cells is attempted. The
    faithful default is `hysteresis=0` (place == dl_th, infinitesimal features like the
    references) -- a large inflation overshoots on a distorted mesh and is itself a source
    of the post-reconnection sign-flips above; the `cooldown` carries the anti-thrash duty.
    An optional per-pass stochastic gate `p_transition` (Okuda
    Fig. 7 uses 0.01/step) is also provided for runs that want Okuda's rate-limited
    reconnection; default 1.0 (the guard + cooldown already prevent thrash).

HANDLE RE-FETCH (the correctness crux).
    A reconnection invalidates the handles for its whole local neighbourhood, so a stale
    candidate must never be reused after an overlapping site mutates. Strategy:
    **body-disjoint batching**. One scan yields all candidates; we apply, in urgency
    order (most-collapsed first), only a subset whose 5-cell neighbourhoods are pairwise
    BODY-DISJOINT. Body-disjoint neighbourhoods share no vertices and no surfaces (a
    shared vertex/surface implies a shared body), so their mutations cannot invalidate
    one another's handles -- the whole batch is safe from a single scan, no intra-batch
    re-walk. Overlapping candidates are simply left for the next scan. We then re-scan and
    repeat up to `max_passes` (scanners re-fetch fresh handles each pass). `max_per_step`
    caps total reconnections per call (runaway guard). Trade-off vs the simpler
    "apply one, re-scan" loop: disjoint batching does more useful work per O(mesh) scan
    and mirrors how the native op drains a work-queue.

ENERGY GATE (optional -- a DEPARTURE from Okuda, off by default).
    Okuda's reconnection trigger is purely GEOMETRIC (edge < dl_th); sorting then emerges
    from long-time self-organisation (a reconnection that creates an unfavourable contact
    later reverses when that contact shrinks). In a finite block over feasible runtimes
    that self-organisation is too slow -- worse, the geometric trigger fires preferentially
    at shrinking HETEROTYPIC faces (highest tension), and I->H there creates a new cap-cap
    contact that is often heterotypic, so ungated reconnection can RAISE het contact area
    (anti-sorting; empirically ~3/4 of triggered reconnections are energetically uphill).
    `energy_gate=True` adds a greedy/Metropolis-at-T=0 acceptance: a tentatively-applied
    reconnection is reversed if it raises the local adhesion energy Sum(lam_ij A_ij). This
    makes reconnection reliably downhill and is what actually drives sorting here. Flagged
    as a modelling departure (PORTING_NOTES); the C++ port can keep it as an option.

CHECK / MUTATE split preserved.
    `apply()` calls the predicate halves (`reconnect.i_to_h_check` / `h_to_i_check`)
    before the mutate halves (`reconnect.i_to_h` / `h_to_i`), keeping the C++-port
    discipline at this layer too.

I<->H preserves cell COUNT and identity (no body is created or destroyed -- only
vertices/surfaces change), so a `bodies` list (and `BodyType.instances`) stays valid for
the whole run; only vertex/surface handles within a touched neighbourhood go stale.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from . import reconnect as rc
from . import topology as topo


# --------------------------------------------------------------------------------------
# parameters + per-call result
# --------------------------------------------------------------------------------------
@dataclass
class OperatorParams:
    dl_th: float                      # Condition-2 trigger length (BELOW equilibrium edge)
    hysteresis: float = 0.0           # placement scale = dl_th*(1+hysteresis); 0 = faithful
    cooldown: int = 8                 # steps a reconnected/reverted 5-cell site is locked out
    volume_guard: bool = False        # reverse a reconnection that inverts a neigh cell (OFF:
                                      # proven a no-op + wrong tool, see faithful_run.py)
    vol_floor: float = 0.0            # ...threshold the volume_guard reverses below (vol <= this)
    energy_gate: bool = False         # reject a reconnection that RAISES local adhesion energy
    energy_tol: float = 1e-9          # slack on the energy gate (accept if dE <= tol)
    temperature: float = 0.0          # Metropolis T for the energy gate: an UPHILL (dE>tol)
                                      # reconnection is KEPT with prob exp(-dE/T) instead of
                                      # always reversed. T=0 => pure greedy (current default,
                                      # nothing changes). T>0 lets the sort escape the greedy
                                      # local optimum (simulated annealing, as in CPM/Potts
                                      # cell sorting). Anneal it toward 0 over a run.
    p_transition: float = 1.0         # per-pass stochastic gate (Okuda Fig.7 ~0.01); 1.0 = off
    max_passes: int = 4               # scan/apply passes per call (handles re-fetched between)
    max_per_step: int = 64            # hard cap on reconnections per call (runaway guard)

    @property
    def place_scale(self) -> float:
        return self.dl_th * (1.0 + self.hysteresis)


@dataclass
class StepStats:
    i_to_h: int = 0
    h_to_i: int = 0
    vetoed: int = 0                   # Condition-4 / structural refusals (pre-mutation)
    reverted: int = 0                 # applied then reversed by the volume guard
    reverted_energy: int = 0          # applied then reversed by the energy gate (uphill, T=0 reject)
    accepted_uphill: int = 0          # uphill reconnection KEPT by the Metropolis gate (T>0)
    failed_revert: int = 0            # guard tripped but reversal could not re-walk (rare/bad)
    passes: int = 0

    @property
    def total(self) -> int:
        return self.i_to_h + self.h_to_i


# --------------------------------------------------------------------------------------
# small read-only helpers
# --------------------------------------------------------------------------------------
def surfaces_of(bodies) -> List:
    """Unique live surfaces of `bodies` (deduped by id; interior faces appear on two)."""
    sd = {}
    for b in bodies:
        for s in b.getSurfaces():
            sd[s.id] = s
    return list(sd.values())


def _neigh_body_ids(cfg) -> frozenset:
    """The 5-cell neighbourhood body ids of an I- or H-config (disjoint-claim + cooldown key)."""
    return frozenset(set(cfg.side_cell_ids) | {cfg.cap_top_id, cfg.cap_bot_id})


def mesh_health(bodies) -> Dict:
    """Cheap validity readout over `bodies`: counts, volumes, validate() failures."""
    vids = {v.id for b in bodies for v in b.getVertices()}
    sids = {s.id for b in bodies for s in b.getSurfaces()}
    vols = [b.volume for b in bodies]
    n_bad_vol = sum(1 for v in vols if (v != v) or v <= 0.0)
    bad_validate = 0
    for b in bodies:
        try:
            if b.validate() is False:
                bad_validate += 1
        except Exception:
            pass
    return dict(
        n_verts=len(vids), n_surfs=len(sids),
        min_vol=min(vols) if vols else 0.0, max_vol=max(vols) if vols else 0.0,
        mean_vol=float(np.mean(vols)) if vols else 0.0,
        n_bad_vol=n_bad_vol, n_bad_validate=bad_validate,
    )


# --------------------------------------------------------------------------------------
# winding-clamp integration -- the stability guard (the "abs-flip" alternative)
# --------------------------------------------------------------------------------------
def _np(p) -> np.ndarray:
    return np.array([p[0], p[1], p[2]], dtype=float)


def _clamp_to(old: np.ndarray, new: np.ndarray, cap: float) -> np.ndarray:
    """Pull `new` back along (new-old) so |result-old| <= cap. Pure (no TF) => unit-testable."""
    d = new - old
    L = float(np.linalg.norm(d))
    if L <= cap or L == 0.0:
        return new
    return old + (cap / L) * d


def stable_step(bodies, rel_frac: float = 0.4, abs_cap: Optional[float] = None) -> Dict:
    """One integration step with a PER-VERTEX DISPLACEMENT LIMITER -- the "winding clamp".

    WHY (the whole point of this guard). The faithful instability is NOT a reconnection bug;
    it is a TF signed-volume WINDING SIGN-FLIP. At finite dt the overdamped integrator can
    overshoot a near-collapsed face, flinging a vertex PAST a neighbour so the face everts;
    the body's signed volume goes negative, the VolumeConstraint force reverses sign, and the
    cell inflates without bound (cells "shoot off") -- see the operator docstring + the
    `faithful-instability-is-winding-signflip` memory and the min_vol -> -218 blow-up in
    scripts/sort_with_video.py runs.

    3DVertVor avoids this NATIVELY with an abs(volume)/orientation repair INSIDE the force, so
    a flipped winding can't reverse the force sign. TF computes volume internally, so that
    cannot be injected from Python, and TF exposes NO Surface winding-reversal (verified
    against tfSurface.h) -- so post-hoc winding repair is impossible from Python. What CAN be
    done is PREVENT the overshoot: cap each vertex's per-step displacement at `rel_frac` of its
    distance to its nearest connected neighbour, so no vertex can cross a neighbour in one step
    and no winding can flip. This is a trust-region limiter on overdamped gradient descent:
    under normal relaxation per-step motion is ~1e-3 (cap ~0.5+; never binds); it binds only on
    the pathological overshoot, where it holds the collapsing edge short until the reconnection
    operator resolves it topologically next pass.

    DEPARTURE from the paper's force-level abs(volume) fix (PORTING_NOTES): the native C++ port
    should use abs(volume) in the VolumeConstraint instead of this position-level clamp.

    `rel_frac` -- cap as a fraction of nearest-neighbour distance (0.5 = cannot cross a neighbour).
    `abs_cap`  -- optional hard ceiling on the cap (None = neighbour-relative only).
    Returns {n_clamped, max_excess}. (`b.volume` is a cache refreshed by positionChanged, so we
    re-fetch it on the bodies whose vertices we moved -- see tfBody.cpp Body::positionChanged.)
    """
    import tissue_forge as tf

    def _fv(a):
        return tf.FVector3(float(a[0]), float(a[1]), float(a[2]))

    # vertex handles + OLD positions (topology is invariant across a pure integration step)
    vmap = {v.id: v for b in bodies for v in b.getVertices()}
    old = {vid: _np(v.position) for vid, v in vmap.items()}

    # implicit-edge adjacency from each surface's ordered vertex ring (consecutive = edge)
    nbr: Dict[int, set] = {vid: set() for vid in vmap}
    for b in bodies:
        for s in b.getSurfaces():
            ring = [v.id for v in s.getVertices() if v.id in vmap]
            for k in range(len(ring)):
                a, c = ring[k], ring[(k + 1) % len(ring)]
                nbr[a].add(c)
                nbr[c].add(a)

    # per-vertex cap = rel_frac * nearest-neighbour distance (can't jump past a neighbour)
    cap: Dict[int, float] = {}
    for vid, ns in nbr.items():
        ds = [float(np.linalg.norm(old[vid] - old[j])) for j in ns if j in old]
        c = rel_frac * min(ds) if ds else (abs_cap if abs_cap is not None else np.inf)
        cap[vid] = min(c, abs_cap) if abs_cap is not None else c

    tf.step()

    n_clamped, max_excess = 0, 0.0
    touched = {}                         # body id -> handle; cached volume must be re-fetched
    for vid, v in vmap.items():
        new = _np(v.position)
        c = cap[vid]
        L = float(np.linalg.norm(new - old[vid]))
        if L > c:
            v.position = _fv(_clamp_to(old[vid], new, c))   # updateChildren=True: refreshes surfaces
            n_clamped += 1
            max_excess = max(max_excess, L - c)
            for b in v.getBodies():
                touched[b.id] = b
    for b in touched.values():
        b.position_changed()             # recompute body volume from the refreshed surfaces
    return dict(n_clamped=n_clamped, max_excess=max_excess)


# --------------------------------------------------------------------------------------
# the operator
# --------------------------------------------------------------------------------------
class ReconnectionOperator:
    """Stateful per-step reconnection driver (carries rng + the cooldown table)."""

    def __init__(self, bodies, stype, params: OperatorParams, rng_seed: int = 0,
                 lam: Optional[Callable[[str, str], float]] = None):
        self.bodies = bodies
        self.bid = {b.id: b for b in bodies}      # cell identity is invariant under I<->H
        self.stype = stype
        self.params = params
        self.rng = np.random.default_rng(rng_seed)
        self.lam = lam                            # type-pair adhesion coeff (for energy_gate)
        self.cooldown: Dict[frozenset, int] = {}  # 5-cell site -> step it unlocks
        self.cum = StepStats()                    # cumulative over the whole run
        if params.energy_gate and lam is None:
            raise ValueError("energy_gate=True requires a lam(name_i, name_j) callable")

    # -- volume guard helpers ----------------------------------------------------------
    def _neigh_min_vol(self, body_ids: frozenset) -> float:
        return min(self.bid[i].volume for i in body_ids if i in self.bid)

    def _neigh_energy(self, body_ids: frozenset) -> float:
        """Local adhesion energy Sum(lam_ij * A_ij) over interior faces touching these
        cells (each face once) -- the quantity differential adhesion minimises. The
        energy gate accepts a reconnection only if it does not raise this."""
        def tname(b):
            t = b.type
            t = t() if callable(t) else t
            return t.name
        seen, E = set(), 0.0
        for i in body_ids:
            b = self.bid.get(i)
            if b is None:
                continue
            for s in b.getSurfaces():
                if s.id in seen:
                    continue
                seen.add(s.id)
                bs = s.getBodies()
                if len(bs) != 2:
                    continue
                E += self.lam(tname(bs[0]), tname(bs[1])) * s.area
        return E

    def _reverse_i_to_h(self, T) -> bool:
        """Undo an I->H by collapsing its triangle back (h_to_i). Returns success."""
        hcfg = topo.h_neighbourhood(T)
        if hcfg is None:
            return False
        return rc.h_to_i(hcfg, self.params.place_scale, check_conditions=False).ok

    def _reverse_h_to_i(self, new_vertex_ids) -> bool:
        """Undo an H->I by re-splitting its recovered edge (i_to_h). Returns success."""
        allv = {v.id: v for b in self.bodies for v in b.getVertices()}
        a, b = new_vertex_ids
        if a not in allv or b not in allv:
            return False
        cfg = topo.i_neighbourhood(allv[a], allv[b]) or topo.i_neighbourhood(allv[b], allv[a])
        if cfg is None:
            return False
        return rc.i_to_h(cfg, self.params.place_scale, self.stype, check_conditions=False).ok

    # -- one quality pass --------------------------------------------------------------
    def apply(self, current_step: int) -> StepStats:
        p = self.params
        place = p.place_scale
        stats = StepStats()

        for _pass in range(p.max_passes):
            stats.passes += 1
            surfaces = surfaces_of(self.bodies)
            edges = topo.find_short_edges(self.bodies, p.dl_th)        # I->H candidates
            tris = topo.find_small_triangles(surfaces, p.dl_th)        # H->I candidates

            cands = [("I", cfg.length, cfg) for (_a, _b, cfg) in edges]
            cands += [("H", h.max_edge, h) for (_t, h) in tris]
            if not cands:
                break
            if p.p_transition < 1.0:
                cands = [c for c in cands if self.rng.random() < p.p_transition]
                if not cands:
                    break
            cands.sort(key=lambda c: c[1])        # most-collapsed first

            claimed: set = set()
            applied = 0
            for kind, _length, cfg in cands:
                bset = _neigh_body_ids(cfg)
                if bset & claimed:
                    continue                       # overlaps an applied site -> next scan
                if self.cooldown.get(bset, -1) > current_step:
                    continue                       # anti-thrash: site locked out

                # ---- check half (predicate) ----
                veto = rc.i_to_h_check(cfg) if kind == "I" else rc.h_to_i_check(cfg)
                if veto is not None:
                    stats.vetoed += 1
                    continue

                # energy BEFORE the tentative mutation (for the energy gate)
                e_pre = self._neigh_energy(bset) if p.energy_gate else 0.0

                # ---- mutate half (created features sized at `place`) ----
                if kind == "I":
                    res = rc.i_to_h(cfg, place, self.stype)
                else:
                    res = rc.h_to_i(cfg, place)
                if not res.ok:
                    stats.vetoed += 1
                    continue

                # ---- guards: reverse if a neighbourhood cell inverted (geometry) or if
                #      the reconnection raised local heterotypic energy (sorting/uphill) ----
                inverted = (p.volume_guard and
                            self._neigh_min_vol(bset) <= p.vol_floor)
                # energy gate: an UPHILL reconnection (dE>tol) is normally reversed; with a
                # finite Metropolis temperature it is KEPT with prob exp(-dE/T), so the sort
                # can climb out of the greedy local optimum (simulated annealing, as in the
                # CPM/Potts cell-sorting model). temperature=0 => pure greedy (default).
                uphill = False
                if p.energy_gate and not inverted:
                    dE = self._neigh_energy(bset) - e_pre
                    if dE > p.energy_tol:
                        if p.temperature > 0.0 and self.rng.random() < np.exp(-dE / p.temperature):
                            stats.accepted_uphill += 1      # keep it (annealing escape)
                        else:
                            uphill = True                   # reject -> reverse below
                if inverted or uphill:
                    ok = (self._reverse_i_to_h(res.new_surface) if kind == "I"
                          else self._reverse_h_to_i(res.new_vertex_ids))
                    if not ok:
                        stats.failed_revert += 1
                    elif uphill:
                        stats.reverted_energy += 1
                    else:
                        stats.reverted += 1
                    self.cooldown[bset] = current_step + p.cooldown
                    claimed |= bset                # don't touch these cells again this pass
                    continue

                # accepted
                claimed |= bset
                self.cooldown[bset] = current_step + p.cooldown
                applied += 1
                if kind == "I":
                    stats.i_to_h += 1
                else:
                    stats.h_to_i += 1
                if stats.total >= p.max_per_step:
                    self._accumulate(stats)
                    return stats

            if applied == 0:
                break

        self._accumulate(stats)
        return stats

    def _accumulate(self, s: StepStats):
        self.cum.i_to_h += s.i_to_h
        self.cum.h_to_i += s.h_to_i
        self.cum.vetoed += s.vetoed
        self.cum.reverted += s.reverted
        self.cum.reverted_energy += s.reverted_energy
        self.cum.accepted_uphill += s.accepted_uphill
        self.cum.failed_revert += s.failed_revert


# --------------------------------------------------------------------------------------
# run loop (shared by the stability + sorting scripts)
# --------------------------------------------------------------------------------------
def run_loop(bodies, stype, params: OperatorParams, n_steps: int,
             types_for_metric=None, lam: Optional[Callable] = None,
             reconnect: bool = True, report_every: int = 25,
             rng_seed: int = 0, on_report: Optional[Callable] = None) -> List[Dict]:
    """Integrate `n_steps`, applying the reconnection operator between steps.

    Records a history row at t=0, every `report_every` steps, and at the final step:
    mesh health + cumulative-since-last-report reconnection counts + (if
    `types_for_metric` given) the sorting metric. Returns the history (list of dicts).
    `reconnect=False` reproduces the frozen control through the same harness.
    """
    import tissue_forge as tf
    from .metrics import contact_summary

    op = (ReconnectionOperator(bodies, stype, params, rng_seed=rng_seed, lam=lam)
          if reconnect else None)
    history: List[Dict] = []
    window = StepStats()

    def record(step: int):
        row = dict(step=step, i_to_h=window.i_to_h, h_to_i=window.h_to_i,
                   vetoed=window.vetoed, reverted=window.reverted,
                   reverted_energy=window.reverted_energy,
                   failed_revert=window.failed_revert)
        row.update(mesh_health(bodies))
        if types_for_metric is not None:
            row.update(contact_summary(types_for_metric, lam=lam))
        history.append(row)
        if on_report is not None:
            on_report(row)

    record(0)
    for i in range(1, n_steps + 1):
        tf.step()
        if op is not None:
            s = op.apply(i)
            window.i_to_h += s.i_to_h
            window.h_to_i += s.h_to_i
            window.vetoed += s.vetoed
            window.reverted += s.reverted
            window.reverted_energy += s.reverted_energy
            window.failed_revert += s.failed_revert
        if i % report_every == 0 or i == n_steps:
            record(i)
            window = StepStats()
    return history
