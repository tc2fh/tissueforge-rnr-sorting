# RNR reconnection + cell sorting — durable learnings (Phase 2)

> Durable copy of the 2026-05-30 session's learnings (Phase 2: wire the I↔H 3D-T1
> reconnection into the dynamics + reproduce sorting). Kept in-repo so it survives an
> assistant-memory wipe. Companions: `CLAUDE.md` (source of truth), `progress.md` (status),
> `rnr/PORTING_NOTES.md` (§4c/§5/§6/§7 API + port seams), project memory
> `phase2-sorting-partial` + `faithful-instability-is-winding-signflip`.

## 0. TL;DR

Phase 2's gate is **met**: `pixi run sort` demixes — **het pairs 194→189** vs a matched
frozen control held at 194, with **bounded** (non-catastrophic) stability. The hard part
was not the reconnection (Phase 1 proved it reversible) but the **dynamics**: under
heterotypic tension a reconnection can trigger a **TF signed-volume winding sign-flip**
that reverses the VolumeConstraint force and inflates a cell without bound. The fix that
makes the demo work is **small dt** (2e-4); the *real* fix for the Okuda-faithful regime
is a **native volume-sign/winding repair** (Phase 3, can't be done from Python).

## 1. The central finding — the winding sign-flip runaway

The full causal chain (measured, `rnr/scripts/faithful_run.py` + `faithful_probe.py`):

1. Linear heterotypic tension (`Adhesion` λ·A on cell–cell surfaces) drives het faces to
   **collapse toward zero** — interior edges reach ~1e-4 in our finite block. (So a
   reference-faithful tiny Δl_th ~1e-3 *is* feasible here; my pre-compaction worry was wrong.)
2. At **too-large dt**, the explicit overdamped integrator **overshoots** a near-collapsed
   face: a vertex crosses *through* it, so that face's **winding flips**.
3. A flipped face makes TF's **signed** cell volume `Body.volume` go **negative even though
   the cell is geometrically intact** (vertices still span a normal region).
4. The `VolumeConstraint` force `∝ (V − V₀)` then **points the wrong way** → the cell
   **inflates without bound** (RUNAWAY: min_vol → −8e4, hulls seen ≤73× target) → cascade →
   the metric **freezes on a corrupted mesh**.

**It is a sign-flip, not a collapse — proven.** `faithful_run.py`'s orientation diagnostic
computes two orientation-FREE volumes for every `Body.volume ≤ 0` cell:
- **convex-hull volume** of the vertices (rules out collapse-to-point/plane), and
- **`V_geo`** = divergence-theorem volume with face normals forced outward by the cell
  centroid (positive iff the cell is *untangled*).
Result at peak badness: **21/25 and 8/8 cells have `V_geo > 0`**, often at *exact magnitude*
vs the negative `Body.volume` (e.g. TF −10.10 / V_geo +10.61; TF −4.347 / +4.347). That is
exactly 3DVertVor's `abs(volume)`+flip case — **not** the catastrophic genuine collapse
(V→−6e5) seen under the OLD negative-tension energetics.

> ⚠️ This **reverses** a pre-compaction claim ("my negatives are genuine collapse, so
> abs-flip would mask them"). That was true only for the old negative-homotypic-λ substrate.
> Under the faithful (non-negative-tension) energetics the negatives are winding sign-flips.

## 2. dt is the master stability lever (not a guard, not the energy gate)

- **Frozen substrate** (reconnection OFF entirely): inverts cells at **dt=1e-3** but is
  **STABLE at dt=2e-4** over the same physical time. Smaller dt ⇒ per-step displacement
  `μ·F·dt` stays **below the face-crossing threshold** ⇒ no overshoot ⇒ no flip.
- **`pixi run sort` runs at dt=2e-4** and stays bounded (worst min_vol ≈ −7, transient)
  while demixing. At dt=1e-3 it runs away (and the old "194→190" was a *coarse-dt
  corrupted-frozen artifact*, not real sorting — caught this session).
- The older finding still holds as a *separate* layer: **negative homotypic λ + dt=0.005
  blows up with reconnection OFF** (faces grow unbounded). Non-negative tensions fixed that;
  small dt fixes the sign-flip layer on top.

## 3. What's faithful vs what's a departure

Read the reference repos (`tvm`, `3DVertVor`) as an oracle only (GPL — never paste).

| Lever | Okuda/references | Our prototype | Status |
|---|---|---|---|
| Trigger | geometric, edge < Lth≈1e-3 | edge < Δl_th | faithful (Condition 2; H→I uses **max** triangle edge) |
| Feature size | place ≈ Lth (infinitesimal) | `hysteresis=0` → place=Δl_th | faithful default |
| Volume | stiff (kv≈10) | `volume_lam` | faithful-ish |
| Acceptance | **none** (purely geometric) | **energy gate** (reverse uphill) | **DEPARTURE** — see below |
| Stability backstop | tiny dt + `abs(volume)`+flip | small dt (+ optional volume guard) | partial — abs-flip is native-only |

**Why the energy gate is needed in Python (a real result):** the bare geometric trigger
fires at shrinking *het* faces, and the new cap–cap contact it creates is often *het* too,
so ungated reconnection **adds** het contacts (**anti-sorting**, ~¾ of reconnections are
uphill). The `energy_gate` reverses uphill reconnections (greedy/Metropolis-T=0) → sorting.
But the gate needs a **finite feature** to have a measurable ΔE: at faithful-*infinitesimal*
features it goes blind and the run anti-sorts again. So the **pragmatic `pixi run sort`**
deliberately uses **big features (Δl_th=0.45)** so the gate works, **plus the volume guard**
(which at big features reverses the direct mutate-time inversions — 834 fires; at small
features it never fires, `cum.reverted=0`, so it's default-OFF in `operator.py`).

The faithful (tiny-feature, no-gate, no-guard) path is the scientifically-correct one but
needs the native repair (§5) before it sorts.

## 4. The two-task deliverable layout

- **`pixi run sort`** → `rnr/scripts/sort_with_reconnect.py` — the PRAGMATIC demo that
  sorts. dt=2e-4, big features + energy gate + volume guard. Output:
  `rnr/exports/sort_with_reconnect.{png,csv}`. Result: het 194→189, het-area 0.342 vs 0.364,
  energy 166 vs 185, bounded stability.
- **`pixi run sort-faithful`** → `rnr/scripts/faithful_run.py` — the reference-faithful
  DIAGNOSTIC (anti-sorts + the orientation diagnostic showing why native repair is needed).
- **`pixi run faithful-probe`** → `rnr/scripts/faithful_probe.py` — frozen substrate:
  edge-collapse feasibility + dt-stability sweep.
- **`pixi run test`** → 11 gates, green + deterministic. (The clean-integration gate was
  de-flaked by lowering the test fixture dt 0.01→0.001 — itself a confirmation of §2.) The
  two newest gates pin `metrics.demixing_index` (§7) on hand-enumerated + Kelvin configs.

## 5. The native repair (Phase-3 unblock — DO NOT start without explicit go-ahead)

The one failure mode Python cannot fix: TF computes `Body.volume` internally, so a
sign-flipped volume feeds a wrong-signed `VolumeConstraint` force we cannot intercept.
Native fix, in the LGPL fork's `source/models/vertex/solver/`:
- make `Body::getVolume()` / the `VolumeConstraint` force **robust to a transient face-winding
  flip** — clamp to `abs`, or use a **one-sided `V>0` penalty** so a flipped sign cannot
  drive runaway inflation; AND
- ensure the reconnection surgery (the future native `MeshQualityOperation`) writes
  **consistent surface winding**.
This unblocks the Okuda-pure infinitesimal-feature trigger and lets the **energy-gate
departure be dropped**. CLAUDE.md keeps the C++ port a *later, separate* project — this is a
note *for* that port, not a cue to start it.

## 6. Gotchas specific to this work

- **Instability corrupts the geometric metrics.** Inflated (sign-flipped) cells gain
  neighbours and area, so het-AREA fraction and even het-PAIR count rise spuriously during a
  blow-up. The het-PAIR count is robust *only while the mesh is healthy* — always read it
  alongside `min_vol`/`n_bad_vol`, and distrust a "sorting" number if the mesh is corrupted.
- **`Δl_th` must sit BELOW the equilibrium edge (~0.707 for the span-8 Kelvin block)** or the
  whole pristine foam reconnects at once. (`test_find_short_edges_below_equilibrium_is_empty`.)
- **Handle re-fetch after any mutation** via body-disjoint batching (disjoint 5-cell
  neighbourhoods share no handles). I↔H preserves cell **count + identity**, so a `bodies`
  list stays valid; only vertex/surface handles in a touched neighbourhood go stale.
- **`contact_summary(bodies=...)`** with an explicit list is required when ON/OFF blocks
  share global types in one universe (else `type.instances` mixes them).
- **The orientation diagnostic recipe** (reusable): for a suspect cell, compare `Body.volume`
  against hull volume + outward-normal `V_geo`. `V_geo>0` & TF<0 ⇒ winding sign-flip;
  `V_geo`≈0/tiny hull ⇒ genuine collapse.

## 7. Scale-up: the demixing index + dt-must-scale-with-block-density (2026-05-30)

Making the demixing **dramatic + quantitatively validated** (still pure Python):

- **Demixing index.** `rnr/metrics.demixing_index` = cell-average `2·(het_frac − ½)` ∈ [−1,+1],
  **signed so more sorted = more negative** (a deliberate departure-note vs 3DVertVor's positive
  convention, chosen so it falls with the het-pair/area/energy curves). It is TOPOLOGICAL
  (neighbour identity), so — unlike het-area-fraction — it does NOT drift under mere shape
  relaxation: in the frozen control and before the first reconnection it is exactly flat. Two
  deterministic gates pin it: a hand-enumerated minimal-[I] config (`D=0.4` for wedges=A/caps=B;
  `D=−1` uniform) and a Kelvin block (planar A|B split `D<−0.4`; salt-and-pepper `|D|<0.25`).
  `contact_summary` now returns `demixing_index`; `pixi run sort` plots `D(t)` ON-vs-OFF as the
  headline panel (6-panel layout).

- **Headline result** (`pixi run sort 8000 0.45 1.0 0.0001 5`, N=5 → 189 cells/block, dt=1e-4):
  **D −0.043 → −0.098 (2.3×)** while the matched control stays pinned at −0.043; **het 460→447
  (−13)** vs frozen 460; het-area 0.385 vs 0.417; adhesion energy 438 vs 459; 285 reconnections.
  Plateaus by ~t=4000 and HOLDS below OFF (sustained). Worst min_vol −1.28, worst bad-cell 7
  (transient, recovered to 0). Artifact: `rnr/exports/sort_with_reconnect_n5_dt1e4.{png,csv}`.

- **THE scale-up finding — dt must scale DOWN with block density** (extends §2). At N=5 the
  *same* run at dt=2e-4 (the N=4 stable dt) accumulated **15–25 simultaneously sign-flipped
  cells** (min_vol −7.9) and the geometric metrics (af, E) went erratic — a denser block has
  more het faces near collapse at once, so more cross the face-crossing threshold per step.
  **Halving dt to 1e-4 reached the same D≈−0.10 plateau cleanly** (bad≤7, recovering). So dt is
  the master stability lever AND its safe value falls as cells/box rises. Practical rule for the
  finite-block demo: when scaling `N_PER_AXIS` up, scale `dt` down (≈÷2 per +1 axis here).

- **Why the demix is partial (D≈−0.1, not −1).** The greedy (T=0) energy gate reaches a local
  optimum: once the easy het-reducing swaps are taken and faces stop collapsing below `dl_th`,
  reconnection supply dries up (recon plateaus ~285) and D freezes. Fuller demixing would need a
  stronger drive (higher het λ → more face collapse), Okuda's rate-limiting `p_transition`, or
  finite-T/annealed acceptance — all follow-ups, all within the Python phase. The native
  abs(volume) repair (§5) is the orthogonal unblock for the *faithful* (no-gate) regime.

- **Operational:** `sort_with_reconnect.py` now (a) takes `N_PER_AXIS` as a 5th CLI arg with
  `SPAN=2·N` auto-tracking the lattice so the equilibrium edge stays ~0.707, (b) checkpoints
  CSV+PNG every 2000 steps so a long background run is inspectable mid-flight and robust to an
  early kill, and (c) takes `TEMP0`/`ANNEAL_TAU` (6th/7th args) for the annealed gate below.

### Step-3 tuning experiments — BOTH negative (greedy D≈−0.10 is near the accessible optimum)

Two knobs were tested to push past the greedy plateau; neither beats it on the demixing index.

- **Stronger heterotypic λ does NOT deepen the plateau** (only speeds reaching it). Under
  uniform λ scaling `dE = λ·ΔA`, so the *sign* of every swap's ΔE — hence the greedy
  accept/reject decision — is λ-independent; λ only sets how fast faces collapse below `dl_th`.
  Empirically (N=5, dt=1e-4): **λ=2 plateaus at D≈−0.099/het≈441 by t≈2600** vs λ=1's
  −0.098/447 by t≈4000 — same level, ~2× faster, then λ=2 *destabilises* (~t=4800, min_vol
  −4.5, bad=7) because the stronger drive collapses more faces at once → more sign-flips (again:
  a stronger drive, like a denser block, wants a smaller dt).

- **Naive Metropolis annealing on the energy gate does NOT beat greedy** (it slightly hurts D).
  Added `operator.OperatorParams.temperature`: an uphill reconnection (dE>tol) is KEPT w.p.
  `exp(−dE/T)` instead of always reversed (T=0 = the greedy default, unchanged → tests stay
  green). Annealed schedule `T = T0·exp(−step/τ)`. Run N=5, dt=1e-4, **T0=0.5, τ=4000**: it
  churned far more (recon **406** vs greedy 285) but **settled at D −0.079 / het-frac 0.459**,
  *worse* than greedy's −0.098 / 0.447. The accepted uphill moves changed coordination (total
  pairs ~957 vs 1001) and raised the per-cell het fraction rather than lowering it; cooling did
  not recover (D flat from t≈3800). **Worse still, it then DESTABILISED catastrophically**: after
  t≈6800 the uphill-acceptance churn triggered a sign-flip cascade — **min_vol → −343, 45/189
  (24%) bad cells by t=8000** (the D/het readings past ~t=6800 are on a corrupting mesh). So
  naive annealing is worse on BOTH counts (sorting AND stability). Caveat: only one (hot) schedule
  tried — a cooler T0 / faster τ might avoid the blow-up — but T0=0.5 is clearly not the easy win.
  The `temperature` knob is kept (tested, default-off) for future schedules + as a C++-port note.

- **Takeaway.** On a finite 189-cell block the I↔H move-set + greedy gate reaches **D≈−0.10**,
  and that is near the *accessible* optimum here — not a tuning failure. Pushing toward full
  demix (D→−0.3+) most likely needs a **larger / periodic** system (3DVertVor uses a periodic
  1728-cell box) where coordination is bulk-like and the move-set isn't boundary-limited, and/or
  a smarter acceptance than naive Metropolis. That is the natural next scale-up, beyond this
  finite-block Python phase. **[§8 tests this hypothesis against the oracle — and largely
  REFUTES the strong form of it.]**

## 8. Oracle comparison: does the periodic/bulk 3DVertVor break our ceiling? (2026-05-30)

We built the `3DVertVor` oracle and compared demixing *trends* (never absolute numbers). The
build + run recipe is in `oracle_run/README.md`; the firsthand oracle characterization (and the
correction of an earlier session's fabricated specifics) is in `docs/oracle_step0_groundtruth.md`.

**Getting it to run was the hard part (all documented, GPL-clean — numbers only, no code copied):**
- The README says it "needs VTK", but VTK is used **only** by the `.vtu` writer. We stripped VTK +
  `python2.7` + `std::experimental::filesystem` and built **Eigen-only** (Eigen is already a pixi
  dep). The base engine is pure C++11.
- **C++11 → 17** (modern conda Eigen needs ≥14); **3 missing-`return` UB traps** (`magForce`,
  `updateSpringArea`, `updateAverageForce`) that clang19 `-O3` compiles to `brk #1` — the authors'
  older compiler tolerated them.
- **pyvoro-mmalahe's `periodic=True` is broken** (returns wall faces + cells outside the box →
  inconsistent mesh; the version incompatibility CLAUDE.md warned about). We generate periodicity
  ourselves with **scipy.spatial.Voronoi + 27-image replication** (`oracle_run/make_periodic_topo.py`,
  OUR code, oracle file format) → a clean periodic 216-cell, 0-non-4-valent foam (Euler-checked).
- **Box size is hard-coded** `Lx_=Ly_=Lz_=12` in `Run::Run()`; it MUST match the generated box
  (we use 6³/216) or the periodic min-image in `updateCenter`/`updateArea`/`Volume` is wrong and
  cell volumes blow up (we saw total V=727 vs the correct 216 before fixing this).

**The energetics map (oracle re-derived from `Energy/{Volume,Interface}.cpp`; ours read from
TF actor source — cited, not copied):**

| term | oracle (3DVertVor) | ours (rnr/TissueForge) |
|---|---|---|
| volume | `kv(V−v0)²`, kv=10, v0=1 | `λ_V(V−V0)²`, λ_V=1, V0=4 (`VolumeConstraint`) |
| surface | `(s−s0)²` (coeff 1), s0=5.6, shape idx 5.6 | `λ_S(A−A0)²`, λ_S=**0.1**, A0≈15.12, shape idx **6.0** (`SurfaceAreaConstraint`, tfSurfaceAreaConstraint.cpp:34) |
| het tension | `σ·A` — **AREA** of het faces, σ=1 | `Adhesion` — **EDGE/LINE** tension `λ·Σ(edge len × #partner-type nbr surfaces)`, λ=1 (tfAdhesion.cpp:137) |

So volume + surface are the **SAME quadratic Hookean form** on BOTH sides (an earlier draft
wrongly said the surface term differed — it does NOT). The genuine functional difference is the
**het tension**: the oracle's is **area-based** `σ·A`, ours (TF `Adhesion`) is **edge/line-based**
(`metrics.contact_summary`'s `Σλ·A` is only an area *proxy*, not the actor energy). Plus the
COEFFICIENTS differ: our surface stiffness 0.1 vs the oracle's 1 (→ our het-to-surface ratio is
~10× larger), volume 1 vs 10, cell size V0 4 vs 1, target shape index 6.0 vs 5.6. **Bottom line:
σ=1 ≈ λ_het=1 is NOT a matched drive** — which is exactly why ours reaches |D|≈0.10 at λ=1 while
the oracle needs σ≈2 to get there. s0=5.6/6.0 is fluid (above the ~5.4 rigidity transition → can
rearrange). `temperature=0` → passive (no active motility). `Run::dumpDemix` computes **exactly
our index with the opposite sign**: `demix = ⟨2(hom_frac−½)⟩ = −D_ours` (count/face-based) — a
clean cross-check of our `metrics.demixing_index`.

**Results (`pixi run compare-oracle` → `rnr/exports/oracle_comparison.{png,csv}`):**

| run | het tension | demix peak (sorted=+) | S_peak=1−hf/hf0 | stability |
|---|---|---|---|---|
| ours (finite 189, greedy gate) | λ_het=1 | \|D\|=**0.098** | S_count 0.058 / S_area 0.20 | stable (dt=1e-4) |
| oracle (periodic 216) | σ=1 | **0.065** | 0.110 | **stable to t=300** |
| oracle (periodic 216) | σ=2 | **0.110** | 0.152 | **CRASHED t=72** |
| oracle (periodic 216) | σ=3 | **0.087** | 0.131 | **CRASHED t=6** |

**THE VERDICT (trend/shape + relative magnitude only):**
1. **The periodic bulk does NOT dramatically break our finite-block ceiling at matched energetics.**
   At σ=1 the oracle plateaus at demix≈0.06 — the **same order** as our |D|≈0.10 — and both show
   the **same shape**: a monotone rise to an early plateau (not a creeping climb). The plateau is
   real in BOTH codes, so our D≈−0.10 is **largely physical**, not just a finite-block / move-set
   artifact. This **revises the §7 hypothesis** ("a periodic/bulk system should sort further").
2. **Stronger drive sorts deeper but destabilises — in BOTH codes.** σ=2 reaches demix≈0.11 (=our
   |D|) and σ=3 climbs fastest, but each **crashes** on the oracle's *own* reconnection:
   `Reconnection Error: c123 and c456 have the same direction` — a **winding/orientation
   degeneracy**, the same failure family as our **winding sign-flip** (§1). Crash time falls with
   σ (t=72 at σ=2, t=6 at σ=3), exactly our "stronger drive / denser block wants smaller dt →
   else more sign-flips" rule (§2, §7).
3. **Implication for the project.** Deeper demixing is gated by a **het-tension-vs-stability
   tradeoff common to both implementations**, not by our being finite or Python. So the highest-
   leverage unlock is the **native winding / abs(volume) reconnection repair** (§5 / Phase-3) —
   which would let *either* code push σ higher without the degeneracy crash — MORE than simply
   going periodic or bigger.

**Caveats (do not over-read the match):** periodic (oracle) vs free-boundary (ours); 216 vs 189
cells; **σ=1 is NOT a matched drive** — volume + surface terms share the SAME quadratic form, but
the het tension differs (oracle AREA `σ·A` vs our EDGE/LINE `Adhesion`) and the coefficients/scale
differ (our surface stiffness 0.1 vs 1 → het-to-surface ratio ~10× larger; volume 1 vs 10; V0 4 vs
1; shape idx 6.0 vs 5.6); different integrators + time units (x normalised to fraction-of-run);
demix is count/face-based
both sides (we also report our area-weighted S, which is larger, 0.20, but drifts under shape
relaxation — the frozen control confirms the *count* metric is the clean one); S is normalised by
hf(0), sensitive to the random-init offset; our greedy energy gate biases us toward sorting (no
such gate in the oracle). The claim is **trend/shape + order-of-magnitude**, never a numeric match.
