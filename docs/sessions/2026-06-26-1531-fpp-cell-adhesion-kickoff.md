# Extensibility layer shipped; next = FPP centroid-to-centroid cell adhesion

## Summary 2026-06-26 15:31 EDT

Built + committed (`35e01c4`, **pushed** to origin) a **user-extensibility layer** on the Warp GPU
3D-vertex engine so new modeling plugs in as hooks WITHOUT editing the validated fused core
(`physics_warp.force_kernel` + host ref `physics_csr.py` UNCHANGED, byte-identical when no hooks).

**What changed and why:**
- `rnr/gpu/extensions.py` — `CellState`: durable per-cell field store, dict-like so it stands in for
  the engine's `phys` dict. `add_field(name, init, dtype=None)` (infers vec3d / int32(incl bool) /
  float64), `cells.polarity`==`cells['polarity']`, `fields()` introspection.
- `rnr/gpu/engine.py` — `forward_step` gained `behaviors=()/forces=()` hook lists (empty == old core
  path, **guarded** by `test_crawl.test_empty_hooks_byte_identical`); `Engine` facade (`@eng.behavior`/
  `@eng.force`, `.step()`).
- `rnr/examples/crawl.py` — `migration_force` (whole-cell SPV propulsion → cell TRANSLATES, stays
  compact) vs `lamellipodium_force` (leading-edge only → protrusion that ELONGATES into spiky shards;
  kept as the documented failure mode) + `persistent_repolarization`. **Lesson:** a leading-edge
  protrusion ≠ migration (it stretches, doesn't move); whole-cell propulsion at v0-scale `f_mag≈0.1–0.3`
  (NOT 2.0) is the migration drive.
- `rnr/examples/energy_terms.py` — `edge_length_penalty` + `face_area_penalty` (global regularizer
  force hooks) supplying the LOCAL edge/face stiffness the engine lacks (it has only per-cell VOLUME +
  per-cell TOTAL-area + tension; no edge-length term, and total-area lets a cell redistribute area among
  faces) → they cure the protrusion's spikes (demo'd: `nv` 331 stable vs ballooning to 409).
- `rnr/scripts/gpu_crawl_video.py` — turntable demo video (`MODE` migration/lamellipodium, `REG`
  edge,face, `REG_K` strength, `N_CRAWLERS` scattered tracked crawlers via farthest-point sampling).

**Key invariant (drives the whole design — internalize it):** bodies never change slots under
reconnection/compaction → a per-body array sized `nb` is **durable** all sim, no remap. Vertices are
renumbered + born-blank → per-vertex persistent state is **fragile** (not built). Express "specific
vertices" as per-cell state + an IN-KERNEL geometric selector, never tagged vertex ids.

**Validation pattern (reuse it for adhesion):** for a coupled energy term, a finite-difference
GRADIENT test (`rnr/tests/test_energy_terms.py`): launch the hook, confirm `force == -dE/dx` by
central-differencing the same energy on the host. It CAUGHT a real 2× in `face_area_penalty` —
`physics_warp.d_area_grad` returns the gradient of the UNHALVED cross-product area (`= -2·dA/dx`; the
engine's K_A convention omits the ½), so it needs a `0.5`. `crawl.py` drives get deterministic
per-vertex tests (`test_crawl.py`).

**Build/test/git state:**
- **Branch `migrate/linux64-wsl2`**, HEAD `35e01c4`, **in sync with origin** (pushed this session).
- Gate **139 passed** earlier this session (134 + test_crawl 3 + test_energy_terms 2) — backs the
  committed code; nothing test-affecting changed since (only this handoff doc + export blobs).
- Engine fork (`feat/native-rnr-reconnection`) **unchanged** this session.
- `git status --short`: only `rnr/exports/*` blobs (videos/CSVs/PNGs/frames — **LEAVE**, prior+this
  session's render artifacts) + this new handoff. Nothing else uncommitted.

```
 M rnr/exports/{dpmax.json,fig1e_*,fig1f_*}                 <- LEAVE (blobs)
?? rnr/exports/gpu_crawl_demo.mp4                            <- LEAVE (5-crawler migration demo)
?? rnr/exports/gpu_crawl_lamellipodium_reg.mp4              <- LEAVE (deformation demo, REG_K=2.5, 3000 steps)
?? rnr/exports/gpu_crawl_frames/                            <- LEAVE
?? rnr/exports/gpu_scalability_*, gpu_sort_*, sort_oracle_* <- LEAVE (prior-session blobs)
?? docs/sessions/2026-06-26-1531-fpp-cell-adhesion-kickoff.md  <- COMMIT (this handoff)
```
(Memory `gpu-extensibility-architecture` + MEMORY.md updated, in `~/.claude`, not the repo.)

---

## Kickoff — next session

You are continuing the Warp/CUDA GPU 3D-vertex engine + RNR (RTX 5090) on branch
`migrate/linux64-wsl2`. Last session shipped a **user-extensibility layer** (force/behavior hooks +
`CellState`, commit `35e01c4`). **This session: add a Hookean centroid-to-centroid CELL-ADHESION
constraint — a CompuCell3D FocalPointPlasticity (FPP) analog — as a force hook.** It's a genuinely
useful constraint (springs that hold/pull cells together by their centers) and a clean fit for the new
architecture. Read the committed `rnr/gpu/extensions.py` + `rnr/examples/energy_terms.py` first — you
will MIRROR `edge_length_penalty`'s structure + the FD-gradient test.

**The model (grounded in the user's CC3D sim — READ IT FIRST):**
`/mnt/d/Work/GlazierFoxLab/CC3D_GPU/Embryo_Model_dev/Embryo/Simulation/` — `Embryo.xml:78–129` (the
`FocalPointPlasticity` plugin, per type-pair) + `EmbryoSteppables.py` (`new_fpp_link(cell, target,
Lambda, TargetDist, MaxDist)`, params `SLinkLambda=10, SLink_TargetDist=1, SLinkMaxDist=5`). FPP = a
spring between two cells' centers of mass: **energy `λ·(d − L0)²`** where `d = |c_i − c_j|`, `L0` =
TargetDistance; a link **breaks** beyond `MaxDistance`; params are per cell-type-pair, with a
`MaxNumberOfJunctions`. (CC3D's `LinkConstituentLaw` lets you swap the formula, e.g. `λ·|d−L0|` or
`λ·d` — the user's Leading↔Substrate lamellipodia link uses `λ·Length`.) This is a reference ORACLE;
reimplement the physics, don't copy code.

**Design (force hook, in priority order):**
1. **Read** the CC3D FPP model above for the exact law + params the user wants; decide energy
   convention (CC3D uses `λ(d−L0)²`, no ½ — or use `(k/2)(d−L0)²`, just be consistent in the FD test).
2. **`rnr/examples/adhesion.py`** — `centroid_adhesion(links, k, l0)` force hook `fn(g, cells, geom)`:
   - **Centroid = VERTEX-MEAN** `c_i = (1/N_i) Σ_{v∈i} x_v`, NOT the engine's area-weighted `bcent`
     (its gradient is messy). Vertex-mean gives the trivial chain rule `dc_i/dx_v = 1/N_i`. **Min-image
     unwrap** each cell's vertices before averaging (periodic box), and min-image the inter-centroid
     distance.
   - Per link `(i,j)`: `F_i = -2λ(d − L0)·ĉ_ij` (toward partner if too far = adhesion; away if too
     close), then **scatter `F_i/N_i` to every vertex of cell i** and `F_j = -F_i` to cell j. That
     translates each cell as a unit → Newton's 3rd law holds.
   - Implementation: a small reduction for per-body vertex-mean centroids + counts, then per-link
     force, then a per-vertex scatter kernel summing over the links its cell is in. `M` (links) is
     small; iterate all links per vertex, or build a cell→links adjacency.
   - **Link state is per-cell-PAIR**, `links (M,2)` body-id array + per-link `L0/λ` (or per-type-pair
     lookup via `body_type`). Durable (bodies stable). It is NOT a `CellState` field (that's per-cell,
     `nb`) — pass `links` into the factory (closed over) or stash on the `Engine`. **v1 = STATIC link
     list** (e.g. predefined pairs / all same-type within a distance at t=0). v2 (later) = dynamic
     form/break behavior hook (full FPP: form links with neighbors within `MaxDistance`, break beyond).
3. **`rnr/tests/test_adhesion.py`** — (a) FD-gradient test (mirror `test_energy_terms.py`: `force ==
   -dE/dx` for `E = Σ λ(d−L0)²`; vertex-mean centroid keeps it clean — watch for the same kind of
   ½/2× convention slip the face-area term had); (b) a 2-cell settle test: two linked cells, core
   forces off, relax → centroid separation → `L0`.
4. **Demo** — a short video (new `gpu_adhesion_video.py` or extend `gpu_crawl_video.py`): two cell
   groups linked by FPP springs cluster / hold at target distance; color linked cells.

**Caveats:** use vertex-mean centroid (not `bcent`); periodic min-image for BOTH centroid build and
inter-centroid distance; recompute `N_i`/centroid every step (a cell's vertex set changes under
reconnection); per-PAIR is a NEW hook shape (existing hooks are per-cell or global) so it may motivate
a tiny `links` abstraction on the `Engine`. Keep the hook capture-safe (alloc-free, no per-step host
readback).

**Commands:** `pixi run test` (139 gate) · `pixi run python -m pytest rnr/tests/test_adhesion.py -q`
(new gate) · demo render once built. **Scope/license:** reimplement from the FPP physics + Okuda 2013
/ our `rnr/`; NEVER copy GPL `tvm/`. Read-only oracles (own `.git`, never commit): `tvm/ 3DVertVor/
cellGPU/ VertAX/ gpu_reference_papers/` + the CC3D model dir. `tissue-forge/` is our active fork
(`feat/native-rnr-reconnection`); unchanged this session. Commit at handoff (standing auth); push only
on explicit ask.
