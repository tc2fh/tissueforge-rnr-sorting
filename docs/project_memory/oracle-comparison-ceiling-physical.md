---
name: oracle-comparison-ceiling-physical
description: "Step-4 3DVertVor oracle comparison — periodic bulk does NOT break our D≈−0.10 ceiling at matched energetics; the limit is largely physical (het-tension-vs-reconnection-stability tradeoff shared by both codes), so the real unlock is the native winding repair, not going periodic/bigger"
metadata: 
  node_type: memory
  type: project
  originSessionId: 173c3c41-a899-4b19-ac9f-7bf805b938de
---

Step 4 (2026-05-30): built the GPL `3DVertVor` oracle **VTK-free** (`oracle_run/`, Eigen-only —
VTK was only the `.vtu` writer; also stripped python2.7 + experimental::filesystem, bumped C++11→17
for modern Eigen, fixed 3 missing-`return` UB traps clang19 -O3 turns into `brk #1`, and generated
periodicity with scipy since pyvoro-mmalahe's `periodic=True` is broken). Ran a periodic 216-cell
two-type sort (γ_homo=0, γ_het=σ, passive) and compared the normalized demixing trend
`S=1−hf/hf0` (`pixi run compare-oracle`).

**Verdict:** the periodic/bulk oracle does **NOT** dramatically exceed our finite-block ceiling at
matched energetics. Matched σ=1 oracle plateaus at demix≈**0.06** (stable to t=300); our prototype
|D|≈**0.098** — same order (~0.06–0.10), both **monotone-rise-then-plateau** (same shape). Stronger
drive σ=2→0.11 (crash t=72), σ=3→0.087 (crash t=6) sorts deeper but the oracle's OWN reconnection
aborts on a **winding degeneracy** ("c123/c456 same direction") — the same failure family as our
**winding sign-flip** [[faithful-instability-is-winding-signflip]]. So the demixing limit is a
**het-tension-vs-reconnection-stability tradeoff common to both implementations**, largely
PHYSICAL — this **revises** the earlier "periodic should sort further" hypothesis
[[phase2-sorting-partial]].

**Why it matters / how to apply:** the highest-leverage next step is the **native winding /
abs(volume) reconnection repair** (lets either code push σ higher without the degeneracy crash),
NOT merely going periodic or to more cells. Useful cross-checks discovered: oracle het tension is
`σ·A` in `Energy/Interface.cpp` (= our `Adhesion`); `Run::dumpDemix` computes exactly our index with
opposite sign (`⟨2(hom_frac−½)⟩ = −D_ours`); oracle box size is hard-coded `Lx_=12` in `Run::Run()`
and MUST match the generated topo box or periodic volumes blow up. **Caveat (σ=1 is NOT a matched
drive):** volume + surface terms share the SAME quadratic Hookean form (TF `SurfaceAreaConstraint`
= `λ(A−A0)²`, tfSurfaceAreaConstraint.cpp:34); the GENUINE difference is the het tension — oracle
is AREA `σ·A`, ours (TF `Adhesion`) is EDGE/LINE `λ·Σ(edge len × #partner-type nbr surfaces)`
(tfAdhesion.cpp:137; `contact_summary`'s Σλ·A is only an area proxy). Coeffs differ too (surface
stiffness 0.1 vs 1 → our het:surface ratio ~10× larger; volume 1 vs 10; V0 4 vs 1), which is why
ours hits |D|≈0.10 at λ=1 while the oracle needs σ≈2. Trend/shape only, never numeric. (An earlier
write of this memory wrongly called the surface term a form difference — corrected.) Build recipe:
`oracle_run/README.md`; science: `docs/rnr_sorting_notes.md §8`;
oracle ground truth: `docs/oracle_step0_groundtruth.md`. License clean: numbers only, no GPL copied.
