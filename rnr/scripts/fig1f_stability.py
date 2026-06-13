"""Reproduce Manning2024 Fig 1F: the demixed state is energetically preferred (stays demixed).

THE TARGET (Manning2024 Fig 1F, p.7): initialize the 3D vertex tissue in a COMPLETELY DEMIXED
(segregated) state and show the boundary stays stable -- DP/DP_max remains near 1 -- confirming
the demixed state is energetically preferred (the dark-magenta "initially demixed" curve). The
companion "initially mixed" curve rises toward it. We pair our demixed-IC runs against the same-
sigma mixed-IC runs from Fig 1E.

WHY THIS IS THE CLEAN COUNTERPART TO FIG 1E HERE: our mixed-IC count-based DP is reconnection/
finite-N limited (it needs domain formation at large N -- see fig1e_demixing.py / PORTING_NOTES
6l), so it stays near 0 at N=216,t=100. The demixed IC SIDESTEPS that: DP starts at DP_max, so the
test is simply whether the het tension HOLDS it there. If demixed stays ~1 while mixed stays ~0,
that is direct evidence the energetics + native RNR are correct (the demixed state is a stable
minimum), even though we cannot reach it from the mixed side at this scale.

DP = -demixing_index (Sahu Eq.2; re-derived, not copied). DP_max from compute_dpmax.py (dpmax.json).
Demixed CSVs: ..._seed{N}_demixed.csv ; mixed CSVs: ..._seed{N}.csv (the Fig 1E runs).

Usage:  pixi run python rnr/scripts/fig1f_stability.py [M] [DT]   (defaults M=6 dt=0.001)
        writes rnr/exports/fig1f_stability.{png,csv}
"""
import csv
import glob
import json
import math
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EXPORTS = os.path.join(ROOT, "rnr", "exports")

M = int(sys.argv[1]) if len(sys.argv) > 1 else 6
DT = sys.argv[2] if len(sys.argv) > 2 else "0.001"
# noise model whose CSVs to plot: "active" (FAITHFUL clamp-free) default | "thermal" (legacy clamp).
MODEL = sys.argv[3] if len(sys.argv) > 3 else "active"
_SUF = r"_active" if MODEL == "active" else ""
_NOISE_LBL = "active motility, clamp-free" if MODEL == "active" else "clamp=0.4 + native repair"
COLORS = {0.1: "#e89bd0", 0.2: "#8d4bd6", 0.5: "#2e1378"}


def load_csv(path):
    with open(path) as f:
        rd = csv.DictReader(f)
        col = {k: [] for k in (rd.fieldnames or [])}
        for row in rd:
            for k in col:
                col[k].append(float(row[k]))
    return col


def mean_se(rows):
    n = len(rows)
    L = min(len(r) for r in rows)
    mean, se = [], []
    for j in range(L):
        vals = [r[j] for r in rows]
        m = sum(vals) / n
        var = sum((v - m) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
        mean.append(m)
        se.append(math.sqrt(var / n) if n > 1 else 0.0)
    return mean, se


def collect(ic):
    """ic='mixed' -> _seed{N}{_SUF}.csv ; ic='demixed' -> _seed{N}{_SUF}_demixed.csv (for the
    chosen MODEL: active = _active infix, thermal = none). -> {sigma: [paths]}."""
    suffix = rf"_seed\d+{_SUF}\.csv$" if ic == "mixed" else rf"_seed\d+{_SUF}_demixed\.csv$"
    by_sigma = {}
    for path in sorted(glob.glob(os.path.join(
            EXPORTS, f"sort_oracle_M{M}_S*_KT0.1_L0.001_dt{DT}_cut1.9_seed*.csv"))):
        base = os.path.basename(path)
        if not re.search(suffix, base):
            continue
        m = re.search(rf"_M{M}_S([0-9.]+)_KT", base)
        if m:
            by_sigma.setdefault(float(m.group(1)), []).append(path)
    return by_sigma


def series(paths, dp_max, dtf):
    cols = [load_csv(p) for p in paths]
    L = min(len(c["step"]) for c in cols)
    t = [s * dtf for s in cols[0]["step"][:L]]
    Dm, Dse = mean_se([c["D"][:L] for c in cols])
    norm = dp_max or 1.0
    y = [-d / norm for d in Dm]
    yse = [s / norm for s in Dse]
    return t, y, yse


def main():
    dp_max = None
    p = os.path.join(EXPORTS, "dpmax.json")
    if os.path.exists(p):
        dp_max = json.load(open(p)).get(str(M), {}).get("DP_max")
    dtf = float(DT)

    mixed = collect("mixed")
    demixed = collect("demixed")
    if not demixed:
        print("NO demixed CSVs yet (run sort_periodic_oracle.py ... demixed). Aborting.")
        return

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    rows_out = []

    # ---- panel 1: sigma=0.5 -- demixed (stays ~1) vs mixed (stays ~0) : the Fig 1F contrast ----
    sig0 = 0.5 if 0.5 in demixed else sorted(demixed)[-1]
    if sig0 in demixed:
        t, y, yse = series(demixed[sig0], dp_max, dtf)
        ax[0].plot(t, y, "-", color="#b3007a", lw=2.2, label=f"initially DEMIXED ($\\sigma$={sig0:g})")
        ax[0].fill_between(t, [a - b for a, b in zip(y, yse)], [a + b for a, b in zip(y, yse)],
                           color="#b3007a", alpha=0.18, lw=0)
        rows_out.append((f"demixed_s{sig0:g}", y[0], y[-1], len(demixed[sig0])))
    if sig0 in mixed:
        t, y, yse = series(mixed[sig0], dp_max, dtf)
        ax[0].plot(t, y, "--", color="#e89bd0", lw=2.2, label=f"initially MIXED ($\\sigma$={sig0:g})")
        ax[0].fill_between(t, [a - b for a, b in zip(y, yse)], [a + b for a, b in zip(y, yse)],
                           color="#e89bd0", alpha=0.18, lw=0)
        rows_out.append((f"mixed_s{sig0:g}", y[0], y[-1], len(mixed[sig0])))
    ax[0].axhline(1.0, color="0.6", lw=0.8, ls=":")
    ax[0].axhline(0.0, color="k", lw=0.5)
    ax[0].set_ylim(-0.08, 1.08)
    ax[0].set_xlabel(r"time  $t = \mathrm{step}\cdot dt$")
    ax[0].set_ylabel(r"$DP / DP_{max}$")
    ax[0].set_title(f"Fig 1F core: demixed state is stable ($\\sigma$={sig0:g})\n"
                    "demixed stays $\\approx$1 (energetically preferred); mixed stays low "
                    "(domain formation\nneeds larger N / longer t -- Fig 1E limit)", fontsize=9)
    ax[0].legend(fontsize=9, loc="center right")
    ax[0].grid(alpha=0.3)

    # ---- panel 2: demixed-IC across sigma -- does the boundary hold at weak tension too? ----
    for sig in sorted(demixed):
        t, y, yse = series(demixed[sig], dp_max, dtf)
        col = COLORS.get(sig, "k")
        ax[1].plot(t, y, "-", color=col, lw=2, label=f"$\\sigma$={sig:g} (n={len(demixed[sig])})")
        ax[1].fill_between(t, [a - b for a, b in zip(y, yse)], [a + b for a, b in zip(y, yse)],
                           color=col, alpha=0.15, lw=0)
        rows_out.append((f"demixed_s{sig:g}", y[0], y[-1], len(demixed[sig])))
    ax[1].axhline(1.0, color="0.6", lw=0.8, ls=":")
    ax[1].set_ylim(-0.08, 1.08)
    ax[1].set_xlabel(r"time  $t = \mathrm{step}\cdot dt$")
    ax[1].set_ylabel(r"$DP / DP_{max}$  (demixed IC)")
    ax[1].set_title("Demixed-IC stability vs interfacial tension\n"
                    "(does the boundary hold even at weak $\\sigma$?)", fontsize=9)
    ax[1].legend(title="het. tension", fontsize=9, loc="lower left")
    ax[1].grid(alpha=0.3)

    fig.suptitle(f"Manning2024 Fig 1F in TissueForge 3D vertex (M={M}, N={M**3}, {_NOISE_LBL}, "
                 f"dt={DT}) — the demixed state is energetically preferred", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _figtag = "" if MODEL == "thermal" else f"_{MODEL}"
    png = os.path.join(EXPORTS, f"fig1f_stability{_figtag}.png")
    fig.savefig(png, dpi=140)

    out = os.path.join(EXPORTS, f"fig1f_stability{_figtag}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["series", "DP/DPmax_0", "DP/DPmax_end", "n_seeds"])
        for r in rows_out:
            w.writerow([r[0]] + [f"{x:.4g}" for x in r[1:]])

    print("=" * 74)
    print(f"Fig 1F (M={M}, N={M**3}, dt={DT}, {_NOISE_LBL}) -- DP_max={dp_max}")
    print(f"{'series':>16} {'DP/DPmax_0':>11} {'DP/DPmax_end':>13} {'seeds':>6}")
    for nm, a, b, n in rows_out:
        print(f"{nm:>16} {a:>11.3f} {b:>13.3f} {n:>6d}")
    print("-" * 74)
    # verdict: demixed stays high?
    dm = {s: series(demixed[s], dp_max, dtf)[1][-1] for s in demixed}
    held = {s: v for s, v in dm.items() if v > 0.8}
    print(f"demixed-IC final DP/DP_max: {[f'{s:g}:{v:.2f}' for s, v in sorted(dm.items())]}")
    print(f"-> demixed state HELD (DP/DP_max>0.8) for sigma: {sorted(held)}"
          if held else "-> demixed state did NOT hold anywhere")
    print(f"saved: {png}\nsaved: {out}")


if __name__ == "__main__":
    main()
