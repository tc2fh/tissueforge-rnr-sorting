"""Compare the Python-active drive vs the native C++ engine drive: are the sort outputs equivalent?

Both are the SAME active self-propulsion model (per-cell director + rotational diffusion Dr, per-
vertex force v0*<incident directors>); "active" runs it in Python (sort_periodic_oracle.add_noise_active),
"native" runs it inside tf.step() via MeshSolver.set_motility (PORTING_NOTES §6o). They use different
RNG streams, so they can NOT be bit-identical -- equivalence is an ENSEMBLE claim: the seed-averaged
demixing curve D(t) and the reconnection rate should match within seed-to-seed spread.

Reads the matched CSV pairs written by run_overnight.py (…_active[_demixed].csv vs …_native[_demixed].csv)
and reports, per sigma:
  * demixing D(t): active mean+-SE vs native mean+-SE, final-D difference vs combined SE -> EQUIVALENT?
  * reconnections: active vs native mean totals
Saves an overlay figure (active solid / native dashed) + prints a verdict table. Mixed IC = Fig 1E
demixing; demixed IC = Fig 1F stability.

Run: pixi run python rnr/scripts/compare_active_native.py [M] [DT]
"""
import csv
import glob
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
COLORS = {0.1: "#e89bd0", 0.2: "#8d4bd6", 0.5: "#2e1378"}


def load_col(path, key):
    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            out.append(float(row[key]))
    return out


def mean_se(rows):
    """rows: list of equal-length sequences -> (mean[], se[]) column-wise (truncated to shortest)."""
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


def collect(model, ic):
    """-> {sigma: [paths]} for the given drive model + initial condition."""
    suf = f"_{model}" + ("_demixed" if ic == "demixed" else "")
    pat = os.path.join(EXPORTS, f"sort_oracle_M{M}_S*_KT0.1_L0.001_dt{DT}_cut1.9_seed*.csv")
    by_sigma = {}
    for path in sorted(glob.glob(pat)):
        base = os.path.basename(path)
        if not re.search(rf"_seed\d+{re.escape(suf)}\.csv$", base):
            continue
        m = re.search(rf"_M{M}_S([0-9.]+)_KT", base)
        if m:
            by_sigma.setdefault(float(m.group(1)), []).append(path)
    return by_sigma


def ensemble(paths):
    """seed-averaged D(t), final-D mean+-SE, mean total reconnections."""
    Ds = [load_col(p, "D") for p in paths]
    steps = load_col(paths[0], "step")
    dmean, dse = mean_se(Ds)
    finals = [d[-1] for d in Ds]
    nf = len(finals)
    fmean = sum(finals) / nf
    fse = math.sqrt(sum((v - fmean) ** 2 for v in finals) / (nf - 1) / nf) if nf > 1 else 0.0
    recons = [load_col(p, "recon")[-1] for p in paths]
    return steps[:len(dmean)], dmean, dse, fmean, fse, sum(recons) / len(recons), nf


def main():
    print(f"=== active vs native equivalence  M={M} dt={DT} ===\n")
    any_data = False
    for ic in ("mixed", "demixed"):
        act, nat = collect("active", ic), collect("native", ic)
        sigmas = sorted(set(act) & set(nat))
        if not sigmas:
            print(f"[{ic}] no matched active+native sigmas yet "
                  f"(active={sorted(act)} native={sorted(nat)})")
            continue
        any_data = True
        fig, ax = plt.subplots(figsize=(7, 5))
        print(f"[{ic}]  sigma | final D active | final D native |  Δ   | combSE | recon a/n | verdict")
        for sg in sigmas:
            sa, ma, ea, fa, fea, ra, na = ensemble(act[sg])
            sn, mn, en, fnv, fen, rn, nn = ensemble(nat[sg])
            comb = math.sqrt(fea ** 2 + fen ** 2)
            d = fa - fnv
            verdict = "EQUIV" if abs(d) <= max(comb, 1e-9) * 2 else "DIFFER"
            print(f"        {sg:4.2f} | {fa:+.4f} (n{na}) | {fnv:+.4f} (n{nn}) | {d:+.4f} | "
                  f"{comb:.4f} | {ra:.0f}/{rn:.0f} | {verdict}")
            c = COLORS.get(sg, "#444")
            ax.plot(sa, ma, "-", color=c, label=f"σ={sg:g} active")
            ax.plot(sn, mn, "--", color=c, label=f"σ={sg:g} native")
        ax.set_xlabel("step"); ax.set_ylabel("demixing index D")
        ax.set_title(f"active (solid) vs native (dashed) — {ic} IC, M={M}")
        ax.legend(fontsize=7, ncol=len(sigmas)); ax.grid(alpha=0.3)
        out = os.path.join(EXPORTS, f"compare_active_native_{ic}_M{M}.png")
        fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
        print(f"        -> {out}\n")
    if not any_data:
        print("\nNothing to compare yet. Run both sweeps first: "
              "`pixi run overnight` (active) and `... overnight 19 100000 native`.")


if __name__ == "__main__":
    main()
