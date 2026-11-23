"""RQ3 figure: one fixed threshold for every frame vs the brightness-adaptive threshold (RiskGate dark rule).

x = % of the frame masked on CLEAN video (utility cost where nothing is wrong), y = exposure on DARK video (dark:2).
A fixed threshold t gives one point (its clean area, its dark exposure); lowering t buys dark privacy with clean
area. The adaptive policy keeps the operating threshold on normal frames (clean area of the op threshold) and uses
thr_dark only when the frame's mean grey level is < 60, so it reaches the low-threshold dark exposure at the
op threshold's clean area: points left of the fixed curve.

  python src/analysis/rq3_pareto.py --dir results/val --prefix rq3_ --out results/val/rq3_pareto.png
"""
import argparse
import glob
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

FIXED, ADAPT = ("#2a78d6", "o"), ("#eb6834", "D")  # validated categorical slots 1-2 (dataviz palette)
INK, MUTED, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DETS = [("yunet_480x640_npu", "YuNet stock (op 0.6)", 0.6), ("yunet_ft_480x640_npu", "YuNet fine-tuned (op 0.7)", 0.7)]
METRICS = {"pct_appearances_leaked": "% appearances exposed on dark video",
           "pct_identities_leaked": "% people exposed on dark video"}


def load(d, prefix, cond):
    files = sorted(glob.glob(str(Path(d) / f"{prefix}{cond}_*.csv")))
    x = pd.concat([pd.read_csv(f) for f in files]) if files else pd.DataFrame()
    if len(x):
        x = x.drop_duplicates(["system", "threshold"], keep="last")
        x["thr_dark"] = x.system.str.extract(r"risk_d\d+_t([\d.]+)")[0].astype(float)
    return x


def points(clean, dark, det, op):
    """Rows (kind, label, clean_area, dark metrics...) for one detector."""
    rows = []
    base = f"{det}+lite+back3"
    fc, fd = clean[clean.system == base], dark[dark.system == base]
    for t in sorted(set(fc.threshold) & set(fd.threshold)):
        c, k = fc[fc.threshold == t].iloc[0], fd[fd.threshold == t].iloc[0]
        rows.append({"detector": det, "policy": "fixed", "threshold": t, "clean_area": c.pct_area_masked,
                     **{m: k[m] for m in METRICS}, "dark_area": k.pct_area_masked})
    for _, k in dark[(dark.detector == det) & dark.thr_dark.notna() & ~dark.system.str.contains("_h|_p")].iterrows():
        c = clean[(clean.system == k.system) & (clean.threshold == op)]
        if len(c):
            rows.append({"detector": det, "policy": "adaptive", "threshold": k.thr_dark,
                         "clean_area": c.iloc[0].pct_area_masked, **{m: k[m] for m in METRICS},
                         "dark_area": k.pct_area_masked})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True)
    p.add_argument("--prefix", default="rq3_", help="files <prefix>clean_*.csv and <prefix>dark2_*.csv")
    p.add_argument("--extra-clean", nargs="*", default=[], help="more csvs with clean op rows (e.g. lite_val_op.csv)")
    p.add_argument("--metric", default="pct_appearances_leaked", choices=list(METRICS))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    clean, dark = load(a.dir, a.prefix, "clean"), load(a.dir, a.prefix, "dark2")
    if a.extra_clean:
        clean = pd.concat([clean] + [pd.read_csv(f) for f in a.extra_clean]).drop_duplicates(["system", "threshold"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), facecolor=BG)
    allp = []
    for ax, (det, title, op) in zip(axes, DETS):
        ax.set_facecolor(BG)
        pts = points(clean, dark, det, op)
        allp.append(pts)
        for kind, (color, marker), label in (("fixed", FIXED, "one fixed threshold everywhere"),
                                             ("adaptive", ADAPT, "op threshold, lower only in dark frames (ours)")):
            s = pts[pts.policy == kind].sort_values("threshold")
            if not len(s):
                continue
            ax.plot(s.clean_area, s[a.metric], color=color, lw=2, marker=marker, ms=7, mec=BG, mew=2, label=label,
                    zorder=3)
            for r in s.itertuples():
                if kind == "fixed" or r.threshold in (0.1, 0.2, 0.3):
                    ax.annotate(f"{r.threshold:g}", (r.clean_area, getattr(r, a.metric)), textcoords="offset points",
                                xytext=(6, 4) if kind == "fixed" else (-6, -12), ha="left" if kind == "fixed" else "right",
                                fontsize=7, color=MUTED)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("% of frame masked on clean video", color=MUTED, fontsize=9)
        ax.grid(color=GRID, lw=0.8)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].set_ylabel(METRICS[a.metric], color=MUTED, fontsize=9)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8, labelcolor=INK, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(a.out, dpi=200)
    pd.concat(allp).round(4).to_csv(a.out.with_suffix(".csv"), index=False)
    print(a.out)


if __name__ == "__main__":
    main()
