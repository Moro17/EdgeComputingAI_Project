"""RQ1 analysis on a benchmark --out dir: CIs, exposure vs track length, recall vs exposure.

  python src/analysis/rq1.py --out results/rq1 --tags yunet_480x640_npu_t0.9 "yunet_480x640_npu+bytetrack+keep_t0.9"
  python src/analysis/rq1.py --out results/rq1 --sweep results/rq1/sweep/sweep.csv

Writes <out>/analysis/: ci.csv, length.csv, length.png, recall_vs_exposure.png (with --sweep).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics.exposure import bootstrap_ci, leak_vs_length  # noqa: E402


def load(out, tag):
    files = sorted((out / "scores" / tag).glob("*.csv"))
    if not files:
        raise SystemExit(f"no scores for {tag} in {out / 'scores'}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tags", nargs="*", default=[], help="score tags = scores/ sub-folders, e.g. yunet_ort_t0.9")
    p.add_argument("--sweep", type=Path, default=None, help="CSV of results.csv rows at several thresholds")
    args = p.parse_args()
    dst = args.out / "analysis"
    dst.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.tags:
        ci, lengths = [], []
        fig, ax = plt.subplots(figsize=(6, 4))
        for tag in args.tags:
            f = load(args.out, tag)
            lo, hi = bootstrap_ci(f)
            ci.append({"tag": tag, "pct_identities_leaked": 100 * f.groupby("identity").leaked.any().mean(),
                       "ci_lo": lo, "ci_hi": hi})
            L = leak_vs_length(f).assign(tag=tag)
            lengths.append(L)
            line, = ax.plot(L.length, L.p_leak, marker="o", label=tag)
            ax.plot(L.length, L.p_independent, ls="--", color=line.get_color(), alpha=0.6)
        pd.DataFrame(ci).round(2).to_csv(dst / "ci.csv", index=False)
        pd.concat(lengths).round(4).to_csv(dst / "length.csv", index=False)
        ax.set(xscale="log", xlabel="window length (frames)", ylabel="P(window contains a leaked frame)",
               title="Exposure vs track length (dashed: independent misses)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(dst / "length.png", dpi=150)
        print(pd.DataFrame(ci).round(2).to_string(index=False))

    if args.sweep:
        s = pd.read_csv(args.sweep)
        fig, ax = plt.subplots(figsize=(6, 4))
        for system, g in s.groupby("system"):
            g = g.sort_values("threshold")
            ax.plot(g.frame_recall, g.pct_identities_leaked, marker="o", label=system)
        ax.set(xlabel="frame recall", ylabel="% identities exposed", title="Recall is not privacy")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(dst / "recall_vs_exposure.png", dpi=150)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
