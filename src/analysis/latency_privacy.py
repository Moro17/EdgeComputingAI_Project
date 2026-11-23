"""Privacy vs latency on val: % identities exposed against the look-ahead W (frames of delay) of each policy.

windowW (our re-implementation of MBAM's temporal fusion), keep+backW and keep+coast+backW all need W future frames,
so at equal W they cost the same latency (W x 40 ms at 25 FPS). One panel per detector, board NPU detections.

  python src/analysis/latency_privacy.py --csv results/val/val_0.9.csv results/val/wsweep_val_0.9.csv \\
      --out results/val/latency_privacy.png
"""
import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# validated default categorical palette, slots 1-3 in order (dataviz skill, references/palette.md)
SERIES = [("window", "windowW (MBAM re-impl.)", "#2a78d6", "o"),
          ("bytetrack+keep+back", "keep+backW (ByteTrack)", "#eb6834", "s"),
          ("bytetrack+keep+coast+back", "keep+coast+backW", "#1baf7a", "^"),
          ("lite+back", "lite tracker+backW (ours, on the board)", "#eda100", "D")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DETS = [("yunet_480x640_npu", "YuNet stock"), ("yunet_ft_480x640_npu", "YuNet fine-tuned")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    d = pd.concat([pd.read_csv(f) for f in a.csv]).drop_duplicates("system", keep="last")
    d["policy"] = d.policy.fillna("")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True, facecolor="#fcfcfb")
    rows = []
    for ax, (det, title) in zip(axes, DETS):
        ax.set_facecolor("#fcfcfb")
        for prefix, label, color, marker in SERIES:
            pts = []
            for r in d[d.detector == det].itertuples():
                m = re.fullmatch(re.escape(prefix) + r"(\d+)", r.policy)
                if m:
                    pts.append((int(m.group(1)), r.pct_identities_leaked, r.pct_area_masked))
            pts.sort()
            if not pts:
                continue
            w, y, area = zip(*pts)
            ax.plot(w, y, color=color, lw=2, marker=marker, ms=7, mec="#fcfcfb", mew=2, label=label, zorder=3)
            rows += [{"detector": det, "policy": label, "W": wi, "pct_identities_exposed": yi, "pct_area_masked": ai}
                     for wi, yi, ai in pts]
        per_frame = d[(d.detector == det) & (d.policy == "")].pct_identities_leaked
        if len(per_frame):
            ax.axhline(per_frame.iloc[0], color=MUTED, lw=1, ls="--", zorder=1)
            ax.text(6.1, per_frame.iloc[0], "per-frame", color=MUTED, fontsize=8, va="bottom", ha="right")
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("look-ahead W (frames; x 40 ms at 25 FPS)", color=MUTED, fontsize=9)
        ax.set_xticks(range(0, 7))
        ax.set_xlim(0.5, 6.3)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].set_ylabel("% people exposed (val, threshold 0.9)", color=MUTED, fontsize=9)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8, labelcolor=INK, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(a.out, dpi=200)
    pd.DataFrame(rows).to_csv(a.out.with_suffix(".csv"), index=False)  # table view of the same numbers
    print(a.out)


if __name__ == "__main__":
    main()
