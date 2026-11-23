"""Figure: % of people exposed per policy on ChokePoint dark:2, Music Video clean and Music Video dark:2.

Reads the results CSVs copied into experiments/ (rq3/dark2_adapt.csv, musicvideo/results_op.csv,
musicvideo/results_op@dark2.csv); window3 and per-frame on ChokePoint dark2 come from experiments/setlevel
(per-frame judge column, same number as benchmark.py).

  python src/analysis/generalization_fig.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

E = Path(__file__).resolve().parents[2] / "experiments"
POL = [("", "per-frame"), ("+window3", "window3 (MBAM re-impl.)"), ("+lite+back3", "tracker + look-back"),
       ("+risk_d60_t0.1+lite+back3", "+ brightness rule")]
COLORS = ["#b9c0c8", "#8a95a3", "#2a78d6", "#eb6834"]


def table():
    sl = pd.read_csv(E / "setlevel" / "summary_test@dark2.csv").set_index("system").pct_ids_frame_judge
    cp = pd.read_csv(E / "rq3" / "dark2_adapt.csv").set_index("system").pct_identities_leaked
    mv = pd.read_csv(E / "musicvideo" / "results_op.csv").set_index("system").pct_identities_leaked
    mvd = pd.read_csv(E / "musicvideo" / "results_op@dark2.csv").set_index("system").pct_identities_leaked
    rows = []
    for det, name in [("yunet_480x640_npu", "YuNet stock (0.6)"), ("yunet_ft_480x640_npu", "YuNet fine-tuned (0.7)")]:
        for sfx, pol in POL:
            s = det + sfx
            rows.append({"model": name, "policy": pol, "ChokePoint dark2": cp.get(s, sl.get(s)),
                         "Music Video": mv[s], "Music Video dark2": mvd[s]})
    return pd.DataFrame(rows)


def main():
    t = table()
    conds = ["ChokePoint dark2", "Music Video", "Music Video dark2"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (model, g) in zip(axes, t.groupby("model", sort=False)):
        x = np.arange(len(conds))
        w = 0.2
        for i, ((_, r), c) in enumerate(zip(g.iterrows(), COLORS)):
            v = [r[k] for k in conds]
            b = ax.bar(x + (i - 1.5) * w, v, w - 0.02, color=c, label=r["policy"])
            ax.bar_label(b, fmt="%.0f", fontsize=7, color="#333")
        ax.set_xticks(x, conds)
        ax.set_title(model, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e5e5")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("% of people exposed (one frame is enough)")
    axes[0].set_ylim(0, 108)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    out = E / "generalization.png"
    fig.savefig(out, dpi=160)
    t.round(1).to_csv(out.with_suffix(".csv"), index=False)
    print(t.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
