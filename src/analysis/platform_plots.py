"""Performance and quantization figures across the three platforms: desktop PC (onnxruntime CPU), Jetson Nano
(TensorRT GPU, tflite CPU), i.MX 8M Plus (NPU, CPU). Numbers are the measured p50s from the docs
(docs/jetson.md, docs/edge-board.md section 4, docs/next-runs.md quantization study, experiments/quant/*.csv).

    python src/analysis/platform_plots.py   # -> experiments/platforms/*.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "platforms"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
PREC = {"FP32": "#2a78d6", "FP16": "#1baf7a", "INT8": "#eb6834", "int16": "#6250d6"}
BUDGET = 40.0
CLIP = 100.0

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def quiet(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def latency():
    """Detector alone, stock YuNet 480x640, p50 ms."""
    rows = [  # platform, precision, ms, note
        ("PC (CPU, onnxruntime)", "FP32", 9.7, ""), ("PC (CPU, onnxruntime)", "INT8", 17.6, ""),
        ("PC (CPU, onnxruntime)", "int16", 21.3, ""),
        ("Jetson Nano GPU (TensorRT)", "FP32", 20.2, ""), ("Jetson Nano GPU (TensorRT)", "FP16", 20.2, ""),
        ("Jetson Nano GPU (TensorRT)", "INT8", 20.2, "fără INT8 nativ"),
        ("Jetson Nano CPU (4x A57)", "FP32", 64.3, ""), ("Jetson Nano CPU (4x A57)", "INT8", 44.8, ""),
        ("i.MX 8M Plus NPU", "FP32", 918, "918 ms"), ("i.MX 8M Plus NPU", "INT8", 11.7, ""),
        ("i.MX 8M Plus NPU", "int16", 2200, "respins, 2.2 s"),
        ("i.MX 8M Plus CPU (4x A53)", "FP32", 61.3, ""), ("i.MX 8M Plus CPU (4x A53)", "INT8", 72.7, ""),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    y, ticks, labels, last = 0, [], [], None
    for plat, prec, ms, note in rows:
        if plat != last and last is not None:
            y += 0.6
        last = plat
        ax.barh(y, min(ms, CLIP), color=PREC[prec], height=0.78, edgecolor=SURF, lw=2)
        txt = note if ms > CLIP else (f"{ms:g} ms" + (f"  ({note})" if note else ""))
        ax.text(min(ms, CLIP) + 1.2, y, txt, va="center", fontsize=9, color=INK)
        ticks.append(y)
        labels.append(f"{plat}  {prec}")
        y += 1
    ax.axvline(BUDGET, color=MUTED, ls="--", lw=1.2)
    ax.text(BUDGET + 0.8, y - 0.2, "buget 40 ms / cadru (tot pipeline-ul)", color=MUTED, fontsize=9)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, CLIP + 18)
    ax.set_xlabel("latența detectorului, p50 (ms) — axa tăiată la 100 ms")
    ax.set_title("YuNet 480x640 pe fiecare platformă și precizie", loc="left", color=INK, fontsize=12, pad=12)
    quiet(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in PREC.values()]
    ax.legend(handles, PREC.keys(), loc="upper right", frameon=False, title="precizie")
    fig.tight_layout()
    fig.savefig(OUT / "latency_by_platform.png", dpi=160)


def pipeline():
    """Whole pipeline, one camera, fine-tuned model, p50 ms per stage."""
    stages = {
        "i.MX 8M Plus\n(NPU INT8, fire în paralel)": {"citire + pre": 1.4, "detector": 10.4, "post (NMS)": 1.8,
                                                       "tracker + reguli": 0.9, "blur": 2.9},
        "Jetson Nano\n(GPU FP16, buclă serială)": {"citire + pre": 2.7, "detector": 25.0, "post (NMS)": 1.4,
                                                    "tracker + reguli": 1.1, "blur": 4.1},
    }
    colors = {"citire + pre": "#52514e", "detector": "#eb6834", "post (NMS)": "#2a78d6",
              "tracker + reguli": "#1baf7a", "blur": "#6250d6"}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
    for i, (board, st) in enumerate(stages.items()):
        left = 0
        for k, v in st.items():
            ax.barh(i, v, left=left, color=colors[k], height=0.55, edgecolor=SURF, lw=2, label=k if i == 0 else None)
            if v >= 3:
                ax.text(left + v / 2, i, f"{v:g}", ha="center", va="center", color="white", fontsize=9)
            left += v
        ax.text(BUDGET + 0.8 if left > BUDGET - 7 else left + 0.8, i, f"{left:.1f} ms", va="center", color=INK, fontsize=9)
    ax.axvline(BUDGET, color=MUTED, ls="--", lw=1.2)
    ax.text(BUDGET + 0.5, -0.38, "40 ms", color=MUTED, fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(stages.keys())
    ax.invert_yaxis()
    ax.set_xlim(0, 48)
    ax.set_xlabel("ms pe cadru, p50, o cameră (model fine-tuned, regula de întuneric, look-back 3)")
    ax.set_title("Unde se duce timpul", loc="left", color=INK, fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=5, frameon=False, fontsize=8.5)
    quiet(ax)
    # cameras in real time
    cams = [("i.MX 8M Plus", 3, "max 89.5 FPS"), ("Jetson Nano", 1, "max 30 FPS")]
    for i, (b, n, note) in enumerate(cams):
        bx.barh(i, n, color="#eb6834" if i == 0 else "#1baf7a", height=0.55)
        bx.text(n + 0.08, i, f"{n}  ({note})", va="center", fontsize=9, color=INK)
    bx.set_yticks([0, 1])
    bx.set_yticklabels([c[0] for c in cams])
    bx.invert_yaxis()
    bx.set_xlim(0, 6.5)
    bx.set_xticks(range(0, 5))
    bx.set_xlabel("camere la 25 FPS în timp real")
    bx.set_title("Câte camere duce placa", loc="left", color=INK, fontsize=12)
    quiet(bx)
    fig.tight_layout()
    fig.savefig(OUT / "pipeline_by_board.png", dpi=160)


def quant_privacy():
    """% people exposed, FP32 vs deployed INT8 vs int16 activations (same graph, onnxruntime, faithful to the NPU)."""
    q = ROOT / "experiments" / "quant"
    sets = [("ChokePoint dark:2\n+ tracker + regula de întuneric", "dark2.csv", "ft_dark2.csv",
             "+risk_d60_t0.1+lite+back3"),
            ("Music Video\n+ tracker (lite+back3)", "mv_clean.csv", "mv_clean.csv", "+lite+back3")]
    recipes = [("FP32", "fp32", "FP32"), ("INT8 (pe placă)", "board", "INT8"), ("int16 activări", "a16", "int16")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (title, f_stock, f_ft, tail) in zip(axes, sets):
        ds, dft = pd.read_csv(q / f_stock), pd.read_csv(q / f_ft)
        for j, (model, d, pref) in enumerate((("original", ds, "yq_"), ("fine-tuned", dft, "yqft_"))):
            for k, (lab, key, col) in enumerate(recipes):
                v = float(d.loc[d.system == f"{pref}{key}{tail}", "pct_identities_leaked"].iloc[0])
                x = j * 3.6 + k
                ax.bar(x, v, color=PREC[col], width=0.86, edgecolor=SURF, lw=2, label=lab if j == 0 else None)
                ax.text(x, v + 1.2, f"{v:.0f}", ha="center", fontsize=9, color=INK)
        ax.set_xticks([1, 4.6])
        ax.set_xticklabels(["YuNet original", "YuNet fine-tuned"])
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("% persoane expuse (mai mic = mai bine)")
    axes[0].set_ylim(0, 100)
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    fig.suptitle("Cuantizarea nu costă confidențialitate semnificativ (teste pereche, toate p ≥ 0.125)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "quantization_privacy.png", dpi=160)


def three_platforms():
    """Laptop vs i.MX vs Jetson, fine-tuned YuNet 480x640, final policy, one camera. Measured 2026-11-24:
    experiments/pc/pipeline_*.json (src/pc_pipeline_bench.py), experiments/edge/final_max.json,
    experiments/jetson/jet_final_ft_*.json."""
    rows = [  # label, detector ms (inference only), pipeline FPS max, colour
        ("Laptop CPU\nRyzen 5 5600H, FP32", 7.1, 85.5, PREC["FP32"]),
        ("Laptop CPU\nRyzen 5 5600H, INT8", 10.0, 66.8, PREC["INT8"]),
        ("i.MX 8M Plus\nNPU, INT8", 10.2, 89.5, PREC["INT8"]),
        ("Jetson Nano\nGPU, FP16", 25.0, 30.0, PREC["FP16"]),
        ("Jetson Nano\nGPU, FP32", 25.1, 29.8, PREC["FP32"]),
    ]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    ys = range(len(rows))
    for y, (lab, ms, fps, c) in zip(ys, rows):
        ax.barh(y, ms, color=c, height=0.66, edgecolor=SURF, lw=2)
        ax.text(ms + 0.5, y, f"{ms:g} ms", va="center", fontsize=9, color=INK)
        bx.barh(y, fps, color=c, height=0.66, edgecolor=SURF, lw=2)
        bx.text(fps + 1.5, y, f"{fps:g} FPS", va="center", fontsize=9, color=INK)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 31)
    ax.set_xlabel("inferența detectorului, p50 (ms)")
    ax.set_title("Inferență YuNet (fine-tuned)", loc="left", color=INK)
    bx.axvline(25, color=MUTED, ls="--", lw=1.2)
    bx.text(26, -0.5, "25 FPS = timp real", color=MUTED, fontsize=9)
    bx.set_xlim(0, 112)
    bx.set_xlabel("tot pipeline-ul, 1 cameră, maxim (FPS)")
    bx.set_title("Pipeline complet (tracker + look-back + blur)", loc="left", color=INK)
    for a in (ax, bx):
        quiet(a)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PREC[k]) for k in ("FP32", "FP16", "INT8")]
    bx.legend(handles, ("FP32", "FP16", "INT8"), loc="lower right", frameon=False, title="precizie")
    fig.tight_layout()
    fig.savefig(OUT / "inference_3_platforms.png", dpi=160)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    latency()
    pipeline()
    quant_privacy()
    three_platforms()
    print("written", sorted(p.name for p in OUT.glob("*.png")))
