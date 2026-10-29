"""Precision study: the board candidates at FP32 vs FP16 on the same runtime (onnxruntime) and the same machine.

Runs the RQ1 benchmark for each (model, precision) into --out, reusing the judge and enrollment from --judge-from
so only detection and scoring run, then writes precision.csv / precision.md with the FP16 - FP32 deltas.
INT8 is measured on the board (NPU), not here.

Usage: python src/precision_bench.py --out experiments/precision [--judge-from experiments/rq1] [--max-frames N]
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from systems.detectors import MODELS

ROOT = Path(__file__).resolve().parents[1]
INSIGHTFACE = Path("~/.insightface/models/buffalo_s").expanduser()
# model -> {precision: (system name, model file)}
CANDIDATES = {
    "YuNet": {"fp32": ("yunet_ort", MODELS / "face_detection_yunet_2023mar.onnx"),
              "fp16": ("yunet_fp16", MODELS / "face_detection_yunet_2023mar_fp16_dyn.onnx")},
    "SCRFD-500M": {"fp32": ("scrfd_500m", INSIGHTFACE / "det_500m.onnx"),
                   "fp16": ("scrfd_500m_fp16", MODELS / "det_500m_fp16.onnx")},
    "YOLOv8n-face": {"fp32": ("yolov8n_face", MODELS / "yolov8n-face.onnx"),
                     "fp16": ("yolov8n_face_fp16", MODELS / "yolov8n-face_fp16.onnx")},
}
COLS = ["frame_recall", "pct_identities_leaked", "pct_appearances_leaked", "max_exposed_run", "pct_area_masked",
        "fps", "p95_ms"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--judge-from", type=Path, default=ROOT / "experiments" / "rq1")
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for name in ["emb", "judge.json", "data_config.json"]:  # same judge and gallery as RQ1
        src, dst = args.judge_from / name, args.out / name
        if src.exists() and not dst.exists() and args.max_frames is None:
            (shutil.copytree if src.is_dir() else shutil.copy)(src, dst)

    systems = [s for m in CANDIDATES.values() for s, _ in m.values()]
    cmd = [sys.executable, str(ROOT / "src" / "benchmark.py"), "--systems", *systems, "--out", str(args.out)]
    if args.max_frames:
        cmd += ["--max-frames", str(args.max_frames)]
    subprocess.run(cmd, check=True)

    res = pd.read_csv(args.out / "results.csv").set_index("system")
    rows = []
    for model, precs in CANDIDATES.items():
        for prec, (system, path) in precs.items():
            rows.append({"model": model, "precision": prec, "system": system,
                         "size_mb": round(path.stat().st_size / 1e6, 2), **res.loc[system, COLS].to_dict()})
    df = pd.DataFrame(rows)
    fp32 = df[df.precision == "fp32"].set_index("model")
    for c in ["frame_recall", "pct_identities_leaked", "fps"]:
        df[f"d_{c}"] = df[c].to_numpy() - fp32.loc[df.model, c].to_numpy()
    df.round(4).to_csv(args.out / "precision.csv", index=False)
    t = df.drop(columns="system").round(3).astype(str)
    (args.out / "precision.md").write_text("\n".join(
        "| " + " | ".join(r) + " |" for r in [list(t.columns), ["---"] * t.shape[1], *t.to_numpy().tolist()]) + "\n")
    (args.out / "precision.json").write_text(json.dumps({"host": json.loads(
        (args.out / "results.json").read_text())["host"], "rows": df.to_dict("records")}, indent=2, default=float))
    print(df.drop(columns="system").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
