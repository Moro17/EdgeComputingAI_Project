"""YOLOv8n-face .pt -> full-INT8 TFLite for the i.MX 8M Plus NPU (VX delegate), calibrated on the ChokePoint S5 frames
used for every INT8 model here (never scored). NMS stays out of the graph: it runs on the CPU in the board scripts.

Needs the export env (torch + ultralytics + tensorflow), separate from .venv:
  uv venv -p 3.12 .venv-export && VIRTUAL_ENV=.venv-export uv pip install ultralytics litert-torch pandas
Usage: .venv-export/bin/python src/export_tflite.py models/yolov8n-face.pt [--imgsz 640]
       -> models/<stem>_int8.tflite
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from loaders.chokepoint import DATA  # noqa: E402
from systems.quantize import CALIB_FRAMES, CALIB_SEQS  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("weights", type=Path)
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()

    from ultralytics import YOLO

    paths = sorted(p for s in CALIB_SEQS for p in (DATA / "frames" / s).glob("*.jpg"))
    assert paths, f"no calibration frames: download the S5 sequences into {DATA / 'frames'}"
    paths = [paths[i] for i in np.linspace(0, len(paths) - 1, CALIB_FRAMES).astype(int)]
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "calib.txt").write_text("\n".join(map(str, paths)))
        (Path(d) / "calib.yaml").write_text(f"path: {d}\ntrain: calib.txt\nval: calib.txt\nnames:\n  0: face\n"
                                                # yolov8n-face is a pose model: 5 landmarks x (x, y, visible)
                                                "kpt_shape: [5, 3]\nflip_idx: [1, 0, 2, 4, 3]\n")
        # ultralytics >= 8.4 exports TFLite via litert_torch; quantize=8 = static INT8 weights + activations
        out = Path(YOLO(str(args.weights)).export(format="litert", quantize=8, imgsz=args.imgsz, nms=False,
                                                  data=str(Path(d) / "calib.yaml"), fraction=1.0))
    src = out if out.suffix == ".tflite" else next(out.glob("*.tflite"))
    dst = args.weights.with_name(f"{args.weights.stem}_int8.tflite")
    if src.resolve() != dst.resolve():
        shutil.copy(src, dst)
    print(f"{dst} ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
