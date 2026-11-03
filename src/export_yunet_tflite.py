"""YuNet ONNX (stock or fine-tuned) -> static-shape INT8 TFLite for the i.MX 8M Plus NPU, calibrated on the ChokePoint
S5 frames used for every INT8 model here (never scored). Output is what src/board_bench.py YuNetTFLite loads.

Needs the export env (.venv-export: onnx, onnxruntime, onnx2tf, tensorflow, ai_edge_litert) and
models/calibration_image_sample_data_20x128x128x3_float32.npy (onnx2tf reads it from the CWD).
Usage: .venv-export/bin/python src/export_yunet_tflite.py SRC.onnx --size H W [--out PATH]
       -> models/<src stem>_<H>x<W>_int8.tflite
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import onnx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from systems.quantize import calibration_frames  # noqa: E402

MODELS = Path(__file__).resolve().parents[1] / "models"
ONNX2TF_DATA = "calibration_image_sample_data_20x128x128x3_float32.npy"


def static_onnx(src: Path, h: int, w: int, dst: Path) -> Path:
    m = onnx.load(str(src))
    want = [f"{k}_{s}" for k in ("cls", "obj", "bbox", "kps") for s in (8, 16, 32)]
    got = [o.name for o in m.graph.output]
    # board_bench.YuNetTFLite decodes output_0..11 by position, so the ONNX order must be exactly this
    assert got == want, f"{src.name} outputs {got} != expected {want}"
    for d, v in zip(m.graph.input[0].type.tensor_type.shape.dim, (1, 3, h, w)):
        d.ClearField("dim_param")
        d.dim_value = v
    del m.graph.value_info[:]  # stale shapes (also the outputs': 640x640 anchor counts); shape inference refills them
    for o in m.graph.output:
        o.type.tensor_type.ClearField("shape")
    onnx.save(onnx.shape_inference.infer_shapes(m), str(dst))
    return dst


def calib_npy(h: int, w: int, dst: Path, n: int = 100) -> Path:
    """Frames placed top-left, zero padded, float32 BGR 0-255 NHWC: exactly what YuNetTFLite feeds at runtime."""
    out = np.zeros((n, h, w, 3), np.float32)
    for i, f in enumerate(calibration_frames(n)):
        fh, fw = f.shape[:2]
        r = min(w / fw, h / fh)
        out[i, : round(fh * r), : round(fw * r)] = f if r == 1 else cv2.resize(f, (round(fw * r), round(fh * r)))
    np.save(dst, out)
    return dst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src", type=Path)
    p.add_argument("--size", type=int, nargs=2, required=True, metavar=("H", "W"))
    p.add_argument("--out", type=Path)
    p.add_argument("--keep-float", action="store_true", help="also save onnx2tf's FP32 and FP16 files next to --out")
    args = p.parse_args()
    h, w = args.size
    out = args.out or MODELS / f"{args.src.stem}_{h}x{w}_int8.tflite"
    seed = MODELS / ONNX2TF_DATA
    if not seed.exists():
        sys.exit(f"missing {seed}: copy onnx2tf's calibration_image_sample_data_20x128x128x3_float32.npy there")

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        shutil.copy(seed, d / ONNX2TF_DATA)
        static = static_onnx(args.src, h, w, d / "static.onnx")
        npy = calib_npy(h, w, d / "calib.npy")
        onnx2tf = Path(sys.executable).with_name("onnx2tf")
        # int16 variants may fail after the integer_quant file is written; only check that file exists
        subprocess.run([str(onnx2tf), "-i", str(static), "-o", str(d / "tf"), "-oiqt", "-qt", "per-tensor",
                        "-cind", "input", str(npy), "[[[[0,0,0]]]]", "[[[[1,1,1]]]]"], cwd=d)
        built = list((d / "tf").glob("*_integer_quant.tflite"))
        if not built:
            sys.exit("onnx2tf produced no *_integer_quant.tflite")
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(built[0], out)
        if args.keep_float:  # quantization before/after: the same graph unquantized
            for prec in ("float32", "float16"):
                for f in (d / "tf").glob(f"*_{prec}.tflite"):
                    shutil.copy(f, out.with_name(out.name.replace("_int8", f"_{'fp32' if prec == 'float32' else 'fp16'}")))
    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
