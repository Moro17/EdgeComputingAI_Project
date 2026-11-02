"""YuNet export self-checks. Run: .venv-export/bin/python tests/test_export.py
The tflite check needs models/yunet_320x416_int8.tflite (build: .venv-export/bin/python src/export_yunet_tflite.py
models/face_detection_yunet_2023mar.onnx --size 320 416) and skips without it."""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import onnx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from board_bench import YuNetTFLite  # noqa: E402
from export_yunet_tflite import static_onnx  # noqa: E402
from systems.detectors import MODELS, SYSTEMS  # noqa: E402
from test_detectors import sample  # noqa: E402

TMP = Path(tempfile.mkdtemp())


def test_static_onnx_fixes_shape_and_keeps_output_names():
    dst = static_onnx(MODELS / "face_detection_yunet_2023mar.onnx", 320, 416, TMP / "y.onnx")
    m = onnx.load(str(dst))
    assert [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim] == [1, 3, 320, 416]
    assert [o.name for o in m.graph.output] == [f"{k}_{s}" for k in ("cls", "obj", "bbox", "kps") for s in (8, 16, 32)]


def test_static_onnx_rejects_reordered_outputs():
    m = onnx.load(str(MODELS / "face_detection_yunet_2023mar.onnx"))
    outs = list(m.graph.output)
    del m.graph.output[:]
    m.graph.output.extend(outs[::-1])
    bad = TMP / "bad.onnx"
    onnx.save(m, str(bad))
    try:
        static_onnx(bad, 320, 416, TMP / "o.onnx")
    except AssertionError:
        return
    raise AssertionError("reordered outputs accepted")


def test_int8_tflite_matches_ort():
    path = MODELS / "yunet_320x416_int8.tflite"
    if not path.exists():
        print("skip test_int8_tflite_matches_ort: no", path)
        return
    g, img = sample()
    small = cv2.resize(img, (416, 312))
    ref = SYSTEMS["yunet_ort"][0](None)(small)
    ref = ref[ref[:, 4] >= 0.5]
    got, _ = YuNetTFLite(path, npu=False)(small)
    assert len(got[got[:, 4] >= 0.5]) == len(ref)
    assert np.abs(got[got[:, 4] >= 0.5][:, :4] - ref[:, :4]).max() < 3
    assert np.abs(got[got[:, 4] >= 0.5][:, 4] - ref[:, 4]).max() < 0.1  # score tracks FP32, not saturated



def test_tflite_reused_buffer_matches_fresh():  # the input buffer is kept between frames; padding must not leak
    path = MODELS / "yunet_480x640_int8.tflite"
    if not path.exists():
        print("skip test_tflite_reused_buffer_matches_fresh: no", path)
        return
    _, img = sample()
    small = cv2.resize(img, (400, 300))
    det = YuNetTFLite(path, npu=False)
    det(img)
    assert np.array_equal(det(small)[0], YuNetTFLite(path, npu=False)(small)[0])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
