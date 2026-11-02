"""Demo self-checks on a real ChokePoint frame. Run: .venv-export/bin/python tests/test_demo.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import demo  # noqa: E402
from board_bench import YuNetTFLite  # noqa: E402
from systems.detectors import MODELS  # noqa: E402
from test_detectors import sample  # noqa: E402

_, img = sample()


def test_overlay_text():
    lines = demo.overlay_text("yunet_ft", 26.04, {"pre": 1.8, "npu": 9.1, "post": 2.0}, 2)
    assert lines == ["yunet_ft", "26.0 FPS", "pre 1.8 | npu 9.1 | post 2.0 ms", "faces 2"]


def test_default_thr():
    assert demo.default_thr("yunet_480x640_int8.tflite") == 0.9
    assert demo.default_thr("yolov8n-face_320_int8.tflite") == 0.49


def test_run_one_frame_cpu():  # end to end on a real frame without the NPU
    out, _, n = demo.process(img, YuNetTFLite(MODELS / "yunet_480x640_int8.tflite", npu=False), thr=0.9, blur_on=True)
    assert out.shape == img.shape and not np.array_equal(out, img) and n >= 1
    off, _, _ = demo.process(img, YuNetTFLite(MODELS / "yunet_480x640_int8.tflite", npu=False), thr=0.9, blur_on=False)
    assert np.array_equal(off, img)


def test_missing_video_exits_with_message():
    import subprocess
    r = subprocess.run([sys.executable, str(Path(demo.__file__)), "--video", "/nonexistent.mp4", "--loop", "--display", "none",
                        "--models", "x_int8.tflite"], capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "cannot open video /nonexistent.mp4" in r.stderr, r.stderr[-300:]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
