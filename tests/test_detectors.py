"""Detector self-checks on a real ChokePoint frame (needs data/chokepoint; models download on first use).
Run: python tests/test_detectors.py"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loaders.chokepoint import frame_path, load_gt  # noqa: E402
from systems.detectors import SYSTEMS  # noqa: E402


def sample():
    seq = "P2E_S1_C1.1"
    gt = load_gt(seq)
    g = gt.iloc[len(gt) // 2]  # a mid-track frontal face
    return g, cv2.imread(str(frame_path(seq, int(g.frame))))


def test_detectors_find_the_chokepoint_face():
    g, img = sample()
    for name in ["yunet", "yolov8n_face", "yunet_fp16", "scrfd_500m_fp16", "yolov8n_face_fp16"]:
        b = SYSTEMS[name][0](None)(img)
        assert b.dtype == np.float32 and b.shape[1] == 5, name
        hit = b[(b[:, 4] >= 0.5) & (abs(b[:, 0] + b[:, 2] / 2 - (g.x + g.w / 2)) < g.w / 2)]
        assert len(hit) >= 1, name


def test_yunet_ort_matches_opencv():
    _, img = sample()
    a, b = SYSTEMS["yunet"][0](None)(img), SYSTEMS["yunet_ort"][0](None)(img)
    a, b = a[a[:, 4] >= 0.5], b[b[:, 4] >= 0.5]
    assert len(a) == len(b) >= 1
    assert np.abs(a[np.argsort(a[:, 0])] - b[np.argsort(b[:, 0])]).max() < 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
