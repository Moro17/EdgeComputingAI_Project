"""Fail fast if the judge or ONNX detectors are not running on CUDA (ORT silently falls back to CPU).

Usage: python src/check_gpu.py
"""
import numpy as np

import onnxruntime as ort
from judge.arcface import Judge
from systems.detectors import SCRFD, Deface, used_providers

print("onnxruntime", ort.__version__, ort.get_available_providers())
j = Judge()
j.embed([np.zeros((112, 112, 3), np.uint8)] * 3)
s = SCRFD("buffalo_s", "det_500m.onnx")
s(np.zeros((600, 800, 3), np.uint8))
d = Deface()
d(np.zeros((600, 800, 3), np.uint8))
for name, net in [("judge", j.net), ("scrfd", s.net), ("deface", d.net)]:
    print(name, used_providers(net))
    assert used_providers(net)[0] == "CUDAExecutionProvider", f"{name} fell back to CPU"
print("GPU OK")
