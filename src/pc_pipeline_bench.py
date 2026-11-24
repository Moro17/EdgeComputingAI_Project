"""The edge pipeline on the laptop CPU, for the platform comparison (docs/edge-board.md, docs/jetson.md): the same
stages as the boards' serial loop (edge_pipeline.py --serial, jetson_pipeline.py), with YuNet 480x640 on onnxruntime:

    OpenCV decode (CPU) --> YuNet ORT FP32 or INT8 (CPU) --> ByteTrack-lite + dark rule --> look-back 3 --> fast_blur

Resources: process CPU time / wall time (% of ONE core, like top), peak working set (Windows), FPS, stage p50.

    python src/pc_pipeline_bench.py --video dist/demo_videos/chokepoint.mp4 --system yqft_fp32 --json out.json
"""
import argparse
import ctypes
import json
import os
import sys
import time
from ctypes import wintypes
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edge_pipeline import LookBack  # noqa: E402
from systems.anonymize import fast_blur, scale_boxes  # noqa: E402
from systems.bytelite import LOW, LiteTracker, RiskGate  # noqa: E402
from systems.detectors import SYSTEMS  # noqa: E402


def peak_rss_mb():
    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                           "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                           "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    k.K32GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.PeakWorkingSetSize / 2 ** 20


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--system", default="yqft_fp32", help="systems/detectors.py name (yqft_fp32, yqft_board = INT8)")
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--json", default=None)
    a = p.parse_args()
    factory, thr = SYSTEMS[a.system]
    det = factory(None)
    tracker, gate, back = LiteTracker(thr), RiskGate(thr, dark=60, thr_dark=0.1), LookBack(3)
    floor = min(LOW, 0.1)
    cap = cv2.VideoCapture(a.video)
    stage = {k: [] for k in ("read", "detect", "track", "blur", "frame")}
    n, t_end, c0, w0 = 0, time.perf_counter() + a.seconds, time.process_time(), time.perf_counter()
    while time.perf_counter() < t_end:
        t = time.perf_counter()
        ok, f = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        f = cv2.resize(f, (640, 480), interpolation=cv2.INTER_AREA)
        t1 = time.perf_counter()
        b = det(f).astype(np.float32)
        b = b[b[:, 4] >= floor] if len(b) else np.zeros((0, 5), np.float32)
        t2 = time.perf_counter()
        b = gate.update(b, float(cv2.cvtColor(np.ascontiguousarray(f[::8, ::8]), cv2.COLOR_BGR2GRAY).mean()))
        b = tracker.update(b)
        t3 = time.perf_counter()
        for it in back.push({"frame": f, "boxes": b}):
            m = np.concatenate(it["mask"])
            if len(m):
                fast_blur(it["frame"], scale_boxes(m, it["frame"].shape))
        t4 = time.perf_counter()
        for k, v in zip(stage, (t1 - t, t2 - t1, t3 - t2, t4 - t3, t4 - t)):
            stage[k].append(1000 * v)
        n += 1
    wall = time.perf_counter() - w0
    out = {"system": a.system, "video": Path(a.video).name, "frames": n, "fps": round(n / wall, 2),
           "stage_ms_p50": {k: round(float(np.median(v)), 2) for k, v in stage.items()},
           "cpu_pct_of_one_core": round(100 * (time.process_time() - c0) / wall, 1), "cores": os.cpu_count(),
           "peak_rss_mb": round(peak_rss_mb(), 1)}
    print(json.dumps(out))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
