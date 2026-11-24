"""The edge pipeline on a Jetson Nano (JetPack 4.6, Python 3.6, TensorRT 8.2): the same privacy policy as the i.MX
8M Plus board (src/edge_pipeline.py), with YuNet as a TensorRT engine on the GPU instead of INT8 on the NPU.

    video --> nvv4l2decoder (NVDEC) + nvvidconv resize to 640x480 --> YuNet TensorRT (GPU) --> decode + NMS (CPU)
          --> ByteTrack-lite + dark rule + look-back W (CPU) --> fast_blur (CPU) [--> display]

One thread, stages in a row (like edge_pipeline.py --serial), so the stage times add up to the frame time.
Engine: trtexec --onnx=yunet_ft_480x640.onnx --fp16 --saveEngine=yunet_ft_480x640_fp16.engine
Run (copy bytelite.py and anonymize.py next to it):
    python3 jetson_pipeline.py --video chokepoint.mp4 --engine yunet_ft_480x640_fp16.engine --risk d60_t0.1 --seconds 30
"""
import argparse
import json
import time
from collections import deque

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401  creates the CUDA context
import pycuda.driver as cuda
import tensorrt as trt

from anonymize import fast_blur, scale_boxes
from bytelite import LOW, LiteTracker, RiskGate

H, W = 480, 640
FLOOR = 0.05
GST = ("filesrc location={} ! qtdemux ! h264parse ! nvv4l2decoder ! nvvidconv ! "
       "video/x-raw,format=BGRx,width={},height={} ! videoconvert ! video/x-raw,format=BGR ! appsink sync=false")


class YuNetTRT:
    def __init__(self, engine):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.host, self.dev, self.names, self.bindings = {}, {}, [], []
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = tuple(self.engine.get_binding_shape(i))
            self.host[name] = cuda.pagelocked_empty(shape, np.float32)
            self.dev[name] = cuda.mem_alloc(self.host[name].nbytes)
            self.bindings.append(int(self.dev[name]))
            if not self.engine.binding_is_input(i):
                self.names.append(name)
        self.grids = {}
        for s in (8, 16, 32):
            rows, cols = np.divmod(np.arange(H // s * (W // s)), W // s)
            self.grids[s] = (cols.astype(np.float32), rows.astype(np.float32))

    def __call__(self, bgr):
        """bgr (480, 640, 3) -> ((n, 5) x y w h score, {pre, gpu, post ms})."""
        t0 = time.perf_counter()
        self.host["input"][0] = bgr.transpose(2, 0, 1)
        t1 = time.perf_counter()
        cuda.memcpy_htod_async(self.dev["input"], self.host["input"], self.stream)
        self.ctx.execute_async_v2(self.bindings, self.stream.handle)
        for n in self.names:
            cuda.memcpy_dtoh_async(self.host[n], self.dev[n], self.stream)
        self.stream.synchronize()
        t2 = time.perf_counter()
        out, boxes = self.host, []
        for s in (8, 16, 32):
            score = np.sqrt(np.clip(out["cls_%d" % s][0, :, 0], 0, 1) * np.clip(out["obj_%d" % s][0, :, 0], 0, 1))
            keep = score >= FLOOR
            if not keep.any():
                continue
            b = out["bbox_%d" % s][0][keep]
            cols, rows = self.grids[s][0][keep], self.grids[s][1][keep]
            cx, cy = (cols + b[:, 0]) * s, (rows + b[:, 1]) * s
            bw, bh = np.exp(b[:, 2]) * s, np.exp(b[:, 3]) * s
            boxes.append(np.stack([cx - bw / 2, cy - bh / 2, bw, bh, score[keep]], 1))
        a = np.concatenate(boxes).astype(np.float32) if boxes else np.zeros((0, 5), np.float32)
        if len(a):
            idx = cv2.dnn.NMSBoxes(a[:, :4].tolist(), a[:, 4].tolist(), FLOOR, 0.3, top_k=5000)
            a = a[np.asarray(idx, int).reshape(-1)]
        t3 = time.perf_counter()
        return a, {"pre": 1000 * (t1 - t0), "gpu": 1000 * (t2 - t1), "post": 1000 * (t3 - t2)}


def risk_knobs(spec):
    """'d60_t0.1' -> {'dark': 60, 'thr_dark': 0.1} (the dark rule of policies.py)."""
    kw = {}
    for part in spec.split("_"):
        if part.startswith("d"):
            kw["dark"] = int(part[1:])
        elif part.startswith("t"):
            kw["thr_dark"] = float(part[1:])
    return kw


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--engine", required=True)
    p.add_argument("--thr", type=float, default=None, help="default: 0.7 fine-tuned (yunet_ft*), 0.6 stock")
    p.add_argument("--back", type=int, default=3)
    p.add_argument("--risk", default="", help="dark rule knobs, e.g. d60_t0.1 (empty: off)")
    p.add_argument("--tracker", choices=["lite", "none"], default="lite")
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--loop", action="store_true")
    p.add_argument("--display", action="store_true", help="show on the Jetson's HDMI screen (cv2.imshow)")
    p.add_argument("--json", default=None)
    a = p.parse_args()
    thr = a.thr if a.thr is not None else (0.7 if "yunet_ft" in a.engine else 0.6)

    det = YuNetTRT(a.engine)
    cap = cv2.VideoCapture(GST.format(a.video, W, H), cv2.CAP_GSTREAMER)
    assert cap.isOpened(), "GStreamer pipeline did not open: " + a.video
    tracker = LiteTracker(thr) if a.tracker == "lite" else None
    gate = RiskGate(thr, **risk_knobs(a.risk)) if a.risk else None
    floor = min(LOW, gate.thr_dark) if gate else (LOW if tracker else thr)
    buf = deque()
    stage = {k: [] for k in ("read", "pre", "gpu", "post", "track", "blur", "frame")}
    n, t_start = 0, time.perf_counter()
    while time.perf_counter() - t_start < a.seconds:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            if not a.loop:
                break
            cap.release()
            cap = cv2.VideoCapture(GST.format(a.video, W, H), cv2.CAP_GSTREAMER)
            continue
        t1 = time.perf_counter()
        boxes, st = det(frame)
        boxes = boxes[boxes[:, 4] >= floor]
        t2 = time.perf_counter()
        if gate:
            grey = float(cv2.cvtColor(np.ascontiguousarray(frame[::8, ::8]), cv2.COLOR_BGR2GRAY).mean())
            boxes = gate.update(boxes, grey)
        boxes = tracker.update(boxes) if tracker else boxes[boxes[:, 4] >= thr]
        for old in buf:  # look-back: the new boxes also mask the W frames still waiting
            old[1].append(boxes)
        buf.append((frame, [boxes]))
        t3 = time.perf_counter()
        if len(buf) > a.back:
            f, masks = buf.popleft()
            m = np.concatenate(masks) if masks else np.zeros((0, 5))
            out = fast_blur(f, scale_boxes(m, f.shape)) if len(m) else f
            if a.display:
                cv2.imshow("jetson", out)
                cv2.waitKey(1)
        t4 = time.perf_counter()
        for k, v in (("read", t1 - t0), ("track", t3 - t2), ("blur", t4 - t3), ("frame", t4 - t0)):
            stage[k].append(1000 * v)
        for k in ("pre", "gpu", "post"):
            stage[k].append(st[k])
        n += 1
    elapsed = time.perf_counter() - t_start
    med = {k: round(float(np.median(v[10:])), 2) for k, v in stage.items() if len(v) > 10}
    summary = {"engine": a.engine, "video": a.video, "back": a.back, "risk": a.risk, "tracker": a.tracker,
               "frames": n, "fps": round(n / elapsed, 2), "stage_ms_p50": med}
    print(json.dumps(summary))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
