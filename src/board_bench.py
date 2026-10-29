"""Board benchmark: a YOLOv8n-face or YuNet INT8 .tflite on the i.MX 8M Plus NPU over frame folders -> the det cache.

Runs on the board with only tflite_runtime, OpenCV and numpy (no pandas). Writes, per sequence:
  <out>/<name>/<seq>.csv   frame, track_id, x, y, w, h, score   (every detection >= FLOOR, like src/benchmark.py)
  <out>/<name>/<seq>.json  latency_ms (preprocess + NPU + decode/NMS per frame), per-stage ms, device
Copy <out>/<name>/ to experiments/<run>/dets/ on the Mac and score it there with src/benchmark.py --systems <name>.

Model I/O (src/export_tflite.py): float32 NCHW (1, 3, S, S) RGB/255 letterboxed; output (1, 20, N) =
cx, cy, w, h (fractions of S), score, 5 landmarks x (x, y, visible).
shortcut: INT8 calibration capped the score logit at 0, so faces saturate at ~0.4986 (operating threshold 0.49);
recalibrate the score head when exporting the fine-tuned model.

Usage (board): python3 board_bench.py --model yolov8n-face_320_int8.tflite --frames /root/chokepoint --out /root/dets
       (Mac sanity check: add --cpu; needs ai-edge-litert)
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

FLOOR = 0.05  # same as src/systems/detectors.py
NMS_IOU = 0.5
VX = "/usr/lib/libvx_delegate.so"


def interpreter(model, npu):
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate
    except ImportError:
        from ai_edge_litert.interpreter import Interpreter, load_delegate
    if npu:  # cache the compiled NPU graph: the first run otherwise takes 8-18 s every start
        os.environ.setdefault("VIV_VX_ENABLE_CACHE_GRAPH_BINARY", "1")
        os.environ.setdefault("VIV_VX_CACHE_BINARY_GRAPH_DIR", str(Path(model).resolve().parent))
    net = Interpreter(str(model), experimental_delegates=[load_delegate(VX)] if npu else None, num_threads=4)
    net.allocate_tensors()
    return net


class YoloTFLite:
    def __init__(self, model, npu=True):
        self.net = interpreter(model, npu)
        self.inp, self.out = self.net.get_input_details()[0], self.net.get_output_details()[0]
        self.size = int(self.inp["shape"][2])
        self.net.set_tensor(self.inp["index"], np.zeros(self.inp["shape"], np.float32))
        self.net.invoke()  # warm-up: NPU graph compile happens here

    def __call__(self, bgr):
        """BGR frame -> ((n, 5) float32 x, y, w, h, score in frame pixels, {stage: ms})."""
        t0 = time.perf_counter()
        h, w = bgr.shape[:2]
        s = self.size
        r = min(s / w, s / h)
        nw, nh = round(w * r), round(h * r)
        px, py = (s - nw) // 2, (s - nh) // 2  # centred letterbox, as ultralytics and src/systems/detectors.YoloFace
        canvas = np.full((s, s, 3), 114, np.uint8)
        canvas[py : py + nh, px : px + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255
        t1 = time.perf_counter()
        self.net.set_tensor(self.inp["index"], x)
        self.net.invoke()
        o = self.net.get_tensor(self.out["index"])[0]
        t2 = time.perf_counter()
        o = o[:, o[4] >= FLOOR]
        cx, cy, bw, bh = o[:4] * s
        boxes = np.stack([(cx - bw / 2 - px) / r, (cy - bh / 2 - py) / r, bw / r, bh / r, o[4]], 1).astype(np.float32)
        keep = cv2.dnn.NMSBoxes(boxes[:, :4].tolist(), boxes[:, 4].tolist(), FLOOR, NMS_IOU)
        boxes = boxes[np.asarray(keep, int).reshape(-1)]
        t3 = time.perf_counter()
        return boxes, {"pre": 1000 * (t1 - t0), "npu": 1000 * (t2 - t1), "post": 1000 * (t3 - t2)}


class YuNetTFLite:
    """YuNet INT8 from onnx2tf: float32 NHWC (1, H, W, 3) raw BGR 0-255, frame at top-left, zero padding.
    Outputs output_0..11 = ONNX order cls_8/16/32, obj_8/16/32, bbox_8/16/32, kps_8/16/32.
    Decoded as src/systems/detectors.YuNetORT (OpenCV FaceDetectorYN)."""

    def __init__(self, model, npu=True):
        self.net = interpreter(model, npu)
        self.run = self.net.get_signature_runner()
        _, self.h, self.w, _ = self.run.get_input_details()["input"]["shape"]
        self.size = f"{self.h}x{self.w}"
        self.run(input=np.zeros((1, self.h, self.w, 3), np.float32))  # warm-up / NPU graph compile

    def __call__(self, bgr):
        t0 = time.perf_counter()
        h, w = bgr.shape[:2]
        r = min(self.w / w, self.h / h)
        x = np.zeros((1, self.h, self.w, 3), np.float32)
        x[0, : round(h * r), : round(w * r)] = bgr if r == 1 else cv2.resize(bgr, (round(w * r), round(h * r)))
        t1 = time.perf_counter()
        o = self.run(input=x)
        t2 = time.perf_counter()
        boxes = []
        for k, s in enumerate((8, 16, 32)):
            rows, cols = np.divmod(np.arange(self.h // s * (self.w // s)), self.w // s)
            cls, obj, b = o[f"output_{k}"][0, :, 0], o[f"output_{k + 3}"][0, :, 0], o[f"output_{k + 6}"][0]
            score = np.sqrt(np.clip(cls, 0, 1) * np.clip(obj, 0, 1))
            cx, cy = (cols + b[:, 0]) * s, (rows + b[:, 1]) * s
            bw, bh = np.exp(b[:, 2]) * s, np.exp(b[:, 3]) * s
            boxes.append(np.stack([cx - bw / 2, cy - bh / 2, bw, bh, score], 1))
        a = np.concatenate(boxes).astype(np.float32)
        a = a[a[:, 4] >= FLOOR]
        keep = cv2.dnn.NMSBoxes(a[:, :4].tolist(), a[:, 4].tolist(), FLOOR, 0.3, top_k=5000)  # YuNet's NMS 0.3
        a = a[np.asarray(keep, int).reshape(-1)]
        a[:, :4] /= r
        t3 = time.perf_counter()
        return a, {"pre": 1000 * (t1 - t0), "npu": 1000 * (t2 - t1), "post": 1000 * (t3 - t2)}


def run(det, frames, out, device):
    paths = sorted(frames.glob("*.jpg"))
    rows, lat, stages = [], [], {"read": [], "pre": [], "npu": [], "post": []}
    for i, p in enumerate(paths):
        t = time.perf_counter()
        img = cv2.imread(str(p))
        read = 1000 * (time.perf_counter() - t)
        boxes, st = det(img)
        if i >= 5:  # skip warm-up, as src/benchmark.py
            lat.append(st["pre"] + st["npu"] + st["post"])
            for k, v in st.items():
                stages[k].append(v)
            stages["read"].append(read)
        rows += [(int(p.stem), -1, *(round(float(v), 2) for v in b[:4]), round(float(b[4]), 4)) for b in boxes]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{frames.name}.csv", "w", newline="") as f:
        csv.writer(f).writerows([("frame", "track_id", "x", "y", "w", "h", "score"), *rows])
    (out / f"{frames.name}.json").write_text(json.dumps({"latency_ms": lat, "stages": stages, "device": device}))
    return np.mean(lat), {k: round(float(np.mean(v)), 1) for k, v in stages.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--frames", type=Path, required=True, help="folder of sequence folders of <frame:08d>.jpg")
    p.add_argument("--seqs", nargs="+", default=None, help="default: every sub-folder of --frames")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--name", default=None, help="system name; default <model stem minus _int8>_npu")
    p.add_argument("--cpu", action="store_true", help="no NPU delegate (Mac sanity check / CPU baseline)")
    args = p.parse_args()

    name = args.name or args.model.stem.replace("-", "_").replace("_int8", "") + ("_cpu" if args.cpu else "_npu")
    device = "cpu-tflite" if args.cpu else "imx8mp-npu-vx"
    det = (YuNetTFLite if "yunet" in args.model.name else YoloTFLite)(args.model, npu=not args.cpu)
    for seq in args.seqs or sorted(d.name for d in args.frames.iterdir() if d.is_dir()):
        ms, st = run(det, args.frames / seq, args.out / name, device)
        print(f"{name} {seq}: {ms:.1f} ms/frame {st}", flush=True)


if __name__ == "__main__":
    main()
