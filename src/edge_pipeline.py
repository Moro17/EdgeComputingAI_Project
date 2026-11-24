"""Pipelined edge anonymizer for the i.MX 8M Plus: every hardware unit works on a different frame at the same time.

    capture thread   VPU decodes H.264, the 2D GPU (g2d) converts colour AND scales to the detector size
          | queue
    infer thread     uint8 frame -> YuNet INT8 on the NPU -> decode + NMS (CPU)
          | queue
    output thread    online privacy policy (look-back delay buffer), fast_blur, display via g2d + Wayland

The serial loop (demo.py) leaves the NPU idle while the CPU decodes, pre/post-processes and blurs, and the CPU idle
while the NPU runs. Here they overlap: throughput is bounded by the slowest stage, not by the sum of the stages.
tflite invoke and the OpenCV calls release the GIL, so Python threads run truly in parallel on the 4 A53 cores.

Policy --back W (src/systems/policies.py backW, offline-equivalent): a frame is shown only after the next W frames
were detected, and it is masked with its own boxes plus theirs. That covers a person's first frames before the
detector fires (most identity leaks) at W frames of latency (W=3: 120 ms at 25 FPS). --back 0 = per-frame baseline.

--stats prints, every 2 s: FPS, end-to-end latency (capture -> shown), per-stage ms, NPU load (galcore debugfs),
per-core CPU load and SoC temperature. Several cameras: --streams K (K capture threads and VPU decoders, ONE NPU
thread: two processes using the NPU at once hung the galcore driver on this board), --source-fps 25 paces each source
like a live camera.

Usage (board, /root/edge with board_bench.py, anonymize.py, the models and a clip):
    python3 edge_pipeline.py --video demo.mp4 --model yunet_ft_480x640_u8.tflite --back 3 --display none --seconds 30
    python3 edge_pipeline.py --frames /root/chokepoint/P2E_S1_C1.1 --model ... --serial   # serial reference
"""
import argparse
import json
import queue
import re
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from board_bench import YuNetTFLite  # noqa: E402

try:
    from systems.anonymize import fast_blur, scale_boxes
    from systems.bytelite import LOW, LiteTracker, RiskGate
except ImportError:  # board: flat layout
    from anonymize import fast_blur, scale_boxes
    from bytelite import LOW, LiteTracker, RiskGate

def risk_knobs(spec):
    """'h80_p3_d60_t0.4' -> RiskGate kwargs (same knobs as policies.py's risk step; no pandas on the board)."""
    m = re.fullmatch(r"(?:_?l([\d.]+))?(?:_?h(\d+))?(?:_?p(\d+))?(?:_?d(\d+)_t([\d.]+))?", spec)
    if not m or not any(m.groups()):
        raise ValueError(f"--risk {spec!r}")
    lo, h, p, dk, td = m.groups()
    return {k: v for k, v in (("low", lo and float(lo)), ("h_min", h and int(h)), ("persist", p and int(p)),
                              ("dark", dk and int(dk)), ("thr_dark", td and float(td))) if v is not None}


GAMMA_DARK, GAMMA_TARGET = 60, 110  # as src/analysis/boost_dets.py (docs/next-runs.md, detection boosts)
_LUTS = {}


def gamma_input(frame):
    """Detector input of a dark frame (mean grey < 60): gamma LUT lifting the mean grey to 110; other frames as they
    are. Only the detector sees it; the blur is applied to the original frame. One LUT per integer mean, cached."""
    m = int(cv2.cvtColor(np.ascontiguousarray(frame[::8, ::8]), cv2.COLOR_BGR2GRAY).mean())
    if m >= GAMMA_DARK:
        return frame
    if m not in _LUTS:
        g = float(np.clip(np.log(GAMMA_TARGET / 255) / np.log(max(m, 1) / 255), 0.25, 1.0))
        _LUTS[m] = (255 * (np.arange(256) / 255) ** g).clip(0, 255).astype(np.uint8)
    return cv2.LUT(frame, _LUTS[m])


# VPU decode; g2d does colour conversion and (with {w}x{h}) the resize to the detector input, so the CPU does neither
GST_SCALED = ("filesrc location={path} ! qtdemux ! h264parse ! vpudec ! imxvideoconvert_g2d ! "
              "video/x-raw,format=BGRx,width={w},height={h} ! videoconvert ! video/x-raw,format=BGR ! "
              "appsink sync=false max-buffers=4 drop=false")
GST_FULL = ("filesrc location={path} ! qtdemux ! h264parse ! vpudec ! imxvideoconvert_g2d ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink sync=false max-buffers=4 drop=false")
SINK = ("appsrc ! videoconvert n-threads=2 ! imxvideoconvert_g2d ! video/x-raw,width=1440,height=1080 ! waylandsink sync=false")
STOP = object()


def draw_zone(tile, zone, boxes, shape):
    """Demo overlay: green outline of the door zone and a green box on every masked face centred in it."""
    th, tw = tile.shape[:2]
    gh, gw = zone.shape
    cnts, _ = cv2.findContours(cv2.resize(zone.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(tile, cnts, -1, (0, 200, 0), 2)
    if cnts:
        x, y, _, _ = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        cv2.putText(tile, "zona usii", (x + 4, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2, cv2.LINE_AA)
    sy, sx = th / shape[0], tw / shape[1]
    for x1, y1, x2, y2 in scale_boxes(boxes, shape) if len(boxes) else []:
        cx, cy = (x1 + x2) / 2 / shape[1], (y1 + y2) / 2 / shape[0]
        if zone[min(int(cy * gh), gh - 1), min(int(cx * gw), gw - 1)]:
            cv2.rectangle(tile, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)), (0, 255, 0), 2)


def default_thr(model):
    return 0.7 if "yunet_ft" in Path(model).name else 0.6  # chosen on val (docs/plan.md, Phase 4)


class Source:
    """Frames from an H.264 file (hardware path) or a folder of jpgs (CPU imread, for accuracy checks)."""

    def __init__(self, a, size):
        self.files, self.cap, self.a = None, None, a
        if a.frames:
            self.files = sorted(Path(a.frames).glob("*.jpg"))
        else:
            h, w = size
            self.pipe = (GST_SCALED.format(path=a.video, w=w, h=h) if a.g2d_resize else GST_FULL.format(path=a.video))
            self.open()
        self.i = 0

    def open(self):
        self.cap = cv2.VideoCapture(self.pipe, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():  # no VPU (desktop): software decode
            self.cap = cv2.VideoCapture(self.a.video)

    def read(self):
        if self.files is not None:
            if self.i >= len(self.files):
                if not self.a.loop:
                    return None
                self.i = 0
            img = cv2.imread(str(self.files[self.i]))
            self.i += 1
            return img
        ok, img = self.cap.read()
        if not ok and self.a.loop:
            self.cap.release()
            self.open()
            ok, img = self.cap.read()
        return img if ok else None


class LookBack:
    """Online backW: emit frame t-W masked with the boxes of frames t-W..t."""

    def __init__(self, w):
        self.w, self.buf = w, deque()

    def push(self, item):
        """item = dict(frame=..., boxes=(n, 5)). Returns the items ready to show (0 or 1, all at flush)."""
        for old in self.buf:  # the new detections also mask the W frames still waiting
            old["mask"].append(item["boxes"])
        item["mask"] = [item["boxes"]]
        self.buf.append(item)
        return [self.buf.popleft()] if len(self.buf) > self.w else []

    def flush(self):
        out, self.buf = list(self.buf), deque()
        return out


class Monitor(threading.Thread):
    """Samples NPU load (galcore), per-core CPU busy % and SoC temperature every second."""

    GC = Path("/sys/kernel/debug/gc/load")
    TEMP = Path("/sys/class/thermal/thermal_zone0/temp")

    def __init__(self):
        super().__init__(daemon=True)
        self.npu, self.cpu, self.temp, self.run_ = [], [], [], True

    @staticmethod
    def cpu_times():
        rows = [ln.split() for ln in open("/proc/stat") if ln.startswith("cpu") and ln[3] != " "]
        return np.array([[int(v) for v in r[1:8]] for r in rows], float)

    def run(self):
        try:
            prev = self.cpu_times()
        except OSError:
            return
        while self.run_:
            time.sleep(1)
            cur = self.cpu_times()
            d = cur - prev
            prev = cur
            self.cpu.append(100 * (1 - (d[:, 3] + d[:, 4]) / np.maximum(d.sum(1), 1)))  # idle + iowait
            try:
                txt = self.GC.read_text()
                loads = [int(x.split(":")[1].strip().rstrip("%")) for x in txt.splitlines() if x.startswith("load")]
                self.npu.append(loads[1] if len(loads) > 1 else np.nan)  # core 1 = VIP8000 NPU (core 0 = 3D GPU)
            except (OSError, ValueError, IndexError):
                self.npu.append(np.nan)
            try:
                self.temp.append(int(self.TEMP.read_text()) / 1000)
            except OSError:
                pass


PAGE = (b"<!doctype html><html><head><title>One Frame Is Enough - i.MX 8M Plus</title></head>"
        b"<body style='margin:0;background:#111;display:flex;justify-content:center;align-items:center;height:100vh'>"
        b"<img src='/stream' style='max-width:100%;max-height:100vh'></body></html>")


def web_threads(canvas, port, fps, stop, stage):
    """MJPEG over HTTP: one thread JPEG-encodes the latest mosaic at `fps`, one HTTP server thread per viewer.
    Like the HDMI display it runs beside the pipeline, so a slow viewer never slows the anonymizer."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    latest, cond = [None], threading.Condition()

    def encode():
        nxt = time.perf_counter()
        while not stop.is_set():
            nxt += 1 / fps
            time.sleep(max(0.0, nxt - time.perf_counter()))
            t = time.perf_counter()
            ok, jpg = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
            stage.setdefault("web_jpeg", []).append(1000 * (time.perf_counter() - t))
            if ok:
                with cond:
                    latest[0] = jpg.tobytes()
                    cond.notify_all()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path != "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(PAGE)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while not stop.is_set():
                    with cond:
                        cond.wait(1.0)
                        jpg = latest[0]
                    if jpg:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.daemon_threads = True
    print(f"browser stream: http://<board>:{port}/", flush=True)
    return [threading.Thread(target=encode, daemon=True), threading.Thread(target=server.serve_forever, daemon=True)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--frames")
    p.add_argument("--model", required=True)
    p.add_argument("--thr", type=float, default=None)
    p.add_argument("--back", type=int, default=3, help="look-back frames (0 = per-frame baseline)")
    p.add_argument("--every", type=int, default=1, help="detect every N-th frame of a camera; the frames in between "
                   "reuse the last keyframe's boxes (policies.py skipN+hold)")
    p.add_argument("--tracker", choices=["lite", "none"], default="lite",
                   help="lite: numpy ByteTrack-lite per camera keeps low-score faces of a track masked (the privacy "
                        "gain needs it: without a tracker look-back alone is worse than a temporal window)")
    p.add_argument("--risk", default="", metavar="KNOBS",
                   help="risk-adaptive threshold before the tracker, knobs as in policies.py without 'risk_', "
                        "e.g. h13_p3_d60_t0.4, h = percent of frame height (empty: off)")
    p.add_argument("--zone", default=None, metavar="NPY",
                   help="door zone of this camera (bool grid, systems/scene.py reflection_zone, written next to the "
                        "demo video by make_demo_videos.py): a detection >= --zone-thr centred in it is masked")
    p.add_argument("--zone-thr", type=float, default=0.1)
    p.add_argument("--gamma", action="store_true", help="dark frames (mean grey < 60): gamma-lift the detector input "
                   "(test dark:2 exposure 34.5%% -> 3.4%% fine-tuned, 10.3%% -> 0%% stock; docs/next-runs.md)")
    p.add_argument("--show-zone", action="store_true", help="demo: green outline of the zone and green boxes on the "
                   "faces masked inside it")
    p.add_argument("--streams", type=int, default=1, help="cameras: one capture thread + VPU decoder each, ONE NPU thread")
    p.add_argument("--source-fps", type=float, default=0, help="pace each source like a live camera (0 = as fast as possible)")
    p.add_argument("--serial", action="store_true", help="one thread, stages one after another (demo.py style)")
    p.add_argument("--g2d-resize", type=int, default=1, help="1: g2d scales to the detector size in the decoder")
    p.add_argument("--display", choices=["none", "sink", "web", "both"], default="none",
                   help="sink: HDMI via g2d + Wayland; web: MJPEG in a browser at http://<board>:--web-port/; "
                        "both: HDMI and browser. With several streams a 2-column mosaic of all cameras")
    p.add_argument("--web-port", type=int, default=8080)
    p.add_argument("--web-fps", type=float, default=15, help="browser stream rate (JPEG encoding runs on the CPU)")
    p.add_argument("--loop", action="store_true")
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--cpu", action="store_true", help="no NPU (desktop test)")
    p.add_argument("--cpu-model", default=None,
                   help="second inference engine on the A53 cores (FP32 .tflite, XNNPACK): takes frames from the same "
                        "queue as the NPU when the NPU is busy (heterogeneous NPU + CPU)")
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--json", default=None, help="write the summary here")
    p.add_argument("--tag", default="")
    a = p.parse_args()
    assert not (a.serial and a.streams > 1), "--serial runs one stream"
    thr = a.thr if a.thr is not None else default_thr(a.model)

    det = YuNetTFLite(a.model, npu=not a.cpu)
    det_cpu = YuNetTFLite(a.cpu_model, npu=False, threads=a.cpu_threads) if a.cpu_model else None
    engine_frames = {"npu": 0, "cpu": 0}
    sources = [Source(a, (det.h, det.w)) for _ in range(a.streams)]
    sink = None
    cols = 1 if a.streams == 1 else 2
    rows_ = -(-a.streams // cols)
    th_, tw_ = (det.h, det.w) if a.streams == 1 else (det.h // 2, det.w // 2)  # mosaic tiles at half size: the
    canvas = np.zeros((rows_ * th_, cols * tw_, 3), np.uint8)  # CPU colour conversion of the sink costs per pixel
    recent = [deque(maxlen=25) for _ in range(a.streams)]
    draw = a.display != "none"  # the mosaic canvas feeds the HDMI sink and/or the browser stream
    if a.display in ("sink", "both"):  # run with XDG_RUNTIME_DIR=/run/user/0 WAYLAND_DISPLAY=wayland-1 (scripts/run_demo.sh)
        sink = cv2.VideoWriter(SINK, cv2.CAP_GSTREAMER, 0, 25, (cols * tw_, rows_ * th_), True)
        assert sink.isOpened(), "waylandsink did not open (HDMI screen? Wayland env?)"
    policies = [LookBack(a.back) for _ in range(a.streams)]
    trackers = [LiteTracker(thr) if a.tracker == "lite" else None for _ in range(a.streams)]
    rk = risk_knobs(a.risk) if a.risk else {}
    rk["h_min"] = rk.get("h_min", 0) * det.h / 100  # percent of the detector input height -> pixels
    zone = np.load(a.zone).astype(bool) if a.zone else None
    if zone is not None:  # door zone rule (policies.py risk _zrT_m0): no motion gate on the board
        rk |= {"thr_zone": a.zone_thr, "motion_min": 0.0, "zone": zone}
    still = np.zeros((60, 80), np.uint8)  # motion map placeholder: motion_min 0 accepts every box
    gates = [RiskGate(thr, **rk) if (a.risk or zone is not None) else None for _ in range(a.streams)]
    held = [np.empty((0, 5), np.float32) for _ in range(a.streams)]  # last keyframe's boxes, for --every
    floor = LOW if a.tracker == "lite" else thr
    if gates[0]:
        floor = min([floor, gates[0].low] + [t for t in (gates[0].thr_dark, gates[0].thr_zone) if t is not None])

    def track(s, boxes, frame):
        t = time.perf_counter()
        if gates[s]:  # mean grey of the detector input, subsampled: what policies.py's dark rule reads offline
            gates[s].frame_hw = frame.shape[:2]
            boxes = gates[s].update(boxes, float(cv2.cvtColor(np.ascontiguousarray(frame[::8, ::8]), cv2.COLOR_BGR2GRAY).mean()),
                                    still if zone is not None else None)
        out = trackers[s].update(boxes) if trackers[s] else boxes[boxes[:, 4] >= thr]
        stage.setdefault("track", []).append(1000 * (time.perf_counter() - t))
        return out
    mon = Monitor()
    mon.start()
    lat = [[] for _ in range(a.streams)]
    stage = {k: [] for k in ("read", "pre", "npu", "post", "out")}
    shown, t_start, stop, lock = [0] * a.streams, [None], threading.Event(), threading.Lock()

    def detect(frame):
        t = time.perf_counter()
        x = gamma_input(frame) if a.gamma else frame
        if a.gamma:
            stage.setdefault("gamma", []).append(1000 * (time.perf_counter() - t))
        boxes, st = det(x)
        for k in ("pre", "npu", "post"):
            stage[k].append(st[k])
        return boxes[boxes[:, 4] >= floor]

    def show(item):
        t = time.perf_counter()
        boxes = np.concatenate(item["mask"]) if item["mask"] else np.empty((0, 5), np.float32)
        out = fast_blur(item["frame"], scale_boxes(boxes, item["frame"].shape)) if len(boxes) else item["frame"]
        if draw:
            si = item["s"]
            recent[si].append(time.perf_counter())
            r_, c_ = divmod(si, cols)
            tile = canvas[r_ * th_:(r_ + 1) * th_, c_ * tw_:(c_ + 1) * tw_]
            tile[:] = out if out.shape[:2] == (th_, tw_) else cv2.resize(out, (tw_, th_), interpolation=cv2.INTER_AREA)
            q = recent[si]
            fps_now = (len(q) - 1) / (q[-1] - q[0]) if len(q) > 1 and q[-1] > q[0] else 0.0
            if a.show_zone and zone is not None:
                draw_zone(tile, zone, boxes, item["frame"].shape)
            for color, th in (((0, 0, 0), 4), ((255, 255, 255), 2)):
                cv2.putText(tile, f"cam {si + 1}  {fps_now:4.1f} FPS  faces {len(item['boxes'])}", (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, th, cv2.LINE_AA)
        now = time.perf_counter()
        stage["out"].append(1000 * (now - t))
        if t_start[0] is None:
            t_start[0] = now
        lat[item["s"]].append(1000 * (now - item["t0"]))
        shown[item["s"]] += 1

    def read(s):
        t = time.perf_counter()
        f = sources[s].read()
        with lock:
            stage["read"].append(1000 * (time.perf_counter() - t))
        return f, t

    deadline = time.perf_counter() + a.seconds
    if a.serial:
        n_serial = 0
        while time.perf_counter() < deadline:
            f, t0 = read(0)
            if f is None:
                break
            if n_serial % a.every == 0:
                held[0] = track(0, detect(f), f)
            n_serial += 1
            for it in policies[0].push({"s": 0, "frame": f, "boxes": held[0], "t0": t0}):
                show(it)
    else:
        q1, q2 = queue.Queue(4 * a.streams), queue.Queue(4 * a.streams)

        def capture(s):
            nxt = time.perf_counter()
            while not stop.is_set():
                if a.source_fps:  # a camera delivers a frame every 1/fps s, whether we are ready or not
                    nxt += 1 / a.source_fps
                    time.sleep(max(0.0, nxt - time.perf_counter()))
                f, t0 = read(s)
                if a.source_fps:
                    t0 = time.perf_counter()  # latency counts from when the frame exists, not from the read
                if f is None:
                    q1.put(STOP)
                    return
                it = {"s": s, "n": seq[s], "frame": f, "t0": t0}
                if a.gamma and seq[s] % a.every == 0:  # here, not in the NPU thread: overlaps the NPU's previous frame
                    t = time.perf_counter()
                    it["det_in"] = gamma_input(f)
                    with lock:
                        stage.setdefault("gamma", []).append(1000 * (time.perf_counter() - t))
                q1.put(it)
                seq[s] += 1

        seq = [0] * a.streams
        qm = queue.Queue(4 * a.streams)

        def infer(engine, d):
            """NPU thread (only the uint8 copy and the NPU call) or, with --cpu-model, a CPU engine that only takes
            a frame when a backlog builds up (the NPU is busy), so it adds throughput without slowing light loads."""
            while True:
                if engine == "cpu":
                    if q1.qsize() < a.streams:
                        time.sleep(0.002)
                        continue
                    try:
                        it = q1.get_nowait()
                    except queue.Empty:
                        continue
                else:
                    it = q1.get()
                if it is STOP:
                    q1.put(STOP)  # let the other engine see it too
                    qm.put(STOP)
                    return
                if it["n"] % a.every:  # not a keyframe: no detector run, output() reuses the last boxes
                    it["held"] = True
                    qm.put(it)
                    continue
                x = it.pop("det_in", it["frame"])
                it["raw"], it["r"], st = d.forward(x)
                it["det"] = d
                if engine == "npu":
                    stage["pre"].append(st["pre"])
                    stage["npu"].append(st["npu"])
                else:
                    stage.setdefault("cpu_infer", []).append(st["pre"] + st["npu"])
                engine_frames[engine] += 1
                qm.put(it)

        def post():  # CPU thread: anchor decoding + NMS of frame t while the NPU runs frame t+1
            while True:
                it = qm.get()
                if it is STOP:
                    q2.put(STOP)
                    return
                if it.get("held"):
                    q2.put(it)
                    continue
                boxes, ms = it.pop("det").decode(it.pop("raw"), it["r"])
                stage["post"].append(ms)
                it["boxes"] = boxes[boxes[:, 4] >= floor]
                q2.put(it)

        def output():
            pending, want = [dict() for _ in range(a.streams)], [0] * a.streams  # two engines can finish out of order
            while True:
                it = q2.get()
                if it is STOP:
                    break
                s = it["s"]
                pending[s][it["n"]] = it
                while want[s] in pending[s]:
                    nxt_ = pending[s].pop(want[s])
                    if not nxt_.get("held"):  # the tracker needs the frames of a camera in order
                        held[s] = track(s, nxt_["boxes"], nxt_["frame"])
                    nxt_["boxes"] = held[s]
                    for r in policies[s].push(nxt_):
                        show(r)
                    want[s] += 1

        threads = [threading.Thread(target=capture, args=(s,), daemon=True) for s in range(a.streams)]
        threads += [threading.Thread(target=infer, args=("npu", det), daemon=True)]
        if det_cpu:
            threads += [threading.Thread(target=infer, args=("cpu", det_cpu), daemon=True)]
        threads += [threading.Thread(target=f, daemon=True) for f in (post, output)]
        if sink:
            def display():  # own thread: a slow screen never slows the anonymizer; shows the latest mosaic at 25 FPS
                nxt = time.perf_counter()
                while not stop.is_set():
                    nxt += 1 / 25
                    time.sleep(max(0.0, nxt - time.perf_counter()))
                    t = time.perf_counter()
                    sink.write(canvas.copy())
                    stage.setdefault("display", []).append(1000 * (time.perf_counter() - t))
            threads.insert(0, threading.Thread(target=display, daemon=True))
        if a.display in ("web", "both"):
            threads[:0] = web_threads(canvas, a.web_port, a.web_fps, stop, stage)
        for t in threads:
            t.start()
        while time.perf_counter() < deadline and threads[-1].is_alive():
            time.sleep(0.2)
        stop.set()
    mon.run_ = False
    end = time.perf_counter()
    elapsed = end - (t_start[0] or end)
    skip = 10  # warm-up frames
    med = lambda v: round(float(np.median(v[skip:])), 2) if len(v) > skip else None  # noqa: E731
    allat = [x for v in lat for x in v[skip:]]
    cpu = np.array(mon.cpu[2:]) if len(mon.cpu) > 2 else np.zeros((1, 4))
    fps_s = [round((n - 1) / elapsed, 2) if elapsed > 0 and n > 1 else None for n in shown]
    npu_ms = med(stage["npu"])
    summary = {
        "tag": a.tag, "model": Path(a.model).name, "tracker": a.tracker, "risk": a.risk, "zone": a.zone, "gamma": a.gamma, "input_dtype": np.dtype(det.dtype).name, "back": a.back, "every": a.every,
        "mode": "serial" if a.serial else "pipelined", "streams": a.streams, "source_fps": a.source_fps,
        "g2d_resize": bool(a.g2d_resize and a.video),
        "fps_total": round(sum(f or 0 for f in fps_s), 2), "fps_per_stream": fps_s,
        "latency_ms_p50": round(float(np.median(allat)), 1) if allat else None,
        "latency_ms_p95": round(float(np.percentile(allat, 95)), 1) if allat else None,
        "stage_ms_p50": {k: med(v) for k, v in stage.items()},
        # NPU duty = share of wall time the NPU stage is busy (frames/s x ms per frame); galcore load is also logged
        "npu_duty_pct": round(sum(f or 0 for f in fps_s) / a.every * npu_ms / 10, 1) if npu_ms else None,  # keyframes only
        "galcore_npu_load_pct": round(float(np.nanmean(mon.npu[2:])), 1) if len(mon.npu) > 2 else None,
        "cpu_busy_pct_per_core": [round(float(c), 1) for c in cpu.mean(0)],
        "cpu_busy_pct_total": round(float(cpu.mean(0).sum()), 1),
        "temp_c_max": max(mon.temp) if mon.temp else None, "frames": sum(shown),
        "cpu_model": Path(a.cpu_model).name if a.cpu_model else None, "frames_by_engine": engine_frames,
    }
    print(json.dumps(summary), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(summary, indent=1))
    if sink:
        sink.release()
    import os
    os._exit(0)  # daemon threads may still block in GStreamer / queue.put


if __name__ == "__main__":
    main()
