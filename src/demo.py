"""Live demo on the i.MX 8M Plus: video file -> hardware decode -> face detector on the NPU -> fast_blur -> HDMI + FPS overlay.

Usage (board): XDG_RUNTIME_DIR=/run/user/0 WAYLAND_DISPLAY=wayland-1 python3 demo.py --video demo.mp4 --loop \
    --models yunet_480x640_int8.tflite yunet_ft_480x640_int8.tflite yolov8n-face_320_int8.tflite
Models are cycled every --switch-every seconds (the board has no keyboard). Default display is a GStreamer waylandsink
(g2d-scaled to 1440x1080, ~0 ms: cv2.imshow on GTK costs 25 ms/frame there); --display window uses cv2.imshow with keys m next model,
b blur on/off, q quit, at the price of ~16 FPS.
Needs board_bench.py and anonymize.py next to this file (or src/ on the Mac)."""
import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from board_bench import YoloTFLite, YuNetTFLite  # noqa: E402

try:
    from systems.anonymize import fast_blur, scale_boxes
except ImportError:  # on the board anonymize.py sits next to this file
    from anonymize import fast_blur, scale_boxes

GST = ("filesrc location={} ! qtdemux ! h264parse ! vpudec ! imxvideoconvert_g2d ! video/x-raw,format=BGRx ! "
       "videoconvert ! video/x-raw,format=BGR ! appsink sync=false")


def is_yunet(name):
    return "yunet" in Path(name).name


def default_thr(name):
    return 0.9 if is_yunet(name) else 0.49  # YOLO's INT8 score saturates at ~0.4986 (board_bench.py)


def load(path, npu=True):
    return (YuNetTFLite if is_yunet(path) else YoloTFLite)(path, npu=npu)


def process(bgr, det, thr, blur_on):
    """-> (anonymized frame, detector stage ms, number of faces)."""
    boxes, st = det(bgr)
    boxes = boxes[boxes[:, 4] >= thr]
    t = time.perf_counter()
    out = fast_blur(bgr, scale_boxes(boxes, bgr.shape)) if blur_on and len(boxes) else bgr
    return out, {**st, "blur": 1000 * (time.perf_counter() - t)}, len(boxes)


def overlay_text(name, fps, stages, n_faces):
    return [name, f"{fps:.1f} FPS", f"pre {stages['pre']:.1f} | npu {stages['npu']:.1f} | post {stages['post']:.1f} ms",
            f"faces {n_faces}"]


def draw(img, lines):
    for text, scale, y in zip(lines, (1.4, 1.0, 0.6, 0.8), (45, 90, 125, 160)):
        for color, th in (((0, 0, 0), 4), ((255, 255, 255), 2)):  # outline keeps text legible on any frame
            cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def open_video(path):
    cap = cv2.VideoCapture(GST.format(path), cv2.CAP_GSTREAMER)
    if not cap.isOpened():  # Mac, or no VPU
        cap = cv2.VideoCapture(str(path))
    return cap


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--thr", type=float, nargs="+", default=None, help="one per model; default 0.9 YuNet, 0.49 YOLO")
    p.add_argument("--loop", action="store_true")
    p.add_argument("--switch-every", type=float, default=10, help="seconds per model, 0 = only on key m")
    p.add_argument("--cpu", action="store_true", help="no NPU (Mac test)")
    p.add_argument("--display", choices=["sink", "window", "none"], default="sink")
    p.add_argument("--save-frame", default=None, help="write one overlaid frame (after 60 frames) to this jpg")
    a = p.parse_args()
    thrs = a.thr or [default_thr(m) for m in a.models]
    assert len(thrs) == len(a.models), "--thr needs one value per model"
    dets = [load(m, npu=not a.cpu) for m in a.models]
    names = [Path(m).stem.replace("_int8", "") for m in a.models]
    sink = None
    if a.display == "sink":  # g2d upscales 4:3 to 1440x1080 on the 1080p screen; fullscreen=true hangs the sink
        sink = cv2.VideoWriter("appsrc ! videoconvert ! imxvideoconvert_g2d ! video/x-raw,width=1440,height=1080 ! waylandsink sync=false",
                               cv2.CAP_GSTREAMER, 0, 30, (800, 600), True)
        assert sink.isOpened(), "waylandsink did not open (set XDG_RUNTIME_DIR / WAYLAND_DISPLAY)"
    if a.display == "window":
        cv2.namedWindow("demo", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("demo", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap, k, blur_on, frames, saved = open_video(a.video), 0, True, 0, a.save_frame is None
    times, acc, last_log, switched = deque(maxlen=30), {}, time.perf_counter(), time.perf_counter()
    fps = 0.0
    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            if not a.loop:
                break
            cap.release()
            cap = open_video(a.video)
            continue
        read = 1000 * (time.perf_counter() - t0)
        out, st, n = process(frame, dets[k], thrs[k], blur_on)
        st["read"] = read
        draw(out, overlay_text(names[k], fps, st, n))
        t1 = time.perf_counter()
        if not saved and frames >= 30 and n:  # first frame with a face after 30
            cv2.imwrite(a.save_frame, out)
            saved = True
        key = -1
        if sink:
            sink.write(out)
        if a.display == "window":
            cv2.imshow("demo", out)
            key = cv2.waitKey(1) & 0xFF
        st["show"] = 1000 * (time.perf_counter() - t1)
        now = time.perf_counter()
        times.append(now - t0)
        fps, frames = len(times) / sum(times), frames + 1
        for s, v in st.items():
            acc.setdefault(s, []).append(v)
        if key == ord("q"):
            break
        if key == ord("b"):
            blur_on = not blur_on
        if key == ord("m") or (a.switch_every and now - switched > a.switch_every):
            k, switched, acc = (k + 1) % len(dets), now, {}
        if now - last_log >= 2:
            print(f"{names[k]} {fps:.1f} FPS " + " ".join(f"{s} {np.mean(v):.1f}" for s, v in acc.items()) + f" ms faces {n}", flush=True)
            last_log, acc = now, {}
    if sink:
        sink.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
