"""Face detectors under test. Each detector maps a BGR frame to an (n, 5) float32 array: x, y, w, h, score."""
import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODELS = Path(__file__).resolve().parents[2] / "models"
YUNET_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/{}"
FLOOR = 0.05  # keep low-score detections so operating thresholds can be swept without re-detecting


def ort_providers():
    import onnxruntime as ort

    ort.set_default_logger_severity(3)  # silence per-batch shape warnings from InsightFace's fixed-batch ONNX graphs
    out = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        # cuDNN 9.x heuristic search finds no conv engine on Colab's T4 and ORT silently drops to CPU
        out.insert(0, ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}))
    return out


def used_providers(net):
    """Providers the ONNX session really ended up on (ORT falls back to CPU at run time on CUDA errors)."""
    s = net if hasattr(net, "get_providers") else getattr(net, "session", None) or getattr(net, "sess", None)
    return s.get_providers() if s is not None else None


def xyxy_to_xywh(a):
    a = np.array(a, dtype=np.float32).reshape(-1, 5)
    a[:, 2:4] -= a[:, 0:2]
    return a


class YuNet:
    def __init__(self, int8=False):
        name = f"face_detection_yunet_2023mar{'_int8' if int8 else ''}.onnx"
        path = MODELS / name
        if not path.exists():
            MODELS.mkdir(exist_ok=True)
            urllib.request.urlretrieve(YUNET_URL.format(name), path)
        self.net = cv2.FaceDetectorYN.create(str(path), "", (320, 320), FLOOR, 0.3, 5000)

    def __call__(self, bgr):
        self.net.setInputSize((bgr.shape[1], bgr.shape[0]))
        faces = self.net.detect(bgr)[1]
        return np.empty((0, 5), np.float32) if faces is None else faces[:, [0, 1, 2, 3, 14]]


class SCRFD:
    def __init__(self, pack, file):
        from insightface.model_zoo import get_model
        from insightface.utils.storage import ensure_available

        path = Path(ensure_available("models", pack, root="~/.insightface")) / file
        self.net = get_model(str(path), providers=ort_providers())
        self.net.prepare(ctx_id=0, input_size=(640, 640), det_thresh=FLOOR)

    def __call__(self, bgr):
        return xyxy_to_xywh(self.net.detect(bgr)[0])


class Deface:
    def __init__(self):
        from deface.centerface import CenterFace

        self.net = CenterFace(in_shape=None, backend="auto")
        if getattr(self.net, "sess", None) is not None:  # deface takes no provider options; apply ours
            p = ort_providers()
            self.net.sess.set_providers([x[0] if isinstance(x, tuple) else x for x in p],
                                        [x[1] if isinstance(x, tuple) else {} for x in p])

    def __call__(self, bgr):
        return xyxy_to_xywh(self.net(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), threshold=FLOOR)[0])


class YoloFace:
    """YOLOv8n-face (akanametov/yolo-face, WIDER FACE, AGPL-3.0). ONNX export with NMS inside:
    output (1, 300, 21) = x1, y1, x2, y2, score, class, 5 landmarks x (x, y, conf), in letterboxed 640x640 pixels."""

    URL = "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.onnx"
    SIZE = 640

    def __init__(self):
        import onnxruntime as ort

        path = MODELS / "yolov8n-face.onnx"
        if not path.exists():
            MODELS.mkdir(exist_ok=True)
            urllib.request.urlretrieve(self.URL, path)
        self.net = ort.InferenceSession(str(path), providers=ort_providers())

    def __call__(self, bgr):
        h, w = bgr.shape[:2]
        r = min(self.SIZE / w, self.SIZE / h)
        nw, nh = round(w * r), round(h * r)
        px, py = (self.SIZE - nw) // 2, (self.SIZE - nh) // 2  # ultralytics centres the letterbox
        canvas = np.full((self.SIZE, self.SIZE, 3), 114, np.uint8)
        canvas[py : py + nh, px : px + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255
        out = self.net.run(None, {"images": x})[0][0]
        out = out[out[:, 4] >= FLOOR]
        boxes = (out[:, :4] - [px, py, px, py]) / r
        return xyxy_to_xywh(np.concatenate([boxes, out[:, 4:5]], 1))


class EgoBlur:
    def __init__(self, path):
        import torch

        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = torch.jit.load(str(path), map_location=self.dev).eval()

    def __call__(self, bgr):
        # EgoBlur's demo feeds the raw BGR uint8 frame as a CHW tensor
        t = self.torch.from_numpy(np.ascontiguousarray(bgr.transpose(2, 0, 1))).to(self.dev)
        with self.torch.no_grad():
            boxes, _, scores, _ = self.net(t)
        a = xyxy_to_xywh(np.concatenate([boxes.cpu().numpy(), scores.cpu().numpy()[:, None]], 1))
        return a[a[:, 4] >= FLOOR]


# name -> (factory(egoblur_path), default operating threshold = each tool's own default)
SYSTEMS = {
    "yunet": (lambda _: YuNet(), 0.9),
    "yunet_int8": (lambda _: YuNet(int8=True), 0.9),
    "scrfd_500m": (lambda _: SCRFD("buffalo_s", "det_500m.onnx"), 0.5),
    "scrfd_10g": (lambda _: SCRFD("buffalo_l", "det_10g.onnx"), 0.5),
    "yolov8n_face": (lambda _: YoloFace(), 0.5),
    "deface": (lambda _: Deface(), 0.2),
    "egoblur": (lambda p: EgoBlur(p), 0.9),
}
