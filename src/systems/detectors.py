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


def to_fp16(src):
    """FP16 copy of an ONNX model (float32 inputs/outputs kept) in models/."""
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16  # onnxconverter-common's breaks YOLOv8

    dst = MODELS / f"{Path(src).stem}_fp16.onnx"
    if not dst.exists():
        m = onnx.load(str(src))
        # a node using one tensor twice (SCRFD's Resize roi=scales) gets two same-named Casts: copy it first
        for i, n in reversed(list(enumerate(m.graph.node))):
            for k, t in enumerate(n.input):
                if t and t in n.input[:k]:
                    n.input[k] = f"{t}_dup{i}_{k}"
                    m.graph.node.insert(i, onnx.helper.make_node("Identity", [t], [n.input[k]]))
        onnx.save(convert_float_to_float16(m, keep_io_types=True), str(dst))
    return dst


def model_for(src, precision, fp32_detector):
    """Model file for a precision. INT8 is built once by static quantization, calibrated on the FP32 detector's inputs."""
    if precision == "fp32":
        return src
    if precision == "fp16":
        return to_fp16(src)
    dst = MODELS / f"{Path(src).stem}_int8.onnx"
    if not dst.exists():
        from systems.quantize import calibration_feeds, to_int8

        to_int8(src, calibration_feeds(fp32_detector()), dst)
    return dst


def dynamic(src):
    """YuNet's ONNX is fixed at 640x640 (OpenCV reshapes it); free H and W so onnxruntime runs it at frame size."""
    import onnx

    dst = MODELS / f"{Path(src).stem}_dyn.onnx"
    if not dst.exists():
        m = onnx.load(str(src))
        d = m.graph.input[0].type.tensor_type.shape.dim
        d[2].dim_param, d[3].dim_param = "h", "w"
        for o in m.graph.output:
            o.type.tensor_type.shape.dim[1].dim_param = "n"
        del m.graph.value_info[:]  # stale 640x640 shape hints
        onnx.save(m, str(dst))
    return dst


class YuNetORT:
    """YuNet on onnxruntime with OpenCV FaceDetectorYN's decoding, so FP32, FP16 and INT8 share one runtime and one decoder.
    Matches cv2.FaceDetectorYN within 1 px (tests/test_detectors.py)."""

    def __init__(self, precision="fp32", path=None):
        import onnxruntime as ort

        YuNet()  # downloads the FP32 model if missing
        src = dynamic(path or MODELS / "face_detection_yunet_2023mar.onnx")  # path=None: stock model
        path = model_for(src, precision, YuNetORT)
        self.net = ort.InferenceSession(str(path), providers=ort_providers())

    def __call__(self, bgr):
        h, w = bgr.shape[:2]
        ph, pw = (h - 1) // 32 * 32 + 32, (w - 1) // 32 * 32 + 32  # pad bottom/right to a multiple of 32
        x = np.zeros((1, 3, ph, pw), np.float32)
        x[0, :, :h, :w] = bgr.transpose(2, 0, 1)
        out = dict(zip([o.name for o in self.net.get_outputs()], self.net.run(None, {"input": x})))
        boxes = []
        for s in (8, 16, 32):
            rows, cols = np.divmod(np.arange(ph // s * (pw // s)), pw // s)
            score = np.sqrt(np.clip(out[f"cls_{s}"][0, :, 0], 0, 1) * np.clip(out[f"obj_{s}"][0, :, 0], 0, 1))
            b = out[f"bbox_{s}"][0]
            cx, cy = (cols + b[:, 0]) * s, (rows + b[:, 1]) * s
            bw, bh = np.exp(b[:, 2]) * s, np.exp(b[:, 3]) * s
            boxes.append(np.stack([cx - bw / 2, cy - bh / 2, bw, bh, score], 1))
        a = np.concatenate(boxes).astype(np.float32)
        a = a[a[:, 4] >= FLOOR]
        keep = cv2.dnn.NMSBoxes(a[:, :4].tolist(), a[:, 4].tolist(), FLOOR, 0.3, top_k=5000)
        return a[np.asarray(keep, int).reshape(-1)]


class SCRFD:
    def __init__(self, pack, file, precision="fp32"):
        from insightface.model_zoo import get_model
        from insightface.utils.storage import ensure_available

        path = Path(ensure_available("models", pack, root="~/.insightface")) / file
        path = model_for(path, precision, lambda: SCRFD(pack, file))
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

    def __init__(self, precision="fp32"):
        import onnxruntime as ort

        path = MODELS / "yolov8n-face.onnx"
        if not path.exists():
            MODELS.mkdir(exist_ok=True)
            urllib.request.urlretrieve(self.URL, path)
        self.net = ort.InferenceSession(str(model_for(path, precision, YoloFace)), providers=ort_providers())

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


class BoardOnly:
    """Detections produced on the i.MX 8M Plus by src/board_bench.py; the Mac only scores the copied dets/ cache."""

    net, dev = None, "imx8mp-npu-vx (board_bench.py)"

    def __call__(self, bgr):
        raise RuntimeError("board-only system: run src/board_bench.py on the board and copy its dets/<name>/ here")


# name -> (factory(egoblur_path), default operating threshold = each tool's own default)
SYSTEMS = {
    "yunet": (lambda _: YuNet(), 0.9),
    "yunet_int8": (lambda _: YuNet(int8=True), 0.9),
    "scrfd_500m": (lambda _: SCRFD("buffalo_s", "det_500m.onnx"), 0.5),
    "scrfd_10g": (lambda _: SCRFD("buffalo_l", "det_10g.onnx"), 0.5),
    "yolov8n_face": (lambda _: YoloFace(), 0.5),
    # precision study (CPU): same model, same runtime (onnxruntime), FP32 vs FP16 vs INT8 (static, per-tensor QDQ)
    "yunet_ort": (lambda _: YuNetORT(), 0.9),
    "yunet_fp16": (lambda _: YuNetORT("fp16"), 0.9),
    "yunet_q8": (lambda _: YuNetORT("int8"), 0.9),
    # shortcut: 0.9 is a placeholder until notebooks/finetune_yunet.ipynb Step 6 picks T on val
    "yunet_ft": (lambda _: YuNetORT(path=MODELS / "yunet_ft.onnx"), 0.9),
    "scrfd_500m_fp16": (lambda _: SCRFD("buffalo_s", "det_500m.onnx", "fp16"), 0.5),
    "scrfd_500m_q8": (lambda _: SCRFD("buffalo_s", "det_500m.onnx", "int8"), 0.5),
    "yolov8n_face_fp16": (lambda _: YoloFace("fp16"), 0.5),
    "yolov8n_face_q8": (lambda _: YoloFace("int8"), 0.5),
    # board NPU, INT8 TFLite at several input sizes; 0.49 = the saturated score bucket (see src/board_bench.py)
    **{f"yolov8n_face_{s}_npu": (lambda _: BoardOnly(), 0.49) for s in (256, 320, 416, 640)},
    **{f"yunet_{s}_npu": (lambda _: BoardOnly(), 0.9) for s in ("608x800", "480x640", "384x512", "320x416")},
    "yunet_ft_480x640_npu": (lambda _: BoardOnly(), 0.7),  # epoch 20; threshold chosen on val (docs/plan.md Phase 5)
    # FP32 TFLite of the same graphs (export --keep-float), run with board_bench.py --cpu: quantization before/after
    "yunet_480x640_fp32": (lambda _: BoardOnly(), 0.6),
    "yunet_ft_480x640_fp32": (lambda _: BoardOnly(), 0.7),
    "deface": (lambda _: Deface(), 0.2),
    "egoblur": (lambda p: EgoBlur(p), 0.9),
}
