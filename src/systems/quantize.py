"""Static INT8 quantization (onnxruntime QDQ, per-tensor: closest to what the i.MX 8M Plus NPU runs).

Calibration frames come from ChokePoint S5 (crowd sequences): on disk, never scored, so no test-set leakage.
Each detector's own preprocessing is captured by recording the inputs its FP32 session receives.
"""
import cv2
import numpy as np

from loaders.chokepoint import DATA

CALIB_SEQS = ["P2E_S5_C1.1", "P2E_S5_C2.1", "P2E_S5_C3.1", "P2L_S5_C1.1", "P2L_S5_C2.1", "P2L_S5_C3.1"]
CALIB_FRAMES = 200


def calibration_frames(n=CALIB_FRAMES):
    paths = sorted(p for s in CALIB_SEQS for p in (DATA / "frames" / s).glob("*.jpg"))
    assert paths, f"no calibration frames: download the S5 sequences into {DATA / 'frames'}"
    return [cv2.imread(str(paths[i])) for i in np.linspace(0, len(paths) - 1, n).astype(int)]


def calibration_feeds(detector, frames=None):
    """Run the FP32 detector and record every input dict its onnxruntime session receives."""
    import onnxruntime as ort

    feeds, run = [], ort.InferenceSession.run

    def record(self, output_names, input_feed, *a, **k):
        feeds.append({name: np.array(x) for name, x in input_feed.items()})
        return run(self, output_names, input_feed, *a, **k)

    ort.InferenceSession.run = record
    try:
        for img in frames if frames is not None else calibration_frames():
            detector(img)
    finally:
        ort.InferenceSession.run = run
    return feeds


def to_int8(src, feeds, dst):
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(feeds)

        def get_next(self):
            return next(self.it, None)

    pre = dst.with_name(dst.stem + "_pre.onnx")
    quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
    # Conv only: decode heads (YOLOv8 concatenates 0-640 px boxes with 0-1 scores) would round every score to 0
    # in one per-tensor INT8 range. NPU deployments also leave decode/NMS in float on the CPU.
    quantize_static(str(pre), str(dst), Reader(), quant_format=QuantFormat.QDQ, per_channel=False,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, op_types_to_quantize=["Conv"])
    pre.unlink()
