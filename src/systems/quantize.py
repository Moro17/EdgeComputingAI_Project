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


def to_int8(src, feeds, dst, per_channel=False, ops=("Conv",), method="minmax", act="u8"):
    """Static QDQ quantization. Defaults are the CPU precision study's; RECIPES holds the edge study's variants."""
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(feeds)

        def get_next(self):
            return next(self.it, None)

    pre = dst.with_name(dst.stem + "_pre.onnx")
    if per_channel:  # per-axis DequantizeLinear needs opset >= 13 (YuNet ships opset 11)
        import onnx
        from onnx import version_converter

        m = onnx.load(str(src))
        if max(o.version for o in m.opset_import if o.domain in ("", "ai.onnx")) < 13:
            src = dst.with_name(dst.stem + "_op13.onnx")
            onnx.save(version_converter.convert_version(m, 13), str(src))
    quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
    if per_channel and src.name.endswith("_op13.onnx"):
        src.unlink()
    # Conv only by default: decode heads (YOLOv8 concatenates 0-640 px boxes with 0-1 scores) would round every score
    # to 0 in one per-tensor INT8 range. NPU deployments also leave decode/NMS in float on the CPU.
    acts = {"u8": QuantType.QUInt8, "s8": QuantType.QInt8, "s16": QuantType.QInt16}
    methods = {"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy,
               "percentile": CalibrationMethod.Percentile}
    quantize_static(str(pre), str(dst), Reader(), quant_format=QuantFormat.QDQ, per_channel=per_channel,
                    activation_type=acts[act], weight_type=QuantType.QInt8,
                    op_types_to_quantize=list(ops) if ops else None, calibrate_method=methods[method],
                    extra_options={"CalibPercentile": 99.999} if method == "percentile" else {})
    pre.unlink()


# Edge quantization study (YuNet, desktop simulation of the board's INT8 arithmetic with onnxruntime QDQ).
# Every op quantized, as onnx2tf's full-integer TFLite; int8 activations as TFLite. "board" mirrors what runs on the
# NPU (onnx2tf -oiqt -qt per-tensor: min-max, 100 clean S5 frames); the others change one thing each.
# calib: list of (degradation or None, share of the frames); degraded S5 frames use dark:1 and dark:3 only, so the
# dark:2 test condition is never seen in calibration.
RECIPES = {
    "board": dict(per_channel=False, method="minmax", act="s8", calib=[(None, 1.0)]),
    "pc": dict(per_channel=True, method="minmax", act="s8", calib=[(None, 1.0)]),
    "pct": dict(per_channel=False, method="percentile", act="s8", calib=[(None, 1.0)]),
    "ent": dict(per_channel=False, method="entropy", act="s8", calib=[(None, 1.0)]),
    "dark": dict(per_channel=False, method="minmax", act="s8", calib=[(None, 1 / 3), ("dark:1", 1 / 3), ("dark:3", 1 / 3)]),
    "a16": dict(per_channel=False, method="minmax", act="s16", calib=[(None, 1.0)]),
}


def recipe_frames(calib, n=100):
    """n calibration frames, the i-th one degraded by the share of the recipe it falls in (seed = i)."""
    from degrade import degrade

    frames, out, k = calibration_frames(n), [], 0
    for deg, share in calib:
        m = round(share * n) if (deg, share) != calib[-1] else n - k
        for i in range(k, k + m):
            f = frames[i]
            out.append(f if deg is None else degrade.apply(f, *degrade.parse(deg), seed=i))
        k += m
    return out
