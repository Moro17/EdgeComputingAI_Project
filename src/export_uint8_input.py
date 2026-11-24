"""Give an INT8 YuNet .tflite a uint8 input tensor instead of float32 (lossless when the input scale is 1).

onnx2tf's INT8 YuNet takes float32 BGR 0-255 and starts with QUANTIZE float32 -> int8 (scale 1.0, zero point -128).
On the board that costs a CPU float conversion of every 640x480x3 frame and a 4x larger tensor copy. Here the input
tensor becomes uint8 with scale 1.0, zero point 0, so the same QUANTIZE op requantizes uint8 -> int8 (x - 128):
exactly the same int8 values reach the first convolution, so detections are bit-identical.

Needs the TFLite flatbuffer schema module (schema_py_generated.py, shipped in the ai-edge-litert and tensorflow
wheels) and the flatbuffers package:
  python src/export_uint8_input.py models/yunet_480x640_int8.tflite models/yunet_480x640_u8.tflite \\
      [--schema-dir DIR_CONTAINING_schema_py_generated.py]
"""
import argparse
import importlib
import sys
from pathlib import Path


def load_schema(schema_dir):
    if schema_dir:
        sys.path.insert(0, str(schema_dir))
        return importlib.import_module("schema_py_generated")
    for name in ("ai_edge_litert.schema_py_generated", "tensorflow.lite.python.schema_py_generated"):
        try:
            return importlib.import_module(name)
        except ImportError:
            pass
    raise SystemExit("no TFLite schema module: pass --schema-dir")


def convert(src, dst, s):
    import flatbuffers

    model = s.ModelT.InitFromObj(s.Model.GetRootAsModel(bytearray(Path(src).read_bytes()), 0))
    g = model.subgraphs[0]
    assert len(g.inputs) == 1, "one input expected"
    t = g.tensors[g.inputs[0]]
    assert t.type == s.TensorType.FLOAT32, "input is not float32"
    first = next(op for op in g.operators if g.inputs[0] in op.inputs)
    code = model.operatorCodes[first.opcodeIndex]
    assert max(code.builtinCode, code.deprecatedBuiltinCode) == s.BuiltinOperator.QUANTIZE, "input must feed QUANTIZE"
    q = g.tensors[first.outputs[0]].quantization
    assert list(q.scale) == [1.0], f"input scale {list(q.scale)} != 1: uint8 input would not be lossless"
    t.type = s.TensorType.UINT8
    t.quantization = s.QuantizationParametersT()
    t.quantization.scale, t.quantization.zeroPoint = [1.0], [0]
    code.version = max(code.version, 2)  # QUANTIZE v2: int8/uint8 requantization
    b = flatbuffers.Builder(1024)
    b.Finish(model.Pack(b), file_identifier=b"TFL3")
    Path(dst).write_bytes(b.Output())
    print(f"{src} -> {dst}: input uint8 (scale 1, zp 0) -> QUANTIZE -> int8 (scale 1, zp {list(q.zeroPoint)[0]})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--schema-dir", type=Path, default=None)
    a = p.parse_args()
    convert(a.src, a.dst, load_schema(a.schema_dir))


if __name__ == "__main__":
    main()
