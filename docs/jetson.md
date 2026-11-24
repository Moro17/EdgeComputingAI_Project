# Jetson Nano: the same pipeline on a GPU edge board (2026-11-24)

Second edge platform, to compare with the i.MX 8M Plus NPU (docs/edge-board.md). Raw logs: `experiments/jetson/`.

## Board

NVIDIA Jetson Nano Developer Kit (4 GB, Maxwell GPU 128 CUDA cores, 4x Cortex-A57), JetPack 4.6.1 (L4T R32.7.1),
CUDA 10.2, TensorRT 8.2.1, Python 3.6.9, OpenCV 4.1.1 (GStreamer, no CUDA), numpy 1.13. Power mode MAXN, clocks
locked with `sudo jetson_clocks` (GPU 921.6 MHz). Extra package: `pycuda 2021.1` (pip, built on the board).
Access: `ssh jetson` (key login, alias in `~/.ssh/config`, board at 192.168.0.129 on the home network).

## Detector alone (trtexec, GPU compute time, 480x640 input)

| Model | FP32 | FP16 |
|---|---|---|
| YuNet stock (`yunet_tfl_480x640.onnx`) | 20.2 ms | 20.2 ms |
| YuNet fine-tuned (`yunet_ft_480x640.onnx`) | 24.1 ms | 24.0 ms |

- FP16 brings nothing: YuNet is depthwise convolutions and small tensors, memory-bound on this GPU. INT8: see below.
- The fine-tuned ONNX is 4 ms slower than the stock one although the architecture is the same: it was exported from
  the training code, the stock one comes back from the board's TFLite (different but equivalent graph). Not chased.
- For comparison, i.MX 8M Plus NPU: 11.7 ms INT8 (benchmark_model), 2x faster than the Nano GPU.

### Quantization on the Nano

- INT8 on the GPU, implicit (`trtexec --int8`, calibration-free timing): 20.2 ms, the same as FP32. TensorRT warns
  "Int8 support requested on hardware without native Int8 support" and falls back to FP32/FP16 kernels
  (`experiments/jetson/trt_stock_int8_implicit.log`).
- INT8 on the GPU, explicit (the QDQ ONNX the i.MX INT8 model was made from, `yunet_ft_480x640_qboard_480x640.onnx`):
  TensorRT 8.2 refuses to parse it ("only activation types allowed as input to this layer" on a constant
  DequantizeLinear; `trt_ft_qdq_int8.log`). Newer TensorRT would take it, but JetPack 4.6 is the last for the Nano,
  and the GPU has no INT8 path anyway.
- CPU, the same .tflite files as the i.MX (`src/jetson_tflite_bench.py`, tflite_runtime 2.5, 4 threads, p50 of 40
  runs, `tfl_cpu.json`): stock FP32 64.3 ms, INT8 44.8 ms; fine-tuned FP32 79.3 ms, INT8 53.6 ms. All over 40 ms.

## Final inference table: i.MX 8M Plus vs Jetson Nano (YuNet, 480x640 input, detector alone, p50)

| Board / engine | Precision | Stock YuNet | Fine-tuned YuNet | Fits 40 ms with the pipeline? |
|---|---|---|---|---|
| i.MX 8M Plus NPU (VX delegate) | INT8 | **11.7 ms** (9.1 uint8 input) | 13.0 ms | yes: 3 cameras at 25 FPS |
| i.MX 8M Plus NPU | int16x8 | rejected by the delegate, 2.2 s on CPU kernels | - | no |
| i.MX 8M Plus NPU | FP32 | 918 ms | - | no |
| i.MX 8M Plus CPU, 4x A53 (XNNPACK) | INT8 / FP32 | 72.7 / 61.3 ms | - / 54-69 ms p95 in the pipeline | no |
| Jetson Nano GPU (TensorRT 8.2) | FP32 | 20.2 ms | 24.1 ms | yes: 1 camera, 30 FPS serial |
| Jetson Nano GPU | FP16 | 20.2 ms | 24.0 ms | yes: 1 camera (deployed on the Nano) |
| Jetson Nano GPU | INT8 | 20.2 ms (FP32 fallback) / QDQ not parsed | - | no gain |
| Jetson Nano CPU, 4x A57 (tflite_runtime) | INT8 / FP32 | 44.8 / 64.3 ms | 53.6 / 79.3 ms | no |

i.MX numbers: `benchmark_model` (docs/edge-board.md section 4, docs/next-runs.md quantization study); Nano CPU numbers
come from the Python interpreter, not `benchmark_model`, and tflite_runtime 2.5 has no XNNPACK by default, so they are
not strictly the same harness as the i.MX CPU row.

Whole pipeline (decode + detect + tracker + dark rule + look-back 3 + fast_blur), fine-tuned model:

| Board | Detector | One camera | Max real-time cameras at 25 FPS |
|---|---|---|---|
| i.MX 8M Plus | INT8 NPU, pipelined threads | 89.5 FPS | 3 (4th saturates the NPU, 22 FPS each) |
| Jetson Nano | FP16 GPU, serial loop | 30.0 FPS (34.4 ms per frame) | 1 (pipelined bound ~40 FPS) |

Reading: quantization is what makes the i.MX work (INT8 NPU is 5x faster than its CPU and 2x faster than the Nano
GPU). On the Nano it buys nothing: Maxwell has no INT8 kernels and YuNet is memory-bound, so FP32 = FP16 = INT8.
The Nano still meets the 40 ms budget for one camera without quantizing, i.e. with no INT8 accuracy risk; the
i.MX is the board that scales to three cameras.

## Whole pipeline, one stream (`src/jetson_pipeline.py`, single thread, stages in a row)

ChokePoint test P2E_S1_C2.1 (`chokepoint.mp4`, src/make_demo_videos.py), 40 s, p50 per stage in ms:

| Config | FPS | read (NVDEC) | pre | GPU | post | track | blur | frame |
|---|---|---|---|---|---|---|---|---|
| fine-tuned FP16, lite + dark rule + back3 (final) | 30.0 | 0.5 | 2.2 | 25.0 | 1.4 | 1.1 | 4.1 | 34.4 |
| fine-tuned FP32, same | 29.8 | 0.5 | 2.3 | 25.1 | 1.4 | 1.0 | 4.2 | 34.7 |
| stock FP16, same | 33.9 | 0.5 | 2.2 | 21.2 | 1.3 | 1.0 | 4.2 | 30.6 |
| fine-tuned FP16, per-frame (no tracker, no look-back) | 33.0 | 0.5 | 2.2 | 25.0 | 1.4 | 0.0 | 1.6 | 30.8 |

- The final configuration runs in real time on one camera (30 FPS, 34 ms per frame, under the 40 ms budget); the
  tracker, dark rule and look-back cost 3.6 ms on the Cortex-A57.
- Serial loop: the GPU waits for the CPU stages. A pipelined version (as on the i.MX, where one stream reaches
  89.5 FPS) would bound the Nano at about 1000 / 25 = 40 FPS for the fine-tuned model: one camera, not three.
- Same detections as on the desktop up to FP16 rounding (TensorRT runs the same ONNX graph); privacy numbers are not
  re-scored on the Nano.

## Laptop vs i.MX 8M Plus vs Jetson Nano: resources (2026-11-24)

Final configuration (fine-tuned YuNet 480x640, ByteTrack-lite + dark rule + look-back 3 + fast_blur), one camera,
ChokePoint clip. Figure: `experiments/platforms/inference_3_platforms.png` (src/analysis/platform_plots.py).

| | Laptop (Lenovo 82JU) | i.MX 8M Plus EVK | Jetson Nano 4 GB |
|---|---|---|---|
| Compute used | CPU Ryzen 5 5600H, 6 cores / 12 threads (RTX 3060 not used) | NPU 2.3 TOPS + 4x Cortex-A53 1.6 GHz | GPU 128-core Maxwell + 4x Cortex-A57 1.48 GHz |
| Precision | FP32 (INT8: slower on this CPU) | INT8 | FP16 |
| Detector p50 | 7.1 ms (INT8 10.0) | 10.2 ms | 25.0 ms |
| Pipeline, max FPS | 85.5 (INT8 66.8), serial loop | 89.5, pipelined threads | 30.0, serial loop |
| Cameras at 25 FPS | not measured (> 3 by FPS) | 3 (p50 latency 146 ms) | 1 |
| CPU load | 7.2 of 12 threads busy at max | 0.4-0.7 of 4 cores at 25 FPS, 1.6 at max | 0.8 of 4 cores at 30 FPS |
| Accelerator load | none | NPU busy 26% at 25 FPS, 91% at max | GPU 66% at 30 FPS |
| Memory | 169 MB process | 126 MB process (+90 MB system, 6 GB board) | +0.7 GB system (CUDA + TensorRT, of 4 GB) |
| Power | not measured (laptop TDP 45 W CPU; dGPU idles at 10 W) | not measured (no current sensor on the EVK) | 6.8 W whole board (3.1 W idle), tegrastats POM_5V_IN |
| Max temperature | not measured | 37 C (SoC) | 47 C (CPU) |

Sources: `experiments/pc/pipeline_yqft_{fp32,board}.json` (src/pc_pipeline_bench.py, 30 s), `experiments/edge/final_*.json`,
`experiments/edge/boost/res_cams1` run and `experiments/jetson/tegrastats_run.txt` (30 s, `sudo tegrastats`).
Reading: the i.MX matches the laptop's throughput with under 2 of its 4 small cores and no fan, because the NPU does
the inference; the Nano needs its GPU at 66% for one camera. The laptop is the reference, not an edge target.

## Reproduce

```bash
scp models/yunet_ft_480x640.onnx src/jetson_pipeline.py src/systems/bytelite.py src/systems/anonymize.py \
    dist/demo_videos/chokepoint.mp4 jetson:edge/
ssh jetson 'cd edge && /usr/src/tensorrt/bin/trtexec --onnx=yunet_ft_480x640.onnx --fp16 \
    --saveEngine=yunet_ft_480x640_fp16.engine --workspace=1024 && sudo jetson_clocks && \
    python3 jetson_pipeline.py --video chokepoint.mp4 --loop --seconds 40 --engine yunet_ft_480x640_fp16.engine \
    --risk d60_t0.1 --back 3'
```
