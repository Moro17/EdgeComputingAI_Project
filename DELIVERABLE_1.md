# Deliverable 1: face detection and anonymization on the i.MX 8M Plus

This report answers the points of the first practical demonstration. The task: blur every face in a video, live, on
the NXP i.MX 8M Plus, at 25 FPS or more. 25 FPS gives 40 ms per frame for the whole loop. The model is a YuNet face
detector, fine-tuned on ChokePoint and quantized to INT8 for the NPU. Dataset, method and the privacy metric:
[README.md](README.md).

All board numbers were measured on the board itself, with [scripts/measure_board.sh](scripts/measure_board.sh) (raw
log: [experiments/d1/board_measure.log](experiments/d1/board_measure.log)) and `src/board_bench.py`. Accuracy is on the
test split: 12 ChokePoint sequences, 8,957 faces, 29 people.

## 1. Inference on the device

The demo (`src/demo.py`, started with `scripts/run_demo.sh`; steps in [docs/runbook.md](docs/runbook.md)) runs this
loop on the board:

1. H.264 file, decoded in hardware (VPU), converted to BGR by the 2D GPU (g2d).
2. YuNet INT8 on the NPU.
3. fast_blur on each face box.
4. Overlay (model name, FPS, faces), shown on HDMI through Wayland.

| Model on the board (INT8, NPU) | Demo FPS on HDMI | Detector p50 | % people exposed (test) |
|---|---|---|---|
| YuNet stock, 480x640 | 28.5 | 20.1 ms | 0.0 |
| **YuNet fine-tuned, 480x640** | **27.9** | 21.4 ms | **0.0** |
| YOLOv8n-face 320 (reference) | 22.1 | 32.9 ms | 0.0 |

Exposure is at the thresholds chosen on val: 0.6 for stock, 0.7 for fine-tuned.

## 2. Hardware constraints

| Resource | Board | Our use |
|---|---|---|
| CPU | 4x Arm Cortex-A53, 1.8 GHz | demo process: 115% of one core on the NPU path, 275-290% on the CPU-only path (of 400%) |
| NPU | VeriSilicon VIP8000, 2.3 TOPS INT8 (NXP spec) | 11.0 ms per frame (stock), 11.9 ms (fine-tuned) |
| RAM | 5.5 GiB visible to Linux | demo process 131 MB resident; `benchmark_model` peak 208-221 MB with the NPU driver, 21 MB on CPU |
| Storage (flash) | 9.6 GB root file system, 2.8 GB free | model files 93 KB (stock) and 118 KB (fine-tuned) INT8; YOLOv8n-face 3.4 MB |
| Latency | 40 ms per frame at 25 FPS | fine-tuned loop: 34 ms per frame (29.3 FPS headless) |
| Energy | not measured | see below |

**Energy:** the board exposes no power sensor we could read, and we had no power meter, so energy was not measured.
Two proxies:
- **CPU load.** The NPU path uses 2.4 times less CPU than the CPU-only path (115% against 275% of one core) and runs
  2.5 times faster. Per frame that is about 6 times less CPU time.
- **SoC temperature.** It rose from 30-32 °C at boot to 44-46 °C after about 10 minutes of benchmarks. This was not a
  controlled measurement.

## 3. Tech stack and inference configuration

| Layer | What we use |
|---|---|
| OS | NXP i.MX Release Distro 6.6 (nanbield), Linux 6.6.3, Weston (Wayland) |
| Runtime | Python 3.11.5, `tflite_runtime` 2.14.0, OpenCV 4.8.1 (with GStreamer), NumPy 1.26.0 |
| NPU access | TFLite external delegate `/usr/lib/libvx_delegate.so` (VX delegate); compiled NPU graph cached on disk (`VIV_VX_ENABLE_CACHE_GRAPH_BINARY=1`) |
| Model format | TFLite, full INT8 (weights and activations), per-tensor, float32 input and outputs |
| Model input | 480x640 (H x W), BGR 0-255, frame at the top left, zero padding |
| Conversion | PyTorch -> ONNX (trainer's exporter) -> static shapes -> onnx2tf, INT8 calibrated on ChokePoint S5 frames (`src/export_yunet_tflite.py`) |
| Video in | `filesrc ! qtdemux ! h264parse ! vpudec ! imxvideoconvert_g2d ! BGRx ! videoconvert ! BGR ! appsink sync=false` |
| Video out | `appsrc ! videoconvert ! imxvideoconvert_g2d ! 1440x1080 ! waylandsink sync=false` |
| Post-processing | YuNet decoding as OpenCV `FaceDetectorYN`, NMS 0.3, score threshold 0.7 (fine-tuned) |
| Anonymization | `fast_blur`: strong Gaussian computed on the face shrunk to 16 px, mask scale 1.3 |
| Training | libfacedetection.train, Colab T4, 20 epochs, WIDER FACE + ChokePoint train |

## 4. NPU against CPU

Model only (`benchmark_model`, average of 200 runs on the NPU, 100 on the CPU):

| Model | NPU | CPU, 1 thread | CPU, 4 threads | NPU speed-up vs best CPU |
|---|---|---|---|---|
| YuNet stock INT8 | **11.0 ms** | 147.9 ms | 61.2 ms | 5.6x |
| YuNet fine-tuned INT8 | **11.9 ms** | 191.1 ms | 73.5 ms | 6.2x |
| YuNet stock FP32 | 729.4 ms | 149.6 ms | **45.5 ms** | NPU 16x slower |
| YuNet fine-tuned FP32 | 918.4 ms | 212.3 ms | **61.8 ms** | NPU 15x slower |
| YuNet stock FP16 | 1050.8 ms | 150.9 ms | 46.2 ms | NPU 23x slower |
| YOLOv8n-face 320 INT8 | 19.8 ms | 287.0 ms | 119.1 ms | 6.0x |

The whole demo loop, headless (`--display none`, 25 s per run):

| Model | NPU path | CPU-only path (`--cpu`, 4 threads) | Speed-up |
|---|---|---|---|
| YuNet stock INT8 | 30.6 FPS (detector 13.2 ms) | 12.3 FPS (detector 63.2 ms) | 2.5x |
| YuNet fine-tuned INT8 | 29.3 FPS (detector 14.2 ms) | 10.7 FPS (detector 74.7 ms) | 2.7x |

- **The NPU only helps an INT8 model.** FP32 and FP16 models run 15-23 times slower on the NPU than on the CPU, so
  quantization is what makes the NPU usable at all.
- **On the CPU, FP32 is faster than INT8:** 45.5 ms against 61.2 ms with 4 threads. The CPU kernels for our
  per-tensor INT8 model are slower than the float ones, so INT8 only pays off on the NPU.
- **Only the NPU path reaches 25 FPS.** On the CPU, the detector alone takes 63-75 ms, above the whole 40 ms
  budget.

## 5. Optimizations: before and after

| Optimization | Before | After | Effect on the results |
|---|---|---|---|
| **INT8 quantization** (static, per-tensor, calibrated on S5) | FP32: 239 KB; 729 ms on the NPU, 45.5 ms on the CPU | INT8: 93 KB; 11.0 ms on the NPU | none on accuracy: table below |
| **NPU instead of CPU** | 61.2 ms (INT8, 4 threads) | 11.0 ms | none: same model file |
| **Input size** | 608x800: detector 32.2 ms | 480x640: 27.0 ms | recall 0.955 to 0.944; exposure unchanged (86.2% at 0.9) |
| **Hardware video decode** (g2d colour conversion) | 22.6 ms (CPU `videoconvert`) | 7.6 ms | none |
| **fast_blur** | strong blur: 104 ms for 2 faces | 2.9 ms | the judge recognises 0.3% of blurred faces; exposure identical for all 10 systems tested |
| **Detector post-processing** (reused input buffer, decode only anchors above the score floor) | detector 27.0 ms (post 4.2 ms) | 20.1 ms (post 2.0 ms) | detections bit-identical. Part of the gain was measured on a freshly booted board. |
| **Display** | `cv2.imshow`: 25 ms per frame | waylandsink: 0.7 ms | none |
| **Operating threshold** (chosen on val) | 0.9 (OpenCV default): 86.2% of people exposed | stock 0.6 / fine-tuned 0.7: 0% | the largest accuracy change. It costs no compute. |
| **Fine-tuning** (at the same threshold 0.9) | stock: recall 0.944, 86.2% exposed | fine-tuned: recall 0.992, 48.3% exposed | +1.3 ms on the NPU (the trainer's network has 59 convolutions against 53) |

Not used: pruning, knowledge distillation. Apart from what onnx2tf does when it builds the graph, no manual
operation fusion.

**Quantization, accuracy before and after** (test split; FP32 TFLite of the same graph run on a Mac CPU, INT8 on the
board NPU; `experiments/d1/results_quant_*.csv`):

| Model | Threshold | Recall FP32 / INT8 | % people exposed FP32 / INT8 | Longest leak (frames) FP32 / INT8 |
|---|---|---|---|---|
| Stock | 0.6 | 0.9999 / 0.9999 | 0.0 / 0.0 | 0 / 0 |
| Fine-tuned | 0.7 | 1.000 / 1.000 | 0.0 / 0.0 | 0 / 0 |
| Stock | 0.9 | 0.941 / 0.944 | 86.2 / 86.2 | 32 / 32 |
| Fine-tuned | 0.9 | 0.990 / 0.992 | 55.2 / 48.3 | 12 / 9 |

INT8 loses nothing at the operating thresholds. At 0.9 it is slightly better than FP32, which is within noise: one
person is 3.4%.

## 6. The bottleneck

Where the time goes, per frame, fine-tuned model on the NPU path (stage means from the demo log):

| Stage | ms | Runs on |
|---|---|---|
| NPU inference (`benchmark_model`) | 11.9 | NPU |
| Copying tensors in and out from Python | ~2.3 | CPU |
| Pre-processing: resize 800x600 to 640x480, convert to float32 | 6.7 | CPU |
| Post-processing: decode and NMS | 1.8 | CPU |
| Waiting for the decoded frame | 1.0 | VPU + g2d |
| Blur | 0-1 | CPU |
| Display on HDMI | 0.7 | g2d + Wayland |
| Not attributed (overlay drawing, Python loop) | ~10 | CPU, not profiled |
| **Frame time** | **~34** (29.3 FPS headless, 27.9 on HDMI) | |

- **The NPU itself is the largest single cost:** 12 ms, about a third of the frame.
- **The CPU work around the NPU is the next one:** pre-processing, tensor copies and the unprofiled loop overhead (~10 ms)
  add up to more than the NPU's share. Because the model takes a float32 input, the CPU first converts each
  640x480x3 frame to float.
- **What would remove it, not done yet:**
  - A model with an 8-bit input tensor, which skips the float conversion and copies 4 times fewer bytes.
  - Resizing to 640x480 in the g2d step of the decode pipeline instead of on the CPU.
  - Profiling the remaining ~10 ms.

  These would leave room under 40 ms for the tracking and missed-face recovery we add next.

## Reproduce

```bash
# on the Mac: INT8 plus FP32/FP16 copies
.venv-export/bin/python src/export_yunet_tflite.py yunet_ft.onnx --size 480 640 --out models/yunet_ft_480x640_int8.tflite --keep-float
# copy the models, src/board_bench.py, src/demo.py and scripts/measure_board.sh to /root on the board, then:
ssh imx 'cd /root && ./measure_board.sh > measure.log 2>&1'
```

Accuracy runs and scoring: [README.md](README.md), section Reproduce.
