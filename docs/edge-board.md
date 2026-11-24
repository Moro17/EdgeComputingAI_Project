# Using the whole i.MX 8M Plus: the edge pipeline

What each hardware unit of the board does in our anonymizer, how far we pushed it, and what did not help.
All numbers are measured on the board (NXP i.MX 8M Plus EVK, Linux 6.6, tflite_runtime 2.14), 30 s per run, with
`scripts/edge_board.sh`. Raw results: [experiments/edge/](../experiments/edge/). Pipeline code: `src/edge_pipeline.py`.

## 1. The units and their jobs

| Unit | What it is (clock measured from `/sys/kernel/debug/gc/clk` and `clk_summary`) | Our use |
|---|---|---|
| VPU (Hantro decoder) | hardware H.264/HEVC decode, 800 MHz | decodes every camera stream (one decoder session per stream) |
| GPU 2D (GC520L, `g2d`) | blitter / colour conversion / scaling, 1 GHz core | colour conversion **and the resize to the detector size**, display scaling to 1440x1080 |
| NPU (VeriSilicon VIP8000, "gpu1" in galcore) | 2.3 TOPS INT8 (NXP spec), running at **1 GHz, its maximum (overdrive)** | YuNet INT8, 9.1 ms per 640x480 frame (`benchmark_model`) |
| CPU (4x Cortex-A53, 1.8 GHz) | general purpose | feeds the NPU, anchor decoding + NMS, privacy policy, blur, Python glue |
| GPU 3D (GC7000UL, "gpu0") | graphics GPU, OpenCL / OpenVX, 1 GHz | measured for inference and rejected (section 4) |
| Not used | Cortex-M7, HiFi4 DSP, ISP + MIPI-CSI cameras | the M7/DSP have no role in vision; the ISP would replace the file decode with live cameras |

```
 camera/H.264 --> VPU decode --> g2d: BGRx + resize to 640x480 --> [queue]
   --> NPU thread: uint8 copy (0.7 ms) + YuNet INT8 on the NPU (~10 ms) --> [queue]
   --> post thread (CPU): anchor decode + NMS (~3.6 ms) --> [queue]
   --> output thread (CPU): per-camera tracker (0.3 ms) + look-back privacy policy, fast_blur --> g2d --> HDMI
```

Each stage works on a different frame at the same time. tflite `invoke` and the OpenCV calls release the Python GIL,
so the threads really run in parallel on the four cores while the NPU computes.

## 2. Optimization steps (fine-tuned YuNet, one stream, as fast as the pipeline goes, no display)

| Step | FPS | Pre-proc. | NPU stage | Latency p50 | CPU busy (of 400%) |
|---|---|---|---|---|---|
| Serial loop as in Deliverable 1: float32 input, resize on the CPU | 39.6 | 7.1 ms | 14.0 ms | 25 ms | 141% |
| + **uint8 input model** (`src/export_uint8_input.py`) | 50.5 | 5.8 ms | 10.1 ms | 19 ms | 148% |
| + **resize in the g2d** step of the decoder | 72.6 | 0.7 ms | 10.0 ms | 14 ms | 88% |
| + **4-stage pipeline** (threads, decode/NMS off the NPU thread) | **88.3** | 0.7 ms | 10.4 ms | 72 ms* | 130% |
| + privacy policy: numpy tracker (0.34 ms) + look-back 3 frames | 87.7 | 0.7 ms | 10.4 ms | 106 ms* | 156% |

\* Unpaced run: the queues fill up, so latency here is queueing. With live cameras (section 3) it is ~134 ms,
of which 120 ms is the deliberate 3-frame look-back.

- **uint8 input.** The INT8 model took float32 BGR 0-255 and its first op quantized it with scale 1.0, zero
  point -128. Changing only the input tensor to uint8 (scale 1, zero point 0) keeps the same int8 values:
  detections are **bit-identical on 100/100 test frames**. The NPU stage drops from 14.4 to 10.2 ms (4x smaller
  tensor copy, no float conversion). Deliverable 1 listed this as the first thing to do.
- **Resize in g2d.** `imxvideoconvert_g2d` scales 800x600 to 640x480 while it converts colour, so the CPU no
  longer resizes (pre-processing 5.9 to 0.7 ms).
- **Pipeline.** The NPU no longer waits for the CPU: NPU duty goes from 73% to 92%. The NPU is now the bottleneck,
  at its maximum clock.

## 3. How many live cameras one board anonymizes

Each source paced at 25 FPS like a real camera, privacy policy on (tracker + 3-frame look-back), fine-tuned model:

| Cameras | FPS per camera | Total FPS | Latency p50 / p95 | NPU duty | CPU busy (of 400%) | SoC temp. |
|---|---|---|---|---|---|---|
| 1 | 25.0 | 25.0 | 134 / 140 ms | 26% | 41% | 39 °C |
| 2 | 25.0 | 50.0 | 143 / 149 ms | 51% | 90% | 40 °C |
| **3** | **25.0** | **74.9** | **145 / 161 ms** | **75%** | 128% | 40 °C |
| 4 | 21.8 (saturated) | 87.4 | 377 ms | 91% | 155% | 41 °C |
| 4, stock YuNet (9.7 ms NPU) | 23.3 (saturated) | 93.1 | 353 ms | 90% | 162% | 41 °C |

**One board protects three 25 FPS cameras in real time**, with the NPU at 75% and two and a half CPU cores free.
A fourth camera saturates the NPU (22 FPS each).

### Live demo: all cameras on HDMI

From the laptop (Windows PowerShell), one command per demo, output on the board's HDMI screen. The command goes
over the USB debug cable (serial console, COM22 by default, `-Port` to change), so it works without any network;
`-Ssh` uses the network instead. Each start stops the running demo, so two NPU processes never overlap. On the
board the same launcher is `demo final`. `--display web` streams MJPEG to a browser instead (port 8080; 3 cameras
still at 25.0 FPS each with the stream open), `--display both` does both.

```powershell
.\scripts\demo.ps1 install    # once, over ssh: copy the pipeline + scripts/demo_board.sh, link /usr/bin/demo
.\scripts\demo.ps1 final      # or cams3, cams4, baseline, stock, models
.\scripts\demo.ps1 stop       # also: status, log
```

A second word picks the video: `chokepoint` (test P2E_S1_C2.1, 24 people), `dark` (the same at dark:2), `mv`
(30 s from each of 4 Music Video clips, letterboxed to 4:3), default the NXP `demo.mp4`; e.g.
`.\scripts\demo.ps1 final dark`. Build them with `src/make_demo_videos.py` (ffmpeg, H.264 for the VPU; kept in
`dist/`, not in git) and copy them with `.\scripts\demo.ps1 install-videos`. `models` (demo.py, serial loop) runs at
~15 FPS on the 960x720 videos because of the HDMI write; `edge_pipeline.py` modes keep 25 FPS.
`baseline` is the per-frame detector without tracker and look-back (the leaks are visible), `models` the
Deliverable 1 demo that cycles stock YuNet / fine-tuned YuNet / YOLOv8n-face. Extra arguments go to
`edge_pipeline.py` (e.g. `--video other.mp4`). By hand:

```bash
ssh imx 'cd /root/edge; XDG_RUNTIME_DIR=/run/user/0 WAYLAND_DISPLAY=wayland-1 nohup python3 edge_pipeline.py \
  --video demo.mp4 --loop --seconds 100000 --streams 3 --source-fps 25 --back 3 \
  --model yunet_ft_480x640_u8.tflite --display sink > mosaic.log 2>&1 < /dev/null &'
ssh imx 'pkill -f "[e]dge_pipeline"'     # stop
```

A 2x2 mosaic, each tile labelled with its camera's FPS and face count. The display runs in its own thread, so a
slow screen never slows the anonymizer: with 3 cameras all three still run at 25.0 FPS (p50 latency 154 ms,
measured before the tracker was added; the tracker costs 0.3 ms per frame).
The screen itself, though, is limited by its mode: on the 4K monitor connected on 2026-11-20 (3840x2160, Weston
composes a 4K frame with the g2d renderer) a frame takes 66.6 ms to show (15 FPS) with the 1440x1080 g2d upscale,
33.3 ms without it. On the 1080p screen of Deliverable 1 the same sink showed 28 FPS. For the presentation, use a
1080p screen (or set `mode=1920x1080` for HDMI-A-1 in `/etc/xdg/weston/weston.ini`; not changed here).

## 4. What we tried and rejected

| Idea | Measurement | Verdict |
|---|---|---|
| Run YuNet on the 3D GPU (OpenCL TFLite delegate) | 190 ms per frame (u8/int8 graph), 212 ms FP32 | 21x slower than the NPU, 3x slower than the CPU: a graphics GPU, not an accelerator for this CNN |
| Run YuNet on the 3D GPU via OpenVX (`USE_GPU_INFERENCE=1`) | 267-546 ms | worse still |
| FP32/FP16 model on the NPU | 918 ms FP32, 1238 ms FP16 | the NPU is INT8-only in practice (as in Deliverable 1) |
| CPU as a second inference engine (FP32 XNNPACK, 2 threads) beside the NPU, 4 cameras | total 69.7 FPS against 87.4 NPU-only; CPU 338%; NPU stage 10.4 to 12.7 ms | **worse**: the A53s are needed to feed the NPU; inference on them starves it |
| Two processes using the NPU at once | both stuck in `gckOS_WaitSignal`, unkillable, board reboot needed | one process, one NPU thread for all cameras |
| CPU governor `performance` instead of `ondemand` | 88.6 vs 88.2 FPS | no effect: NPU-bound (governor restored) |
| Measure power | the only power sensor (USB-C PD, hwmon2) reads 0 | not measurable on this board without a meter; CPU load and temperature reported instead |

Same network on every engine (`benchmark_model`, `experiments/edge/accelerators.txt`):

| Engine | uint8-input INT8 | INT8 (float input) | FP32 |
|---|---|---|---|
| NPU (VX delegate) | **9.1 ms** | 12.0 ms | 918 ms |
| CPU 4 threads (XNNPACK) | 72.2 ms | 72.7 ms | 61.3 ms |
| 3D GPU (OpenCL delegate) | 190 ms | 190 ms | 212 ms |
| 3D GPU (OpenVX) | 267 ms | 270 ms | 546 ms |

## 5. The privacy policy on the board

Three parts, all in the output thread, per camera (`src/edge_pipeline.py`):

1. **Tracker** (`src/systems/bytelite.py`, numpy only, our ByteTrack-lite): keeps masking a face whose detector
   score dips below the threshold, as long as a low-score detection (>= 0.1) attaches to its track. 0.33 ms per frame.
2. **Look-back** (`LookBack`): a frame is shown only after the next 3 frames were detected, and it is masked with
   their boxes too, which covers a person's first frames before the detector fires. 120 ms of latency at 25 FPS,
   no compute.

3. **Brightness-adaptive threshold** (`--risk d60_t0.1`, `RiskGate`, RQ3): the mean grey level of a 1/8-subsampled
   frame (0.3 ms); below 60 the detector's threshold drops to 0.1 (stock) / 0.3 (fine-tuned), because in the dark
   faces come out with low scores. On test dark video it cuts people exposed from 34% to 10% (stock) and 72% to 45%
   (fine-tuned); on normal frames it never fires, so it masks nothing extra. Whole gate 0.8 ms per frame: one
   camera 87.6 FPS (88.7 without), three cameras still 25.0 FPS each, p50 latency 147 ms.

The online code gives exactly the masks of the offline policy `lite+back3` that we score
(`tests/test_policies.py::test_online_tracker_and_lookback_equal_offline_lite_back`). Chosen on val; at the same
3-frame delay it exposes 34% / 10% of people (stock / fine-tuned, threshold 0.9) against 83% / 48% for our MBAM
re-implementation (window3), at the same masked area (figure `experiments/rq2/latency_privacy.png`, numbers in
`docs/next-runs.md`). The tracker matters: look-back without it exposes 90% / 69%.

## 6. Where the time goes now and what is left

- NPU: 9.1 ms pure inference at its maximum clock, 92% busy at 88 FPS. Only a cheaper network (smaller input for
  far cameras, or pruning) would add cameras.
- CPU: decode + NMS 3.6 ms, tracker 0.3 ms, Python glue; 2.7 cores free at three cameras.
- Not done: live MIPI-CSI cameras through the ISP (file input here), INT8 outputs (skips 12 DEQUANTIZE ops),
  a power meter on the 12 V input.

## Reproduce

```bash
python src/export_uint8_input.py models/yunet_ft_480x640_int8.tflite models/yunet_ft_480x640_u8.tflite --schema-dir DIR
scp src/edge_pipeline.py src/board_bench.py src/systems/anonymize.py src/systems/bytelite.py scripts/edge_board.sh models/*.tflite imx:/root/edge/
ssh imx 'cd /root/edge && ./edge_board.sh > edge_board.log 2>&1'
scp -r imx:/root/edge/results/. experiments/edge/
```

`--schema-dir`: a folder with `schema_py_generated.py` (in the `ai-edge-litert` or `tensorflow` wheels).
