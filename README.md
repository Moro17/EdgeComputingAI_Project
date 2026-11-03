# One Frame Is Enough

Face anonymization for video on an NXP i.MX 8M Plus, measured at the level of identities: one frame in which a face
is recognisable is enough to expose that person. ECA course project (Valerian, Moro).

Deliverable 1 shows a YuNet face detector fine-tuned on ChokePoint, quantized to INT8 and running on the board's NPU in a
live anonymization demo at 25 FPS, compared with the stock YuNet and with YOLOv8n-face as the "big detector" reference.
**Deliverable 1 report** (hardware constraints, tech stack, NPU against CPU, optimizations before and after, the
bottleneck): [DELIVERABLE_1.md](DELIVERABLE_1.md). Board setup and demo: [docs/runbook.md](docs/runbook.md).
The plan and the measured budget are in [docs/plan.md](docs/plan.md); why YOLO is not the edge detector is in
[docs/decisions/why-not-yolo320.md](docs/decisions/why-not-yolo320.md).

## Dataset

**ChokePoint** ([Zenodo 815657](https://zenodo.org/records/815657), CC-BY-NC-4.0): people walking through two
portals (P1, P2), filmed by three cameras each, 800x600 at 30 FPS. The ground truth gives eye coordinates per person
and frame; face boxes are derived from the eyes (`src/loaders/chokepoint.py`), so every box has the same
eye-based geometry. Persons without eye annotations are skipped (one, in P2E_S3_C1.1, a val sequence).

| Split | Sequences | Use |
|---|---|---|
| train | P1E/P1L, S1-S4, C1-C3 (24) | fine-tuning |
| val | P2E/P2L, S2-S4, C1-C3, .1/.2 (36) | checkpoint and threshold choice |
| test | P2E/P2L, S1, C1-C3, .1/.2 (12; 8,957 GT faces, 29 identities) | every reported number |
| calib | S5 | INT8 calibration only |

Defined in `src/splits.py`. Train and test use different portals, but the **same 29 people** appear in both.

**Training labels** (`src/train_labels.py`, libfacedetection labelv2 format): every 5th frame. Boxes from the GT eyes;
the two eye landmarks from the GT, nose and mouth corners pseudo-labelled with SCRFD-10G. A frame is dropped when
SCRFD finds a face (score >= 0.7) that overlaps no GT box (IoU < 0.3), since that face would be trained as background:
train keeps 6,314 frames (0.6% dropped), val 5,846 (7.8% dropped).

**WIDER FACE** (train split) is mixed in so the model does not forget general faces; ChokePoint is repeated to make up
about half of each epoch (25,504 samples per epoch).

**Leak judge:** ArcFace R50 (InsightFace `buffalo_l`), each sequence is judged against a gallery of the same people's clean, eye-aligned faces from
the other recording sessions; the threshold is the
similarity at 1% false accept rate on impostor pairs. InsightFace models are for non-commercial research only.

## Method

- **Scoring:** a GT face is protected when the blur mask covers at least 80% of its box. Any other face is cropped from
  the anonymized frame and compared with the person's gallery by the judge; above threshold it is a leaked frame.
  A person is exposed when at least one of their frames leaks.
- **Anonymization:** `fast_blur` (`src/systems/anonymize.py`): a strong 3-pass Gaussian computed on the face shrunk to
  16 px, then upscaled; mask scale 1.3. 2.9 ms on the board; the judge recognises 0.3% of blurred faces.
- **Budget:** 25 FPS = 40 ms per frame for decode (7.6 ms, hardware) + detector + fast_blur (2.9) + tracker reserve (2),
  leaving 27.5 ms for the detector. YuNet 480x640 (27.0 ms) is the largest size that fits.
- **Fine-tuning** (`notebooks/finetune_yunet.ipynb`, Colab T4): libfacedetection.train (BSD-3) from the released
  `yunet_n` weights, lr 0.001 (1/10 of default), square 640 crops, checkpoint chosen on val.
- **Quantization** (`src/export_yunet_tflite.py`): ONNX -> static INT8 TFLite at 480x640 with onnx2tf, per-tensor,
  for the VX delegate on the NPU.

## Results (test split, i.MX 8M Plus NPU, INT8)

| System | Threshold | Detector p50 ms | Frame recall | % identities exposed | Max exposed run (frames) |
|---|---|---|---|---|---|
| YuNet stock 480x640 | 0.9 (OpenCV default) | 20.1 | 0.944 | 86.2 | 32 |
| YuNet fine-tuned 480x640 | 0.9 | 21.4 | 0.992 | 48.3 | 9 |
| YuNet stock 480x640 | 0.6 (chosen on val) | 20.1 | 0.9999 | 0.0 | 0 |
| YuNet fine-tuned 480x640 | 0.7 (chosen on val) | 21.4 | 1.000 | 0.0 | 0 |
| YOLOv8n-face 256 | 0.49 | 23.2 | 0.995 | 10.3 | 5 |
| YOLOv8n-face 320 | 0.49 | 32.9 (over budget) | 0.999 | 0.0 | 0 |

YuNet rows: `experiments/d1/` (the per-threshold tables are `results_*.csv`). YOLO rows and the other input
sizes: `experiments/board/results.csv`; their timings predate the faster YuNet decoding (27.0 ms -> 20.1 ms for stock).

What this shows:
- At OpenCV's default threshold 0.9, fine-tuning raises frame recall from 0.944 to 0.992 and cuts exposed people
  from 86% to 48%.
- With the threshold chosen on val, both stock and fine-tuned YuNet expose nobody on clean test video. On clean
  ChokePoint, the threshold matters more than fine-tuning. Degraded video (the stress tier) is where the
  detectors will separate; that is the next phase.
- The fine-tuned network is slightly larger (trainer's `yunet_n`: 59 convolutions against 53): +1.3 ms on the NPU.

Val results (checkpoint and threshold choice): [docs/plan.md](docs/plan.md), Phase 4.

**Live demo on HDMI, whole loop** (decode + detect + fast_blur + overlay + display, 277 s clip of three test
sequences): YuNet stock 28.5 FPS (27.2-28.8), YuNet fine-tuned 27.9 FPS (26.7-28.6), YOLOv8n-face 320 22.1 FPS
(21.8-22.3); ranges over 2 s windows.

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/loaders/chokepoint_download.py --all       # ~12 GB into data/chokepoint/
.venv/bin/python src/loaders/chokepoint.py                      # GT CSVs into gt/chokepoint/
cd src && ../.venv/bin/python benchmark.py --split test --systems yunet_ort yunet_ft --out ../experiments/mac   # Mac/CPU
```

Fine-tuning: `.venv/bin/python src/train_labels.py train --every 5 --out data/yunet_ft/` (and `val`), zip `data/yunet_ft`,
upload it to Google Drive and run `notebooks/finetune_yunet.ipynb`.

Checkpoint to ONNX, with the trainer cloned next to it:
`python -m yunet_train.cli.export_onnx yunet_ft_e20.pth --variant yunet_n --dynamic-export --verify --output-file yunet_ft.onnx`.
INT8 export (separate env with onnx2tf): `.venv-export/bin/python src/export_yunet_tflite.py yunet_ft.onnx --size 480 640 --out models/yunet_ft_480x640_int8.tflite`.

On the board (copy `src/board_bench.py`, `src/demo.py`, `src/systems/anonymize.py`, `scripts/run_demo.sh` and the
`.tflite` models to `/root`):

```bash
python3 board_bench.py --model yunet_ft_480x640_int8.tflite --frames /root/chokepoint --out /root/dets   # copy dets/ back, score on the Mac:
#   cd src && ../.venv/bin/python benchmark.py --seqs <12 test seqs> --systems yunet_480x640_npu yunet_ft_480x640_npu \
#     --thr yunet_480x640_npu=0.6 yunet_ft_480x640_npu=0.7 --out ../experiments/d1
./run_demo.sh --video demo.mp4 --loop --models yunet_480x640_int8.tflite yunet_ft_480x640_int8.tflite yolov8n-face_320_int8.tflite
```

Tests: `.venv/bin/python tests/test_core.py`, `tests/test_loaders.py`; `.venv-export/bin/python tests/test_export.py`,
`tests/test_demo.py`.

Data, models, detections and face embeddings are not in the repository.
