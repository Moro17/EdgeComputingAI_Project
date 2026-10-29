# Project plan: can a small fine-tuned YuNet pipeline match a big YOLO?

Updated 2026-10-29. Target: 25 FPS on the i.MX 8M Plus, i.e. 40 ms per frame for the whole pipeline
(decode + detect + track/recovery + blur). YuNet is the edge detector and YOLOv8n-face is the reference;
the reasons are in [decisions/why-not-yolo320.md](decisions/why-not-yolo320.md). Task-level detail for the
pipeline and the degradations is in
[superpowers/plans/2026-10-28-miss-aware-pipeline.md](superpowers/plans/2026-10-28-miss-aware-pipeline.md)
(Tasks 6-11); this file is the order and the gates.

## Final comparison (what the paper reports)

Rows, all scored on the same test split, same judge, same blur, mask scale 1.3:

| | Per-frame | + recovery pipeline |
|---|---|---|
| YuNet stock | baseline | effect of the pipeline alone |
| YuNet fine-tuned | effect of fine-tuning alone | **ours** |
| YOLOv8n-face 640 FP32 (desktop) | accuracy ceiling | - |
| YOLOv8n-face 320 INT8 (board) | big YOLO at edge cost | - |

Columns: frame recall, % identities exposed, % appearances exposed, max exposed run, % area masked,
end-to-end FPS and p50/p95 ms on the board. On clean ChokePoint and on the stress tier.

## Done

- RQ1 harness: loaders, coverage matcher, ArcFace judge (1% FAR), exposure metrics.
- Board: YOLOv8n-face 320/416/640 and YuNet 608x800 as INT8 TFLite on the NPU, scored on ChokePoint P2 S1
  (`experiments/board/`).

## Phase 1: budget sweep on the board (gate for everything after)

- Time YuNet INT8 at 608x800, 480x640, 384x512, 320x416 and YOLO INT8 at 256 (pre + NPU + post).
- Measure the rest of the frame: hardware video decode (GStreamer), blur, and a tracker stub.
- Score each size on the test split.
- **Output:** the YuNet input size that fits the detector's share of 40 ms with room for the pipeline.
  Fine-tuning happens at this size. If YOLO-256 fits and still exposes 0%, revisit the decision doc.

**Outcome (measured on the i.MX 8M Plus, 12 test sequences; anonymization = fast_blur):**

| System | NPU only ms | p50 pre / npu / post ms | p50 detector ms | % identities exposed | frame recall |
|---|---|---|---|---|---|
| YuNet 608x800 | 17.8 | 1.7 / 24.2 / 5.8 | 32.2 | 86.2 | 0.955 |
| **YuNet 480x640** | 11.3 | 5.6 / 17.3 / 4.2 | **27.0** | 86.2 | 0.944 |
| YuNet 384x512 | 6.6 | 3.7 / 10.0 / 3.2 | 16.8 | 96.6 | 0.885 |
| YuNet 320x416 | 5.1 | 3.8 / 7.0 / 2.6 | 13.4 | 100.0 | 0.741 |
| YOLOv8n-face 256 | 13.4 | 5.0 / 17.6 / 0.6 | 23.2 | 10.3 | 0.995 |
| YOLOv8n-face 320 | 19.8 | - | 32.9 | 0.0 | 0.999 |

- Decode (800x600 H.264, 300 frames): vpudec + imxvideoconvert_g2d + videoconvert, `appsink sync=false`:
  p50 7.6 ms. With CPU videoconvert instead of g2d: 22.6 ms. (With appsink's default sync the video is paced to
  30 FPS and reads 33 ms; that is not decode cost.)
- Anonymization, 2 face-sized boxes: fast_blur (the strong blur computed at 16 px) p50 2.89 ms; the full strong blur
  104 ms, which alone broke the budget. fast_blur passes the judge check (0.3% recognised, at most 2% required) and
  reproduces the full blur's exposure on every system. Pixelate (6 cells, 1.1 ms) was cheaper but leaked more on
  partly covered faces (YOLO-320 3.4%, YOLO-256 34.5%), so it was dropped.
- Budget: detector = 40 - 7.6 (decode) - 2.9 (fast_blur) - 2 (tracker reserve) = 27.5 ms.
- **Chosen: YuNet 480x640** (largest size with p50 <= 27.5 ms: 27.0 ms, 0.5 ms of margin; 608x800 at 32.2 ms is over).
- Fallback if the live demo misses 25 FPS: fast_blur with side=8 (1.94 ms, not yet checked for privacy).
  Stock YuNet exposes 86% of identities at this size; closing that gap is the fine-tuning's job.
- YOLO-256 fits the budget (23.2 ms) but exposes 10.3% of identities, so the reversal condition (fits and 0%)
  does not trigger. YOLO-320 exposes 0% but is over budget (32.9 ms).

## Phase 2: split ChokePoint

| Split | Sequences | Use |
|---|---|---|
| train | P1 S1-S4 (all cameras, 24) | fine-tuning |
| val | P2 S2-S4 (36) | checkpoint and operating threshold |
| test | P2 S1 (12, the current benchmark) | every reported number |
| calib | P2 S5 | INT8 calibration only (unchanged) |

- Different portal for train and test. The same 29 people appear in both; the paper says so.
- Test = the sequences already benchmarked, so the cached detections in `experiments/board/dets/` stay valid;
  re-scoring is a check, not a rerun.
- Download P1 and P2 S2-S4 (size to check).

## Phase 3: training labels

- Boxes from the ChokePoint eye GT (same geometry as the benchmark GT); 5 landmarks pseudo-labelled with
  SCRFD-10G, as `fill_eyes` does for Music Video.
- Check for unlabelled faces in train frames (they would become negatives); pseudo-label or drop those frames.
- Mix in WIDER FACE so YuNet does not overfit 29 faces or learn only our box shape. Disclose the eye-derived boxes.

## Phase 4: fine-tune YuNet (FP32)

- libfacedetection.train from the pretrained yunet_n weights, on Colab, at the Phase 1 input size,
  with degradation augmentation (blur, low light, downscale).
- Pick checkpoint and threshold on val. Export ONNX, check it matches the PyTorch model.

## Phase 5: quantize and benchmark the detector

- FP32 and FP16: Mac/CPU reference rows (the NPU runs integer models only).
- INT8 TFLite for the NPU via onnx2tf, calibrated on S5. Check the score is not saturated.
- Benchmark on test: Mac FP32/FP16, board INT8. Fills the "per-frame" column.

## Phase 6: stress tier

- Degradations (dark, blur, downscale, occlusion; 3 severities) on the test split (pipeline plan, Task 11).
- Needed because on clean ChokePoint a good detector already exposes 0%; the comparison has to be made where
  detectors miss.

## Phase 7: recovery pipeline

- Policies on cached detections (pipeline plan, Tasks 6-10): ByteTrack (B2), detect every N frames (B3),
  coasting, pose fallback.
- Budget rule: tracker and coasting run every frame on the CPU; pose runs only when a track is lost
  (or desktop only) unless the board numbers say it fits.

## Phase 8: benchmark the pipeline

- Fill the 2x2 grid on clean and stressed test data, end-to-end latency on the board.
- Answer: does fine-tuned YuNet + pipeline reach the YOLO-640 ceiling, and beat YOLO-320 at equal cost?

## Phase 9: demo and write-up

- `demo.py` on the board: video file, NPU, blur, HDMI, FPS overlay, stock vs fine-tuned toggle.
- README dataset section, results table, push to the team repo.

## Deliverable 1 (2026-11-03 16:00)

Needs a fine-tuned model deployed on the board, a live demo, code on GitHub and dataset details:
Phases 1-5 and the Phase 9 demo. Phases 6-8 come after.
