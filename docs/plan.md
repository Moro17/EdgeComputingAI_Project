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
