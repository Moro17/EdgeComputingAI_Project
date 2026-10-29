# Why YOLOv8n-face at 320 is not our edge detector

Decision (2026-10-29): YuNet is the edge baseline we fine-tune and deploy. YOLOv8n-face stays in every table as the
"big detector" reference: 640 FP32 is the accuracy ceiling, and 320 INT8 on the board is big YOLO at edge cost.
The main research question becomes: can a small fine-tuned YuNet with the recovery pipeline match a big YOLO?

All numbers below are from the 12 test sequences (ChokePoint P2 S1, 8,957 GT faces, 800x600 frames), in
`experiments/board/results.csv`.

## 1. Speed: no room left for the pipeline at 25 FPS

25 FPS gives 40 ms per frame for the whole pipeline (decode + detect + track/recovery + anonymization).
Measured on the board: hardware decode (vpudec + imxvideoconvert_g2d, `appsink sync=false`) 7.6 ms, pixelate
1.1 ms (the earlier strong blur took 104 ms and was replaced), tracker reserve 2 ms. The detector gets
40 - 7.6 - 1.1 - 2 = 29.3 ms.

| Model on the board (INT8, NPU) | Detector ms (pre + NPU + post, p50) | NPU only (benchmark_model) | % identities exposed |
|---|---|---|---|
| YOLOv8n-face 256 | 23.2 | 13.4 | 34.5 |
| YOLOv8n-face 320 | 32.9 | 19.8 | 3.4 |
| YOLOv8n-face 416 | 49.4 | - | 0.0 |
| YOLOv8n-face 640 | 104.0 | - | 0.0 |
| YuNet 608x800 | 32.2 | 17.8 | 86.2 |
| YuNet 480x640 | 27.0 | 11.3 | 86.2 |

YOLO at 320 is over the 29.3 ms detector share and leaves 7 ms of the 40 ms for everything else, so
none of the recovery pipeline (tracker, pose fallback) fits next to it. Measured end to end with JPEG reading,
the YOLO-320 loop runs at about 17 FPS. 416 and 640 are far over. Making YOLO faster means going below 320,
which makes the face-size problem in 2 worse.

YuNet 608x800 costs about the same as YOLO-320 (32.2 ms), so speed alone does not rule YOLO out at that size.
YuNet 480x640 fits (27.0 ms), and YOLO-256 also fits (23.2 ms) but exposes 34.5% of identities under pixelation:
going below 320 loses faces (on clean video, recall 0.995). YuNet is the model we can fine-tune at a size that fits.
YuNet has the room to get there: about 75k parameters against about 3M for YOLOv8n-face, and it already
processes 4.75x more pixels than YOLO-320 in the same time.

## 2. Resolution: 320 shrinks faces toward the detection limit

| YOLO input | Scale | Median face after resize | Faces narrower than 16 px |
|---|---|---|---|
| 640 | 0.80 | 42 px | 0 |
| 416 | 0.52 | 28 px | 9 |
| 320 | 0.40 | 21 px | 2,014 (22%) |

On clean video YOLO-320 still reaches recall 0.999 and exposes 3.4% of identities (1 of 29), so this costs
little on clean video. But it leaves no margin: under blur, low light or smaller faces (the stress tier), these faces are lost
first. Going below 320 to save time would make this worse. The 4:3 frame also fills only 320x240 of the
320x320 input, so a quarter of the NPU's work is padding.

## 3. Confidence: the INT8 score is saturated

In our INT8 TFLite export, every detected face scores the same value, 0.4986. The score head's activation is
quantized with scale 0.249 and zero point 127, so its largest value is 0. The board model therefore gives no
confidence signal, and RQ3 (choosing synthetic replacement or conservative masking by confidence) needs one.
Recalibrating could fix this, but it is one more reason the current board model cannot be used as is.

## 4. License

The YOLOv8n-face weights (akanametov/yolo-face, Ultralytics) are AGPL-3.0. YuNet (OpenCV Zoo) is MIT, and it
ships in OpenCV as `cv2.FaceDetectorYN`, the default face detector on edge devices.

## What would reverse this

If YOLO at a smaller input (256; measured: fits at 23.2 ms but exposes 34.5%) leaves room for the pipeline in 40 ms and still exposes 0% on the stress
tier, the budget rule picks YOLO. The finding would then be "a good detector is enough", and the paper would
say so. The rule is: lowest exposure under the detector's share of 40 ms, measured on the board.
