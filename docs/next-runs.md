# Next runs: stress tier, policies, set-level attacker

What is ready to run (2026-11-18) and the exact commands. Direction and why: [research-gap.md](research-gap.md).
Everything scores on the ChokePoint test split (12 sequences, 8,957 faces, 29 people), same judge, same fast_blur,
mask scale 1.3.

## What exists

| Piece | File | Spec / flag |
|---|---|---|
| Degradations (stress tier) | `src/degrade/degrade.py` | `--degrade KIND:SEV`, KIND in `dark blur down occl jpeg`, SEV 1-3 |
| Per-frame baseline B1 | - | `yunet_480x640_npu` |
| Detect every N + hold (B3) | `src/systems/policies.py` | `+skip3+hold` |
| ByteTrack (B2, supervision) | same | `+bytetrack` |
| Our re-implementation of MBAM's temporal fusion | same | `+window3` (simplified, see docstring) |
| Fail-closed tracking (ours) | same | `+bytetrack+keep` |
| Coasting (ours v1) | same | `+bytetrack+keep+coast` |
| Exposure budget per track (ours v2) | same | `+bytetrack+keep+budget`, knobs `+budget_d0.7_g1.8_a10-60` (decay, grow_max, coast ages) |
| Equal-masked-area control | same | `+growP` last, e.g. `+window3+grow145`: every final box x1.45 |
| Risk-adaptive threshold (RQ3, ours, board) | same + `systems/bytelite.py` | `+risk_d60_t0.1+lite+back3` (also `_h13`, `_p3`) |
| Music Video dataset (tier 2) | `src/loaders/musicvideo.py`, `scripts/mv_board.sh` | `--dataset musicvideo` |
| Val detections on the board (chunked, resumable) | `scripts/val_board.sh` | val frames (4.9 GB) do not fit the board's 2.8 GB free; copies one archive at a time |
| Set-level attacker | `src/judge/setlevel.py` | separate script, reads the benchmark caches |
| CIs, exposure vs length, recall-vs-exposure plot | `src/analysis/rq1.py` | separate script |

Not done: pose fallback (plan Task 10), conformal calibration of the threshold, extra datasets (plan Task 14),
report/Pareto script (plan Task 12), latency of skipN in the table (fps column is still the detector's every frame).

## First results (board NPU detections, clean test split, YuNet stock, threshold 0.9)

| System | Frame recall | % identities exposed |
|---|---|---|
| per-frame | 0.944 | 86.2 |
| + skip3 + hold | 0.892 | 100 |
| + ByteTrack (vanilla) | 0.968 | **89.7**: higher recall, MORE people exposed |
| + ByteTrack + keep (fail-closed) | 0.977 | 65.5 |

Why vanilla ByteTrack exposes more: it drops 80 faces the detector had found at score >= 0.9, 29 of them in the first
two frames of an appearance (a new track is unconfirmed for one frame) and the rest when a track is re-born after a
lost association. `keep` masks every raw detection as well. (The keep+coast number from the same day was computed
with a bug that coasted keep's untracked rows as one fake track; rerun it.)

## Results 2026-11-19: look-back fill beats our MBAM re-implementation at equal latency and area

Board NPU detections, % identities exposed / % frame area masked. Knob choice (W=3, latency-matched to window3)
made on VAL; test numbers reported after.

Val (P2 S2-S4, 36 sequences, 29 people), threshold 0.9:

| Model | window3 (MBAM re-impl.) | keep | keep+coast | keep+budget | **keep+back3** | keep+coast+back3 |
|---|---|---|---|---|---|---|
| stock | 82.8 / 1.37 | 96.6 / 1.05 | 89.7 / 1.46 | 86.2 / 2.15 | **72.4 / 1.40** | 55.2 / 1.87 |
| fine-tuned | 48.3 / 1.48 | 44.8 / 1.09 | 44.8 / 1.51 | 44.8 / 2.20 | **13.8 / 1.45** | 13.8 / 1.93 |

At the operating thresholds (0.6 / 0.7) everything with keep or a window is at 0-3.4% on val.

Test, 6 conditions x 2 thresholds x 2 models (`results/.../stress/back_*.csv`, `npu_*.csv`): keep+back3 exposes no
more identities than window3 in all 24 cells and fewer in most, at the same or smaller masked area, e.g. clean@0.9
stock 38 vs 55, occl2@0.9 stock 45 vs 62, dark2@op 52/66 vs 62/83, jpeg2@0.9 FT 41 vs 52. Adding coast helps
where misses are long (dark2@op: 34/52) at +0.5% area.

- Equal-area control (`+growP`): growing window3 or keep masks to the budget's area changes nothing (a miss has no
  box to grow); coast+grow135 matches budget. The exposure budget is no better than coast with a uniformly larger
  mask -> it stays an ablation, not the contribution.
- Why look-back works: most leaks are an appearance's first frames, before the detector fires; forward-only
  tracking/coasting cannot cover them, a 3-frame delay buffer can. Vanilla ByteTrack hides the same frames.

## Results 2026-11-20: it is the tracker, and the board's numpy tracker is the best variant

Ablation (`+back3` without a tracker, `+spread3` = windowW as a union without NMS, `+lite` = `systems/bytelite.py`,
the numpy ByteTrack-lite that runs on the board), % identities exposed (stock / fine-tuned), threshold 0.9:

| | window3 | spread3 | back3 (no tracker) | keep+back3 (supervision ByteTrack) | **lite+back3** |
|---|---|---|---|---|---|
| val | 83 / 48 | 83 / 38 | 90 / 69 | 72 / 14 | **34 / 10** |
| test clean | 55 / 10 | 48 / 10 | 66 / 21 | 38 / 7 | **17 / 7** |
| test occl2 | 62 / 14 | 55 / 10 | 72 / 28 | 45 / 10 | **24 / 7** |
| test jpeg2 | 86 / 52 | 86 / 52 | 93 / 69 | 86 / 41 | **62 / 31** |

Masked area: lite+back3 1.37-1.48% of the frame, window3 1.16-1.48%.

- Look-back alone is WORSE than a symmetric window: the gain comes from the tracker keeping low-score
  detections (0.1 <= s < thr) of an existing track. NMS costs window3 a little (spread3 <= window3), not the gap.
- lite beats supervision's ByteTrack: lost tracks keep moving (re-found where they went) and low-score detections
  can re-attach to any live track. Same masked area.
- Paired (same people), lite+back3 vs window3: val stock 14 people exposed only by window3, 0 only by ours
  (sign test p = 0.0001); val fine-tuned 11 vs 0 (p = 0.001); test, all conditions pooled: 66 vs 2.
  `src/analysis/paired.py`, csvs `paired_*_window3_vs_liteback3.csv`.
- Latency sweep on test: at W <= 3 keep+backW wins almost everywhere; at W = 5-6 windowW catches up in some
  conditions (clean and occl2 fine-tuned, 3.4% both) at 200+ ms.
- Exception: dark2 at the operating threshold, fine-tuned: lite+back3 72 vs keep+back3 66 (window3 83); lite+spread3
  48 there (needs 3 more frames' boxes and +0.3% area).

## Results 2026-11-21: what is left is a threshold problem (no pose), and RQ3 as a brightness-adaptive threshold

**Where the remaining leaks come from** (`src/analysis/leak_anatomy.py`, `experiments/rq3/leak_anatomy.csv`): for
every face still exposed by lite+back3, is there a raw detection on it (IoU >= 0.3, any score >= 0.05)?
- Every condition except dark: 100% of leaked faces have a detection, >= 99% at score >= 0.3. The detector saw the
  face; the threshold dropped it.
- dark2: 4.5-28% of leaked faces have no detection at all, but at most ONE person per condition is exposed only
  through undetected frames. A pose fallback (a second network on the NPU) could save at most one person per
  condition -> **pose fallback dropped from RQ2**, with this number as the reason.

**RQ3, reformulated (quantitative, runs on the board):** does a context-adaptive threshold beat every fixed threshold
on the exposure / masked-area trade-off? `+risk` step (`systems/bytelite.py` RiskGate, online, numpy), three rules
for a detection below the operating threshold: tall box (`_hP`, % of frame height), persistent 3 frames (`_pK`),
dark frame (`_dL_tT`: mean grey of the frame < L -> threshold T). L = 60 from frame brightness on val: clean, blur,
jpeg, occl, down frames are 81-110, dark:1-3 are 10-56.
- Height and persistence rules: at most one person on val dark, at a larger area cost -> not kept.
- Dark rule on val (`experiments/rq3/rq3_pareto_val.png`): in the dark it gives exactly the low threshold's exposure
  (stock: 28.6% -> 9.4% of appearances exposed at T = 0.1); on clean video it never fires, so it keeps the
  operating threshold's masked area (1.57%) where the fixed 0.1 masks 1.81% (+15%).
- Choice on val (criterion: lowest T whose dark masked area <= 1.3x the clean area): stock T = 0.1, fine-tuned T = 0.3
  (the fine-tuned model fires many false positives on dark noise: 3.9% area at 0.1). Caveat: this criterion was
  written down after the test sweep had run, so report the whole T curve on test, not only the chosen point.

Test (`experiments/rq3/*.csv`, `rq3_pareto_test.png`, `paired_test_dark2.txt`), operating thresholds:

| % people exposed, dark2 | op threshold | dark rule T=0.1 | T=0.15 | T=0.2 | T=0.3 |
|---|---|---|---|---|---|
| stock | 34.5 | **10.3** (p = 0.016) | 13.8 | 13.8 (p = 0.031) | 20.7 (p = 0.125) |
| fine-tuned | 72.4 | 37.9 (p = 0.002) | 41.4 | 44.8 (p = 0.008) | **44.8** (p = 0.008) |

p: paired sign test vs the op threshold, same 29 people; no person is exposed only by the adaptive policy.
Clean, blur2, occl2, jpeg2, down2: masks identical to the operating threshold (rule never fires), 0% exposed.

The other dark severities (`dark1_adapt.csv`, `dark3_adapt.csv`, `paired_test_dark{1,3}.txt`): the rule helps only in
the middle of the range.
- dark1: nobody is exposed even at the op threshold; the rule fires (frames 37-56 < 60) and only costs area
  (1.49% -> 1.78% at T = 0.1, 1.52% at T = 0.3).
- dark3: every person is exposed with or without it (identity level saturated; fine-tuned T = 0.1: 96.6%). Per
  appearance it still helps: stock 62.5% -> 47.5% of appearances exposed, mean longest exposed run 6.6 -> 3.1 frames,
  at 4.5% area (fine-tuned T = 0.1: 17% area, dark-noise false positives; T = 0.3 keeps area < 1%).
- So the claim is "at moderate darkness the brightness rule removes most of the exposure for free on normal frames",
  not "it fixes the dark"; below that, the detector itself must improve (fine-tune on dark data, or IR camera).
On the board: 0.8 ms per frame for the gate (luma of a 1/8-subsampled frame + rules); 3 cameras still 25 FPS
(`experiments/edge/cams3_ft_risk.json`).

## Results 2026-11-21: generalization to Music Video (tier 2, nothing tuned on it)

Setup (`src/loaders/musicvideo.py`, `musicvideo_landmarks.py`, `scripts/mv_board.sh`; `experiments/musicvideo/`):
8 music videos, 50 people, 95k GT faces, each video split in two halves; the attacker's gallery for a person comes
from the other half of the same video (same clothes and lighting: an **upper bound** on exposure). Eyes from
RetinaFace on clean frames for 94-99.9% of faces. Judge: threshold 0.339 at 1% FAR, clean TAR 83.7% (ChokePoint
99.99%: smaller, blurrier faces). Detections: NPU on the board, INT8 480x640, thresholds chosen on ChokePoint val.

Bug found on the way: `benchmark.enroll` stored identities in GT row order but embeddings in frame order. Harmless on
ChokePoint (GT already frame-sorted; checked: 48/48 embedding files identical), fatal on Music Video (GT sorted by
track: TAR 6-9%). Fixed, with `tests/test_core.py::test_enroll_ids_follow_crop_order_when_gt_is_sorted_by_track`.

| % people exposed (50) | per-frame | window3 | bytetrack+keep+back3 | lite+back3 | + dark rule T=0.1 | T=0.3 |
|---|---|---|---|---|---|---|
| stock, 0.6 (frame recall 0.88 per-frame) | 94 | 80 | 80 | 76 | **72** | 74 |
| fine-tuned, 0.7 (0.87) | 96 | 82 | 82 | 80 | **72** | 76 |
| stock, 0.9 | 98 | 98 | 96 | 96 | 90 | 92 |
| fine-tuned, 0.9 | 98 | 96 | 96 | 94 | 88 | 92 |

- RQ1 again, on a second dataset: 88% frame recall exposes 94% of people.
- Vs per-frame (`paired_vs_frame.txt`), at the op thresholds: lite+back3 94 -> 76 (stock, 9 vs 0 people, p = 0.004),
  96 -> 80 (fine-tuned, p = 0.008); with the dark rule 94 -> 72 (p = 0.001), 96 -> 72 (p < 0.001). At 0.9 nothing
  is significant (98% exposed per-frame, frame recall 0.35-0.45).
- Same ordering as ChokePoint, but identity level is near saturation, so vs window3 no paired test is significant
  (best: fine-tuned dark rule 82 -> 72, 5 vs 0 discordant, p = 0.06).
- **RQ3 generalizes** (`results_op@dark2.csv`, `paired_dark2.txt`; Music Video + synthetic dark:2, op thresholds):

  | % people exposed, MV dark2 | per-frame | window3 | lite+back3 | dark rule T=0.1 | T=0.3 |
  |---|---|---|---|---|---|
  | stock | 98 | 88 | 76 | **60** (8 vs 0, p = 0.008) | 72 (p = 0.5) |
  | fine-tuned | 96 | 86 | 84 | **56** (14 vs 0, p < 0.001) | 76 (p = 0.13) |

  Longest exposed run 8.7 -> 1.5 frames (stock T = 0.1), at 10.4% -> 20.1% masked area (T = 0.3: 12.8%). The T chosen
  on ChokePoint val for fine-tuned (0.3) is too timid here; T = 0.1 wins on both datasets in exposure. The longest exposed run shrinks more:
  stock 0.6 per-frame 11.4 -> lite+back3 7.6 -> dark rule 6.6 frames; 0.9: 78.7 -> 70.8 -> 55.8.
- The dark rule fires on real stage lighting here (clean frames, no synthetic darkness), at +3 points of area
  (lite+back3 11.4% -> 14.3% at T = 0.1, 12.1% at T = 0.3; faces are large in music videos, so area is 8-14%).
- Where faces are missed: small ones. 1280x720 frames are squeezed to 640x480 for YuNet, so a 20-40 px face becomes
  13-27 px. Per-frame protection 44% for faces <= 20 px, 78% for 20-40 px, >= 93% above 40 px (lite+back3: 68 / 90
  / >= 97%). Big close-ups are fine. A larger input or tiling would help, at NPU cost (RQ4).

## Results 2026-11-22: deface baseline (CenterFace, per-frame, desktop CPU)

`deface` 1.5.0 (CenterFace ONNX, its own default threshold 0.2, full frame resolution), with our mask scale 1.3 and
fast_blur, scored like every other system. CPU only (onnxruntime, desktop); it does not run on the NPU.
Results: `experiments/deface/`.

| % people exposed | ChokePoint test clean | ChokePoint test dark:2 | Music Video clean |
|---|---|---|---|
| deface per-frame | 0 | 13.8 | 70 |
| deface + lite+back3 | 0 | 3.4 | 32 |
| deface + lite+back3 + dark rule T=0.1 | 0 | 0 | 32 |
| *YuNet stock NPU, same three* | *0 / 0 / 0* | *100 / 34 / 10* | *94 / 76 / 72* |

- deface is far more robust than YuNet per frame: CenterFace runs at full resolution (800x600; Music Video 1280x720,
  where YuNet squeezes small faces to 640x480) and is a larger network. Cost: 61 ms per frame on the desktop CPU on
  ChokePoint, 198 ms on Music Video (p50), against 9 ms for YuNet INT8 on the board NPU. It does not fit the 40 ms board
  budget.
- **Our recovery is detector-agnostic:** tracker + look-back cuts deface's exposure 13.8 -> 3.4% (dark:2) and
  70 -> 32% (Music Video); the dark rule takes ChokePoint dark:2 to 0. On Music Video the rule adds nothing to deface
  (32 -> 32): CenterFace's scores in stage lighting are already above its threshold.
- Area: deface masks 1.05-1.15% of the frame on ChokePoint (YuNet 1.1%), 8.4% on Music Video (YuNet 8.5%): same
  utility cost.

## Results 2026-11-22: quantization study (RQ4), desktop simulation of the board's INT8

What runs on the board: onnx2tf full-integer INT8, per-tensor, min-max calibration on 100 clean ChokePoint S5 frames.
Question: does a different quantization recipe lose fewer people? Six recipes, each changing one thing
(`systems/quantize.RECIPES`):

| recipe | change |
|---|---|
| board | as deployed: per-tensor, min-max, int8 activations, clean S5 |
| pc | per-channel weights |
| pct | percentile calibration (99.999%) instead of min-max: clips activation outliers |
| ent | entropy (KL) calibration |
| dark | calibration frames 1/3 clean, 1/3 dark:1, 1/3 dark:3 (never dark:2, the test condition) |
| a16 | int16 activations, int8 weights (the "int16x8" mode) |

How: onnxruntime static QDQ quantization of the same graph, every op quantized (as onnx2tf), input 480x640
placed top-left as on the board. The fine-tuned model is only on the board as TFLite; its FP32 TFLite was converted
back to ONNX with tf2onnx (stock control `yq_tfl_fp32`: identical detections and identical results to the original
ONNX, so the conversion is exact). The board's onnx2tf export needs `ai-edge-litert`, which is not installed on this
PC, so the recipes are simulated here and not yet run on the NPU. Fidelity check: ORT's "board" recipe against the
real NPU detections, paired, on dark:2: per-frame 1 vs 0 discordant people, lite+back3 4 vs 6 (stock) and 3 vs 2
(fine-tuned), all p >= 0.75: the simulation is faithful.

ChokePoint test, dark:2, % people exposed (29 people, 1 person = 3.4 points); per-frame / lite+back3 / + dark rule.
Clean: 0% for every recipe and policy.

| recipe | stock (0.6) | fine-tuned (0.7) |
|---|---|---|
| FP32 | 100 / 34 / 17 | 97 / 62 / 41 |
| board (deployed) | 97 / 41 / 10 | 93 / 69 / 34 |
| pc | 100 / 52 / 17 | 97 / 66 / 45 |
| pct | 100 / 34 / 10 | 93 / 66 / 38 |
| ent | 97 / 41 / 10 | 93 / 69 / 34 |
| dark | 100 / 48 / 14 | 93 / **52** / 34 |
| a16 | 97 / 31 / 10 | 93 / 62 / 34 |

Paired sign tests against FP32 (`experiments/quant/paired_*dark2.txt`; tables `experiments/quant/`): no recipe differs from FP32, for
either model, under any policy (smallest p = 0.125, per-channel, stock, lite+back3, where it is worse).
- **INT8 as deployed costs no measurable privacy**, also on degraded video. Per-tensor min-max is enough; per-channel
  does not help (trend worse). int16 activations recover nothing measurable and are 1.2-1.9x slower on the desktop
  CPU; on the board they are unusable (2.2 s per frame, see the conclusion below).
- Entropy calibration gave exactly the board recipe's ranges here (identical results).
- One trend to check: dark calibration on the fine-tuned model, lite+back3: 69 -> 52% against the board recipe
  (6 vs 1 discordant people, p = 0.125). Replication on val dark:2 and Music Video dark:2: see below. It is free to
  deploy (only the calibration set changes).
- At 0.9 the earlier result stands (`experiments/d1/results_quant_t0.9_0.9.csv`): INT8 at 48% vs FP32 55% exposed,
  within noise.

Music Video (clean, 50 people, 1 person = 2 points; `experiments/quant/mv_clean.csv`), per-frame / lite+back3:

| recipe | stock (0.6) | fine-tuned (0.7) |
|---|---|---|
| FP32 | 94 / 76 | 96 / 80 |
| board | 94 / 84 | 96 / 86 |
| pc | 96 / 86 | 96 / 88 |
| a16 | 94 / 78 | 96 / 80 |

- Here INT8 per-tensor is worse with tracking for both models: FP32 vs board 0 vs 4 people (stock, p = 0.125) and
  0 vs 3 (fine-tuned, p = 0.25); every discordant person goes the same way. int16 activations recover it (a16 vs
  board 3 vs 0 for each model, p = 0.25; a16 vs FP32 within 1 person). Per-channel is again no better.
- Reading: on fast, varied video the INT8 detector loses a few low-confidence detections that the tracker needed;
  not significant on its own, but the same direction on two models and the opposite of nothing. Worth one NPU run
  of an int16x8 export (needs the onnx2tf export environment) and its latency before claiming anything.

Replication of the dark-calibration trend, val dark:2 (29 people; `experiments/quant/val_dark2.csv`), lite+back3:
stock FP32 / board / dark 93 / 93 / 97, fine-tuned 97 / 97 / 100 (per-frame 100 everywhere). Not replicated:
dark calibration is 1 person worse for each model (p = 1), FP32 and board are identical. Caveat: without the dark
rule val dark:2 is saturated, so this test has little room; the test-split trend (6 vs 1) is treated as noise.
Music Video dark:2 (50 people; `experiments/quant/mv_dark2.csv`), lite+back3, FP32 / board / dark: stock 78 / 84 /
82, fine-tuned 82 / 80 / 84 (per-frame 98 everywhere). Dark calibration again not better than board (fine-tuned 0 vs 2
people, worse; stock 3 vs 2, p = 1). INT8 vs FP32: stock 1 vs 4 (p = 0.375, same direction as MV clean), fine-tuned
1 vs 0. Two replications fail, so dark calibration is dropped.

On the board itself (FP32 TFLite on the CPU, `board_bench.py --cpu`, against the INT8 NPU detections; ChokePoint
test; `experiments/quant/board_fp32_*.csv`): clean 0% for both. dark:2, per-frame / lite+back3 / + dark rule,
FP32 vs INT8: stock 100 / 34 / 17 vs 100 / 34 / 10, fine-tuned 97 / 62 / 41 vs 97 / 72 / 38. Paired: discordant
people go both ways (4 vs 4, 2 vs 0, 2 vs 5, 2 vs 1), all p >= 0.45. The board FP32 numbers equal the desktop ORT
FP32 ones exactly. Latency p95: FP32 on the CPU 54 / 69 ms against 21 / 22 ms INT8 on the NPU (2.6-3.2x).

Conclusion (RQ4, quantization): the deployed per-tensor min-max INT8 is kept. No recipe beats it on any of four
test sets (ChokePoint test dark:2, val dark:2, Music Video clean and dark:2). The only repeated trend is INT8 losing a
few people with tracking against FP32 (worse in 5 of the 8 model x test-set comparisons, equal in 2, better in 1;
never significant), which int16 activations recover on the desktop. int16 is not an option on the board: the
int16x8 export (`export_yunet_tflite.py --a16`; TFLite needs per-channel weights for it) is rejected by the VX
delegate ("Int16 output is not supported") and runs on CPU reference kernels at 2.2 s per frame on the NPU path and
2.2 s with 4 CPU threads, against 11.7 ms INT8 on the NPU and 44 ms FP32 on 4 CPU threads
(`experiments/quant/board_a16_benchmark.txt`, benchmark_model, 480x640 stock). So the INT8 trend cannot be bought
back with int16 on this board; FP32 on the CPU would cost 54-69 ms per frame, over the 40 ms budget.

## Results 2026-11-23: detection interval N and the final configuration on the board (RQ4)

Final configuration: fine-tuned YuNet INT8 (uint8 input) on the NPU, lite tracker, dark rule d60_t0.1, look-back 3,
fast_blur (`scripts/edge_final.sh`, `experiments/edge/final_*.json`, `every*.json`). One stream 89.5 FPS; live
cameras at 25 FPS: 1-3 cameras hold 25 FPS (latency p50 134-146 ms, includes the 120 ms look-back), 4 cameras drop
to 22.5 FPS (NPU 92% busy). `--every N` (detector on every N-th frame, boxes held in between): N = 2 brings 4 cameras
to 25 FPS (p50 139 ms); max single-stream FPS 132 / 138 / 140 at N = 2 / 3 / 5 (decode and CPU bound beyond N = 2).

Privacy cost (`policies.py` skipN+lite+hold+back3, NPU detections; `experiments/skip/`), % people exposed, fine-tuned
/ stock, with the dark rule, N = 1 / 2 / 3 / 5: ChokePoint dark:2 38/10, 38/10, 45/3, 52/21; ChokePoint clean 0 up to
N = 3, N = 5 3/14; Music Video 72/72, 80/78, 82/80, 94/96. N = 2 is free on ChokePoint and costs 6-8 points on Music
Video (fast motion); N >= 3 is not worth it. Without the dark rule, N = 3 on dark:2 jumps to 72-79%.

## Results 2026-11-23: other detectors on the current protocol (comparison table)

Desktop CPU (onnxruntime), each detector's default threshold (SCRFD 0.5, YOLOv8n-face 0.5), our mask scale and
fast_blur; `experiments/others/`. % people exposed, per-frame -> lite+back3 -> + dark rule T = 0.1:

| detector (desktop p50) | ChokePoint clean | ChokePoint dark:2 | Music Video |
|---|---|---|---|
| SCRFD-500M (12-15 ms) | 0 / 0 / 0 | 76 / 3 / 0 | 96 / 72 / 64 |
| SCRFD-10G (88 ms) | 0 / 0 / 0 | 31 / 3 / 0 | 82 / 36 / 32 |
| YOLOv8n-face (41 ms) | 0 / 0 / 0 | 52 / 28 / 28 | 86 / 60 / 56 |
| deface / CenterFace (61-198 ms) | 0 / 0 / 0 | 14 / 3 / 0 | 70 / 32 / 32 |
| YuNet stock, NPU INT8 (11 ms on the board) | 0 / 0 / 0 | 100 / 34 / 10 | 94 / 76 / 72 |
| EgoBlur Gen1 face, T = 0.9 (88-164 ms, RTX 3060 GPU) | 0 / 0 / 0 | 76 / 7 / 7 | 92 / 52 / 48 |

- The recovery is detector-agnostic: lite+back3 cuts exposure on every detector, most on the strong ones (SCRFD-10G
  Music Video 82 -> 36%). The dark rule adds on top where scores drop in the dark (SCRFD-500M dark:2 3 -> 0, at 4.5%
  area: its dark false positives).
- Not run on the board: SCRFD-500M (12-15 ms desktop CPU) is the one small enough to try on the NPU next.
- EgoBlur (heavy per-frame ceiling, Gen1 face, its default T = 0.9; `experiments/others/egoblur_*.csv`): even the
  ~104M-parameter detector shows the one-frame problem. ChokePoint dark:2: 96.7% frame recall, 76% of people exposed
  (max run 16 frames); lite+back3 -> 7%. Music Video: 92% recall, 92% exposed -> 52% -> 48% with the dark rule, so at
  its default threshold it is not better than SCRFD-10G (36 / 32%). At T = 0.5 (SCRFD/YOLO default; rescored from
  the same cached dets, `egoblur_t05_*.csv`): dark:2 31 -> 7 -> 7, Music Video 68 -> 36 -> 34, i.e. tied with
  SCRFD-10G; even 99.3% frame recall still exposes 31% of the people in dark:2. Mask area 1.1% (clean) / 7.9% (MV) per-frame.
  Model loading: `import torchvision` before `torch.jit.load` (nms op) and load on CPU then `.to(cuda)`.

## Results 2026-11-24: longer look-back, door-reflection zone, dark confirm (RQ2/RQ3)

Motivation: in the ChokePoint demo a face reflected in the door glass is masked late, and on Music Video the dark rule
also masks bodies. Three knobs (`policies.py`): `backN` (look-back N frames instead of 3), `risk_zT_mM` (promote a
low-score box, score >= T, if its centre is in the face zone learnt from the same camera's val GT boxes and its
per-box motion >= M; `src/systems/scene.py`), and `_c` (dark confirm: a dark-rule box needs IoU >= 0.3 with a dark
candidate of the previous frame). NPU INT8 detections, chosen on val dark:2, reported on test; `experiments/round2/`.

Val dark:2, zone alone (no dark rule), % people exposed stock / fine-tuned: lite+back3 ~93 / 97 (desktop run of the
same INT8 model), zone T = 0.1 M = 2: 83 / 90, M = 5: 86 / 93, T = 0.3: 93 / 93. The dark rule alone gives 66 / 83,
and adding the zone to it changes nothing (identical), so the zone is not kept.

Test, % people exposed (max exposed run, frames; % area masked), stock / fine-tuned:

| Policy | ChokePoint clean | ChokePoint dark:2 | Music Video |
|---|---|---|---|
| lite+back3 | 0 / 0 (1.5) | 34 (20) / 72 (18); 1.4 / 1.3 | 76 (53) / 80 (53); 11.4 / 10.8 |
| + dark rule (deployed) | 0 / 0 (1.5) | 10 (4) / 38 (9); 2.1 / 4.9 | 72 (48) / 72 (46); 14.3 / 13.4 |
| + dark rule, back6 | 0 / 0 (1.9) | 7 (4) / 28 (9); 2.7 / 6.1 | 70 (45) / 68 (43); 17.0 / 15.7 |
| + dark rule, back10 | 0 / 0 (2.3) | 7 (4) / 31 (9); 3.4 / 7.2 | 62 (41) / 68 (41); 20.0 / 18.4 |
| + dark rule + confirm | 0 / 0 (1.5) | 10 (4) / 45 (13); 1.7 / 3.8 | 72 (48) / 72 (46); 13.4 / 12.5 |

- Longer look-back helps a little (dark:2 fine-tuned 38 -> 28%, Music Video stock 72 -> 62% at back10) but costs
  area (+25-40%) and latency (6 frames = 240 ms, 10 = 400 ms at 25 FPS). back6 is the reasonable option if the
  delay is acceptable; back3 stays the default.
- Dark confirm removes some of the dark-rule false masks (area 4.9 -> 3.8% fine-tuned dark:2, 14.3 -> 13.4% Music
  Video) at no exposure cost on Music Video and stock, but it loses 2 people with the fine-tuned model on dark:2
  (38 -> 45%). Optional switch for the demo, not the default.
- The door reflection is a limitation: reflections are not in the GT (not scored), and neither the zone nor longer
  look-back is a principled fix. Mention it in the paper as a qualitative failure case.

## Results 2026-11-24: sensor noise (stress tier, NPU INT8 on the board)

`noise:1-3` = Gaussian sensor noise sigma 10 / 20 / 30 at normal brightness (src/degrade/degrade.py), detections on
the i.MX NPU, ChokePoint test; `experiments/noise/`. % people exposed (frame recall), per-frame / lite+back3 /
+ dark rule, stock / fine-tuned:

| noise | stock | fine-tuned |
|---|---|---|
| 1 (sigma 10) | 0 / 0 / 0 (100%) | 0 / 0 / 0 (100%) |
| 2 (sigma 20) | 31 / 3 / 3 (99.6%) | 38 / 0 / 0 (99.5%) |
| 3 (sigma 30) | 100 / 34 / 34 (91.9%) | 100 / 41 / 41 (91.5%) |

- The one-frame effect again: at sigma 20 the detector still finds 99.6% of the faces and a third of the people leak
  per-frame; at sigma 30, 92% recall leaks everybody. Tracking + look-back recovers sigma 20 almost fully and cuts
  sigma 30 to 34-41%.
- The dark rule does nothing here (frames are not dark: it reads brightness, not noise). A noise-aware trigger
  (e.g. high-frequency energy of the frame) would be the analogue; not tried.

## Results 2026-11-24: faces outside the GT (door reflections) found by the attacker

The demo showed faces reflected in the door glass masked late. They are not in the GT, so no metric saw them.
`src/analysis/outside_gt.py` mines them: boxes that EgoBlur (>= 0.5) and SCRFD-10G (>= 0.5) both find, that overlap no
GT box, and that the ArcFace attacker matches (1% FAR threshold, leave-session-out gallery) to a person of the
sequence; "duplicate" = that person has a GT box in the same frame (a reflection), "single" = within 3 s of their
annotation (half-hidden by the door frame, or far down the corridor). ChokePoint test: 7412 boxes, 4666 identifiable
faces (382 reflections of 29 people, in P2E C3 and P2L C1). Checked by eye: clear, recognisable reflections.
Scored like GT faces (coverage >= 0.8, else the judge on the anonymized frame); `experiments/outside_gt/`.

Deployed pipeline (NPU INT8, dark rule + lite + back3), % of the 29 people recognised (GT faces / GT or outside):

| | stock clean | fine-tuned clean | stock dark:2 | fine-tuned dark:2 |
|---|---|---|---|---|
| GT only (the benchmark) | 0 | 0 | 10 | 38 |
| GT or outside the GT | **45** (13 by reflection) | **59** (17 by reflection) | 17 | 52 |

So the benchmark's identity exposure is a lower bound: on clean ChokePoint, where it reports 0%, half the people are
recognisable through a door reflection. EgoBlur per-frame: 27 of 29 people through faces outside the GT.

Fix, the reflection zone (`risk_..._zrT_mM`, systems/scene.py `reflection_zone`): the cells where the attacker found
reflections on VAL (our own YuNet boxes >= 0.1 outside the GT matched to a person in the same frame; 1709 on val,
`experiments/outside_gt/reflections_val.csv`; 0-10% of the frame per camera, on the door glass), where a box with
score >= T is promoted (as the dark rule does in dark frames). A zone blob needs >= 12 cells AND reflections in >= 3
different val frames: one wrong match (P2E_S4_C3.1 frame 4836, a box on a person's dark hair matched to her own face)
had made a 35-cell blob on the floor of P2E C3, visible in the demo; the support rule removes it and the 2-reflection
zone of P2E C2 (rerun 2026-11-24 with the boosts below: every exposure and people count unchanged, area at most 0.05 points
lower, i.e. the tables here are unchanged at two decimals). Test, people
recognised through reflections, stock / fine-tuned, clean:

| policy | through reflections | GT exposure | area masked |
|---|---|---|---|
| dark rule + lite + back3 (deployed) | 13 / 17 | 0 / 0 | 1.52 / 1.52% |
| + reflection zone T = 0.1, no motion gate | **9 / 8** | 0 / 0 | 1.54 / 1.53% |
| + reflection zone T = 0.3 | 10 / 12 | 0 / 0 | 1.53 / 1.52% |
| + reflection zone T = 0.1, motion >= 2 | 9 / 10 | 0 / 0 | 1.53 / 1.53% |

- T = 0.1 halves the people recognised through reflections (fine-tuned 17 -> 8) at no measurable cost (area +0.01
  points, GT exposure unchanged); on dark:2 it changes nothing because the dark rule already applies 0.1 there.
  Candidate for the deployed rule; T chosen by reasoning (= the dark rule's T), not tuned on test.
- What remains (16 faces, 8 people, all P2E C3): 12 have no YuNet box above 0.1 (10 none at all) and 5 lie outside the
  val-learned zone. Neither is fixable by thresholds: it needs a detector that sees reflections (fine-tune on the
  attacker-mined hard examples) or masking the whole glass while something moves on it (privacy-zone masking, ~10%
  area while active).

## Results 2026-11-24: detection boosts (dark-frame enhancement, door zoom)

`src/analysis/boost_dets.py`, simulated with the faithful ORT INT8 models (`yq_board` stock T 0.6, `yqft_board`
fine-tuned T 0.7); every row uses the deployed policy `risk_d60_t0.1+lite+back3`. CSVs: `experiments/boost/`,
reflection leaks: `experiments/outside_gt/boost_*`.
- gam / clahe: on dark frames only (mean grey < 60) the frame is enhanced BEFORE the detector instead of the raw one
  (gamma LUT lifting the mean to 110; CLAHE on LAB L). Same number of inferences.
- zoom: a second inference on the largest blob of the camera's reflection zone (padded 5%), CLAHE on the crop, merged
  with NMS 0.3.

Enhancement chosen on val dark:2 (% people exposed / area masked / max exposed run):

| Model | none | gamma | CLAHE |
|---|---|---|---|
| stock | 66 / 2.35% / 11 | **21 / 2.11% / 3** | 38 / 5.27% / 6 |
| fine-tuned | 83 / 3.97% / 25 | **45 / 4.44% / 7** | 62 / 13.1% / 11 |

Gamma wins on both models. Test dark:2 (GT exposure / people via reflections / area):

| Model | deployed | + gamma | + CLAHE |
|---|---|---|---|
| stock | 10.3% / 5 / 2.51% | **0% / 4 / 2.34%** | 3.4% / 1 / 6.92% |
| fine-tuned | 34.5% / 9 / 5.13% | **3.4% / 3 / 6.28%** | 3.4% / 5 / 14.9% |

Test clean: GT exposure is 0% everywhere (no dark frames, gamma never fires). People recognised through reflections
(of 29) / area:

| Model | deployed | + zone 0.1 | + zoom | + zone 0.1 + zoom |
|---|---|---|---|---|
| stock | 15 / 1.53% | 11 / 1.55% | 14 / 1.54% | **9 / 2.31%** |
| fine-tuned | 15 / 1.51% | 8 / 1.53% | 11 / 1.52% | **7 / 1.54%** |

(The ORT simulation gives 15 / 15 for the deployed row where the board's NPU dets gave 13 / 17; same trend.)
On dark:2 zoom changes nothing. Music Video (% people exposed, 50 people): stock 80 -> 76 with gamma, fine-tuned
76 -> 80 (two people either way; MV exposure comes from shot changes and motion, not darkness). CLAHE is never better.

Take-aways:
- Gamma on dark frames is the largest single gain since the dark rule: test dark:2 stock 10.3% -> 0%, fine-tuned
  34.5% -> 3.4%, max exposed run 10 -> 1 frame. Board cost: one LUT on dark frames only, no extra inference
  (to measure on the NPU). Chosen on val, confirmed on test; neutral on MV.
- Door zoom + zone halves reflection leaks of the fine-tuned model (15 -> 7) for +0.03 area points; it costs a second
  NPU inference per frame (about +12 ms; run it only on frames with motion in the zone).
- Privacy numbers are desktop simulations of the INT8 model. `edge_pipeline.py --gamma` implements the enhancement
  on the board (detector input only; LUT cached per integer mean; matches boost_dets.py within 1 grey level); it runs
  in the capture thread so it overlaps the NPU. Measured on the i.MX (fine-tuned final config, 30 s,
  `experiments/edge/boost/`):

  | Run | FPS per camera | latency p50 / p95 | CPU busy | gamma p50 |
  |---|---|---|---|---|
  | dark clip, 1 camera max, no gamma | 86.5 | 124 / 200 ms | 2.2 cores | - |
  | dark clip, 1 camera max, gamma | 78.5 | 208 / 286 ms | 2.6 cores | 5.0 ms |
  | dark clip, 2 cameras at 25 FPS, gamma | 25.0 | 152 / 167 ms | 1.7 cores | 4.5 ms |
  | dark clip, 3 cameras at 25 FPS, no gamma | 25.0 | 151 / 196 ms | 1.8 cores | - |
  | dark clip, 3 cameras at 25 FPS, gamma | 24.7 | 318 / 708 ms | 2.5 cores | 6.6 ms |
  | clean clip, 3 cameras at 25 FPS, gamma | 25.0 | 153 / 173 ms | 1.9 cores | 1.3 ms |
  | door zone demo (reflection clip, zone 0.1, web display) | 25.0 | 148 / 154 ms | 1.0 core | - |

  Gamma is real time for 1-2 cameras; with 3 cameras that are ALL dark the A53 cores saturate (latency doubles).
  It costs only the mean (1.3 ms) on lit frames. With gamma in the NPU thread instead (first try) one camera fell to
  65 FPS.

## Setup (Mac; Windows: `.venv\Scripts\python`)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt     # supervision must be < 0.27 (ByteTrack)
.venv/bin/python src/loaders/chokepoint_download.py P2E_S1 P2L_S1       # test split only, ~1.3 GB
.venv/bin/python tests/test_core.py && .venv/bin/python tests/test_degrade.py && .venv/bin/python tests/test_policies.py
```

Data outside the repo: `export ECA_DATA=/path/to/data` (the folder that contains `chokepoint/`).

## 1. Board detections (NPU)

Clean detections for every model are on the board in `/root/dets/`. Stress tier on the board (about 30 min per
model and degradation, 23,109 frames; resumable):

```bash
scp src/board_bench.py src/degrade/degrade.py scripts/stress_board.sh imx:/root/
ssh imx 'cd /root && chmod +x stress_board.sh && setsid nohup ./stress_board.sh > stress.log 2>&1 < /dev/null &'
ssh imx 'grep ms/frame /root/stress.log | tail -3'        # progress
ssh imx 'cd /root && tar czf - dets' | tar xzf - -C results/rq1/   # copy back when done
```

Board cache dirs are `dets/<system>@<kind><sev>/`, the same names `benchmark.py --degrade` uses.

## 2. Score policies (desktop, seconds per system once the judge is cached)

```bash
cd src
OUT=../results/rq1   # must contain dets/yunet_480x640_npu etc.
S="yunet_480x640_npu yunet_480x640_npu+skip3+hold yunet_480x640_npu+bytetrack yunet_480x640_npu+window3 \
   yunet_480x640_npu+bytetrack+keep yunet_480x640_npu+bytetrack+keep+coast yunet_480x640_npu+bytetrack+keep+budget"
for t in 0.5 0.6 0.7 0.8 0.9; do
  ../.venv/bin/python benchmark.py --split test --systems $S --thr yunet_480x640_npu=$t --out $OUT
  cp $OUT/results.csv $OUT/results_t$t.csv
done
# stress: the same with --degrade dark:2 (needs dets/yunet_480x640_npu@dark2 from the board); writes results@dark2.csv
../.venv/bin/python benchmark.py --split test --systems $S --thr yunet_480x640_npu=0.6 --degrade dark:2 --out $OUT
```

The first run in a new `--out` embeds the gallery (all 8,957 faces, a few minutes on CPU). A spec takes its
detector's threshold unless `--thr` names the spec itself.

## 3. Set-level attacker

```bash
../.venv/bin/python judge/setlevel.py --out $OUT --systems yunet_480x640_npu yunet_480x640_npu+bytetrack+keep \
    --thr yunet_480x640_npu=0.9 [--degrade dark:2]
```

Prints, per system: % identities exposed by the per-frame judge (must equal benchmark.py's number: a built-in check),
by the mean embedding of the unprotected faces, and by the mean embedding of ALL faces (blurred ones included).

Result 2026-11-21 (ChokePoint test, op thresholds, `experiments/setlevel/`; setlevel.py now passes frame brightness
to the dark rule, before it was silently off there):

| % people exposed, dark2 | per-frame judge (max over frames) | set: mean of unprotected | set: mean of all faces |
|---|---|---|---|
| stock per-frame / window3 / lite+back3 / + dark rule | 100 / 62 / 34 / 10 | 100 / 62 / 31 / 7 | 59 / 28 / 10 / 7 |
| fine-tuned, same order | 97 / 83 / 72 / 38 | 97 / 83 / 66 / 28 | 76 / 48 / 17 / 3 |

Clean: 0% for every system and every attacker. Averaging never makes the attacker stronger: mean embeddings of
weak frames dilute the one good frame, and blurred faces pull the mean away from the person. **The max-over-frames
judge is the strongest of the three, so "one frame is enough" is also the right worst case**; fast_blur does not leak
in aggregate. Same ordering of policies under every attacker.

## 4. Plots and CIs

```bash
../.venv/bin/python analysis/rq1.py --out $OUT --tags yunet_480x640_npu_t0.9 "yunet_480x640_npu+bytetrack+keep_t0.9"
```

## Before results go in the paper

- Thresholds: report at the val-chosen operating thresholds (stock 0.6, fine-tuned 0.7) AND at 0.9; choose any new
  policy knob (budget decay, grow, ages; window size) on val, never on test.
- Exposure-budget and window defaults are knobs, not tuned yet.
