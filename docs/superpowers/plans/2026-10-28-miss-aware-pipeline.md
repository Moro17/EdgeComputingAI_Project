# Miss-Aware Anonymization Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close RQ1 (sanity checks, more data, more edge detectors, length and threshold curves), then build the comparison of the paper: edge baselines (B1 per-frame, B2 ByteTrack, B3 detect-every-N) against our miss-aware method (tracking + coasting + pose fallback), on clean, degraded and harder datasets, with one results table.

**Architecture:** Everything after the detector is a *policy*: a pure function that turns cached detections (`dets/<detector>/<seq>.csv`) into the boxes that get blurred. A system is a spec string `detector+step+step…` (e.g. `yunet_int8+skip3+bytetrack+coast+pose`). The benchmark detects once per detector, then runs any number of policies on the cache, so baselines and ablations cost seconds, not GPU hours. Datasets plug in through one loader interface. Latency of a system = detector ms on keyframes + pose ms where pose ran + policy ms.

**Tech Stack:** Python 3.10+, numpy, pandas, opencv-python, onnxruntime, insightface (judge), `supervision` (ByteTrack), `rtmlib` (YOLOX + RTMPose), `mediapipe` (BlazeFace), matplotlib (report).

**Spec:** [CLAUDE.md](../../../CLAUDE.md) plus decisions from the 2026-10-28 conversation:
- Main question: on the same edge detector and the same board, does our method expose fewer people than B1/B2/B3, at what extra masked area, at 25 FPS?
- Real-time target 25 FPS = 40 ms per frame for the whole pipeline.
- We do not train detectors; the contribution is what happens after the detector, shown on ≥ 2 edge detectors.
- Edge detector candidates: YuNet INT8, SCRFD-500M, YOLOv8n-face, BlazeFace full-range. SCRFD-10G, deface, EgoBlur are reference rows only.
- The proposal PDF in the repo is not binding.

## Global Constraints

- Same blur, mask scale 1.3, protected = coverage ≥ 0.8, judge ArcFace R50 at 1% FAR, for every system.
- A system and its baseline differ only in the policy, never in the detector or detector threshold.
- Unified GT CSV: `frame, track_id, identity, x, y, w, h` (+ `lx, ly, rx, ry` eyes when known). Unified box CSV: `frame, track_id, x, y, w, h, score` (+ `kind`: `det`, `hold`, `coast`, `pose`, `pose_new`).
- Every run writes config + results JSON (`results.json`, already done by `benchmark.py`).
- Existing caches on Drive (`dets/*.csv` + `.json` with `latency_ms` skipping the first 5 frames) keep working: no change to the detection cache format.
- Tests are plain-assert files runnable as `python tests/<file>.py` (match `tests/test_core.py`).
- Run locally with `.venv/bin/python`; GPU runs in Colab from `MyDrive/eca_project`.
- Re-implementations are named "our re-implementation of X". The judge uses GT eye positions: the paper states this is a strong (worst-case) attacker.
- Licences: InsightFace (SCRFD, ArcFace) non-commercial; YOLOv8n-face AGPL-3.0; BlazeFace Apache-2.0; YuNet MIT. All fine for the paper.

## Review Focus

1. A sequence or frame with zero detections: every policy returns an empty DataFrame with the right columns (Tasks 6, 8, 9).
2. Coasted boxes that drift off-frame: dropped when fully outside, clipped otherwise (Task 9 `test_coast_drops_offframe`).
3. `skipN` with N larger than the sequence: only the first frame is a keyframe (Task 6).
4. ByteTrack ids must not leak across sequences: fresh tracker per sequence (Task 8 `test_bytetrack_fresh_per_call`).
5. Cache collisions: different specs, thresholds, degradations or datasets never share a `scores/` dir (Task 7 `test_tags_unique`).

---

## File map

| File | Status | Responsibility |
|---|---|---|
| `src/checks/blur_check.py` | create | does our blur defeat the judge on fully covered faces |
| `src/checks/gt_gaps.py` | create | estimate visible faces missing from eye GT |
| `src/analysis/rq1.py` | create | exposure-vs-length curve, threshold sweep, CIs |
| `src/systems/detectors.py` | modify | add `YoloFace`, `BlazeFace` |
| `src/systems/policies.py` | create | `parse_spec`, `skip`, `hold`, `bytetrack`, `coast`, `run_policy` |
| `src/systems/pose.py` | create | rtmlib wrapper, `head_box`, `pose_fix` |
| `src/degrade/degrade.py` | create | `parse`, `apply` |
| `src/loaders/{plusfiaq,musicvideo,own}.py` | create | extra datasets → unified GT |
| `src/benchmark.py` | modify | spec systems, policy + pose stages, `--degrade`, `--dataset`, lazy/external detectors |
| `src/metrics/exposure.py` | modify | `bootstrap_ci` |
| `src/report.py` | create | merge runs → `experiments/<name>/` |
| `tests/test_*.py` | create | self-checks per task |

---

## Part A: close RQ1

### Task 1: Repo setup and RQ1 results import

**Files:** Create `.gitignore`, `experiments/rq1/`; Modify `CLAUDE.MD` → `CLAUDE.md`, `requirements.txt`

- [ ] **Step 1:** Sync first: if the Drive copy (`MyDrive/eca_project`) has edits, download it and diff against local `src/`. Local and Drive must be identical before Task 2.
- [ ] **Step 2:** Connect to the team repo (`https://github.com/Moro17/EdgeComputingAI_Project.git`, `main` holds only the proposal PDF): `git init -b main && git remote add origin <url> && git pull origin main`, then `git switch -c benchmark`. `.gitignore`: `data/ models/ dist/ .venv/ __pycache__/ *.jit results/`.
- [ ] **Step 3:** Rename `CLAUDE.MD` → `CLAUDE.md` (case-insensitive FS: `mv CLAUDE.MD tmp && mv tmp CLAUDE.md`).
- [ ] **Step 4:** Copy `~/Downloads/eca_results/rq1/` (`judge.json`, `data_config.json`, `dets/`, `scores/`, any `results.*`) into `experiments/rq1/`. Only the small JSON/CSV summaries go into git; `dets/`, `scores/`, `poses/`, `emb/` (biometric) stay local and on Drive (`.gitignore`).
- [ ] **Step 5:** Add `supervision rtmlib mediapipe matplotlib` to `requirements.txt`; install into `.venv` with `uv pip install`.
- [ ] **Step 6:** `.venv/bin/python tests/test_core.py` → all `ok`.
- [ ] **Step 7:** Commit `chore: RQ1 benchmark and results`. Push `benchmark` and open a PR only after Valerian confirms (Moro shares the repo).

---

### Task 2: Sanity checks that protect the RQ1 numbers

**Files:** Create `src/checks/blur_check.py`, `src/checks/gt_gaps.py`, `tests/test_checks.py`

**Interfaces:**
- Consumes: `judge.arcface` (`Judge`, `eye_align`, `templates`), `systems.anonymize.blur/scale_boxes`, `loaders.chokepoint`, the `emb/` + `judge.json` cache in an `--out` dir.
- Produces:
  - `blur_check.py --out DIR [--sample 2000]` → `DIR/checks/blur.json` = `{"n": int, "pct_recognised": float, "threshold": float}`: GT faces blurred over their own 1.3× box, aligned with GT eyes, compared to their template.
  - `gt_gaps.py --out DIR --system scrfd_10g --thr 0.7` → `DIR/checks/gt_gaps.csv` (one row per detection with no GT box at IoU ≥ 0.1: `seq, frame, x, y, w, h, score`) and `DIR/checks/gt_gaps/<i>.jpg` crops for 100 random rows, for manual labelling (face missing from GT / false positive).
  - `unmatched(dets: pd.DataFrame, gt: pd.DataFrame, iou: float = 0.1) -> pd.DataFrame` in `gt_gaps.py`.

- [ ] **Step 1: Failing test** (`tests/test_checks.py`):

```python
def test_unmatched():
    gt = pd.DataFrame({"frame": [1], "x": [100], "y": [100], "w": [50], "h": [50]})
    d = pd.DataFrame({"frame": [1, 1, 2], "x": [105, 400, 100], "y": [100, 400, 100],
                      "w": [50, 40, 50], "h": [50, 40, 50], "score": [.9, .9, .9]})
    out = unmatched(d, gt)
    assert sorted(zip(out.frame, out.x)) == [(1, 400), (2, 100)]
```
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement both scripts. **Step 4:** Run → `ok`.
- [ ] **Step 5:** Run on `experiments/rq1` (blur check on CPU is fine at 2000 samples). Pass criterion: `pct_recognised` ≤ 2 % (≈ FAR). If higher: the blur is too weak; fix the blur in `anonymize.py` (stronger kernel or pixelate-then-blur) **before** any other experiment, and re-score everything.
- [ ] **Step 6:** Label the 100 gap crops by hand (a CSV column `is_face`). Report "estimated visible faces missing from GT = n_gaps × fraction is_face" in `checks/gt_gaps.json`. If > 5 % of GT size, the paper states RQ1 exposure is a lower bound, and PLUSFiaQ (Task 14) becomes high priority.
- [ ] **Step 7:** Commit `feat: blur and GT-gap sanity checks`.

---

### Task 3: More ChokePoint data

**Files:** none new; uses `src/loaders/chokepoint_download.py --all`

- [ ] **Step 1:** In Colab: `python src/loaders/chokepoint_download.py --all --out /content/data/chokepoint` (all portals P1E, P1L, P2E, P2L, sessions S1–S4, plus the face crops). Check: `gt/chokepoint_metadata.csv` regenerated with 72 GT rows, `n_frames_on_disk > 0` for all of them.
- [ ] **Step 2:** Re-run the RQ1 benchmark with all GT sequences into a **new** `--out` (`$OUT/rq1_all`; `data_config.json` refuses to mix). Expected: more identities (ChokePoint has 54 in the video portals), the same direction of results as `rq1`.
- [ ] **Step 3:** Finish the missing rows: `deface` on all sequences, `egoblur` with the gated `.jit`; confirm `device` is CUDA in `results.csv` (else re-run after `check_gpu.py`).
- [ ] **Step 4:** Copy small outputs (`results.*`, `judge.json`, `dets/`) to `experiments/rq1_all/`; commit `exp: RQ1 on all ChokePoint sessions`.

---

### Task 4: Two more edge detectors

**Files:** Modify `src/systems/detectors.py`; Test `tests/test_detectors.py`

**Interfaces:**
- Produces: `YoloFace()` and `BlazeFace()` callables with the same contract as the others (BGR frame → `(n, 5)` float32 `x, y, w, h, score`, scores ≥ `FLOOR`); `SYSTEMS["yolov8n_face"] = (…, 0.5)`, `SYSTEMS["blazeface"] = (…, 0.5)`.

Sources (verified reachable 2026-10-28):
- YOLOv8n-face: `https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.onnx` (WIDER FACE trained, AGPL-3.0). Run with onnxruntime via `ort_providers()`: letterbox to 640, decode, NMS 0.45 with `cv2.dnn.NMSBoxes`. Inspect the output shape first (`[1, 5, 8400]` expected = cx, cy, w, h, conf); if it differs, decode to what it is and note it in the class docstring.
- BlazeFace full-range: `mediapipe` legacy `mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=FLOOR)`; convert relative boxes to pixels. If the legacy API is missing in the installed mediapipe, run `https://storage.googleapis.com/mediapipe-assets/face_detection_full_range.tflite` with `ai-edge-litert` instead and add `# shortcut:` notes for anchor decoding.
- Both download into `models/` like YuNet.

- [ ] **Step 1: Failing test** (`tests/test_detectors.py`, skips gracefully when the model can't be downloaded):

```python
def test_new_detectors_find_the_chokepoint_face():
    from loaders.chokepoint import frame_path, load_gt
    seq = "P2E_S1_C1.1"
    g = load_gt(seq).iloc[len(load_gt(seq)) // 2]          # a mid-track frontal face
    img = cv2.imread(str(frame_path(seq, int(g.frame))))
    for name in ["yolov8n_face", "blazeface"]:
        det = SYSTEMS[name][0](None)
        b = det(img)
        assert b.dtype == np.float32 and b.shape[1] == 5, name
        hit = b[(b[:, 4] >= 0.5) & (abs(b[:, 0] + b[:, 2] / 2 - (g.x + g.w / 2)) < g.w / 2)]
        assert len(hit) >= 1, name
```
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → `ok`.
- [ ] **Step 5:** Add both to the RQ1 runs (`--systems yolov8n_face blazeface` into `$OUT/rq1_all`).
- [ ] **Step 6:** Ask Moro to check TFLite INT8 conversion for these two on the i.MX NPU. The paper's main detectors are the two that convert cleanly and are fastest.
- [ ] **Step 7:** Commit `feat: YOLOv8n-face and BlazeFace detectors`.

---

### Task 5: RQ1 analysis: length curve, threshold sweep, CIs

**Files:** Modify `src/metrics/exposure.py`; Create `src/analysis/rq1.py`; Test `tests/test_core.py`

**Interfaces:**
- Produces:
  - `bootstrap_ci(faces: pd.DataFrame, n: int = 1000, seed: int = 0) -> tuple[float, float]`: 95 % CI of % identities leaked, resampling identities with replacement. `summarise` adds `pct_identities_leaked_lo`, `_hi`.
  - `leak_vs_length(faces: pd.DataFrame, lengths=(1, 5, 10, 25, 50, 100)) -> pd.DataFrame` with columns `length, p_leak, p_independent, n_windows`: for each appearance `(seq, identity)` with ≥ L GT frames, every contiguous window of L frames is leaked if any frame in it is leaked; `p_leak` = mean over windows. `p_independent = 1 - (1 - p_frame)^L` where `p_frame` = overall leaked-frame rate (the naive prediction from frame-level numbers).
  - `rq1.py --out DIR --systems … [--sweep 0.3:0.95:0.05]` → `DIR/analysis/length.csv`, `length.png` (one line per system + dashed independence line), and with `--sweep`: re-scores every system at each threshold through `benchmark.score` (cached per tag) → `sweep.csv`, `recall_vs_exposure.png` (x = frame recall, y = % identities leaked, one curve per detector).

- [ ] **Step 1: Failing tests:**

```python
def test_bootstrap_ci():
    faces = pd.DataFrame({"seq": "s", "identity": [1, 1, 2, 3, 4], "leaked": [True, False, False, False, True]})
    lo, hi = bootstrap_ci(faces)
    assert 0 <= lo <= 50 <= hi <= 100

def test_leak_vs_length():
    faces = pd.DataFrame({"seq": "s", "identity": 1, "frame": range(10),
                          "leaked": [False] * 4 + [True] + [False] * 5})
    r = leak_vs_length(faces, lengths=(1, 5, 10)).set_index("length")
    assert r.loc[1, "p_leak"] == 0.1 and r.loc[10, "p_leak"] == 1.0
    assert abs(r.loc[5, "p_leak"] - 5 / 6) < 1e-9       # windows starting 0..5, five contain frame 4
```
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → `ok`.
- [ ] **Step 5:** Run on `$OUT/rq1_all` with `--sweep`. Expected: `p_leak` climbs toward 1 with length for YuNet; real curve above or below `p_independent` tells whether misses cluster (report either way).
- [ ] **Step 6:** Commit `feat: RQ1 length curve, threshold sweep and CIs`.

**RQ1 is answered when:** headline table on all ChokePoint sessions with CIs (≥ 6 detectors), length curve, recall-vs-exposure curve, stress curve (Task 11), and the two sanity checks pass or are reported.

---

## Part B: baselines and our method (RQ2)

### Task 6: Policy core: spec parser, skip, hold (baseline B3)

**Files:** Create `src/systems/policies.py`, `tests/test_policies.py`

**Interfaces:**
- Produces:
  - `COLS = ["frame", "track_id", "x", "y", "w", "h", "score", "kind"]`
  - `parse_spec(spec: str) -> tuple[str, list[tuple[str, int | None]]]`
  - `keyframes(frames: list[int], n: int) -> list[int]` (= `frames[::n]`)
  - `skip(dets: pd.DataFrame, keys: list[int]) -> pd.DataFrame`
  - `hold(boxes: pd.DataFrame, frames: list[int], keys: list[int]) -> pd.DataFrame`

Spec grammar: first token is the detector; then any of `skip<N>`, `bytetrack`, `hold`, `coast`, `pose`, `posenew`, in that order. `hold` and `coast` exclusive; `coast` requires `bytetrack`; `pose`/`posenew` require `coast`. Violations raise `ValueError`.

- [ ] **Step 1: Failing tests** in `tests/test_policies.py` (same header/runner as `test_core.py`):

```python
def dets(rows):  # (frame, x, y, w, h, score)
    return pd.DataFrame([(f, -1, x, y, w, h, s) for f, x, y, w, h, s in rows],
                        columns=["frame", "track_id", "x", "y", "w", "h", "score"])

def test_parse_spec():
    assert parse_spec("yunet_int8") == ("yunet_int8", [])
    assert parse_spec("yunet_int8+skip3+bytetrack+coast+pose") == (
        "yunet_int8", [("skip", 3), ("bytetrack", None), ("coast", None), ("pose", None)])
    for bad in ["yunet+coast", "yunet+bytetrack+hold+coast", "yunet+pose", "yunet+foo", "yunet+skip0"]:
        try:
            parse_spec(bad)
            assert False, bad
        except ValueError:
            pass

def test_skip_and_hold():
    frames = list(range(10, 17))
    keys = keyframes(frames, 3)
    assert keys == [10, 13, 16]
    d = dets([(f, 100, 100, 50, 50, 0.9) for f in frames])
    out = hold(skip(d, keys), frames, keys)
    assert sorted(out.frame) == frames
    assert (out[out.frame == 11].kind == "hold").all()
    assert (out[out.frame == 13].kind == "det").all()
    assert list(out.columns) == COLS

def test_skip_longer_than_sequence():
    frames = [1, 2, 3]
    keys = keyframes(frames, 10)
    assert keys == [1]
    out = hold(skip(dets([(2, 0, 0, 10, 10, .9)]), keys), frames, keys)
    assert out.empty and list(out.columns) == COLS

def test_empty_dets():
    assert list(hold(skip(dets([]), [1]), [1, 2], [1]).columns) == COLS
```
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement. `skip` keeps rows with `frame in keys`, `kind="det"`. `hold` copies, for each non-key frame, the rows of the latest keyframe ≤ it with `kind="hold"` (`pd.merge_asof` on frame).
- [ ] **Step 4:** Run → `ok`. **Step 5:** Commit `feat: policy spec parser with skip and hold (B3)`.

---

### Task 7: Benchmark runs spec systems on the detection cache

**Files:** Modify `src/benchmark.py`, `src/systems/policies.py`; Test `tests/test_policies.py`

**Interfaces:**
- Consumes: Task 6.
- Produces:
  - `run_policy(steps, dets, frames, thr, shape, poses=None) -> tuple[pd.DataFrame, set[int]]`: boxes (`COLS`) and the frames where pose ran. Without `bytetrack`, dets are filtered `score >= thr` first; `bytetrack` receives dets with `score >= 0.1` and uses `thr` as its high threshold.
  - `tag(spec: str, thr: float, deg: str | None) -> str` = `f"{spec}_t{thr}"` + `f"@{deg}"` when degraded.
  - `frame_latency(frames, keys, det_lat, pose_lat, pose_frames, policy_ms) -> list[float]`.
  - `score(...)` takes the boxes DataFrame instead of raw dets + threshold.

Changes to `benchmark.py`:
- `--systems` accepts spec strings (validated by `parse_spec`; the detector must be in `SYSTEMS` **or** have a complete `dets/<detector>/` cache: this is how board detections come in, Task 13).
- Detect stage loops over the unique detectors; detectors are instantiated only on a cache miss.
- Latency: `det_lat = dict(zip(frames[5:], latency_ms))` (cache format unchanged). Per frame = det ms if keyframe + pose ms if frame in pose set + `policy_ms / len(frames)`; frames without det ms (warm-up) skipped. `fps`, `p50_ms`, `p95_ms` from this list.
- `results.csv` gains `detector`, `policy`, `n` (1 if no skip), `degrade`, `dataset`.
- `device` from the detector, or from `"device"` in the dets JSON for external detectors.

- [ ] **Step 1: Failing tests:**

```python
def test_run_policy_b1_filters_threshold():
    d = dets([(1, 0, 0, 10, 10, 0.95), (1, 50, 50, 10, 10, 0.3)])
    out, pose_frames = run_policy([], d, [1], thr=0.9, shape=(600, 800))
    assert len(out) == 1 and pose_frames == set()

def test_tags_unique():
    from benchmark import tag
    tags = {tag("yunet_int8", 0.9, None), tag("yunet_int8+skip3+hold", 0.9, None),
            tag("yunet_int8", 0.8, None), tag("yunet_int8", 0.9, "dark:2")}
    assert len(tags) == 4

def test_frame_latency():
    from benchmark import frame_latency
    frames = list(range(10))
    det_lat = {f: 30.0 for f in frames[5:]}
    lat = frame_latency(frames, keyframes(frames, 2), det_lat, {}, set(), policy_ms=10.0)
    assert lat == [1.0, 31.0, 1.0, 31.0, 1.0]  # frames 5..9 (warm-up skipped); keyframes 6, 8 add detector ms
```
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement. `bytetrack`/`coast`/`pose` raise `NotImplementedError` until their tasks.
- [ ] **Step 4:** Run → `ok`. Regression: `.venv/bin/python src/benchmark.py --systems yunet yunet_int8 --out experiments/rq1` reproduces the RQ1 rows (yunet: recall 0.956, 86.2 % identities leaked).
- [ ] **Step 5:** First B3 numbers: `--systems yunet_int8 yunet_int8+skip{2,3,5}+hold scrfd_500m scrfd_500m+skip{2,3,5}+hold --out experiments/rq1`. Expected: leaks and fps both rise with N.
- [ ] **Step 6:** Commit `feat: benchmark scores spec systems from the detection cache (B1, B3)`.

---

### Task 8: ByteTrack step (baseline B2)

**Files:** Modify `src/systems/policies.py`; Test `tests/test_policies.py`

**Interfaces:** Produces `bytetrack(dets, keys, thr) -> pd.DataFrame` with integer `track_id`, `kind="det"`.

`supervision.ByteTrack(track_activation_threshold=thr, lost_track_buffer=30, minimum_matching_threshold=0.8, frame_rate=30, minimum_consecutive_frames=1)`, a **new instance per call**, `update_with_detections` once per keyframe (`sv.Detections.empty()` when none). Label: "ByteTrack (supervision implementation)".

- [ ] **Step 1: Failing tests:**

```python
def test_bytetrack_keeps_identity_and_low_scores():
    rows = [(f, 100 + 5 * f, 100, 50, 50, 0.95) for f in range(6)]
    rows[3] = (3, 115, 100, 50, 50, 0.4)            # low-score detection B1 at 0.9 would drop
    out = bytetrack(dets(rows), list(range(6)), thr=0.9)
    assert set(out.frame) == set(range(6)) and out.track_id.nunique() == 1

def test_bytetrack_fresh_per_call():
    d = dets([(0, 100, 100, 50, 50, 0.95)])
    assert bytetrack(d, [0], 0.9).track_id.tolist() == bytetrack(d, [0], 0.9).track_id.tolist()

def test_bytetrack_empty():
    assert list(bytetrack(dets([]), [0, 1], 0.9).columns) == COLS
```
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement; wire into `run_policy`. **Step 4:** Run → `ok`; `--systems yunet_int8+bytetrack scrfd_500m+bytetrack yunet_int8+skip3+bytetrack+hold`. Expected: `yunet_int8+bytetrack` leaks ≤ `yunet_int8`.
- [ ] **Step 5:** Commit `feat: ByteTrack policy (B2)`.

---

### Task 9: Coasting (our method v1)

**Files:** Modify `src/systems/policies.py`; Test `tests/test_policies.py`

**Interfaces:** Produces `coast(tracks, frames, shape, max_age=15, grow=1.25) -> pd.DataFrame` (input rows plus `kind="coast"` rows).

Algorithm (pinned):
```
for each track_id, observed frames sorted:
    for each frame f in `frames` after an observation where this track has no row:
        k = index distance (in `frames`) from the last observation; stop when k > max_age or the track is observed again
        v = (centre_last - centre_prev) / index distance between them   (0 if only one observation)
        centre = centre_last + v * k;  w, h = last w, h * grow
        drop the box if fully outside shape (H, W); otherwise emit it (scale_boxes clips)
```
`max_age=15` ≈ 0.6 s at 25 FPS. Both values are ablation knobs (Task 15), not tuned on the test set.

- [ ] **Step 1: Failing tests:**

```python
def tracks(rows):  # (frame, tid, x, y, w, h)
    return pd.DataFrame([(f, t, x, y, w, h, .9, "det") for f, t, x, y, w, h in rows], columns=COLS)

def test_coast_fills_gap_with_velocity():
    t = tracks([(0, 1, 100, 100, 40, 40), (1, 1, 110, 100, 40, 40), (4, 1, 140, 100, 40, 40)])
    out = coast(t, list(range(5)), (600, 800))
    g = out[out.frame == 2].iloc[0]
    assert g.kind == "coast" and abs((g.x + g.w / 2) - 140) < 1e-6 and abs(g.w - 50) < 1e-6
    assert len(out[out.frame == 4]) == 1

def test_coast_stops_after_max_age():
    out = coast(tracks([(0, 1, 100, 100, 40, 40)]), list(range(30)), (600, 800), max_age=15)
    assert out.frame.max() == 15

def test_coast_drops_offframe():
    t = tracks([(0, 1, 700, 100, 40, 40), (1, 1, 760, 100, 40, 40)])
    assert (coast(t, list(range(10)), (600, 800)).x < 800).all()
```
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement; wire into `run_policy`. **Step 4:** Run → `ok`; `--systems yunet_int8+bytetrack+coast yunet_int8+skip3+bytetrack+coast scrfd_500m+bytetrack+coast`. Expected: fewer identities leaked than the matching B2 row, slightly higher `pct_area_masked`.
- [ ] **Step 5:** Commit `feat: track coasting policy (ours v1)`.

---

### Task 10: Pose fallback (our method v2)

**Before starting:** Valerian reads SecurePose (2026) and Foresee and Protect You (2026). If either is reproducible in a few days, add it as a baseline row ("our re-implementation of X"); otherwise write down how ours differs.

**Files:** Create `src/systems/pose.py`, `tests/test_pose.py`; Modify `src/benchmark.py`, `src/systems/policies.py`

**Interfaces:**
- Produces:
  - `class Pose: __call__(bgr) -> np.ndarray` `(m, 17, 3)` COCO x, y, conf; wraps `rtmlib.Body(mode="lightweight", backend="onnxruntime", device="cuda" if available else "cpu")`.
  - `head_box(kps, min_conf=0.3) -> tuple[float, float, float, float] | None`
  - `pose_fix(boxes, poses: dict[int, np.ndarray], keys, new: bool) -> tuple[pd.DataFrame, set[int]]`
  - Cache `poses/<seq>.npz` (`frames`, `counts`, `kps` concatenated) + `poses/<seq>.json` latency, built for all frames once (first 5 frames' latency skipped).

`head_box` (pinned): COCO 0–4 (nose, eyes, ears) with conf ≥ `min_conf`; need ≥ 2 else `None`. Centre = nose if visible else mean. `span` = x-range of visible head points; `sw` = |x5 − x6| if both shoulders visible else 0. `w = max(1.6 * span, 0.5 * sw)`, `h = 1.35 * w`, top = cy − 0.57·h. `None` if `w < 8`.

`pose_fix`: each `coast` row whose box contains a person's head-box centre is replaced by that head box, `kind="pose"`; pose ran on those frames. If `new`: on every keyframe, each person whose head-box centre is inside no output box is added with `track_id=-1, kind="pose_new"`; pose ran on every keyframe.

- [ ] **Step 1: Failing tests** (`tests/test_pose.py`):

```python
def person(eyes=((100, 100), (140, 100)), nose=(120, 124), ears=((90, 105), (150, 105)), conf=0.9):
    k = np.zeros((17, 3))
    for i, p in enumerate([nose, *eyes, *ears]):
        k[i] = (*p, conf)
    return k

def test_head_box_covers_face():
    x, y, w, h = head_box(person())
    assert x < 100 and x + w > 140 and y < 100 < 124 < y + h and 70 <= w <= 110

def test_head_box_needs_two_points():
    assert head_box(person(conf=0.1)) is None

def test_pose_fix_replaces_coast_and_adds_new():
    boxes = pd.DataFrame([(5, 1, 80, 70, 90, 90, .9, "coast")], columns=COLS)
    poses = {5: np.stack([person(), person(eyes=((400, 300), (440, 300)), nose=(420, 324),
                                           ears=((390, 305), (450, 305)))])}
    out, ran = pose_fix(boxes, poses, keys=[5], new=True)
    assert set(out.kind) == {"pose", "pose_new"} and ran == {5}
    out, _ = pose_fix(boxes, poses, keys=[5], new=False)
    assert set(out.kind) == {"pose"}
```
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement `pose.py`, the pose cache stage (only when a spec contains `pose`/`posenew`), `run_policy` dispatch. **Step 4:** Run → `ok`; Colab: `--systems yunet_int8+bytetrack+coast+pose yunet_int8+bytetrack+coast+posenew`. Expected: `posenew` lowers YuNet INT8 below 100 % identities leaked; check `pct_area_masked` and fps cost.
- [ ] **Step 5:** Commit `feat: pose fallback for coasted and missed faces (ours v2)`.

---

## Part C: harder conditions and data

### Task 11: Stress tier: degradations

**Files:** Create `src/degrade/degrade.py`, `tests/test_degrade.py`; Modify `src/benchmark.py`

**Interfaces:** `parse(s) -> tuple[str, int]` (`"dark:2"` → `("dark", 2)`; kinds `dark, blur, down, occl`, severity 1–3, else `ValueError`); `apply(img, kind, sev, seed) -> np.ndarray` (same shape/dtype); benchmark `--degrade KIND:SEV`.

| kind | 1 | 2 | 3 |
|---|---|---|---|
| dark: `255·(img/255)^γ` + Gaussian noise σ | γ 2, σ 3 | γ 3, σ 6 | γ 4, σ 10 |
| blur: horizontal motion kernel length | 7 | 15 | 25 |
| down: INTER_AREA down by f, INTER_LINEAR back up | 2 | 3 | 4 |
| occl: grey rectangles, 10 % of frame side each, random position | 2 | 4 | 8 |

`seed` = frame number. Enrollment and judge threshold always use **clean** frames. Degraded frames are made on the fly by `load_frame(seq, n, deg)` used by detect, pose and score. Dets cache dir: `dets/<detector>@<kind><sev>/`.

- [ ] **Step 1: Failing test:**

```python
def test_degrade():
    img = np.full((60, 80, 3), 128, np.uint8)
    img[20:40, 30:50] = 255
    assert parse("dark:2") == ("dark", 2)
    for bad in ["dark:0", "dark:4", "fog:1", "dark"]:
        try:
            parse(bad); assert False, bad
        except ValueError:
            pass
    for kind in ["dark", "blur", "down", "occl"]:
        out = apply(img, kind, 2, seed=7)
        assert out.shape == img.shape and out.dtype == np.uint8 and not np.array_equal(out, img)
        assert np.array_equal(out, apply(img, kind, 2, seed=7))
    assert apply(img, "dark", 3, 0).mean() < apply(img, "dark", 1, 0).mean()
```
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → `ok`; eyeball one frame per kind/severity in the notebook.
- [ ] **Step 5:** RQ1 stress curve: B1 for all detectors at every kind/severity → `rq1.py` adds `stress.png` (x = severity, y = % identities leaked, one line per detector, one panel per kind).
- [ ] **Step 6:** Commit `feat: synthetic degradations for the stress tier`.

---

### Task 12: Report

**Files:** Create `src/report.py`

`python src/report.py --runs OUT [OUT ...] --name NAME` → `experiments/NAME/results.csv` (all rows + `run` column), `table.md` (detector, policy, n, degrade, dataset, frame_recall, pct_identities_leaked [lo–hi], max_exposed_run, pct_area_masked, fps, p95_ms, device), `pareto.png` (x = % area masked, y = % identities leaked; colour per policy family, marker per detector; fps ≥ 25 filled, else hollow).

- [ ] **Step 1:** Implement; run `--runs experiments/rq1 --name rq1` → three files exist, `table.md` rows equal `results.csv` rows.
- [ ] **Step 2:** Commit `feat: report table and Pareto plot`.

---

### Task 13: Board detections plug-in (contract with Moro)

**Files:** Modify `CLAUDE.md` (contract)

Contract:
- Moro runs the INT8 detector on the board over the same frames and writes per sequence `dets/<name>/<seq>.csv` (`frame,track_id,x,y,w,h,score`, all detections ≥ 0.05, `track_id=-1`) and `dets/<name>/<seq>.json` = `{"latency_ms": [...], "device": "imx8mp-npu"}`, first 5 frames excluded.
- Names: `<detector>_npu` (e.g. `yunet_int8_npu`); thresholds as the desktop counterpart via `--thr`.
- Moro also reports the board ms of: ByteTrack update, coasting, blur of an 800×600 frame, RTMPose-t per call. Policy latency in the paper table = these board numbers.

- [ ] **Step 1:** Fake test: copy `dets/yunet_int8/` → `dets/yunet_int8_npu/`, add `"device": "imx8mp-npu"`, run `--systems yunet_int8_npu yunet_int8_npu+bytetrack+coast --thr yunet_int8_npu=0.9`. Expected: same privacy numbers as `yunet_int8`, `device` = `imx8mp-npu`, no detector instantiated.
- [ ] **Step 2:** Delete the fake; commit `docs: board detection contract`.

---

### Task 14: Extra datasets

**Files:** Create `src/loaders/plusfiaq.py`, `src/loaders/musicvideo.py`, `src/loaders/own.py`; Modify `src/benchmark.py` (`--dataset`), `src/judge/arcface.py` (box-only alignment); Test `tests/test_loaders.py`

**Interfaces:**
- Every loader module exposes `sequences() -> list[str]`, `frame_numbers(seq)`, `frame_path(seq, n)`, `load_gt(seq) -> DataFrame` (unified GT; eyes `lx…ry` NaN when unknown), `session(seq) -> str` (the leave-out group for enrollment). `loaders.chokepoint` gains `sequences()`. `benchmark.py --dataset NAME` imports `loaders.NAME`; `dataset` goes into `data_config.json` and the tag.
- Box-only GT: eyes filled once per dataset by `fill_eyes(seq)`: SCRFD-10G keypoints on the **clean** frame, matched to the GT box by IoU ≥ 0.5; rows still without eyes are aligned by resizing the GT box crop to 112×112 and flagged `aligned=False` (reported separately).

Datasets, in priority order:
1. **PLUSFiaQ** (ChokePoint S5 crowd boxes): blocked on the licence email to robert.joechl@plus.ac.at. Identities = ChokePoint ids; `session` groups by portal/session so S5 is judged against S1–S4 templates.
2. **Own recording** with the board camera: written consent from everyone filmed; GT boxes from SCRFD-10G + manual correction in CVAT, identities labelled by hand; gallery from separate enrollment clips of each person. Small (3–5 people, 2–3 clips each) but shows the real deployment.
3. **Music Video dataset** (Zhang et al.): first verify a working download link (none found 2026-10-28); `session` = video, enrollment from held-out frames of other shots.

- [ ] **Step 1: Failing test** with a 3-frame fixture per loader (`tests/fixtures/<dataset>/`): `load_gt` columns equal the unified list and `session` differs across two fixture sequences.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement whichever loaders have data. **Step 4:** Run → `ok`; RQ1 B1 rows on each new dataset.
- [ ] **Step 5:** Commit per loader, `feat: <dataset> loader`.

---

### Task 15: Experiment matrix and paper table

**Files:** Modify `notebooks/colab_benchmark.ipynb` (one cell per block), rebuild `dist/eca_benchmark.zip`

Main detectors `D` = the two best of {yunet_int8, scrfd_500m, yolov8n_face, blazeface} by board latency and NPU conversion (Task 4 Step 6); `N ∈ {1, 2, 3, 5}`:

| Family | Spec |
|---|---|
| B1 / B3 | `D[+skipN]+hold` (N = 1: just `D`) |
| B2 | `D[+skipN]+bytetrack+hold` |
| Ours v1 | `D[+skipN]+bytetrack+coast` |
| Ours v2 | `D[+skipN]+bytetrack+coast+pose`, `…+posenew` |
| Prior work | re-implementation from Task 10, if any |
| Reference | `scrfd_10g`, `deface`, `egoblur` (N = 1) |

- [ ] **Step 1:** Clean, all ChokePoint → `$OUT/main`. Expected: ours below B2 at equal N on the Pareto plot.
- [ ] **Step 2:** Stress: same rows minus reference with `--degrade` ∈ `dark:{1,2,3}`, `blur:{1,2,3}`, `down:{1,2,3}`, `occl:2`. Expected: the gap between ours and B2 grows with severity.
- [ ] **Step 3:** Each extra dataset from Task 14: same rows.
- [ ] **Step 4:** Ablation (clean): `max_age ∈ {5, 15, 30}`, `grow ∈ {1.0, 1.25, 1.5}` as spec params `coast15g125`; extend `parse_spec` with a test first.
- [ ] **Step 5:** `report.py` per block (`rq2_main`, `rq2_stress`, `rq2_<dataset>`, `ablation`).
- [ ] **Step 6:** Replace desktop fps with board numbers once Task 13 data arrives; the headline table uses board fps only.
- [ ] **Step 7:** Commit `exp: RQ2 main, stress, datasets and ablation results`.

---

## Moro's track (parallel)

1. TFLite INT8 conversion for the i.MX 8M Plus NPU of YuNet, SCRFD-500M, YOLOv8n-face, BlazeFace; report which convert cleanly and their ms per 800×600 frame.
2. Pick N so that detector ms / N + tracking + coasting + blur (+ pose when it runs) ≤ 40 ms.
3. Deliver detections and timings per the Task 13 contract.
4. End-to-end demo: camera in, blurred video out, measured FPS; record the own-recording clips (Task 14) with the same camera.

## Not in this plan

- Utility metric ("is the blurred video still useful": e.g. person-detector recall on anonymized frames). Proposed, not yet agreed.
- RQ3 (synthetic face replacement vs masking): go/no-go after Task 15.
