# Deliverable 1: Fine-Tuned YuNet on the Board Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** By 2026-11-03 16:00, a YuNet fine-tuned on a ChokePoint train split runs INT8 on the i.MX 8M Plus NPU in a live video demo on HDMI, benchmarked against stock YuNet and YOLOv8n-face on the held-out test split, with code on GitHub and the dataset documented.

**Architecture:** Reuse the existing harness: detections go to the det cache (`experiments/<run>/dets/<system>/<seq>.csv`), and `src/benchmark.py` scores them. Board systems run through `src/board_bench.py` and are scored on the Mac as `BoardOnly` entries. Fine-tuning uses libfacedetection.train (BSD-3, no mmdet, `--variant yunet_n`, released `weights/yunet_n.pth`) on Colab. One new export script turns any YuNet ONNX into a static INT8 TFLite at a given size, so stock and fine-tuned models take the same path to the NPU.

**Tech Stack:** Python 3.10+ (`.venv`: numpy, pandas, opencv-python, onnxruntime, insightface), `.venv-export` (Python 3.12: onnx2tf, tensorflow, ultralytics), board (tflite_runtime 2.14, OpenCV 4.8.1 with GStreamer, numpy, VX delegate `/usr/lib/libvx_delegate.so`), Colab GPU (PyTorch, libfacedetection.train).

**Spec:** [docs/plan.md](../../plan.md) Phases 1-5 and the Phase 9 demo; reasons for YuNet in [docs/decisions/why-not-yolo320.md](../../decisions/why-not-yolo320.md); project rules in [CLAUDE.md](../../../CLAUDE.md).

## Global Constraints

- Real-time target: 25 FPS on the i.MX 8M Plus, i.e. 40 ms per frame for the whole pipeline (decode + detect + track/recovery + blur).
- Every reported number is on the test split: ChokePoint P2 S1, the 12 sequences in `experiments/board/data_config.json`. Never train, calibrate or pick thresholds on them.
- Same judge (ArcFace R50 `buffalo_l`, 1% FAR), same strong blur, mask scale 1.3, coverage >= 80% for every system.
- INT8 calibration uses only the S5 frames (`systems.quantize.CALIB_SEQS`, `CALIB_FRAMES = 200`).
- Board code imports only tflite_runtime, cv2, numpy (no pandas); `src/systems/anonymize.py` is the one repo module allowed on the board.
- Commits: branch `valerian/benchmark`, author `valerian.ccn@gmail.com`. Never commit `data/`, `models/`, `dets/`, `scores*/`, `emb/`, face crops, `.tflite`, `.pt`, `.pth`, `.onnx`, training images or labels built from them. Ask before pushing to the team repo.
- Tests are plain scripts: `.venv/bin/python tests/<file>.py` prints `ok <test>` per test (no pytest installed).
- A shortcut with a known limit gets a code comment: `shortcut: <the limit>, <when to upgrade>`.

## Review Focus

1. Fine-tuned ONNX export with different output names or order than the 2023mar model: the decoder silently reads the wrong tensors and finds no faces. Task 1's test checks names and that TFLite matches ORT on a real frame.
2. A train/val/test sequence leaking across splits, or S5 entering train: inflated results. Task 3's test asserts the splits are disjoint and test equals the cached 12.
3. Train frames with people the ChokePoint GT does not label: those faces become negatives and teach YuNet to miss faces. Task 4 drops such frames and its test pins the rule.
4. Landmarks missing for a GT face (SCRFD did not match): must be written as `-1` and not as zeros at the image corner. Task 4's test pins it.
5. INT8 score saturation (all faces scoring one value, as with YOLO's 0.4986): makes threshold choice meaningless. Task 1 asserts more than 10 distinct scores above the floor on a real frame.

---

## File map

| File | Responsibility |
|---|---|
| `src/export_yunet_tflite.py` (new) | any YuNet ONNX + input size -> static INT8 TFLite via onnx2tf, calibrated on S5 |
| `src/splits.py` (new) | ChokePoint train/val/test/calib sequence lists |
| `src/train_labels.py` (new) | ChokePoint split -> labelv2 file (boxes + 5 landmarks) for libfacedetection.train |
| `notebooks/finetune_yunet.ipynb` (new) | Colab: data, fine-tune, export ONNX, val selection |
| `src/demo.py` (new, runs on board) | video -> NPU detector -> blur -> HDMI with FPS overlay and model toggle |
| `src/systems/detectors.py` | `YuNetORT(path=...)` for fine-tuned ONNX; new SYSTEMS entries |
| `src/benchmark.py` | `--split` selects sequences from `splits.py` |
| `src/board_bench.py` | unchanged except the YuNet class is imported by `demo.py` |
| `README.md` | dataset section, how to reproduce, results table |

---

### Task 0: Commit the pending work

**Files:** everything in `git status` except data and biometrics.

- [ ] **Step 1:** Check `.gitignore` covers `experiments/*/dets/`, `experiments/*/emb/`, `experiments/*/scores*/`, `models/`, `.venv-*/`. Add any missing line.
- [ ] **Step 2:** Run `.venv/bin/python tests/test_core.py && .venv/bin/python tests/test_loaders.py`. Expected: every line `ok ...`.
- [ ] **Step 3:** Commit in two commits: `feat: board benchmark (INT8 TFLite on the i.MX 8M Plus NPU) and TFLite export` (board_bench, export_tflite, detectors BoardOnly, experiments/board results + configs) and `feat: Music Video loader and dataset-agnostic benchmark` (musicvideo, benchmark, arcface, chokepoint, test_loaders), then `docs: project plan, YOLO-320 decision, 25 FPS target` (docs/, CLAUDE.md). Check `git show --stat HEAD~2..HEAD` lists no file under `emb/`, `dets/`, `scores`.

### Task 1: YuNet ONNX -> INT8 TFLite export script

Automates the conversion done by hand for `models/yunet_608x800_int8.tflite`, so the timing sweep (Task 2) and the fine-tuned model (Task 6) use one path.

**Files:**
- Create: `src/export_yunet_tflite.py`
- Test: `tests/test_export.py`

**Interfaces:**
- Consumes: `systems.quantize.calibration_frames(n)`; `board_bench.YuNetTFLite` (decoder under test) and `systems.detectors.YuNetORT` (reference). The test env `.venv-export` needs onnxruntime and ai-edge-litert installed.
- Produces: CLI `.venv-export/bin/python src/export_yunet_tflite.py SRC.onnx --size H W [--out PATH]` -> `models/<src stem>_<H>x<W>_int8.tflite` (default out). Function `static_onnx(src: Path, h: int, w: int, dst: Path) -> Path`.

- [ ] **Step 1: Write the failing test** `tests/test_export.py`:

```python
def test_static_onnx_fixes_shape_and_keeps_output_names():
    dst = static_onnx(MODELS / "face_detection_yunet_2023mar.onnx", 320, 416, TMP / "y.onnx")
    m = onnx.load(str(dst))
    assert [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim] == [1, 3, 320, 416]
    assert [o.name for o in m.graph.output] == [f"{k}_{s}" for k in ("cls", "obj", "bbox", "kps") for s in (8, 16, 32)]

def test_int8_tflite_matches_ort():  # runs only if models/yunet_320x416_int8.tflite exists (built in Step 3)
    g, img = sample()  # tests/test_detectors.sample: mid-track frontal face on P2E_S1_C1.1
    small = cv2.resize(img, (416, 312))
    ref = SYSTEMS["yunet_ort"][0](None)(small); ref = ref[ref[:, 4] >= 0.9]
    got, _ = YuNetTFLite(MODELS / "yunet_320x416_int8.tflite", npu=False)(small)
    assert len(got[got[:, 4] >= 0.9]) == len(ref)
    assert np.abs(got[got[:, 4] >= 0.9][:, :4] - ref[:, :4]).max() < 3
    assert len(np.unique(got[got[:, 4] >= 0.05][:, 4].round(3))) > 10  # score not saturated
```

- [ ] **Step 2:** Run `.venv-export/bin/python tests/test_export.py`. Expected: FAIL, `static_onnx` not defined.
- [ ] **Step 3: Implement** `static_onnx` (set the 4 input dims, drop `value_info`, run `onnx.shape_inference`) and `main`: write `calibration_frames(100)` resized/padded top-left to H×W as float32 BGR NHWC `.npy`, run `onnx2tf -i <static> -o <tmp> -oiqt -qt per-tensor -cind input <npy> "[[[[0,0,0]]]]" "[[[[1,1,1]]]]"` from a temp dir holding a copy of `calibration_image_sample_data_20x128x128x3_float32.npy` (onnx2tf loads it from the CWD), copy `*_integer_quant.tflite` to the output. Build `models/yunet_320x416_int8.tflite` from the stock ONNX.
- [ ] **Step 4:** Run the test again. Expected: `ok` for both. Also rebuild 608×800 and check it matches the hand-built `models/yunet_608x800_int8.tflite` within 1 px on the sample frame.
- [ ] **Step 5:** Commit `feat: YuNet ONNX -> static INT8 TFLite export`.

### Task 2: Board timing sweep and input-size decision

**Files:**
- Modify: `src/systems/detectors.py` (SYSTEMS entries)
- Modify: `docs/plan.md` (Phase 1 outcome), `docs/decisions/why-not-yolo320.md` (replace estimates)

**Interfaces:**
- Consumes: Task 1 CLI; `board_bench.py`; `src/export_tflite.py --imgsz 256`.
- Produces: SYSTEMS entries `yunet_<H>x<W>_npu` (threshold 0.9) for each size and `yolov8n_face_256_npu` (0.49); the chosen size `H×W`, written in `docs/plan.md`, used by Tasks 5-7.

- [ ] **Step 1:** Build INT8 TFLite for YuNet at 480×640, 384×512, 320×416 (Task 1) and YOLO at 256 (`export_tflite.py --imgsz 256`). `scp` them to `imx:/root/`.
- [ ] **Step 2:** On the board, `benchmark_model --graph=<m> --external_delegate_path=/usr/lib/libvx_delegate.so` for each. Record NPU-only ms.
- [ ] **Step 3:** On the board, `python3 board_bench.py --model <m> --frames /root/chokepoint --out /root/dets` for each; copy `dets/<name>/` to `experiments/board/dets/`.
- [ ] **Step 4:** Measure the rest of the frame on the board, 300 frames each: (a) hardware decode of an H.264 800×600 video, `cv2.VideoCapture("filesrc location=test.mp4 ! qtdemux ! h264parse ! vpudec ! videoconvert ! video/x-raw,format=BGR ! appsink", cv2.CAP_GSTREAMER)` (make `test.mp4` from `P2E_S1_C1.1` frames with ffmpeg at 30 fps); (b) `anonymize.blur` on 2 boxes. Record ms per frame.
- [ ] **Step 5:** Add the SYSTEMS entries; run `cd src && ../.venv/bin/python benchmark.py --systems <all new + existing board systems> --out ../experiments/board`. Expected: one results row per system.
- [ ] **Step 6: Decide.** Detector budget = 40 ms − decode − blur − 2 ms (tracker reserve). Chosen size = the largest YuNet size whose p50 detector ms fits that budget. Write the table (size, NPU ms, detector ms, % identities exposed) and the choice into `docs/plan.md` Phase 1; replace the "estimate" lines in the decision doc with the measured decode and blur ms. If `yolov8n_face_256_npu` fits the budget and exposes 0%, stop and tell Valerian before continuing (the decision doc's reversal condition).
- [ ] **Step 7:** Commit `feat: board input-size sweep; choose YuNet <H>x<W>`.

### Task 3: ChokePoint split

**Files:**
- Create: `src/splits.py`
- Modify: `src/benchmark.py` (`--split`)
- Test: `tests/test_loaders.py`

**Interfaces:**
- Produces: `TRAIN, VAL, TEST, CALIB: list[str]` and `split_of(seq: str) -> str` (`"train" | "val" | "test" | "calib"`, `KeyError` otherwise) in `src/splits.py`; benchmark `--split {train,val,test}` (default sequences unchanged when absent; `--seqs` and `--split` are mutually exclusive).
- Values: TRAIN = P1E/P1L × S1-S4 × C1-C3 (24, names like `P1E_S1_C1`); VAL = P2E/P2L × S2-S4 × C1-C3 × .1/.2 (36); TEST = P2E/P2L × S1 × C1-C3 × .1/.2 (12); CALIB = `quantize.CALIB_SEQS`.

- [ ] **Step 1: Write the failing test** in `tests/test_loaders.py`:

```python
def test_splits_are_disjoint_and_test_is_the_cached_benchmark():
    sets = [set(s) for s in (splits.TRAIN, splits.VAL, splits.TEST, splits.CALIB)]
    assert sum(map(len, sets)) == len(set().union(*sets))
    assert splits.TEST == json.loads((ROOT / "experiments/board/data_config.json").read_text())["seqs"]
    assert (len(splits.TRAIN), len(splits.VAL), len(splits.TEST)) == (24, 36, 12)
    assert all(s.startswith("P1") for s in splits.TRAIN) and splits.split_of("P2E_S5_C1.1") == "calib"
```

- [ ] **Step 2:** Run `.venv/bin/python tests/test_loaders.py`. Expected: FAIL, no module `splits`.
- [ ] **Step 3: Implement** `src/splits.py` and `--split` in `benchmark.py`. Check every listed name has a GT xml in `data/chokepoint/groundtruth/` (fix the lists if ChokePoint names differ, e.g. P1 has no `.1/.2`).
- [ ] **Step 4:** Run the tests. Expected: all `ok`.
- [ ] **Step 5:** Download: `.venv/bin/python src/loaders/chokepoint_download.py P1E P1L P2E_S2 P2E_S3 P2E_S4 P2L_S2 P2L_S3 P2L_S4` (check the archive names against `chokepoint_meta.zenodo_files()` first and the free disk space). Check `ls data/chokepoint/frames | wc -l` covers all TRAIN + VAL.
- [ ] **Step 6:** Re-score test: `cd src && ../.venv/bin/python benchmark.py --split test --systems <board + Mac systems> --out ../experiments/board`. Expected: `results.csv` identical to before (the detections are cached and test is the same 12 sequences).
- [ ] **Step 7:** Commit `feat: ChokePoint train/val/test split (train P1, val P2 S2-S4, test P2 S1)`.

### Task 4: Training labels from ChokePoint

**Files:**
- Create: `src/train_labels.py`
- Test: `tests/test_loaders.py`

**Interfaces:**
- Consumes: `loaders.chokepoint.load_gt`, `frame_path`; `splits.TRAIN`, `splits.VAL`; `loaders.musicvideo.iou`; SCRFD-10G (`systems.detectors.SCRFD("buffalo_l", "det_10g.onnx").net`).
- Produces: CLI `.venv/bin/python src/train_labels.py {train,val} --every 5 --out data/yunet_ft/` -> `data/yunet_ft/<split>/labelv2.txt` + `images/<seq>/<frame>.jpg` (symlinks or copies, as the trainer needs). Functions `label_lines(frame_rel: str, w: int, h: int, faces: pd.DataFrame) -> list[str]` and `unlabelled(gt_xywh: np.ndarray, scrfd: np.ndarray) -> bool`.

- [ ] **Step 0: Pin the label format.** Read libfacedetection.train's WIDER labelv2 parser (search the repo for `labelv2`). Write the exact line format and missing-landmark value in the `src/train_labels.py` docstring. Expected (SCRFD labelv2): a header `# <path> <w> <h>`, then one line per face `x1 y1 x2 y2` + 5 × `x y vis` (landmarks lx_eye, rx_eye, nose, l_mouth, r_mouth). If the parser differs, the tests below follow the parser.
- [ ] **Step 1: Write the failing tests:**

```python
def test_label_lines_box_and_missing_landmarks():
    faces = pd.DataFrame([dict(x=10, y=20, w=30, h=40, kps=None)])
    lines = train_labels.label_lines("P1E_S1_C1/00000100.jpg", 800, 600, faces)
    assert lines[0] == "# P1E_S1_C1/00000100.jpg 800 600"
    v = lines[1].split()
    assert v[:4] == ["10.0", "20.0", "40.0", "60.0"] and v[4:] == ["-1.0"] * 15

def test_frame_with_an_unlabelled_face_is_dropped():
    gt = np.array([[100, 100, 50, 60]], float)
    assert not train_labels.unlabelled(gt, np.array([[102, 98, 49, 62, 0.95]]))
    assert train_labels.unlabelled(gt, np.array([[102, 98, 49, 62, 0.95], [500, 80, 40, 50, 0.9]]))
    assert not train_labels.unlabelled(gt, np.array([[500, 80, 40, 50, 0.3]]))  # low-score SCRFD box ignored
```

- [ ] **Step 2:** Run the tests. Expected: FAIL, no module `train_labels`.
- [ ] **Step 3: Implement.** Rules: take every `--every`-th GT frame (consecutive frames are near duplicates); landmarks = GT eyes + SCRFD nose/mouth when SCRFD matches the GT box at IoU >= `musicvideo.EYE_IOU` (0.5), else all `-1`; `unlabelled` = any SCRFD box with score >= 0.7 and IoU < 0.3 with every GT box; drop those frames and print how many.
- [ ] **Step 4:** Run the tests. Expected: all `ok`.
- [ ] **Step 5:** Build both splits. Print and record in the docstring: frames kept, frames dropped as unlabelled, faces, faces with landmarks. Expected: dropped < 10% (else look at a sample before training).
- [ ] **Step 6:** Commit `feat: libfacedetection labelv2 labels from the ChokePoint split` (code only).

### Task 5: Fine-tune YuNet on Colab

**Files:**
- Create: `notebooks/finetune_yunet.ipynb`
- Modify: `src/systems/detectors.py` (`YuNetORT(path=...)`, SYSTEMS)

**Interfaces:**
- Consumes: Task 4 labels zipped as `yunet_ft.zip` (upload to Drive; not committed); chosen size `H×W` from Task 2.
- Produces: `models/yunet_ft.onnx` (dynamic export) and `models/yunet_ft.pth`; `YuNetORT(precision="fp32", path=None)` (path defaults to the stock model); SYSTEMS `yunet_ft` (FP32 ORT, threshold chosen in Step 6).

- [ ] **Step 1:** Notebook cells: mount Drive, clone `ShiqiYu/libfacedetection.train`, `pip install -e ".[dev]"`, download WIDER FACE train/val + labelv2 (README layout under `data/widerface`), unzip `yunet_ft.zip`. Run `python -m yunet_train.cli.train --help` and record how to (a) start from `weights/yunet_n.pth`, (b) add a second dataset or a combined label file, (c) set the train input size. Expected: all three are supported; if (b) is not, merge ChokePoint into the WIDER train labelv2 file with image paths relative to one root.
- [ ] **Step 2:** Smoke run: 1 epoch on 200 images. Expected: loss printed and decreasing within the epoch; a checkpoint saved.
- [ ] **Step 3:** Fine-tune from `weights/yunet_n.pth` on WIDER train + ChokePoint TRAIN (ChokePoint oversampled so it is about half of each epoch), input size `H×W`, lower learning rate than from scratch (start at 1/10 of the default), 40 epochs, checkpoint every 5, to `work_dirs/yunet_ft` on Drive. shortcut: no degradation augmentation in this run; add blur/dark/downscale augmentation when the stress tier (docs/plan.md Phase 6) exists.
- [ ] **Step 4:** Export each checkpoint: `python -m yunet_train.cli.export_onnx <ckpt> --variant yunet_n --dynamic-export --verify --output-file yunet_ft_e<N>.onnx`. Check output names are `cls_8 … kps_32` in the 2023mar order (Task 1 test's list); download to `models/`.
- [ ] **Step 5:** Add `path` to `YuNetORT` (`dynamic()` is applied to it as to the stock model) and `yunet_ft` to SYSTEMS. Add `"yunet_ft"` to `test_detectors_find_the_chokepoint_face`'s list; run `.venv/bin/python tests/test_detectors.py`. Expected: `ok`.
- [ ] **Step 6: Select on val** (never test): for each checkpoint, `benchmark.py --split val --systems yunet_ft --thr yunet_ft=T` for T in 0.5, 0.6, 0.7, 0.8, 0.9 (enroll/calibrate the judge on val as the benchmark does). Pick the checkpoint and T with the lowest % identities exposed; ties go to the higher T (less area masked). Copy it to `models/yunet_ft.onnx`, set T as `yunet_ft`'s threshold. Record the val table in the notebook.
- [ ] **Step 7:** Commit `feat: YuNet fine-tuning notebook and fine-tuned detector entry` (notebook with outputs cleared of images, detectors.py, test).

### Task 6: Quantize the fine-tuned YuNet and benchmark on test

**Files:**
- Modify: `src/systems/detectors.py` (SYSTEMS)
- Create: `experiments/d1/` (results only committed)

**Interfaces:**
- Consumes: `models/yunet_ft.onnx`, Task 1 CLI, `model_for` (FP16 / INT8 QDQ), chosen size `H×W`.
- Produces: SYSTEMS `yunet_ft_fp16`, `yunet_ft_q8` (Mac ORT) and `yunet_ft_<H>x<W>_npu` (BoardOnly); `models/yunet_ft_<H>x<W>_int8.tflite`; `experiments/d1/results.csv`.

- [ ] **Step 1:** Add `yunet_ft_fp16 = YuNetORT("fp16", path=…)` and `yunet_ft_q8 = YuNetORT("int8", path=…)` (the `model_for` file names derive from the stem, so the stock and fine-tuned caches do not collide; check this).
- [ ] **Step 2:** `.venv-export/bin/python src/export_yunet_tflite.py models/yunet_ft.onnx --size H W`. Run Task 1's TFLite-vs-ORT test against it (parametrize the test by model path). Expected: same face count, < 3 px, score not saturated.
- [ ] **Step 3:** On the board: `python3 board_bench.py --model yunet_ft_<H>x<W>_int8.tflite --frames /root/chokepoint --out /root/dets`; copy `dets/yunet_ft_<H>x<W>_npu/` to `experiments/d1/dets/` together with the stock entries from `experiments/board/dets/` (and the `emb/`, `judge.json`, `data_config.json` as in `experiments/board`).
- [ ] **Step 4:** `cd src && ../.venv/bin/python benchmark.py --split test --systems yunet_ort yunet_<H>x<W>_npu yunet_ft yunet_ft_fp16 yunet_ft_q8 yunet_ft_<H>x<W>_npu yolov8n_face yolov8n_face_320_npu --out ../experiments/d1`. Expected: one row each; the stock YuNet and YOLO rows equal `experiments/board/results.csv`.
- [ ] **Step 5:** Commit `feat: fine-tuned YuNet FP16/INT8 and the Deliverable 1 test table` (`experiments/d1/results.*`, configs; no dets/emb/scores).

### Task 7: Live demo on the board

**Files:**
- Create: `src/demo.py`
- Test: `tests/test_demo.py`

**Interfaces:**
- Consumes: `board_bench.YuNetTFLite`, `board_bench.YoloTFLite`; `systems/anonymize.py` (`scale_boxes`, `blur`; copied to the board next to `demo.py`); the decode pipeline from Task 2 Step 4.
- Produces: CLI on the board `python3 demo.py --video test.mp4 --models yunet_<H>x<W>_int8.tflite yunet_ft_<H>x<W>_int8.tflite [--thr 0.9 T] [--loop]`; keys: `m` next model, `b` blur on/off, `q` quit. Pure function `overlay_text(name: str, fps: float, stages: dict, n_faces: int) -> list[str]`.

- [ ] **Step 1: Write the failing test** `tests/test_demo.py`:

```python
def test_overlay_text():
    lines = demo.overlay_text("yunet_ft", 26.04, {"pre": 1.8, "npu": 9.1, "post": 2.0}, 2)
    assert lines == ["yunet_ft", "26.0 FPS", "pre 1.8 | npu 9.1 | post 2.0 ms", "faces 2"]

def test_run_one_frame_cpu():  # end to end on a real frame without the NPU
    out = demo.process(img, YuNetTFLite(MODELS / "yunet_320x416_int8.tflite", npu=False), thr=0.9, blur_on=True)
    assert out.shape == img.shape and not np.array_equal(out, img)
```

- [ ] **Step 2:** Run `.venv-export/bin/python tests/test_demo.py` (needs ai_edge_litert). Expected: FAIL, no module `demo`.
- [ ] **Step 3: Implement** `process(bgr, det, thr, blur_on) -> np.ndarray` (detect, keep score >= thr, `blur(img, scale_boxes(...))`), `overlay_text`, and `main`: open the video with the GStreamer pipeline (fall back to `cv2.VideoCapture(path)` if it does not open), load all models at start (NPU compile is cached next to the model), loop with `cv2.imshow` full screen, FPS = 1 / moving average of the last 30 frame times (decode included).
- [ ] **Step 4:** Run the tests. Expected: `ok`.
- [ ] **Step 5:** On the board: copy `demo.py`, `anonymize.py`, `board_bench.py`, models and `test.mp4`; run with `XDG_RUNTIME_DIR=/run/user/0 WAYLAND_DISPLAY=wayland-1 python3 demo.py …`. Expected on HDMI: blurred faces, overlay; FPS ≥ 25 with the chosen YuNet size. Record the FPS for both models in `docs/plan.md`. If FPS < 25, record which stage is over budget; do not change the plan's target.
- [ ] **Step 6:** Commit `feat: live anonymization demo on the i.MX 8M Plus`.

### Task 8: README, results and push

**Files:**
- Modify: `README.md` (create if missing)

- [ ] **Step 1:** Dataset section: ChokePoint (Zenodo record 815657, citation, license/terms), what was used (portals, sessions, cameras, frames, 29 people), the split table from `docs/plan.md`, how GT boxes come from the eye coordinates (`BOX_W, BOX_UP, BOX_DOWN`), the SCRFD pseudo-landmarks and dropped-frame counts from Task 4, WIDER FACE for fine-tuning, S5 for calibration, and that the same people appear in train and test.
- [ ] **Step 2:** Results: the `experiments/d1/results.csv` table, the Task 2 timing table, the demo FPS. How to reproduce: commands from Tasks 2-7.
- [ ] **Step 3:** Commit `docs: README dataset, results and reproduction`.
- [ ] **Step 4:** Ask Valerian before pushing. Once Moro has added the collaborator access, `git push -u origin valerian/benchmark` and open a PR with the results table in the description.

## Schedule

| Day | Tasks |
|---|---|
| Wed 29 Oct (evening) | 0, 1 |
| Thu 30 Oct | 2, 3 (download overnight) |
| Fri 31 Oct | 4, 5 Steps 1-3 (training runs overnight) |
| Sat 1 Nov | 5 Steps 4-7, 6 |
| Sun 2 Nov | 7, 8 |
| Mon 3 Nov | buffer; demo rehearsal before 16:00 |
