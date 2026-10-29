"""ChokePoint -> libfacedetection.train (yunet_train) labelv2 training data.

Format pinned from ShiqiYu/libfacedetection.train @ dca340a (yunet_train/tasks/face/widerface.py, dataset.py):
  - one record per image: header `# <relpath> <w> <h>` (widerface.py:85-91, exactly 3 tokens), then one line per face.
  - face line: `x1 y1 x2 y2` + 5 x `x y vis` = 19 floats (widerface.py:17-36). Box is corner format, pixels.
  - landmark order: left eye, right eye, nose, left mouth, right mouth (image-left first; flip order [1,0,2,4,3],
    transforms.py:356). Same as SCRFD kps and the ChokePoint loader's l/r eyes.
  - missing landmarks: all 15 values -1 (a point with x=y=vis=-1 gets weight 0, widerface.py:37-39). Present point:
    vis must be >= 0 and is overwritten with weight 1, so we write 1.
  - a record with no face lines is skipped in train mode (widerface.py:69); a 4-value line raises in train mode.
  - <relpath> is resolved as `img_prefix / relpath` (dataset.py:44); `--ann-file` and `--img-prefix` take ONE file/dir
    (cli/train.py:39-40), so ChokePoint + WIDER must be merged on Colab: concatenate the two labelv2.txt files and put
    ChokePoint's images/<seq>/ next to the WIDER class folders under WIDER_train/images/ (paths here are <seq>/<frame>.jpg).

Rules: every --every-th GT frame; landmarks = GT eyes + SCRFD-10G nose/mouth when SCRFD matches the GT box at
IoU >= EYE_IOU, else all -1; frames where SCRFD (score >= 0.7) sees a face no GT box overlaps (IoU < 0.3) are dropped,
since an unlabelled face would train as background. Images are copied (not symlinked) so data/yunet_ft/ can be zipped.

Usage: python src/train_labels.py {train,val} --every 5 --out data/yunet_ft/
Counts (--every 5, 2026-10-30):
  train: kept=6314 dropped_unlabelled=39 (0.6%) faces=6314 with_landmarks=6145
  val:   kept=5846 dropped_unlabelled=492 (7.8%) faces=5846 with_landmarks=5839
"""
import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loaders import chokepoint  # noqa: E402
from loaders.musicvideo import EYE_IOU, iou  # noqa: E402
from splits import TRAIN, VAL  # noqa: E402

SCORE, MISS_IOU = 0.7, 0.3


def _f(v):
    return str(round(float(v), 2))


def label_lines(frame_rel, w, h, faces):
    """`faces`: columns x, y, w, h (xywh) and kps ((5, 2) array or None)."""
    lines = [f"# {frame_rel} {w} {h}"]
    for f in faces.itertuples():
        x1, y1, x2, y2 = max(f.x, 0), max(f.y, 0), min(f.x + f.w, w), min(f.y + f.h, h)
        if x2 <= x1 or y2 <= y1:
            continue
        kp = [-1.0] * 15 if f.kps is None else [v for p in f.kps for v in (*p, 1.0)]
        lines.append(" ".join(map(_f, (x1, y1, x2, y2, *kp))))
    return lines


def unlabelled(gt_xywh, scrfd):
    """True when a confident SCRFD box (x, y, w, h, score) overlaps no GT box."""
    s = scrfd[scrfd[:, 4] >= SCORE]
    return bool(len(s)) and bool((iou(s[:, :4], gt_xywh).max(1) < MISS_IOU).any())


def build(split, every, out):
    from systems.detectors import SCRFD

    net = SCRFD("buffalo_l", "det_10g.onnx").net
    root = Path(out) / split
    lines, kept, dropped, nfaces, nkps = [], 0, 0, 0, 0
    drop_log = []
    for seq in TRAIN if split == "train" else VAL:
        gt = chokepoint.load_gt(seq)
        for n in sorted(gt.frame.unique())[::every]:
            g = gt[gt.frame == n].reset_index(drop=True)
            img = cv2.imread(str(chokepoint.frame_path(seq, n)))
            H, W = img.shape[:2]
            det, kps = net.detect(img)
            d = np.c_[det[:, :2], det[:, 2:4] - det[:, :2], det[:, 4]] if len(det) else np.zeros((0, 5))
            gxywh = g[["x", "y", "w", "h"]].to_numpy()
            if unlabelled(gxywh, d):
                dropped += 1
                drop_log.append((seq, int(n)))
                continue
            m = iou(gxywh, d[:, :4]) if len(d) else np.zeros((len(g), 0))
            k = []
            for i, r in g.iterrows():
                j = m[i].argmax() if m.shape[1] else -1
                ok = j >= 0 and m[i, j] >= EYE_IOU
                k.append(np.array([[r.lx, r.ly], [r.rx, r.ry], *kps[j][2:]]) if ok else None)
            g["kps"] = k
            rel = f"{seq}/{n:08d}.jpg"
            (root / "images" / seq).mkdir(parents=True, exist_ok=True)
            shutil.copy(chokepoint.frame_path(seq, n), root / "images" / rel)
            lines += label_lines(rel, W, H, g)
            kept, nfaces, nkps = kept + 1, nfaces + len(g), nkps + sum(x is not None for x in k)
    (root / "labelv2.txt").write_text("\n".join(lines) + "\n")
    print(f"{split}: kept={kept} dropped_unlabelled={dropped} faces={nfaces} with_landmarks={nkps}")
    return drop_log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["train", "val"])
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--out", default="data/yunet_ft/")
    a = ap.parse_args()
    build(a.split, a.every, a.out)
