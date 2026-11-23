"""Eye positions for the Music Video GT boxes, so the judge aligns faces as on ChokePoint (attacker knows the eyes).

The GT has boxes only. Box-only crops give the judge 6% TAR at 1% FAR on clean faces (unaligned crops; members of
one band look alike), which would make every system look safe. So, once, on CLEAN frames: RetinaFace (InsightFace
buffalo_l det_10g, the judge's own detector family, not a system under test) runs on each GT box with a 50% margin;
the detection with the highest IoU to the GT box (>= 0.3) gives the two eyes. Faces without one keep NaN eyes and
fall back to the box crop (count reported). Output: $ECA_DATA/musicvideo/landmarks/<seq>.csv
(frame, track_id, lx, ly, rx, ry), read by loaders/musicvideo.load_gt.

  python src/loaders/musicvideo_landmarks.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loaders import musicvideo as mv  # noqa: E402
from systems.bytelite import iou  # noqa: E402

MARGIN, SIZE = 0.5, 192  # crop = GT box grown by 50% each side, detector input SIZE x SIZE


def detector():
    from insightface.model_zoo import get_model
    from insightface.utils.storage import ensure_available

    from systems.detectors import ort_providers

    path = Path(ensure_available("models", "buffalo_l", root="~/.insightface")) / "det_10g.onnx"
    m = get_model(str(path), providers=ort_providers())
    m.prepare(ctx_id=0, det_thresh=0.3, input_size=(SIZE, SIZE))
    return m


def eyes(det, img, r):
    h, w = img.shape[:2]
    x0, y0 = max(int(r.x - MARGIN * r.w), 0), max(int(r.y - MARGIN * r.h), 0)
    x1, y1 = min(int(r.x + r.w * (1 + MARGIN)), w), min(int(r.y + r.h * (1 + MARGIN)), h)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    boxes, kps = det.detect(crop, input_size=(SIZE, SIZE))
    if boxes is None or not len(boxes):
        return None
    b = boxes[:, :4].copy()
    b[:, 2:] -= b[:, :2]
    m = iou(np.array([[r.x - x0, r.y - y0, r.w, r.h]]), b)[0]
    i = int(m.argmax())
    if m[i] < 0.3:
        return None
    (lx, ly), (rx, ry) = kps[i][0], kps[i][1]  # insightface order: left eye, right eye (image left / right)
    return lx + x0, ly + y0, rx + x0, ry + y0


def main():
    det = detector()
    out = mv.DATA / "landmarks"
    out.mkdir(exist_ok=True)
    for seq in mv.sequences():
        path = out / f"{seq}.csv"
        if path.exists():
            continue
        gt = mv.load_gt(seq, landmarks=False)
        rows = []
        for n, g in gt.groupby("frame"):
            img = cv2.imread(str(mv.frame_path(seq, n)))
            for r in g.itertuples():
                e = eyes(det, img, r)
                rows.append((n, r.track_id, *(e if e else (np.nan,) * 4)))
        d = pd.DataFrame(rows, columns=["frame", "track_id", "lx", "ly", "rx", "ry"])
        d.round(2).to_csv(path, index=False)
        print(f"{seq}: {len(d)} faces, eyes found for {d.lx.notna().mean():.1%}", flush=True)


if __name__ == "__main__":
    main()
