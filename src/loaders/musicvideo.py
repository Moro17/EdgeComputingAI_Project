"""Music Video dataset (Zhang et al., 8 YouTube music videos, face tracklets with identity labels) -> unified GT.

Layout under data/musicvideo/ (ECA_DATA overrides data/):
  <video>/frames/00000001.jpg ...  one jpg per frame, numbered from 1 (`extract` below writes them from the video file)
  <video>/gt.txt                   raw boxes, one face per line: frame, identity, x, y, w, h[, ...] (comma or space separated)
  <video>/gt_unified.csv           built by `gt` below: unified columns, eyes from SCRFD-10G (regenerable, not committed)
shortcut: the raw annotation format is unverified (no download link found 2026-10-29); fix parse_raw once the files are in hand.

Identities exist only inside one video, so each video is split at its middle frame into sequences '<video>.1' and
'<video>.2'. Each half is its own session, judged against templates from the other half (other shots, same people).
Box-only GT: eyes come from SCRFD-10G keypoints on the clean frame (IoU >= 0.5 with the GT box); faces it misses keep
NaN eyes and the judge aligns them from the box (judge.arcface.align).

Usage: python src/loaders/musicvideo.py extract VIDEO FILE.mp4
       python src/loaders/musicvideo.py gt [VIDEO ...]   # default: every video with frames/ and gt.txt
"""
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("ECA_DATA", ROOT / "data")) / "musicvideo"
COLS = ["frame", "track_id", "identity", "x", "y", "w", "h", "lx", "ly", "rx", "ry"]
EYE_IOU = 0.5


def videos(data=DATA):
    return sorted(p.name for p in data.iterdir() if (p / "frames").is_dir() and (p / "gt.txt").exists())


def sequences(data=DATA):
    return [f"{v}.{h}" for v in videos(data) for h in (1, 2)]


def session(seq):
    return seq  # each half is its own leave-out group


def split(seq):
    video, half = seq.rsplit(".", 1)
    return video, int(half)


def video_frames(video, data=DATA):
    return sorted(int(p.stem) for p in (data / video / "frames").glob("*.jpg"))


def frame_numbers(seq, data=DATA):
    video, half = split(seq)
    n = video_frames(video, data)
    return n[: len(n) // 2] if half == 1 else n[len(n) // 2 :]


def frame_path(seq, n, data=DATA):
    return data / split(seq)[0] / "frames" / f"{n:08d}.jpg"


def parse_raw(path, video):
    """Raw annotation -> unified columns without eyes. Identities become '<video>:<id>' (unique across videos)."""
    rows = [re.split(r"[,\s]+", line.strip())[:6] for line in Path(path).read_text().splitlines() if line.strip()]
    df = pd.DataFrame(np.array(rows, float), columns=["frame", "identity", "x", "y", "w", "h"])
    df["frame"] = df.frame.astype(int)
    df["identity"] = [f"{video}:{int(i)}" for i in df.identity]
    # shortcut: one track per identity per half; a real tracklet id (if the raw files have one) belongs here
    df["track_id"] = df.identity
    df[["lx", "ly", "rx", "ry"]] = np.nan
    return df[COLS]


def iou(a, b):
    """IoU between (n, 4) and (m, 4) xywh boxes -> (n, m)."""
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2) - np.maximum(a[:, None, 0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2) - np.maximum(a[:, None, 1], b[:, 1]), 0, None)
    inter = iw * ih
    return inter / (a[:, 2:].prod(1)[:, None] + b[:, 2:].prod(1) - inter)


def fill_eyes(df, video, data=DATA):
    """Eyes from SCRFD-10G keypoints (image-left eye first, as in the ChokePoint loader) for GT boxes it matches."""
    from systems.detectors import SCRFD

    net = SCRFD("buffalo_l", "det_10g.onnx").net
    df = df.copy()
    for n, g in df.groupby("frame"):
        det, kps = net.detect(cv2.imread(str(data / video / "frames" / f"{n:08d}.jpg")))
        if not len(det):
            continue
        d = det[:, :4].copy()
        d[:, 2:] -= d[:, :2]
        m = iou(g[["x", "y", "w", "h"]].to_numpy(), d)
        best = m.argmax(1)
        for idx, j, ok in zip(g.index, best, m.max(1) >= EYE_IOU):
            if ok:
                (x0, y0), (x1, y1) = sorted(kps[j][:2].tolist())
                df.loc[idx, ["lx", "ly", "rx", "ry"]] = x0, y0, x1, y1
    return df


def load_gt(seq, data=DATA):
    video, _ = split(seq)
    df = pd.read_csv(data / video / "gt_unified.csv")
    return df[df.frame.isin(frame_numbers(seq, data))].reset_index(drop=True)


def extract(video, src, data=DATA):
    out = data / video / "frames"
    out.mkdir(parents=True, exist_ok=True)
    cap, n = cv2.VideoCapture(str(src)), 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        n += 1
        cv2.imwrite(str(out / f"{n:08d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert n, f"no frames decoded from {src}"
    print(f"{video}: {n} frames")


def build_gt(video, data=DATA):
    df = fill_eyes(parse_raw(data / video / "gt.txt", video), video, data)
    df.round(2).to_csv(data / video / "gt_unified.csv", index=False)
    print(f"{video}: {len(df)} faces, {df.identity.nunique()} identities, eyes found for {df.lx.notna().mean():.1%}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    cmd, *rest = sys.argv[1:]
    if cmd == "extract":
        extract(*rest)
    else:
        for v in rest or videos():
            build_gt(v)
