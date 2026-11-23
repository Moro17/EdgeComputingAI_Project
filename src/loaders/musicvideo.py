"""Music Video dataset (Zhang et al., ECCV 2016): 8 music videos, per-frame face boxes, one id per person per video.

Tier 2 ("hard") of the benchmark: shot changes, fast motion, big and small faces, stage lighting. Downloaded from
the authors' Dropbox (FaceTracking_ECCV_2016_data: the videos + GT.zip; no licence stated, cite the paper).

Layout under $ECA_DATA/musicvideo:
  raw/<video>.mp4|.mov, raw/GT.zip         as downloaded
  GT/<gt name>_gt.xml                      unzipped GT.zip (also has Big Bang Theory GT; those videos are not public)
  frames/<seq>/<frame:08d>.jpg             extracted by this script, frame numbers as in the GT (first frame = 1)

Sequences: each video is split at its middle frame into two halves, '<video>.1' and '<video>.2'. A half is its own
session, so the attacker's gallery for a person comes from the OTHER half of the same video (identities exist only
within one video). That is easier for the attacker than ChokePoint's other-day gallery (same clothes, same
lighting), so exposure here is an upper bound; say so in the paper.

GT boxes are tight face boxes, no eye landmarks: musicvideo_landmarks.py adds eyes once from clean frames (box-only
crops gave the judge 6% TAR at 1% FAR); faces without eyes fall back to the box crop (judge/arcface.align).

  python src/loaders/musicvideo.py            # extract frames (~6 GB of JPEG, q 90)
"""
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ.get("ECA_DATA", Path(__file__).resolve().parents[2] / "data")) / "musicvideo"
# video file stem -> GT xml stem
VIDEOS = {"Apink": "Apink", "BrunoMars": "BrunoMars", "Darling": "Darling", "GirlsAloud": "GirlsAloud",
          "HelloBubble": "HelloBubble", "PussycatDolls": "stickwitu", "T-ara": "Tara", "Westlife": "Westlife"}


def _gt_xml(video, data=DATA):
    return data / "GT" / f"{VIDEOS[video]}_gt.xml"


def _video_frames(video, data=DATA):
    return int(ET.parse(_gt_xml(video, data)).getroot().get("end_frame"))


def _split(seq):
    video, _, half = seq.rpartition(".")
    return video, int(half)


def _range(seq, data=DATA):
    video, half = _split(seq)
    n = _video_frames(video, data)
    mid = n // 2
    return (1, mid) if half == 1 else (mid + 1, n)


def sequences(data=DATA):
    """Every half whose frames are extracted."""
    return sorted(p.name for p in (data / "frames").iterdir() if p.is_dir()) if (data / "frames").exists() else []


def session(seq):
    return seq


def frame_numbers(seq, data=DATA):
    a, b = _range(seq, data)
    return list(range(a, b + 1))


def frame_path(seq, n, data=DATA):
    return data / "frames" / seq / f"{n:08d}.jpg"


def load_gt(seq, data=DATA, landmarks=True):
    """GT rows; eyes (lx, ly, rx, ry) from landmarks/<seq>.csv (musicvideo_landmarks.py) when present, else NaN."""
    video, _ = _split(seq)
    a, b = _range(seq, data)
    rows = [(int(f.get("frame_no")), int(t.get("obj_id")), float(f.get("x")), float(f.get("y")),
             float(f.get("width")), float(f.get("height")))
            for t in ET.parse(_gt_xml(video, data)).getroot().iter("Trajectory") for f in t.iter("Frame")
            if a <= int(f.get("frame_no")) <= b and f.get("observation", "1") == "1"]
    df = pd.DataFrame(rows, columns=["frame", "track_id", "x", "y", "w", "h"]).sort_values(["frame", "track_id"])
    df["identity"] = [f"{video}:{t}" for t in df.track_id]  # people exist only within one video
    lm = data / "landmarks" / f"{seq}.csv"
    if landmarks and lm.exists():
        df = df.merge(pd.read_csv(lm), on=["frame", "track_id"], how="left")
    for c in ("lx", "ly", "rx", "ry"):
        if c not in df:
            df[c] = np.nan
    return df[["frame", "track_id", "identity", "x", "y", "w", "h", "lx", "ly", "rx", "ry"]].reset_index(drop=True)


def extract(data=DATA, quality=90):
    import cv2

    for video in VIDEOS:
        src = next((data / "raw").glob(f"{video}.*"))
        n = _video_frames(video, data)
        halves = [f"{video}.1", f"{video}.2"]
        if all((data / "frames" / s / f"{frame_numbers(s, data)[-1]:08d}.jpg").exists() for s in halves):
            print("done", video)
            continue
        cap, k = cv2.VideoCapture(str(src)), 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            k += 1
            seq = halves[0] if k <= n // 2 else halves[1]
            p = frame_path(seq, k, data)
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        assert k == n, f"{video}: {k} frames decoded, GT says {n}"
        print(video, k, "frames")


if __name__ == "__main__":
    extract(*(Path(a) for a in sys.argv[1:2]))
