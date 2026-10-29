"""ChokePoint eye-coordinate GT -> unified GT CSV.

Columns: frame, track_id, identity, x, y, w, h, lx, ly, rx, ry (pixels, top-left origin; l/r = image-left/right eye).

Usage: python src/loaders/chokepoint.py [SEQ ...]   # writes gt/chokepoint/<seq>.csv (default: every GT file)
"""
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("ECA_DATA", ROOT / "data")) / "chokepoint"  # ECA_DATA: data outside the code dir (Colab)

# Face box from the two pupils, in units of inter-ocular distance d: width 2.2d centred on the eyes,
# from 1.1d above the eye line to 1.9d below it. Matches YuNet's box within ~2 px on a frontal sample.
# shortcut: frontal geometry; under strong yaw d shrinks so the box is too small. Revisit if overlays look off.
BOX_W, BOX_UP, BOX_DOWN = 2.2, 1.1, 1.9


def frame_numbers(seq, data=DATA):
    return [int(f.get("number")) for f in ET.parse(data / "groundtruth" / f"{seq}.xml").getroot().iter("frame")]


def frame_path(seq, n, data=DATA):
    return data / "frames" / seq / f"{n:08d}.jpg"


def load_gt(seq, data=DATA):
    rows = [
        (int(f.get("number")), int(p.get("id")), *(float(p.find(e).get(k)) for e in ("leftEye", "rightEye") for k in "xy"))
        for f in ET.parse(data / "groundtruth" / f"{seq}.xml").getroot().iter("frame")
        for p in f.iter("person")
    ]
    df = pd.DataFrame(rows, columns=["frame", "identity", "lx", "ly", "rx", "ry"])
    d = np.hypot(df.rx - df.lx, df.ry - df.ly)
    df["x"] = (df.lx + df.rx) / 2 - BOX_W * d / 2
    df["y"] = (df.ly + df.ry) / 2 - BOX_UP * d
    df["w"], df["h"] = BOX_W * d, (BOX_UP + BOX_DOWN) * d
    df["track_id"] = df.identity  # one pass per person per sequence in ChokePoint
    return df[["frame", "track_id", "identity", "x", "y", "w", "h", "lx", "ly", "rx", "ry"]]


def main(seqs):
    out = ROOT / "gt" / "chokepoint"
    out.mkdir(parents=True, exist_ok=True)
    for seq in seqs or sorted(p.stem for p in (DATA / "groundtruth").glob("*.xml")):
        load_gt(seq).round(1).to_csv(out / f"{seq}.csv", index=False)
        print(seq)


if __name__ == "__main__":
    main(sys.argv[1:])
