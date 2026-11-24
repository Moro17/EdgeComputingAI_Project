"""Scene prior for a fixed camera (risk's zone rule, systems/bytelite.py RiskGate): where faces appear, and when
something moves there.

zone:   bool grid (GRID_H x GRID_W cells over the frame) of the cells that a GT face box covered in the VAL sequences
        of the same portal and camera (ChokePoint: P2E_S1_C1 -> P2E_S2-S4 C1), dilated by one cell. Learned on val
        only, like an installer marking the door on a fixed camera. Never from the test sequence itself.
motion: per frame, a MOTION_W x MOTION_H map of the absolute grey difference to the previous frame (0-255); the
        gate reads the mean of the map inside each detection box. Measured on 2026-11-23: the mean over the whole
        zone hardly separates frames with and without faces (median 2.0 vs 0.7, 90th percentile 1.4), and noise
        lifts it in every frame; inside a box it tells a moving person from static texture. 80x60 averages 10x10
        pixels, which keeps sensor noise low. The board would compute it on the Y plane it already has.
"""
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import splits

REFLECTIONS = Path(__file__).resolve().parents[2] / "experiments" / "outside_gt" / "reflections_val.csv"

GRID_H, GRID_W = 30, 40
MIN_BLOB = 12  # cells; a door zone smaller than this is a stray box
SMALL = (80, 60)


def camera(seq):
    """'P2E_S1_C1.1' -> ('P2E', 'C1')."""
    m = re.fullmatch(r"(P\d[EL])_S\d_(C\d)\.\d", seq)
    if not m:
        raise ValueError(f"{seq}: zone needs a ChokePoint sequence (fixed camera)")
    return m.groups()


def face_zone(ds, seq, shape):
    """Zone grid of seq's camera from the val GT (systems/scene.py docstring)."""
    val = [s for s in splits.VAL if camera(s) == camera(seq)]
    if not val:
        raise ValueError(f"{seq}: no val sequence of the same camera")
    return box_grid(pd.concat([ds.load_gt(s) for s in val]), shape)


def reflection_zone(seq, shape, path=REFLECTIONS):
    """Grid of the cells where faces were REFLECTED in the val sequences of seq's camera (door glass): boxes outside
    the GT that the attacker matched to a person annotated in the same frame (src/analysis/outside_gt.py --mine-val).
    Empty grid for a camera without reflections."""
    r = pd.read_csv(path)
    r = r[[camera(s) == camera(seq) for s in r.seq]]
    if "kind" in r:  # reflections only: faces entering unannotated ("single") are all over the walkway (44-76% of
        r = r[r.kind == "duplicate"]  # the frame on val) and the deployed pipeline already masks them on test
    if not len(r):
        return np.zeros((GRID_H, GRID_W), bool)
    g = box_grid(r, shape)
    n, lab, st, _ = cv2.connectedComponentsWithStats(g.astype(np.uint8), connectivity=8)
    keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= MIN_BLOB]  # drop stray cells (one odd box)
    return np.isin(lab, keep)


def box_grid(gt, shape):
    """Cells covered by any box (x, y, w, h columns), dilated by one cell."""
    h, w = shape[:2]
    grid = np.zeros((GRID_H, GRID_W), bool)
    if len(gt):
        x0 = np.clip((gt.x * GRID_W / w).astype(int), 0, GRID_W - 1)
        x1 = np.clip(((gt.x + gt.w) * GRID_W / w).astype(int), 0, GRID_W - 1)
        y0 = np.clip((gt.y * GRID_H / h).astype(int), 0, GRID_H - 1)
        y1 = np.clip(((gt.y + gt.h) * GRID_H / h).astype(int), 0, GRID_H - 1)
        for a, b, c, d in zip(y0, y1, x0, x1):
            grid[a:b + 1, c:d + 1] = True
    return cv2.dilate(grid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def motion(ds, seq, frames, load_frame, deg=None, cache=None):
    """{frame: (SMALL[1], SMALL[0]) uint8 map of |grey - previous grey|}; first frame all 0. Cached as .npy."""
    if cache is not None and cache.exists():
        return dict(zip(frames, np.load(cache)))
    out, prev = [], None
    for n in frames:
        g = cv2.cvtColor(cv2.resize(load_frame(ds.frame_path(seq, n), n, deg), SMALL, interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY).astype(np.int16)
        out.append(np.zeros(g.shape, np.uint8) if prev is None else np.abs(g - prev).astype(np.uint8))
        prev = g
    out = np.stack(out)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, out)
    return dict(zip(frames, out))
