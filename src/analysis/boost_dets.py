"""Two detection boosts for the board's INT8 YuNet, simulated with the faithful onnxruntime INT8 model (yq_board,
yqft_board; docs/next-runs.md quantization study), written as cached detections that src/benchmark.py scores:

  enh   dark frames only (mean grey < DARK, as the dark rule): enhance the frame before the detector, instead of it
        (one inference, as on the board). gam = gamma LUT that lifts the mean grey to TARGET; clahe = CLAHE on the
        LAB lightness channel (clip 2, 8x8 tiles).
  zoom  a SECOND inference on the camera's reflection zone (systems/scene.py reflection_zone, learnt on val), cropped
        and upscaled to the detector input, CLAHE on the crop (glass is low contrast); boxes mapped back and merged
        with the full-frame ones (NMS 0.3). On the board it would run only on frames with motion in the zone.

Systems written (dets/<name><@deg>/<seq>.csv): <base>_gam, <base>_clahe, <base>_zoom, <base>_gamzoom, <base>_clahezoom.
The parts are cached separately (dets/<base>_part_<gam|clahe|zoom><@deg>/), so combinations cost no inference.

    python src/analysis/boost_dets.py --out results/val --split val --degrade dark:2 --parts gam clahe
    python src/analysis/boost_dets.py --out results/repro_desktop --split test --parts gam clahe zoom
    python src/analysis/boost_dets.py --out results/musicvideo --dataset musicvideo --parts gam clahe
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import splits  # noqa: E402
from benchmark import degrade_suffix  # noqa: E402
from degrade.degrade import load as load_frame  # noqa: E402
from systems import scene  # noqa: E402
from systems.detectors import SYSTEMS  # noqa: E402

DARK, TARGET = 60, 110
BASES = {"yq_board": 0.6, "yqft_board": 0.7}
COLS = ["frame", "track_id", "x", "y", "w", "h", "score"]
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def grey_mean(img):
    return float(cv2.cvtColor(cv2.resize(img, (160, 120), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).mean())


def gamma(img, mean):
    g = np.clip(np.log(TARGET / 255) / np.log(max(mean, 1) / 255), 0.25, 1.0)
    lut = (255 * (np.arange(256) / 255) ** g).clip(0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _CLAHE.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def zone_rect(seq, shape, pad=0.05):
    """Bounding rectangle (x0, y0, x1, y1) of the largest blob of the camera's reflection zone, padded; None without one."""
    try:
        z = scene.reflection_zone(seq, shape)
    except ValueError:
        return None
    if not z.any():
        return None
    h, w = shape[:2]
    n, lab, st, _ = cv2.connectedComponentsWithStats(z.astype(np.uint8), connectivity=8)
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))  # the glass: largest blob (stray cells elsewhere are dropped)
    ys, xs = np.nonzero(lab == big)
    x0, x1 = xs.min() * w / z.shape[1], (xs.max() + 1) * w / z.shape[1]
    y0, y1 = ys.min() * h / z.shape[0], (ys.max() + 1) * h / z.shape[0]
    px, py = pad * w, pad * h
    return int(max(0, x0 - px)), int(max(0, y0 - py)), int(min(w, x1 + px)), int(min(h, y1 + py))


def nms_merge(a, b):
    a = np.concatenate([a, b]) if len(b) else a
    if len(a) < 2:
        return a
    keep = cv2.dnn.NMSBoxes(a[:, :4].tolist(), a[:, 4].tolist(), 0.0, 0.3, top_k=5000)
    return a[np.asarray(keep, int).reshape(-1)]


def save(path, rows, lat, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=COLS)
    df.round({"x": 2, "y": 2, "w": 2, "h": 2, "score": 4}).to_csv(path, index=False)
    path.with_suffix(".json").write_text(json.dumps({"latency_ms": lat, **extra}))


def parts(ds, out, base, seq, deg, which):
    """Compute the missing part caches for one sequence."""
    sfx = degrade_suffix(deg)
    todo = [p for p in which if not (out / "dets" / f"{base}_part_{p}{sfx}" / f"{seq}.csv").exists()]
    if not todo:
        return
    det = SYSTEMS[base][0](None)
    frames = ds.frame_numbers(seq)
    shape = cv2.imread(str(ds.frame_path(seq, frames[0]))).shape
    rect = zone_rect(seq, shape) if "zoom" in todo else None
    rows = {p: [] for p in todo}
    lat = {p: [] for p in todo}
    dark = []  # frames the enhancement ran on (their full-frame detections are replaced, even by nothing)
    for n in frames:
        img = load_frame(ds.frame_path(seq, n), n, deg)
        m = grey_mean(img) if {"gam", "clahe"} & set(todo) else 255
        if m < DARK:
            dark.append(int(n))
        for p in todo:
            t0 = time.perf_counter()
            if p in ("gam", "clahe"):
                if m >= DARK:
                    continue  # not a dark frame: the full-frame detections stand
                a = det(gamma(img, m) if p == "gam" else clahe(img))
            else:
                if rect is None:
                    continue
                x0, y0, x1, y1 = rect
                a = det(clahe(np.ascontiguousarray(img[y0:y1, x0:x1])))
                a[:, 0] += x0
                a[:, 1] += y0
            lat[p].append(1000 * (time.perf_counter() - t0))
            rows[p] += [(n, -1, *b) for b in a.tolist()]
    for p in todo:
        save(out / "dets" / f"{base}_part_{p}{sfx}" / f"{seq}.csv", rows[p], lat[p],
             **({"dark_frames": dark} if p != "zoom" else {"rect": rect}))


def compose(out, base, seq, deg):
    """Systems from the full-frame cache and the parts (an enhanced frame REPLACES the full frame's detections)."""
    sfx = degrade_suffix(deg)
    rd = lambda name: pd.read_csv(out / "dets" / f"{name}{sfx}" / f"{seq}.csv")  # noqa: E731
    full = rd(base)
    have = {p: (out / "dets" / f"{base}_part_{p}{sfx}" / f"{seq}.csv").exists() for p in ("gam", "clahe", "zoom")}
    lat_full = json.loads((out / "dets" / f"{base}{sfx}" / f"{seq}.json").read_text())["latency_ms"]
    for name, enh, zoom in (("gam", "gam", False), ("clahe", "clahe", False), ("zoom", None, True),
                            ("gamzoom", "gam", True), ("clahezoom", "clahe", True)):
        if (enh and not have[enh]) or (zoom and not have["zoom"]):
            continue
        d = full
        if enh:
            e = rd(f"{base}_part_{enh}")
            dark_frames = set(json.loads((out / "dets" / f"{base}_part_{enh}{sfx}" / f"{seq}.json").read_text())
                              ["dark_frames"])
            d = pd.concat([d[~d.frame.isin(dark_frames)], e], ignore_index=True)
        if zoom:
            z = rd(f"{base}_part_zoom")
            if len(z):
                zb = dict(list(z.groupby("frame")))
                merged = []
                for f, g in d.groupby("frame"):
                    a = g[["x", "y", "w", "h", "score"]].to_numpy()
                    if f in zb:
                        a = nms_merge(a, zb.pop(f)[["x", "y", "w", "h", "score"]].to_numpy())
                    merged += [(f, -1, *r) for r in a.tolist()]
                for f, g in zb.items():
                    merged += [(f, -1, *r) for r in nms_merge(g[["x", "y", "w", "h", "score"]].to_numpy(), np.zeros((0, 5))).tolist()]
                d = pd.DataFrame(merged, columns=COLS)
        save(out / "dets" / f"{base}_{name}{sfx}" / f"{seq}.csv", d.sort_values("frame")[COLS].values.tolist(), lat_full)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dataset", default="chokepoint")
    p.add_argument("--split", choices=["val", "test"], default=None)
    p.add_argument("--seqs", nargs="+", default=None)
    p.add_argument("--degrade", default=None)
    p.add_argument("--bases", nargs="+", default=list(BASES))
    p.add_argument("--parts", nargs="+", default=["gam", "clahe", "zoom"])
    a = p.parse_args()
    ds = importlib.import_module(f"loaders.{a.dataset}")
    seqs = a.seqs or (getattr(splits, a.split.upper()) if a.split else ds.sequences())
    for base in a.bases:
        for s in seqs:
            t = time.time()
            parts(ds, a.out, base, s, a.degrade, a.parts)
            compose(a.out, base, s, a.degrade)
            print(f"{base} {s} {a.degrade or 'clean'} {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
