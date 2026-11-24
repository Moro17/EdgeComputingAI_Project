"""Faces the ground truth does not contain: door-glass reflections, people half-hidden by the door frame, people far
down the corridor before their annotation starts. The benchmark never scores them, so identity exposure is a lower
bound. Here they are mined and scored:

1. mine: a box found by BOTH strong detectors (EgoBlur >= 0.5 and SCRFD-10G >= 0.5, IoU >= 0.3) that overlaps no GT
   box (IoU < 0.1), on the clean test frames.
2. the attacker labels them: the clean crop is embedded by the ArcFace judge and compared with every gallery identity
   (leave-session-out templates, as the benchmark). Kept only if the best similarity is above the 1% FAR threshold and
   that identity appears in the sequence's GT: an identifiable face of a person who is in the video.
   Faces with no GT box in the same frame must be within NEAR frames of that person's annotation.
   kind = "duplicate" if the same identity has a GT box in the same frame (a reflection, or the person twice), else
   "single" (a real face the annotation skipped: partial at the door frame, or far away).
3. score: for each system, coverage of these boxes by its mask (>= 0.8 protected); an unprotected one is cropped
   from the ANONYMIZED (degraded) frame and judged against its identity's template, as GT faces are.

    python src/analysis/outside_gt.py --systems yunet_ft_480x640_npu+risk_d60_t0.1+lite+back3 --degrade dark:2
Cached under results/repro_desktop/outside_gt/. Output: experiments/outside_gt/<tag>.csv
"""
import argparse
import os
import sys
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import splits  # noqa: E402
from benchmark import degrade_suffix, luma  # noqa: E402
from degrade.degrade import load as load_frame  # noqa: E402
from judge.arcface import Judge, align, templates  # noqa: E402
from loaders import chokepoint as ds  # noqa: E402
from metrics.exposure import PROTECTED_COVERAGE, coverage  # noqa: E402
from systems.anonymize import box_mask, fast_blur, scale_boxes  # noqa: E402
from systems import scene  # noqa: E402
from systems.detectors import SYSTEMS  # noqa: E402
from systems.policies import parse_spec, run_policy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = Path(os.environ.get("ECA_RESULTS", ROOT / "results")) / "repro_desktop"  # the benchmark's --out of the test split
THR = {"yunet_480x640_npu": 0.6, "yunet_ft_480x640_npu": 0.7, "egoblur": 0.9, "scrfd_10g": 0.5, "deface": 0.2}
NEAR = 75  # frames, 3 s
Row = namedtuple("Row", "x y w h lx ly rx ry")


def iou(a, b):
    a2, b2 = a[:, :2] + a[:, 2:4], b[:, :2] + b[:, 2:4]
    wh = np.clip(np.minimum(a2[:, None], b2[None]) - np.maximum(a[:, None, :2], b[None, :, :2]), 0, None)
    inter = wh.prod(-1)
    return inter / (a[:, 2:4].prod(1)[:, None] + b[:, 2:4].prod(1)[None] - inter)


def mine(seq):
    gt = ds.load_gt(seq)
    e = pd.read_csv(RES / "dets" / "egoblur" / f"{seq}.csv")
    c = pd.read_csv(RES / "dets" / "scrfd_10g" / f"{seq}.csv")
    e, c = e[e.score >= 0.5], c[c.score >= 0.5]
    gb, cb = dict(list(gt.groupby("frame"))), dict(list(c.groupby("frame")))
    out = []
    for f, g in e.groupby("frame"):
        b = g[["x", "y", "w", "h"]].to_numpy()
        ok = iou(b, gb[f][["x", "y", "w", "h"]].to_numpy()).max(1) < 0.1 if f in gb else np.ones(len(b), bool)
        ok &= iou(b, cb[f][["x", "y", "w", "h"]].to_numpy()).max(1) >= 0.3 if f in cb else np.zeros(len(b), bool)
        out += [(seq, f, *r) for r in b[ok]]
    return pd.DataFrame(out, columns=["seq", "frame", "x", "y", "w", "h"])


def crop(img, r):
    return align(img, Row(r.x, r.y, r.w, r.h, np.nan, np.nan, np.nan, np.nan))


def label(judge, cand, tmpl, thr):
    """Attacker labels each mined box with an identity; keeps the identifiable faces of people in the sequence."""
    rows = []
    for seq, g in cand.groupby("seq"):
        gt = ds.load_gt(seq)
        keys = sorted(tmpl[seq])
        T = np.stack([tmpl[seq][k] for k in keys])
        crops = []
        for f, gf in g.groupby("frame"):
            img = cv2.imread(str(ds.frame_path(seq, int(f))))
            crops += [crop(img, r) for r in gf.itertuples()]
        g = g.sort_values("frame", kind="stable").copy()
        sims = judge.embed(crops) @ T.T
        g["identity"] = np.array(keys)[sims.argmax(1)]
        g["sim_clean"] = sims.max(1)
        ids_seq = set(gt.identity.unique())
        g = g[(g.sim_clean >= thr) & g.identity.isin(ids_seq)]
        same = gt.groupby("frame").identity.apply(set).to_dict()
        g["kind"] = ["duplicate" if r.identity in same.get(r.frame, ()) else "single" for r in g.itertuples()]
        # a "single" must be of a person annotated within +-NEAR frames (people pass one at a time): drops the
        # attacker's argmax-over-identities false labels (4 people near the threshold on the clean test set)
        gf = {i: f.to_numpy() for i, f in gt.groupby("identity").frame}
        g = g[[r.kind == "duplicate" or np.abs(gf[r.identity] - r.frame).min() <= NEAR for r in g.itertuples()]]
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def mine_val(judge):
    """Faces outside the GT on VAL (for the door zone, systems/scene.py reflection_zone): our own detector's boxes
    (stock or fine-tuned, NPU, score >= 0.1; >= 0.3 in frames without any GT) outside the GT that the attacker matches
    to a person annotated in the same frame (duplicate: a reflection) or within NEAR frames (single: entering past
    the door frame, or before the annotation starts)."""
    val = RES.parent / "val"
    thr = __import__("json").loads((val / "judge.json").read_text())["threshold"]
    emb = {s: tuple(np.load(val / "emb" / f"{s}.npz")[k] for k in ("ids", "emb")) for s in splits.VAL}
    rows = []
    for seq in splits.VAL:
        tmpl = templates(emb, seq, ds.session)
        keys = sorted(tmpl)
        T = np.stack([tmpl[k] for k in keys])
        gt = ds.load_gt(seq)
        gb = dict(list(gt.groupby("frame")))
        d = pd.concat([pd.read_csv(val / "dets" / m / f"{seq}.csv") for m in ("yunet_480x640_npu", "yunet_ft_480x640_npu")])
        d = d[d.score >= 0.1]
        cand = []
        for f, g in d.groupby("frame"):
            if f not in gb:
                g = g[g.score >= 0.3]  # frames without annotation: fewer, surer boxes
            b = g[["x", "y", "w", "h"]].to_numpy()
            if f in gb:
                b = b[iou(b, gb[f][["x", "y", "w", "h"]].to_numpy()).max(1) < 0.1]
            if len(b) > 1:  # the two models' boxes on the same face: keep one
                keep = [0]
                for i in range(1, len(b)):
                    if iou(b[i:i + 1], b[keep]).max() < 0.3:
                        keep.append(i)
                b = b[keep]
            cand += [(seq, f, *r) for r in b]
        if not cand:
            continue
        c = pd.DataFrame(cand, columns=["seq", "frame", "x", "y", "w", "h"])
        crops = []
        for f, gf in c.groupby("frame", sort=False):
            img = cv2.imread(str(ds.frame_path(seq, int(f))))
            crops += [crop(img, r) for r in gf.itertuples()]
        c = pd.concat([gf for _, gf in c.groupby("frame", sort=False)])
        sims = judge.embed(crops) @ T.T
        c["identity"], c["sim_clean"] = np.array(keys)[sims.argmax(1)], sims.max(1)
        gf_ = {i: f.to_numpy() for i, f in gt.groupby("identity").frame}
        same = np.array([r.frame in gb and r.identity in set(gb[r.frame].identity) for r in c.itertuples()], bool)
        near = np.array([r.identity in gf_ and np.abs(gf_[r.identity] - r.frame).min() <= NEAR for r in c.itertuples()], bool)
        c = c[(c.sim_clean >= thr).to_numpy() & (same | (near & ~same))]
        c["kind"] = np.where([r.frame in gb and r.identity in set(gb[r.frame].identity) for r in c.itertuples()], "duplicate", "single")
        rows.append(c)
        print(seq, len(c), flush=True)
    out = pd.concat(rows, ignore_index=True).round(2)
    scene.REFLECTIONS.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(scene.REFLECTIONS, index=False)
    print(f"{len(out)} faces outside the GT on val ({(out.kind == 'duplicate').sum()} reflections) -> {scene.REFLECTIONS}")


def system_boxes(spec, seq, frames, shape, deg):
    det, steps = parse_spec(spec)
    d = pd.read_csv(RES / "dets" / f"{det}{degrade_suffix(deg)}" / f"{seq}.csv")
    t = THR.get(det) or SYSTEMS[det][1]
    if steps:
        risk = next((dict(v) for n, v in steps if n == "risk"), {})
        lum = luma(ds, seq, frames, RES, deg) if "dark" in risk else None
        zone = mot = None
        if "thr_zone" in risk:
            zone = scene.reflection_zone(seq, shape) if risk.get("zone_src") == "reflection" else scene.face_zone(ds, seq, shape)
            mot = scene.motion(ds, seq, frames, load_frame, deg, cache=RES / "motion" / f"{seq}{degrade_suffix(deg)}.npy")
        return run_policy(steps, d, frames, t, shape, lum=lum, zone=zone, motion=mot)
    return d[d.score >= t]


def score(judge, faces, spec, tmpl, thr, deg):
    rows = []
    for seq, g in faces.groupby("seq"):
        frames = ds.frame_numbers(seq)
        shape = cv2.imread(str(ds.frame_path(seq, frames[0]))).shape
        boxes = dict(list(system_boxes(spec, seq, frames, shape, deg).groupby("frame")))
        crops, idx = [], []
        for f, gf in g.groupby("frame"):
            d = boxes.get(f)
            xyxy = scale_boxes(d[["x", "y", "w", "h"]].to_numpy() if d is not None else np.empty((0, 4)), shape)
            mask = box_mask(xyxy, shape)
            anon = None
            for r in gf.itertuples():
                cov = coverage(mask, r.x, r.y, r.w, r.h)
                rows.append({**r._asdict(), "coverage": cov})
                if cov < PROTECTED_COVERAGE:
                    if anon is None:
                        anon = fast_blur(load_frame(ds.frame_path(seq, int(f)), int(f), deg), xyxy)
                    crops.append(crop(anon, r))
                    idx.append(len(rows) - 1)
        if crops:
            e = judge.embed(crops)
            for k, i in enumerate(idx):
                rows[i]["sim"] = float(e[k] @ tmpl[seq][rows[i]["identity"]])
    out = pd.DataFrame(rows).drop(columns="Index", errors="ignore")
    if "sim" not in out:
        out["sim"] = np.nan
    out["leaked"] = out.sim >= thr
    return out


def gt_leaked(spec, deg):
    """Identities the benchmark already counts as leaked (GT faces), from its cached scores."""
    det = parse_spec(spec)[0]
    d = RES / "scores" / f"{spec}_t{THR.get(det) or SYSTEMS[det][1]}{degrade_suffix(deg)}"
    if not d.exists():
        return None
    f = pd.concat([pd.read_csv(d / f"{s}.csv") for s in splits.TEST])
    return set(f[f.leaked].identity.astype(str).unique()), set(f.identity.astype(str).unique())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--systems", nargs="+", default=[])
    p.add_argument("--mine-val", action="store_true", help="write the val reflections for the reflection zone")
    p.add_argument("--degrade", default=None)
    p.add_argument("--tag", default=None)
    a = p.parse_args()
    cache = RES / "outside_gt"
    cache.mkdir(exist_ok=True)
    judge = Judge()
    if a.mine_val:
        mine_val(judge)
        return
    import json
    thr = json.loads((RES / "judge.json").read_text())["threshold"]
    emb = {s: tuple(np.load(RES / "emb" / f"{s}.npz")[k] for k in ("ids", "emb")) for s in splits.TEST}
    tmpl = {s: templates(emb, s, ds.session) for s in splits.TEST}
    fpath = cache / "faces.csv"
    if not fpath.exists():
        cand = pd.concat([mine(s) for s in splits.TEST], ignore_index=True)
        faces = label(judge, cand, tmpl, thr)
        faces.to_csv(fpath, index=False)
        print(f"mined {len(cand)} boxes outside GT, {len(faces)} identifiable faces of people in the video")
    faces = pd.read_csv(fpath)
    print(faces.groupby("kind").agg(faces=("frame", "size"), people=("identity", "nunique")).to_string())
    table = []
    for spec in a.systems:
        s = score(judge, faces, spec, tmpl, thr, a.degrade)
        s.to_csv(cache / f"{spec}{degrade_suffix(a.degrade)}.csv", index=False)
        leaked_out = set(s[s.leaked].identity.astype(str))
        g = gt_leaked(spec, a.degrade)
        row = {"system": spec, "degrade": a.degrade or "", "faces_outside_gt": len(s),
               "pct_protected": round(100 * (s.coverage >= PROTECTED_COVERAGE).mean(), 1),
               "pct_faces_leaked": round(100 * s.leaked.mean(), 1),
               "people_leaked_outside_gt": len(leaked_out)}
        for k in ("duplicate", "single"):
            row[f"people_leaked_{k}"] = s[s.leaked & (s.kind == k)].identity.nunique()
        if g:
            leaked_gt, people = g
            row |= {"people": len(people), "pct_people_leaked_gt": round(100 * len(leaked_gt) / len(people), 1),
                    "pct_people_leaked_gt_or_outside": round(100 * len(leaked_gt | leaked_out) / len(people), 1),
                    "people_only_outside_gt": len(leaked_out - leaked_gt)}
        table.append(row)
        print(pd.DataFrame([row]).to_string(index=False), flush=True)
    out = ROOT / "experiments" / "outside_gt"
    out.mkdir(parents=True, exist_ok=True)
    tag = a.tag or ("clean" if not a.degrade else degrade_suffix(a.degrade)[1:])
    pd.DataFrame(table).to_csv(out / f"{tag}.csv", index=False)


if __name__ == "__main__":
    main()
