"""RQ1 benchmark: per-frame face anonymizers on ChokePoint, scored at identity level.

Stages (each cached under --out, so an interrupted run resumes where it stopped):
  1. enroll  emb/<seq>.npz           clean GT faces embedded by the ArcFace judge
  2. judge   judge.json              threshold at 1% FAR (impostor = face vs another identity's template)
  3. detect  dets/<system>/<seq>.csv all detections >= FLOOR + per-frame latency
  4. score   scores/<system>_t<thr>/<seq>.csv one row per GT face: coverage, protected, leaked
  5. table   results.csv + results.json

Usage:
  python src/benchmark.py --out results/rq1                          # all systems, every GT sequence on disk
  python src/benchmark.py --systems yunet deface --max-frames 200 --out results/smoke
  python src/benchmark.py --egoblur /path/ego_blur_face.jit --thr yunet=0.6 --out results/rq1
"""
import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from judge.arcface import Judge, eye_align, session, templates
from loaders.chokepoint import DATA, frame_numbers, frame_path, load_gt
from metrics.exposure import PROTECTED_COVERAGE, coverage, summarise
from systems.anonymize import MASK_SCALE, blur, box_mask, scale_boxes
from systems.detectors import FLOOR, SYSTEMS, used_providers

FAR = 0.01


def default_seqs():
    return sorted(p.name for p in (DATA / "frames").iterdir() if (DATA / "groundtruth" / f"{p.name}.xml").exists())


def enroll(judge, seq, frames, out):
    path = out / "emb" / f"{seq}.npz"
    if not path.exists():
        gt = load_gt(seq)
        gt = gt[gt.frame.isin(frames)]
        crops = []
        for n, g in gt.groupby("frame"):
            img = cv2.imread(str(frame_path(seq, n)))
            crops += [eye_align(img, (r.lx, r.ly), (r.rx, r.ry)) for r in g.itertuples()]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, ids=gt.identity.to_numpy(), emb=judge.embed(crops))
    z = np.load(path)
    return z["ids"], z["emb"]


def calibrate(emb, out):
    """Threshold at FAR on clean faces vs other identities' leave-session-out templates; TAR as a sanity check."""
    path = out / "judge.json"
    if not path.exists():
        genuine, impostor = [], []
        for seq, (ids, e) in emb.items():
            t = templates(emb, seq)
            keys = np.array(sorted(t))
            sims = e @ np.stack([t[k] for k in keys]).T
            own = ids[:, None] == keys[None, :]
            genuine += list(sims[own])
            impostor += list(sims[~own])
        thr = float(np.quantile(impostor, 1 - FAR))
        res = {"far": FAR, "threshold": thr, "tar_clean": float(np.mean(np.array(genuine) >= thr)),
               "n_genuine": len(genuine), "n_impostor": len(impostor)}
        path.write_text(json.dumps(res, indent=2))
    return json.loads(path.read_text())


def detect(system, det, seq, frames, out):
    path = out / "dets" / system / f"{seq}.csv"
    if path.exists():
        return pd.read_csv(path), json.loads(path.with_suffix(".json").read_text())["latency_ms"]
    rows, lat = [], []
    for i, n in enumerate(frames):
        img = cv2.imread(str(frame_path(seq, n)))
        t0 = time.perf_counter()
        boxes = det(img)
        if i >= 5:  # skip warm-up
            lat.append(1000 * (time.perf_counter() - t0))
        rows += [(n, -1, *b) for b in boxes.tolist()]
    df = pd.DataFrame(rows, columns=["frame", "track_id", "x", "y", "w", "h", "score"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.round(2).to_csv(path, index=False)
    path.with_suffix(".json").write_text(json.dumps({"latency_ms": lat}))
    return df, lat


def score(judge, dets, thr, seq, frames, tmpl, out, tag):
    path = out / "scores" / tag / f"{seq}.csv"
    if path.exists():
        return pd.read_csv(path), json.loads(path.with_suffix(".json").read_text())["masked"]
    gt = load_gt(seq)
    gt = gt[gt.frame.isin(frames)]
    gt_by_frame = dict(list(gt.groupby("frame")))
    det_by_frame = dict(list(dets[dets.score >= thr].groupby("frame")))
    shape = cv2.imread(str(frame_path(seq, frames[0]))).shape
    keys = sorted(tmpl)
    null_of = {k: keys[(i + 1) % len(keys)] for i, k in enumerate(keys)}  # a fixed wrong identity per person

    rows, masked, crops = [], [], []
    for n in frames:
        d = det_by_frame.get(n)
        xyxy = scale_boxes(d[["x", "y", "w", "h"]].to_numpy() if d is not None else np.empty((0, 4)), shape)
        mask = box_mask(xyxy, shape)
        masked.append(float(mask.mean()))
        if n not in gt_by_frame:
            continue
        anon = None
        for r in gt_by_frame[n].itertuples():
            cov = coverage(mask, r.x, r.y, r.w, r.h)
            row = {"seq": seq, "frame": n, "identity": r.identity, "coverage": cov, "protected": cov >= PROTECTED_COVERAGE}
            if not row["protected"] and r.identity in tmpl:
                if anon is None:
                    anon = blur(cv2.imread(str(frame_path(seq, n))), xyxy)
                crops.append(eye_align(anon, (r.lx, r.ly), (r.rx, r.ry)))
                row["probe"] = len(crops) - 1
            rows.append(row)

    faces = pd.DataFrame(rows)
    e = judge.embed(crops)
    sim = np.full(len(faces), np.nan)
    sim_null = np.full(len(faces), np.nan)
    if crops:
        p = faces.probe.dropna().astype(int)
        sim[p.index] = [e[i] @ tmpl[faces.identity[j]] for j, i in p.items()]
        sim_null[p.index] = [e[i] @ tmpl[null_of[faces.identity[j]]] for j, i in p.items()]
    faces = faces.drop(columns="probe", errors="ignore")
    faces["sim"], faces["sim_null"] = sim, sim_null
    faces["leaked"] = faces.sim >= judge.threshold
    faces["leaked_null"] = faces.sim_null >= judge.threshold
    path.parent.mkdir(parents=True, exist_ok=True)
    faces.round(4).to_csv(path, index=False)
    path.with_suffix(".json").write_text(json.dumps({"masked": masked}))
    return faces, masked


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=list(SYSTEMS))
    p.add_argument("--seqs", nargs="+", default=None, help="default: every sequence with frames and GT on disk")
    p.add_argument("--max-frames", type=int, default=None, help="first N frames per sequence (smoke tests)")
    p.add_argument("--thr", nargs="*", default=[], metavar="SYSTEM=T", help="override operating thresholds")
    p.add_argument("--egoblur", type=Path, default=None, help="path to ego_blur_face.jit (gated download)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    thr = {k: t for k, (_, t) in SYSTEMS.items()} | {k: float(v) for k, v in (s.split("=") for s in args.thr)}
    systems = [s for s in args.systems if s != "egoblur" or (args.egoblur and args.egoblur.exists())]
    if systems != args.systems:
        print("skip egoblur: pass --egoblur with the path to ego_blur_face.jit")
    seqs = args.seqs or default_seqs()
    if len({session(s) for s in seqs}) < 2:
        p.error("need sequences from >= 2 recording sessions: each sequence is judged against a gallery from the others")
    frames = {s: frame_numbers(s)[: args.max_frames] for s in seqs}
    args.out.mkdir(parents=True, exist_ok=True)
    data_cfg, cfg_path = {"seqs": seqs, "max_frames": args.max_frames}, args.out / "data_config.json"
    if cfg_path.exists() and json.loads(cfg_path.read_text()) != data_cfg:
        p.error(f"{args.out} was built with {cfg_path.read_text()}; caches would mix. Use a new --out.")
    cfg_path.write_text(json.dumps(data_cfg))

    judge = Judge()
    emb = {s: enroll(judge, s, frames[s], args.out) for s in seqs}
    cal = calibrate(emb, args.out)
    judge.threshold = cal["threshold"]
    print(f"judge: threshold {cal['threshold']:.3f} at {FAR:.0%} FAR, clean TAR {cal['tar_clean']:.1%}")
    tmpl = {s: templates(emb, s) for s in seqs}

    table = []
    for system in systems:
        det = SYSTEMS[system][0](args.egoblur)
        faces, masked, lat = [], [], []
        for s in seqs:
            print(f"{system} {s}", flush=True)
            d, l = detect(system, det, s, frames[s], args.out)
            f, m = score(judge, d, thr[system], s, frames[s], tmpl[s], args.out, f"{system}_t{thr[system]}")
            faces.append(f)
            masked += m
            lat += l
        table.append({"system": system, "threshold": thr[system], **summarise(pd.concat(faces), masked, lat),
                      "device": str(used_providers(det.net) or getattr(det, "dev", "opencv-cpu"))})
        print(pd.DataFrame(table[-1:]).round(3).to_string(index=False), flush=True)

    df = pd.DataFrame(table)
    df.round(4).to_csv(args.out / "results.csv", index=False)
    (args.out / "results.json").write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(),
        "config": {"seqs": seqs, "max_frames": args.max_frames, "thresholds": thr, "det_floor": FLOOR,
                   "mask_scale": MASK_SCALE, "protected_coverage": PROTECTED_COVERAGE, "far": FAR},
        "judge": cal | {"providers": used_providers(judge.net)},
        "host": {"platform": platform.platform(), "processor": platform.processor(), "gpu": gpu_name()},
        "results": table,
    }, indent=2, default=float))
    print(df.round(3).to_string(index=False))


def gpu_name():
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        return None


if __name__ == "__main__":
    main()
