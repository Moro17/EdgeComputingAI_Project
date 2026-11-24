"""RQ1 benchmark: per-frame face anonymizers on a dataset (--dataset, default ChokePoint), scored at identity level.

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
  python src/benchmark.py --dataset musicvideo --out experiments/musicvideo   # see src/loaders/musicvideo.py
"""
import argparse
import importlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import splits
from degrade.degrade import load as load_frame, parse as parse_degrade
from judge.arcface import Judge, align, templates
from metrics.exposure import PROTECTED_COVERAGE, coverage, summarise
from systems.anonymize import MASK_SCALE, box_mask, fast_blur, scale_boxes
from systems.detectors import FLOOR, SYSTEMS, used_providers
from systems import scene
from systems.policies import parse_spec, run_policy

FAR = 0.01


def enroll(ds, judge, seq, frames, out):
    path = out / "emb" / f"{seq}.npz"
    if not path.exists():
        gt = ds.load_gt(seq)
        gt = gt[gt.frame.isin(frames)]
        crops, ids = [], []  # ids in crop order: groupby sorts by frame, the GT need not be (Music Video is by track)
        for n, g in gt.groupby("frame"):
            img = cv2.imread(str(ds.frame_path(seq, n)))
            crops += [align(img, r) for r in g.itertuples()]
            ids += g.identity.tolist()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, ids=np.asarray(ids), emb=judge.embed(crops))
    z = np.load(path)
    return z["ids"], z["emb"]


def calibrate(emb, session, out):
    """Threshold at FAR on clean faces vs other identities' leave-session-out templates; TAR as a sanity check."""
    path = out / "judge.json"
    if not path.exists():
        genuine, impostor = [], []
        for seq, (ids, e) in emb.items():
            t = templates(emb, seq, session)
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


def degrade_suffix(deg):
    """'dark:2' -> '@dark2' (cache dirs and tags of a degraded run); None -> ''."""
    return "" if deg is None else "@{}{}".format(*parse_degrade(deg))


def luma(ds, seq, frames, out, deg=None):
    """{frame: mean grey level 0-255} of the (degraded) frames, cached in lum/. The board gets the same number from
    the Y plane of its detector input; here the frame is shrunk to 160x120 first (a mean does not need more)."""
    path = out / "lum" / f"{seq}{degrade_suffix(deg)}.npy"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.array([cv2.cvtColor(cv2.resize(load_frame(ds.frame_path(seq, n), n, deg), (160, 120),
                                                        interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).mean()
                                for n in frames], np.float32))
    return dict(zip(frames, np.load(path).tolist()))


def detect(ds, system, det, seq, frames, out, deg=None):
    path = out / "dets" / f"{system}{degrade_suffix(deg)}" / f"{seq}.csv"
    if path.exists():
        return pd.read_csv(path), json.loads(path.with_suffix(".json").read_text())["latency_ms"]
    rows, lat = [], []
    for i, n in enumerate(frames):
        img = load_frame(ds.frame_path(seq, n), n, deg)
        t0 = time.perf_counter()
        boxes = det(img)
        if i >= 5:  # skip warm-up
            lat.append(1000 * (time.perf_counter() - t0))
        rows += [(n, -1, *b) for b in boxes.tolist()]
    df = pd.DataFrame(rows, columns=["frame", "track_id", "x", "y", "w", "h", "score"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.round({"x": 2, "y": 2, "w": 2, "h": 2, "score": 4}).to_csv(path, index=False)  # 2-dp scores flip thresholds
    path.with_suffix(".json").write_text(json.dumps({"latency_ms": lat}))
    return df, lat


def score(ds, judge, dets, thr, seq, frames, tmpl, out, tag, deg=None):
    """deg: the attacker sees the anonymized DEGRADED frame; enrollment and the judge threshold stay on clean frames."""
    path = out / "scores" / tag / f"{seq}.csv"
    if path.exists():
        return pd.read_csv(path), json.loads(path.with_suffix(".json").read_text())["masked"]
    gt = ds.load_gt(seq)
    gt = gt[gt.frame.isin(frames)]
    gt_by_frame = dict(list(gt.groupby("frame")))
    det_by_frame = dict(list(dets[dets.score >= thr].groupby("frame")))
    shape = cv2.imread(str(ds.frame_path(seq, frames[0]))).shape
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
                    anon = fast_blur(load_frame(ds.frame_path(seq, n), n, deg), xyxy)
                crops.append(align(anon, r))
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
    p.add_argument("--dataset", default="chokepoint", choices=["chokepoint", "musicvideo"], help="loader in src/loaders/")
    p.add_argument("--systems", nargs="+", default=list(SYSTEMS), metavar="SPEC",
                   help="detector[+skipN][+bytetrack][+hold|+coast] (src/systems/policies.py)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--seqs", nargs="+", default=None, help="default: every sequence with frames and GT on disk")
    g.add_argument("--split", choices=["train", "val", "test"], default=None, help="fixed ChokePoint split from src/splits.py")
    p.add_argument("--max-frames", type=int, default=None, help="first N frames per sequence (smoke tests)")
    p.add_argument("--thr", nargs="*", default=[], metavar="SYSTEM=T", help="override operating thresholds")
    p.add_argument("--egoblur", type=Path, default=None, help="path to ego_blur_face.jit (gated download)")
    p.add_argument("--degrade", default=None, metavar="KIND:SEV", help="stress tier, e.g. dark:2 (src/degrade/degrade.py)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    for spec in args.systems:
        try:
            if parse_spec(spec)[0] not in SYSTEMS:
                p.error(f"{spec}: unknown detector, choose from {list(SYSTEMS)}")
        except ValueError as e:
            p.error(str(e))
    if args.degrade:
        parse_degrade(args.degrade)
    sfx = degrade_suffix(args.degrade)

    thr = {k: t for k, (_, t) in SYSTEMS.items()} | {k: float(v) for k, v in (s.split("=") for s in args.thr)}
    thr |= {s: thr.get(s, thr[parse_spec(s)[0]]) for s in args.systems}  # a spec defaults to its detector's threshold
    systems = [s for s in args.systems if s != "egoblur" or (args.egoblur and args.egoblur.exists())]
    if systems != args.systems:
        print("skip egoblur: pass --egoblur with the path to ego_blur_face.jit")
    ds = importlib.import_module(f"loaders.{args.dataset}")
    seqs = args.seqs or (getattr(splits, args.split.upper()) if args.split else ds.sequences())
    if len({ds.session(s) for s in seqs}) < 2:
        p.error("need sequences from >= 2 recording sessions: each sequence is judged against a gallery from the others")
    frames = {s: ds.frame_numbers(s)[: args.max_frames] for s in seqs}
    args.out.mkdir(parents=True, exist_ok=True)
    data_cfg = {"seqs": seqs, "max_frames": args.max_frames} | ({"dataset": args.dataset} if args.dataset != "chokepoint" else {})
    cfg_path = args.out / "data_config.json"
    if cfg_path.exists() and json.loads(cfg_path.read_text()) != data_cfg:
        p.error(f"{args.out} was built with {cfg_path.read_text()}; caches would mix. Use a new --out.")
    cfg_path.write_text(json.dumps(data_cfg))

    judge = Judge()
    emb = {s: enroll(ds, judge, s, frames[s], args.out) for s in seqs}
    cal = calibrate(emb, ds.session, args.out)
    judge.threshold = cal["threshold"]
    print(f"judge: threshold {cal['threshold']:.3f} at {FAR:.0%} FAR, clean TAR {cal['tar_clean']:.1%}")
    tmpl = {s: templates(emb, s, ds.session) for s in seqs}

    table = []
    for system in systems:
        detector, steps = parse_spec(system)
        det = SYSTEMS[detector][0](args.egoblur)
        faces, masked, lat = [], [], []
        for s in seqs:
            print(f"{system} {s}", flush=True)
            d, l = detect(ds, detector, det, s, frames[s], args.out, args.degrade)
            t = thr[system]
            if steps:  # policy boxes are final: score() must not re-filter them (ByteTrack keeps low-score matches)
                shape = cv2.imread(str(ds.frame_path(s, frames[s][0]))).shape
                lum = luma(ds, s, frames[s], args.out, args.degrade) if any(
                    n == "risk" and "dark" in dict(v) for n, v in steps) else None
                zone = mot = None
                if any(n == "risk" and "thr_zone" in dict(v) for n, v in steps):
                    refl = any(n == "risk" and dict(v).get("zone_src") == "reflection" for n, v in steps)
                    zone = scene.reflection_zone(s, shape) if refl else scene.face_zone(ds, s, shape)
                    mot = scene.motion(ds, s, frames[s], load_frame, args.degrade,
                                       cache=args.out / "motion" / f"{s}{sfx}.npy")
                d, t = run_policy(steps, d, frames[s], t, shape, lum=lum, zone=zone, motion=mot), -1.0
            # shortcut: latency is the detector's on every frame, also with skipN (frame_latency, plan Task 7)
            f, m = score(ds, judge, d, t, s, frames[s], tmpl[s], args.out, f"{system}_t{thr[system]}{sfx}", args.degrade)
            faces.append(f)
            masked += m
            lat += l
        table.append({"system": system, "detector": detector, "policy": "+".join(n + (str(v) if v else "") for n, v in steps),
                      "threshold": thr[system], "degrade": args.degrade or "", **summarise(pd.concat(faces), masked, lat),
                      "device": str(used_providers(det.net) or getattr(det, "dev", "opencv-cpu"))})
        print(pd.DataFrame(table[-1:]).round(3).to_string(index=False), flush=True)

    df = pd.DataFrame(table)
    df.round(4).to_csv(args.out / f"results{sfx}.csv", index=False)
    (args.out / f"results{sfx}.json").write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(),
        "config": {"dataset": args.dataset, "degrade": args.degrade, "seqs": seqs, "max_frames": args.max_frames, "thresholds": thr, "det_floor": FLOOR,
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
