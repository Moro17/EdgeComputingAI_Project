"""Set-level attacker: aggregate a person's anonymized faces over a whole appearance before matching.

The per-frame judge (benchmark.py) asks "is any single frame recognisable?". A stronger attacker holding the video
averages the ArcFace embeddings of every face crop of one appearance (one person in one sequence) taken from the
ANONYMIZED video, and matches that set template against the person's gallery, at the same 1%-FAR threshold.
Frames that are individually below threshold can add up.

Probe sets per appearance:
  unprotected  faces whose mask coverage is < 80% (what the per-frame judge also sees)
  all          every face, blurred ones included (tests whether fast_blur leaks in aggregate)

Reads the caches of a benchmark --out dir (emb/, judge.json, dets/), so run benchmark.py first for the same
systems, thresholds and degradation. Writes <out>/setlevel/<tag>.csv (one row per appearance) and prints a summary.

  python src/judge/setlevel.py --out results/rq1 --split test --systems yunet_480x640_npu+bytetrack+keep \\
      --thr yunet_480x640_npu=0.9 [--degrade dark:2]
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import splits  # noqa: E402
from benchmark import degrade_suffix, enroll, luma  # noqa: E402
from degrade.degrade import load as load_frame  # noqa: E402
from judge.arcface import Judge, align, templates  # noqa: E402
from metrics.exposure import PROTECTED_COVERAGE, coverage  # noqa: E402
from systems.anonymize import box_mask, fast_blur, scale_boxes  # noqa: E402
from systems.detectors import SYSTEMS  # noqa: E402
from systems.policies import parse_spec, run_policy  # noqa: E402


def appearance_probes(ds, judge, seq, frames, boxes, thr, deg):
    """Embeddings of every GT face of seq cut from the anonymized frames, with coverage. -> DataFrame + (n, 512)."""
    gt = ds.load_gt(seq)
    gt = gt[gt.frame.isin(frames)]
    by = dict(list(boxes[boxes.score >= thr].groupby("frame")))
    shape = cv2.imread(str(ds.frame_path(seq, frames[0]))).shape
    rows, crops = [], []
    for n, g in gt.groupby("frame"):
        d = by.get(n)
        xyxy = scale_boxes(d[["x", "y", "w", "h"]].to_numpy() if d is not None else np.empty((0, 4)), shape)
        mask = box_mask(xyxy, shape)
        anon = fast_blur(load_frame(ds.frame_path(seq, n), n, deg), xyxy)
        for r in g.itertuples():
            rows.append({"seq": seq, "frame": n, "identity": r.identity, "coverage": coverage(mask, r.x, r.y, r.w, r.h)})
            crops.append(align(anon, r))
    return pd.DataFrame(rows), judge.embed(crops)


def set_score(e, t):
    """Cosine similarity of the mean of L2-normalised embeddings e with template t (nan for an empty set)."""
    if not len(e):
        return np.nan
    m = e.mean(0)
    return float(m / np.linalg.norm(m) @ t)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="chokepoint")
    p.add_argument("--systems", nargs="+", required=True, metavar="SPEC")
    p.add_argument("--split", default="test")
    p.add_argument("--thr", nargs="*", default=[], metavar="SYSTEM=T")
    p.add_argument("--degrade", default=None, metavar="KIND:SEV")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    ds = importlib.import_module(f"loaders.{args.dataset}")
    seqs = getattr(splits, args.split.upper())
    data_cfg = json.loads((args.out / "data_config.json").read_text())
    if data_cfg["seqs"] != seqs or data_cfg.get("max_frames"):
        raise SystemExit(f"{args.out} was built for other sequences; run benchmark.py --split {args.split} there first")
    frames = {s: ds.frame_numbers(s) for s in seqs}
    thr = {k: t for k, (_, t) in SYSTEMS.items()} | {k: float(v) for k, v in (s.split("=") for s in args.thr)}
    judge = Judge()
    judge.threshold = json.loads((args.out / "judge.json").read_text())["threshold"]
    emb = {s: enroll(ds, judge, s, frames[s], args.out) for s in seqs}
    tmpl = {s: templates(emb, s, ds.session) for s in seqs}
    sfx = degrade_suffix(args.degrade)
    dst = args.out / "setlevel"
    dst.mkdir(exist_ok=True)

    summary = []
    for spec in args.systems:
        detector, steps = parse_spec(spec)
        t = thr.get(spec, thr[detector])
        rows = []
        for s in seqs:
            print(f"{spec} {s}", flush=True)
            dets = pd.read_csv(args.out / "dets" / f"{detector}{sfx}" / f"{s}.csv")
            shape = cv2.imread(str(ds.frame_path(s, frames[s][0]))).shape
            lum = luma(ds, s, frames[s], args.out, args.degrade) if any(
                n == "risk" and "dark" in dict(v) for n, v in steps) else None  # as benchmark.py: dark rule needs it
            boxes, bt = (run_policy(steps, dets, frames[s], t, shape, lum=lum), -1.0) if steps else (dets, t)
            faces, e = appearance_probes(ds, judge, s, frames[s], boxes, bt, args.degrade)
            faces["sim"] = [e[i] @ tmpl[s][k] if k in tmpl[s] else np.nan for i, k in enumerate(faces.identity)]
            for k, g in faces.groupby("identity"):
                if k not in tmpl[s]:
                    continue
                unp = (g.coverage < PROTECTED_COVERAGE).to_numpy()
                ei = e[g.index.to_numpy()]
                rows.append({"seq": s, "identity": k, "n_faces": len(g), "n_unprotected": int(unp.sum()),
                             "max_frame_sim": float(g.sim[unp].max()) if unp.any() else np.nan,
                             "set_sim_unprotected": set_score(ei[unp], tmpl[s][k]),
                             "set_sim_all": set_score(ei, tmpl[s][k])})
        r = pd.DataFrame(rows)
        for c in ["max_frame_sim", "set_sim_unprotected", "set_sim_all"]:
            r[c.replace("sim", "leak")] = r[c] >= judge.threshold
        tag = f"{spec}_t{t}{sfx}"
        r.round(4).to_csv(dst / f"{tag}.csv", index=False)
        pid = lambda col: 100 * r.groupby("identity")[col].any().mean()  # noqa: E731
        summary.append({"system": spec, "threshold": t, "degrade": args.degrade or "",
                        "pct_ids_frame_judge": pid("max_frame_leak"),
                        "pct_ids_set_unprotected": pid("set_leak_unprotected"),
                        "pct_ids_set_all": pid("set_leak_all")})
        print(pd.DataFrame(summary[-1:]).round(2).to_string(index=False), flush=True)
    pd.DataFrame(summary).round(3).to_csv(dst / f"summary{sfx}.csv", index=False)


if __name__ == "__main__":
    main()
