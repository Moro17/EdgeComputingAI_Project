"""Does our blur defeat the judge? Fully cover sampled GT faces with the benchmark's own blur and try to recognise them.

A protected face (coverage >= 0.8) is never judged in the benchmark, so this check is what makes that rule safe.
Pass: blurred faces are recognised at about the judge's false-match rate (<= 2 %). Clean faces are the control.

Usage: python src/checks/blur_check.py --out experiments/rq1 [--sample 2000]   # needs emb/ + judge.json in --out
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from judge.arcface import Judge, eye_align, templates  # noqa: E402
from loaders.chokepoint import frame_path, load_gt  # noqa: E402
from systems.anonymize import blur, scale_boxes, weak_blur  # noqa: E402


def pixelate(img, xyxy, cells=6):
    out = img.copy()
    for x1, y1, x2, y2 in xyxy:
        if x2 > x1 and y2 > y1:
            small = cv2.resize(out[y1:y2, x1:x2], (cells, cells), interpolation=cv2.INTER_AREA)
            out[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    return out


def solid(img, xyxy):
    out = img.copy()
    for x1, y1, x2, y2 in xyxy:
        out[y1:y2, x1:x2] = 0
    return out


OPS = {"blur": blur, "weak_blur": weak_blur, "pixelate": pixelate, "solid": solid}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sample", type=int, default=2000)
    p.add_argument("--op", choices=list(OPS), default="blur", help="obfuscation to test; blur = the benchmark's")
    args = p.parse_args()
    op = OPS[args.op]

    seqs = json.loads((args.out / "data_config.json").read_text())["seqs"]
    thr = json.loads((args.out / "judge.json").read_text())["threshold"]
    emb = {s: (z["ids"], z["emb"]) for s in seqs for z in [np.load(args.out / "emb" / f"{s}.npz")]}
    gt = pd.concat([load_gt(s).assign(seq=s) for s in seqs])
    gt = gt.sample(min(args.sample, len(gt)), random_state=0)

    clean, blurred = [], []
    for r in gt.itertuples():
        img = cv2.imread(str(frame_path(r.seq, r.frame)))
        eyes = (r.lx, r.ly), (r.rx, r.ry)
        clean.append(eye_align(img, *eyes))
        blurred.append(eye_align(op(img, scale_boxes([[r.x, r.y, r.w, r.h]], img.shape)), *eyes))

    judge = Judge()
    tmpl = {s: templates(emb, s) for s in seqs}
    t = np.stack([tmpl[r.seq][r.identity] for r in gt.itertuples()])
    keys = sorted(tmpl[seqs[0]])
    wrong = np.stack([tmpl[r.seq][keys[(keys.index(r.identity) + 1) % len(keys)]] for r in gt.itertuples()])
    e_clean, e_blur = judge.embed(clean), judge.embed(blurred)
    gt["sim_clean"] = (e_clean * t).sum(1)
    gt["sim_blur"] = (e_blur * t).sum(1)
    gt["sim_blur_wrong"] = (e_blur * wrong).sum(1)  # same probes vs a fixed wrong identity: the chance level
    # rank-1: is the true identity the closest template? chance = 1 / number of identities
    gt["rank1_blur"] = [max(tmpl[r.seq], key=lambda k: e @ tmpl[r.seq][k]) == r.identity
                        for r, e in zip(gt.itertuples(), e_blur)]

    gt["size"] = pd.cut(gt.w, [0, 40, 60, 80, 1e9], labels=["<40px", "40-60px", "60-80px", ">=80px"])
    by_size = gt.groupby("size", observed=True).agg(
        n=("w", "size"), pct_clean=("sim_clean", lambda s: 100 * (s >= thr).mean()),
        pct_blur=("sim_blur", lambda s: 100 * (s >= thr).mean()))
    res = {"op": args.op, "n": len(gt), "threshold": thr,
           "pct_recognised_clean": 100 * float((gt.sim_clean >= thr).mean()),
           "pct_recognised": 100 * float((gt.sim_blur >= thr).mean()),
           "pct_recognised_wrong_identity": 100 * float((gt.sim_blur_wrong >= thr).mean()),
           "pct_rank1_blur": 100 * float(gt.rank1_blur.mean()), "pct_rank1_chance": 100 / len(keys),
           "mean_sim_clean": float(gt.sim_clean.mean()), "mean_sim_blur": float(gt.sim_blur.mean()),
           "mean_sim_blur_wrong": float(gt.sim_blur_wrong.mean()),
           "by_face_width": by_size.round(2).reset_index().to_dict("records")}
    assert res["pct_recognised_clean"] > 90, "control failed: clean faces not recognised, check the pipeline"
    (args.out / "checks").mkdir(exist_ok=True)
    (args.out / "checks" / f"{args.op}.json").write_text(json.dumps(res, indent=2, default=str))
    for i, r in enumerate(gt.head(8).itertuples()):  # a few blurred probes to eyeball
        cv2.imwrite(str(args.out / "checks" / f"{args.op}_probe_{i}.jpg"), np.hstack([clean[i], blurred[i]]))
    print(json.dumps(res, indent=2, default=str))
    print("PASS" if res["pct_recognised"] <= 2 else "FAIL: blur too weak, protected faces can leak")


if __name__ == "__main__":
    main()
