"""Remaining leaks of a policy: did the face detector see anything there (a raw detection at IoU >= 0.3, any score
>= floor)? Leaks with NO detection are the only ones a pose fallback could add; leaks with a weak detection are a
threshold problem (RQ3). Per condition, test split (repro_desktop) and val."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loaders.chokepoint import load_gt  # noqa: E402
from systems.bytelite import iou  # noqa: E402

R = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "results"  # dir with repro_desktop/ and val/
out = []
for root in ("repro_desktop", "val"):
    for sd in sorted((R / root / "scores").glob("*+lite+back3_t*")):
        name = sd.name
        det, rest = name.split("+lite+back3_t")
        thr, _, deg = rest.partition("@")
        dd = R / root / "dets" / (det + (f"@{deg}" if deg else ""))
        if not dd.exists():
            continue
        leaks = []
        for f in sorted(sd.glob("*.csv")):
            s = pd.read_csv(f)
            lk = s[s.leaked == True]  # noqa: E712
            if not len(lk):
                continue
            gt = load_gt(f.stem).set_index(["frame", "identity"])
            d = pd.read_csv(dd / f.name)
            byf = dict(list(d.groupby("frame")))
            for r in lk.itertuples():
                g = gt.loc[(r.frame, r.identity)]
                best = 0.0
                if r.frame in byf:
                    b = byf[r.frame]
                    m = iou(np.array([[g.x, g.y, g.w, g.h]]), b[["x", "y", "w", "h"]].to_numpy())[0]
                    best = b.score.to_numpy()[m >= 0.3].max() if (m >= 0.3).any() else 0.0
                leaks.append((r.identity, best))
        x = pd.DataFrame(leaks, columns=["identity", "best"])
        if not len(x):
            out.append((root, det.replace("_480x640_npu", ""), thr, deg or "clean", 0, 0, np.nan, np.nan, 0))
            continue
        per = x.groupby("identity").best.agg(lambda v: (v == 0).all())  # identity whose every leak had no detection
        out.append((root, det.replace("_480x640_npu", ""), thr, deg or "clean", len(x), x.identity.nunique(),
                    round((x.best == 0).mean() * 100, 1), round((x.best >= 0.3).mean() * 100, 1), int(per.sum())))
t = pd.DataFrame(out, columns=["split", "det", "thr", "cond", "leaked_faces", "leaked_ids", "pct_no_detection",
                               "pct_det_ge_0.3", "ids_only_undetected"])
print(t.to_string(index=False))
t.to_csv(R / "val" / "leak_anatomy.csv", index=False)
