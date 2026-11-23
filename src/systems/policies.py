"""Policies: everything after the detector, as pure functions on cached detections.

A system is a spec string "detector+step+step...", e.g. "yunet_ort+skip3+bytetrack+coast". The benchmark detects
once per detector; any number of policies then run on that cache, so baselines and ablations cost seconds.

Steps, in this order:
  skipN      detector runs only on every N-th frame (keyframes)                  baseline B3
  risk       risk-adaptive threshold (RQ3, systems/bytelite.py RiskGate): a detection in [low, thr) is   ours, board
             promoted to thr if its box is tall (_hH, percent of frame height) or it persisted K frames (_pK); knobs
             or the frame is dark (_dL_tT: mean grey < L -> threshold T); knobs risk_l0.3_h13_p3_d60_t0.4 =
             low, h_min, persist, dark, thr_dark (at least one rule)
  windowW    every frame also gets the detections of the W frames before and     our re-implementation of
             after it, merged by NMS (IoU 0.5); W frames of extra latency        MBAM's temporal fusion
  bytetrack  ByteTrack (supervision implementation) over keyframe detections     baseline B2
  lite       numpy ByteTrack-lite (systems/bytelite.py, the tracker that runs on the board): every   ours, board
             detection >= thr plus low-score ones associated to a track
  keep       fail-closed: also mask every raw detection >= thr that the tracker   ours (needs bytetrack)
             dropped (ByteTrack hides a new track's first frame and re-born tracks)
  hold       copy the last keyframe's boxes onto the frames in between           B3 (exclusive with coast)
  coast      continue lost tracks with constant velocity, box grown              ours v1 (needs bytetrack)
  budget     per-track exposure budget: risk from on-device signals (missed or    ours v2 (needs bytetrack;
             low-score frames) grows the mask and lengthens coasting              replaces coast)
             knobs in the spec: budget_d0.7_g1.8_a10-60 = decay, grow_max, age_min-age_max (each optional)
  pose / posenew  pose fallback                                                  ours v2 (not implemented yet)
  backW      every final box is also masked on the W frames before it (look-     ours: causal coasting covers
             ahead of W frames, like windowW): covers an appearance's first      the future, back the start
             frames, before the detector found it
  spreadW    every final box is also masked on the W frames before AND after it  ablation: windowW without
             (union, no NMS); same W-frame latency as windowW                    NMS (does NMS hurt?)
  growP      scale every final box by P percent around its centre (grow140 =     control: compare policies
             1.4x), on top of the benchmark's mask scale                         at equal masked area

Output boxes: COLS, kind in {det, hold, coast}.
"""
import re

import numpy as np
import pandas as pd

COLS = ["frame", "track_id", "x", "y", "w", "h", "score", "kind"]
ORDER = ["skip", "risk", "window", "bytetrack", "lite", "keep", "hold", "coast", "budget", "pose", "posenew", "back", "spread",
         "grow"]


def parse_spec(spec):
    """'yunet+skip3+bytetrack+coast' -> ('yunet', [('skip', 3), ('bytetrack', None), ('coast', None)])."""
    det, *tokens = spec.split("+")
    steps = []
    for t in tokens:
        m = re.fullmatch(r"(skip|window|back|spread|grow)(\d+)", t)
        b = re.fullmatch(r"budget(?:_d([\d.]+))?(?:_g([\d.]+))?(?:_a(\d+)-(\d+))?", t)
        r = re.fullmatch(r"risk(?:_l([\d.]+))?(?:_h(\d+))?(?:_p(\d+))?(?:_d(\d+)_t([\d.]+))?", t)
        if r:
            lo, h, p, dk, td = r.groups()
            kw = {k: v for k, v in (("low", lo and float(lo)), ("h_min", h and int(h)), ("persist", p and int(p)),
                                    ("dark", dk and int(dk)), ("thr_dark", td and float(td))) if v is not None}
            if not kw.get("h_min") and kw.get("persist", 0) < 2 and not kw.get("dark"):
                raise ValueError(f"{spec}: risk needs _hH (height), _pK with K >= 2 (persistence) and/or _dL_tT (dark)")
            steps.append(("risk", tuple(sorted(kw.items()))))
        elif b and t != "budget":
            d, g, a0, a1 = b.groups()
            kw = {k: v for k, v in (("decay", d and float(d)), ("grow_max", g and float(g)),
                                    ("age_min", a0 and int(a0)), ("age_max", a1 and int(a1))) if v is not None}
            if not 0 <= kw.get("decay", .5) < 1 or kw.get("grow_max", 1) < 1 or kw.get("age_min", 0) > kw.get("age_max", 1e9):
                raise ValueError(f"{spec}: budget knobs out of range")
            steps.append(("budget", tuple(sorted(kw.items()))))
        elif m:
            if int(m.group(2)) < 1 or (m.group(1) == "grow" and int(m.group(2)) < 100):
                raise ValueError(f"{spec}: {m.group(1)}N needs N >= 1 (grow: N >= 100 percent)")
            steps.append((m.group(1), int(m.group(2))))
        elif t in ORDER[1:] and t not in ("risk", "window", "back", "spread", "grow"):
            steps.append((t, None))
        else:
            raise ValueError(f"{spec}: unknown step {t!r}")
    names = [s for s, _ in steps]
    if names != sorted(names, key=ORDER.index) or len(set(names)) != len(names):
        raise ValueError(f"{spec}: steps must appear once, in the order {ORDER}")
    if sum(n in names for n in ("hold", "coast", "budget")) > 1:
        raise ValueError(f"{spec}: hold, coast and budget are exclusive")
    if "budget" in names and "bytetrack" not in names:
        raise ValueError(f"{spec}: budget needs bytetrack")
    if "lite" in names and set(names) & {"bytetrack", "keep", "coast", "budget"}:
        raise ValueError(f"{spec}: lite is its own tracker with keep built in; combine it with hold/back/spread/grow only")
    if "skip" in names and not set(names) & {"hold", "coast", "budget"}:  # else in-between frames go unmasked
        raise ValueError(f"{spec}: skip needs hold, coast or budget")
    if "keep" in names and "bytetrack" not in names:
        raise ValueError(f"{spec}: keep needs bytetrack")
    if "coast" in names and "bytetrack" not in names:
        raise ValueError(f"{spec}: coast needs bytetrack")
    if ("pose" in names or "posenew" in names) and "coast" not in names and "budget" not in names:
        raise ValueError(f"{spec}: pose needs coast")
    return det, steps


def empty():
    return pd.DataFrame({c: pd.Series(dtype=float if c in "xywhscore" else object) for c in COLS})[COLS]


def keyframes(frames, n):
    return list(frames[::n])


def skip(dets, keys):
    """Detections on keyframes only, kind='det'."""
    out = dets[dets.frame.isin(keys)].copy()
    out["kind"] = "det"
    return out[COLS] if len(out) else empty()


def hold(boxes, frames, keys):
    """Each non-key frame gets the boxes of the latest keyframe at or before it (kind='hold')."""
    keys = sorted(keys)
    rows = [boxes]
    by_frame = dict(list(boxes.groupby("frame")))
    for f in frames:
        if f in keys:
            continue
        prev = [k for k in keys if k <= f]
        if prev and prev[-1] in by_frame:
            b = by_frame[prev[-1]].copy()
            b["frame"], b["kind"] = f, "hold"
            rows.append(b)
    out = pd.concat([r for r in rows if len(r)], ignore_index=True) if any(len(r) for r in rows) else empty()
    return out[COLS].sort_values(["frame", "track_id"], kind="stable").reset_index(drop=True)


def nms(b, iou=0.5):
    """Greedy NMS on a COLS frame of one frame's boxes, highest score first."""
    b = b.sort_values("score", ascending=False).reset_index(drop=True)
    x1, y1 = b.x.to_numpy(), b.y.to_numpy()
    x2, y2 = x1 + b.w.to_numpy(), y1 + b.h.to_numpy()
    area = (x2 - x1) * (y2 - y1)
    keep, alive = [], np.ones(len(b), bool)
    for i in range(len(b)):
        if not alive[i]:
            continue
        keep.append(i)
        iw = np.clip(np.minimum(x2[i], x2) - np.maximum(x1[i], x1), 0, None)
        ih = np.clip(np.minimum(y2[i], y2) - np.maximum(y1[i], y1), 0, None)
        inter = iw * ih
        alive &= inter / (area[i] + area - inter + 1e-9) <= iou
    return b.iloc[keep]


def window(boxes, frames, w):
    """Our re-implementation of MBAM's temporal box fusion (Khalili et al., IROS 2026), simplified: the boxes of
    frames f-w..f+w are merged into frame f by NMS. A miss is recovered if a neighbour within w frames saw the face.
    Not reimplemented: MBAM's IoU voting/clustering details, hybrid heavy/light scheduling, frame decorrelation."""
    if not len(boxes):
        return empty()
    by = dict(list(boxes.groupby("frame")))
    rows = []
    for i, f in enumerate(frames):
        near = [by[g] for g in frames[max(0, i - w): i + w + 1] if g in by]
        if not near:
            continue
        m = nms(pd.concat(near, ignore_index=True))
        m = m.assign(frame=f, kind=np.where(m.frame == f, m.kind, "window"))
        rows.append(m)
    return pd.concat(rows, ignore_index=True)[COLS]


def bytetrack(dets, keys, thr):
    """ByteTrack over keyframes: a fresh tracker per call (ids never leak across sequences).

    Gets every detection >= 0.1 (its low-score association) and uses thr as the high threshold. Integer track_id.
    """
    import supervision as sv

    tracker = sv.ByteTrack(track_activation_threshold=thr, lost_track_buffer=30, minimum_matching_threshold=0.8,
                           frame_rate=30, minimum_consecutive_frames=1)
    tracker.det_thresh = thr  # supervision starts new tracks at thr + 0.1 (never, at thr 0.9); start them at thr, like B1
    d = dets[dets.score >= 0.1]
    by_frame = dict(list(d.groupby("frame")))
    rows = []
    for k in keys:
        g = by_frame.get(k)
        if g is None:
            det = sv.Detections.empty()
        else:
            xyxy = g[["x", "y", "w", "h"]].to_numpy(np.float32, copy=True)
            xyxy[:, 2:] += xyxy[:, :2]
            det = sv.Detections(xyxy=xyxy, confidence=g.score.to_numpy(np.float32))
        t = tracker.update_with_detections(det)
        for (x1, y1, x2, y2), s, tid in zip(t.xyxy, t.confidence, t.tracker_id):
            rows.append((k, int(tid), float(x1), float(y1), float(x2 - x1), float(y2 - y1), float(s), "det"))
    return pd.DataFrame(rows, columns=COLS) if rows else empty()


def coast(tracks, frames, shape, max_age=15, grow=1.25):
    """Continue each track through frames where it has no box: constant velocity, box grown by `grow`.

    Stops after max_age frames (index distance in `frames`) or when the track is observed again. Boxes fully
    outside the frame are dropped (partly outside ones are clipped later by scale_boxes).
    """
    if not len(tracks):
        return empty()
    H, W = shape[:2]
    index = {f: i for i, f in enumerate(frames)}
    rows = []
    for tid, g in tracks[tracks.track_id >= 0].groupby("track_id"):  # keep's raw rows (track_id -1) are not a track
        g = g.sort_values("frame")
        obs = [(index[r.frame], r) for r in g.itertuples() if r.frame in index]
        for j, (i, r) in enumerate(obs):
            nxt = obs[j + 1][0] if j + 1 < len(obs) else len(frames)
            cx, cy = r.x + r.w / 2, r.y + r.h / 2
            if j > 0:
                pi, p = obs[j - 1]
                v = np.array([cx - (p.x + p.w / 2), cy - (p.y + p.h / 2)]) / (i - pi)
            else:
                v = np.zeros(2)
            w, h = r.w * grow, r.h * grow
            for k in range(1, min(max_age, nxt - i - 1) + 1):
                c = np.array([cx, cy]) + v * k
                x, y = c[0] - w / 2, c[1] - h / 2
                if x >= W or y >= H or x + w <= 0 or y + h <= 0:
                    break
                rows.append((frames[i + k], tid, x, y, w, h, r.score, "coast"))
    out = pd.concat([tracks, pd.DataFrame(rows, columns=COLS)], ignore_index=True) if rows else tracks
    return out[COLS].sort_values(["frame", "track_id"], kind="stable").reset_index(drop=True)


def budget(tracks, frames, shape, thr, decay=0.8, grow_max=1.6, age_min=15, age_max=45):
    """Per-track exposure budget (ours v2). Each track carries a risk r in [0, 1], updated every frame:

        r <- decay * r + (1 - decay) * m,   m = 1 if the track is unobserved or its score < thr, else 0

    using only what the device knows (no judge). Risk acts in two places:
      - every box of the track, detected or coasted, is scaled by 1 + (grow_max - 1) * r (a fragile track gets a
        bigger mask: localisation is less certain and a partly covered face is what leaks);
      - a lost track is coasted for age_min + (age_max - age_min) * r frames instead of a fixed 15.
    A new track starts at r = 1 (most leaks in tracking baselines are a track's first frames) and calms down as it is
    seen confidently. Constant-velocity motion as in coast(). Defaults are ablation knobs, not tuned on test.
    """
    if not len(tracks):
        return empty()
    H, W = shape[:2]
    index = {f: i for i, f in enumerate(frames)}
    rows = []
    for tid, g in tracks[tracks.track_id >= 0].groupby("track_id"):
        obs = {index[r.frame]: r for r in g.itertuples() if r.frame in index}
        seen = sorted(obs)
        r, last, prev, i = 1.0, None, None, seen[0]
        while i < len(frames):
            o = obs.get(i)
            r = decay * r + (1 - decay) * (0.0 if o is not None and o.score >= thr else 1.0)
            s = 1 + (grow_max - 1) * r
            if o is not None:
                prev, last = last, (i, o)
                cx, cy, w, h, sc, kind = o.x + o.w / 2, o.y + o.h / 2, o.w, o.h, o.score, o.kind
            else:
                k = i - last[0]
                if k > age_min + (age_max - age_min) * r:
                    nxt = [j for j in seen if j > i]
                    if not nxt:
                        break
                    i = nxt[0]  # wait for the track to be observed again
                    continue
                lo = last[1]
                v = np.zeros(2)
                if prev is not None:
                    po = prev[1]
                    v = np.array([lo.x + lo.w / 2 - po.x - po.w / 2, lo.y + lo.h / 2 - po.y - po.h / 2]) / (last[0] - prev[0])
                cx, cy = lo.x + lo.w / 2 + v[0] * k, lo.y + lo.h / 2 + v[1] * k
                w, h, sc, kind = lo.w, lo.h, lo.score, "coast"
            bw, bh = w * s, h * s
            x, y = cx - bw / 2, cy - bh / 2
            if not (x >= W or y >= H or x + bw <= 0 or y + bh <= 0):
                rows.append((frames[i], tid, x, y, bw, bh, sc, kind))
            i += 1
    extra = tracks[tracks.track_id < 0]  # keep's raw detections: no track, masked as they are
    out = pd.concat([pd.DataFrame(rows, columns=COLS), extra], ignore_index=True)
    return out[COLS].sort_values(["frame", "track_id"], kind="stable").reset_index(drop=True)


def back(boxes, frames, w, offsets=None, kind="back"):
    """Each box is copied onto the w frames before it (kind='back'); the masks of a frame are a union, so
    duplicates are harmless. Needs a w-frame delay buffer (w / fps seconds of latency)."""
    if not len(boxes):
        return boxes
    index = {f: i for i, f in enumerate(frames)}
    rows = [boxes]
    for k in offsets or range(1, w + 1):
        i = boxes.frame.map(index) - k
        ok = (i >= 0) & (i < len(frames))
        b = boxes[ok].copy()
        b["frame"] = [frames[j] for j in i[ok].astype(int)]
        b["kind"] = kind
        rows.append(b)
    out = pd.concat(rows, ignore_index=True)
    return out[COLS].sort_values(["frame", "track_id"], kind="stable").reset_index(drop=True)


def run_policy(steps, dets, frames, thr, shape, lum=None):
    """Boxes to blur for one sequence. Without bytetrack, detections are filtered at thr first.
    lum: {frame: mean grey level}, needed by risk's dark rule (benchmark.luma)."""
    names = dict(steps)
    keys = keyframes(frames, names.get("skip", 1))
    if names.get("skip", 1) > 1:  # the detector never ran on the other frames (as on the board)
        dets = dets[dets.frame.isin(set(keys))]
    if "pose" in names or "posenew" in names:
        raise NotImplementedError("pose fallback: docs/superpowers/plans/2026-10-28-miss-aware-pipeline.md Task 10")
    if "risk" in names:
        from systems.bytelite import RiskGate
        kw = dict(names["risk"])
        kw["h_min"] = kw.get("h_min", 0) * shape[0] / 100  # percent of frame height -> pixels
        gate, parts, idx, prev = RiskGate(thr, **kw), [], {f: i for i, f in enumerate(frames)}, None
        for k, g in dets.groupby("frame", sort=True):  # online over the frames that have detections; a frame
            g = g.copy()                                 # without any breaks persistence, as on the board
            if prev is not None and idx.get(k, -9) - idx.get(prev, -99) != 1:
                gate.hist = []
            prev = k
            g["score"] = gate.update(g[["x", "y", "w", "h", "score"]].to_numpy(), lum[k] if lum else None)[:, 4]
            parts.append(g)
        dets = pd.concat(parts, ignore_index=True) if parts else dets
    if "lite" in names:
        from systems.bytelite import LiteTracker
        tr, by, rows = LiteTracker(thr), dict(list(dets.groupby("frame"))), []
        for k in keys:
            g = by.get(k)
            m = tr.update(g[["x", "y", "w", "h", "score"]].to_numpy() if g is not None else np.empty((0, 5)))
            rows += [(k, -1, *b[:4], b[4], "det") for b in m]
        boxes = pd.DataFrame(rows, columns=COLS) if rows else empty()
    elif "bytetrack" in names:
        boxes = bytetrack(dets, keys, thr)
        if "keep" in names:
            raw = skip(dets[dets.score >= thr], keys)
            boxes = pd.concat([b for b in (boxes, raw) if len(b)], ignore_index=True)[COLS] if len(raw) else boxes
    else:
        boxes = skip(dets[dets.score >= thr], keys)
    if "window" in names:
        boxes = window(boxes, frames, names["window"])
    if "hold" in names:
        boxes = hold(boxes, frames, keys)
    if "coast" in names:
        boxes = coast(boxes, frames, shape)
    if "budget" in names:
        boxes = budget(boxes, frames, shape, thr, **dict(names["budget"] or ()))
    if "back" in names:
        boxes = back(boxes, frames, names["back"])
    if "spread" in names:
        w = names["spread"]
        boxes = back(boxes, frames, w, offsets=[k for k in range(-w, w + 1) if k], kind="spread")
    if "grow" in names and len(boxes):
        g = names["grow"] / 100
        boxes = boxes.assign(x=boxes.x - boxes.w * (g - 1) / 2, y=boxes.y - boxes.h * (g - 1) / 2,
                             w=boxes.w * g, h=boxes.h * g)
    return boxes
