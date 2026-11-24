"""ByteTrack-lite: numpy-only online tracker for the board (no supervision, scipy or pandas there).

What the privacy gain needs from ByteTrack (ablation, docs/next-runs.md): faces whose score dips below the threshold
for a few frames stay masked because a low-score detection (>= LOW) is associated to an existing track. So:

  1. predict every track with constant velocity;
  2. match detections >= thr to tracks by IoU (greedy, highest IoU first, IoU >= IOU_HIGH);
  3. match the still-unmatched tracks to low-score detections (LOW <= s < thr) at IoU >= IOU_LOW;
  4. unmatched detections >= thr start tracks; tracks unseen for MAX_LOST frames are dropped.

Boxes to mask = every detection >= thr ("keep": fail-closed, nothing the tracker hides) + the matched low-score ones.
Greedy matching instead of ByteTrack's Hungarian and no Kalman covariance: our re-implementation, not ByteTrack.
"""
import numpy as np

LOW, IOU_HIGH, IOU_LOW, MAX_LOST = 0.1, 0.2, 0.5, 30  # ByteTrack defaults: match_thresh 0.8 (cost), 0.5, buffer 30


def iou(a, b):
    """(n, 4) x (m, 4) xywh -> (n, m) IoU."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2) - np.maximum(a[:, 0, None], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2) - np.maximum(a[:, 1, None], b[:, 1]), 0, None)
    inter = iw * ih
    return inter / (a[:, 2:].prod(1)[:, None] + b[:, 2:].prod(1) - inter + 1e-9)


def greedy(m, thr):
    """Pairs (i, j) of a similarity matrix, best first, each row/column once, sim >= thr."""
    pairs, used_i, used_j = [], set(), set()
    if m.size:
        for k in np.argsort(-m, axis=None):
            i, j = divmod(int(k), m.shape[1])
            if m[i, j] < thr:
                break
            if i not in used_i and j not in used_j:
                pairs.append((i, j))
                used_i.add(i)
                used_j.add(j)
    return pairs


class LiteTracker:
    def __init__(self, thr, low=LOW, max_lost=MAX_LOST):
        self.thr, self.low, self.max_lost = thr, low, max_lost
        self.box = np.zeros((0, 4))  # xywh
        self.vel = np.zeros((0, 2))
        self.lost = np.zeros(0, int)

    def update(self, dets):
        """dets (n, 5) x, y, w, h, score of one frame (any score) -> (k, 5) boxes to mask."""
        dets = np.asarray(dets, float).reshape(-1, 5)
        high, low = dets[dets[:, 4] >= self.thr], dets[(dets[:, 4] >= self.low) & (dets[:, 4] < self.thr)]
        pred = self.box.copy()
        pred[:, :2] += self.vel
        matched_t, out_low = set(), []
        first = greedy(iou(pred, high[:, :4]), IOU_HIGH)
        for i, j in first:
            self._set(i, high[j, :4])
            matched_t.add(i)
        rest = [i for i in range(len(pred)) if i not in matched_t]
        for k, j in greedy(iou(pred[rest], low[:, :4]), IOU_LOW):
            i = rest[k]
            self._set(i, low[j, :4])
            matched_t.add(i)
            out_low.append(low[j])
        new = [j for j in range(len(high)) if j not in {jj for _, jj in first}]
        self.lost += 1
        for i in range(len(self.lost)):
            if i in matched_t:
                self.lost[i] = 0
            else:
                self.box[i, :2] += self.vel[i]  # a lost track keeps moving, so it can be re-found where it went
        if new:
            self.box = np.vstack([self.box, high[new, :4]])
            self.vel = np.vstack([self.vel, np.zeros((len(new), 2))])
            self.lost = np.concatenate([self.lost, np.zeros(len(new), int)])
        alive = self.lost <= self.max_lost
        self.box, self.vel, self.lost = self.box[alive], self.vel[alive], self.lost[alive]
        return np.vstack([high, np.array(out_low).reshape(-1, 5)]) if out_low else high

    def _set(self, i, b):
        c_old = self.box[i, :2] + self.box[i, 2:] / 2
        c_new = b[:2] + b[2:] / 2
        self.vel[i] = 0.5 * self.vel[i] + 0.5 * (c_new - c_old)  # smoothed centre velocity, pixels per frame
        self.box[i] = b


def box_motion(maps, boxes, frame_hw):
    """Mean of a small motion map (systems/scene.py) inside each xywh box given in frame pixels -> (n,)."""
    mh, mw = maps.shape
    h, w = frame_hw
    out = np.zeros(len(boxes))
    for i, (x, y, bw, bh) in enumerate(np.asarray(boxes)[:, :4]):
        x0, y0 = int(np.clip(x * mw / w, 0, mw - 1)), int(np.clip(y * mh / h, 0, mh - 1))
        x1 = int(np.clip(np.ceil((x + bw) * mw / w), x0 + 1, mw))
        y1 = int(np.clip(np.ceil((y + bh) * mh / h), y0 + 1, mh))
        out[i] = maps[y0:y1, x0:x1].mean()
    return out


class RiskGate:
    """Risk-adaptive threshold (RQ3), online and numpy-only like the tracker, applied to a frame's detections before it.

    A detection below thr but >= low is promoted to thr (treated as a sure face) when it is risky:
      - big:        its box height >= h_min pixels (policies.py / edge_pipeline.py pass percent of frame height) (big faces are the recognisable ones; false positives at low score
                    are mostly small texture), or
      - persistent: a detection >= low overlapped it (IoU >= 0.3) in each of the previous persist-1 frames (a real
                    face stays where it is; low-score false positives flicker).
    And, whatever the box: in a dark frame (mean grey level < dark) every detection >= thr_dark is promoted
    (the detector's scores drop in the dark; the board reads the mean off the Y plane it already has).
      - confirm:    the dark rule promotes a box only if a dark candidate (>= thr_dark) overlapped it (IoU >= 0.3)
                    in the previous frame: one-frame false positives on skin and texture (Music Video) stay out.
      - zone:       a fixed camera's face zone (bool grid over the frame, systems/scene.py, learned on val): a
                    detection >= thr_zone whose centre is in the zone and whose box moved (mean abs grey difference
                    to the previous frame inside the box >= motion_min, on a small motion map) is promoted, so a
                    face entering the scene is masked before the detector is sure of it.
    h_min = 0 / persist = 0 / dark = 0 / thr_zone = None switch a rule off. Cost: one IoU matrix per frame per
    remembered frame.
    """

    def __init__(self, thr, low=0.3, h_min=0, persist=0, dark=0, thr_dark=None, confirm=False, thr_zone=None,
                 motion_min=0.0, zone=None, frame_hw=None):
        self.thr, self.low, self.h_min, self.persist = thr, low, h_min, persist
        self.dark, self.thr_dark = dark, thr if thr_dark is None else thr_dark
        self.confirm, self.prev_dark = confirm, np.zeros((0, 4))
        self.thr_zone, self.motion_min, self.zone, self.frame_hw = thr_zone, motion_min, zone, frame_hw
        self.hist = []  # candidate boxes (score >= low) of the previous persist-1 frames, oldest first

    def in_zone(self, boxes):
        gh, gw = self.zone.shape
        h, w = self.frame_hw
        cx = np.clip(((boxes[:, 0] + boxes[:, 2] / 2) * gw / w).astype(int), 0, gw - 1)
        cy = np.clip(((boxes[:, 1] + boxes[:, 3] / 2) * gh / h).astype(int), 0, gh - 1)
        return self.zone[cy, cx]

    def update(self, dets, lum=None, motion=None):
        dets = np.asarray(dets, float).reshape(-1, 5).copy()
        cand = dets[dets[:, 4] >= self.low, :4]
        weak = (dets[:, 4] >= self.low) & (dets[:, 4] < self.thr)
        risky = np.zeros(len(dets), bool)
        if self.dark and lum is not None and lum < self.dark:
            dark = (dets[:, 4] >= self.thr_dark) & (dets[:, 4] < self.thr)
            if self.confirm:
                seen = (iou(dets[:, :4], self.prev_dark) >= 0.3).any(1) if len(self.prev_dark) else np.zeros(len(dets), bool)
                self.prev_dark = dets[dets[:, 4] >= self.thr_dark, :4]
                dark &= seen
            risky |= dark
        elif self.confirm:
            self.prev_dark = np.zeros((0, 4))
        if self.thr_zone is not None and self.zone is not None and motion is not None and len(dets):
            moving = box_motion(motion, dets, self.frame_hw) >= self.motion_min
            risky |= (dets[:, 4] >= self.thr_zone) & (dets[:, 4] < self.thr) & self.in_zone(dets) & moving
        if self.h_min:
            risky |= weak & (dets[:, 3] >= self.h_min)
        if self.persist > 1 and len(self.hist) == self.persist - 1:
            seen = np.ones(len(dets), bool)
            for prev in self.hist:
                seen &= (iou(dets[:, :4], prev) >= 0.3).any(1) if len(prev) else False
            risky |= weak & seen
        dets[risky, 4] = self.thr
        if self.persist > 1:
            self.hist = (self.hist + [cand])[-(self.persist - 1):]
        return dets
