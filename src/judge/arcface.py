"""Leak judge: ArcFace R50 (InsightFace buffalo_l, w600k_r50) on faces aligned by GT eye positions.

The attacker knows where the eyes are (worst case): probes and gallery are aligned the same way.
"""
from pathlib import Path

import cv2
import numpy as np

from loaders.chokepoint import session  # default leave-out grouping

ARC_EYES = np.array([[38.2946, 51.6963], [73.5318, 51.5014]])  # ArcFace 112x112 template eye points


def eye_align(img, left, right):
    """112x112 crop from the similarity transform mapping (left, right) eyes onto ArcFace's template eyes."""
    (a0, a1), (b0, b1) = (complex(*left), complex(*right)), (complex(*ARC_EYES[0]), complex(*ARC_EYES[1]))
    s = (b1 - b0) / (a1 - a0)
    t = b0 - s * a0
    m = np.array([[s.real, -s.imag, t.real], [s.imag, s.real, t.imag]])
    return cv2.warpAffine(img, m, (112, 112), borderValue=0)


def align(img, r):
    """Judge crop for a GT row: eye alignment when the row has eyes, else the GT box resized to 112x112.
    shortcut: box-only crops are a weaker attack than eye-aligned ones; report how many faces needed it."""
    if not np.isnan(r.lx):
        return eye_align(img, (r.lx, r.ly), (r.rx, r.ry))
    x1, y1 = max(int(r.x), 0), max(int(r.y), 0)
    crop = img[y1 : int(r.y + r.h), x1 : int(r.x + r.w)]
    return cv2.resize(crop, (112, 112)) if crop.size else np.zeros((112, 112, 3), np.uint8)


class Judge:
    def __init__(self):
        from insightface.model_zoo import get_model
        from insightface.utils.storage import ensure_available

        from systems.detectors import ort_providers

        path = Path(ensure_available("models", "buffalo_l", root="~/.insightface")) / "w600k_r50.onnx"
        self.net = get_model(str(path), providers=ort_providers())
        self.net.prepare(ctx_id=0)

    def embed(self, crops):
        """List of 112x112 BGR crops -> (n, 512) L2-normalised embeddings."""
        if not crops:
            return np.empty((0, 512), np.float32)
        e = np.concatenate([self.net.get_feat(crops[i : i + 256]) for i in range(0, len(crops), 256)])
        return e / np.linalg.norm(e, axis=1, keepdims=True)


def templates(emb, seq, session=session):
    """Leave-session-out gallery: identity -> mean embedding over sequences from OTHER sessions.

    emb: {seq: (identities array, embeddings array)} of clean GT faces. session: the dataset's leave-out grouping.
    shortcut: plain mean over every gallery face, no quality selection; add top-k by quality if TAR is low.
    """
    ids = np.concatenate([emb[s][0] for s in emb if session(s) != session(seq)])
    e = np.concatenate([emb[s][1] for s in emb if session(s) != session(seq)])
    out = {}
    for i in np.unique(ids):
        t = e[ids == i].mean(0)
        out[i.item()] = t / np.linalg.norm(t)  # int (ChokePoint) or str ('<video>:<id>')
    return out
