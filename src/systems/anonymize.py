"""Shared anonymization applied to every system's detections: scale boxes by 1.3, then fast_blur them.

fast_blur is the validated strong blur computed at 16 px: the judge recognises 0.3 % of faces under it (<= 2 % required;
experiments/board/checks/fast_blur.json) and exposure matches the full blur, but it costs 2.9 ms on the i.MX 8M Plus
against 104 ms. pixelate (6 cells, 1.1 ms) leaked more on partly covered faces (src/checks/blur_check.py).
blur, weak_blur and pixelate stay as reference ops for that check.
shortcut: if the live demo misses 25 FPS try side=8 (1.9 ms), not yet judge-checked."""
import cv2
import numpy as np

MASK_SCALE = 1.3


def scale_boxes(boxes, shape, scale=MASK_SCALE):
    """(n, >=4) x, y, w, h -> (n, 4) int x1, y1, x2, y2 scaled about the centre, clipped to the frame."""
    b = np.asarray(boxes, dtype=np.float32)[:, :4]
    c, s = b[:, :2] + b[:, 2:] / 2, b[:, 2:] * scale
    xyxy = np.concatenate([c - s / 2, c + s / 2], 1).round().astype(int)
    return np.clip(xyxy, 0, [shape[1], shape[0], shape[1], shape[0]])


def box_mask(xyxy, shape):
    m = np.zeros(shape[:2], bool)
    for x1, y1, x2, y2 in xyxy:
        m[y1:y2, x1:x2] = True
    return m


def pixelate(img, xyxy, cells=6):
    out = img.copy()
    for x1, y1, x2, y2 in xyxy:
        if x2 > x1 and y2 > y1:
            small = cv2.resize(out[y1:y2, x1:x2], (cells, cells), interpolation=cv2.INTER_AREA)
            out[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    return out


def blur(img, xyxy):
    """Strong blur: 3 Gaussian passes with a kernel the size of the box. ArcFace recognises ~0.2 % (chance) of
    faces under it, against 37 % under weak_blur (src/checks/blur_check.py)."""
    out = img.copy()
    for x1, y1, x2, y2 in xyxy:
        if x2 > x1 and y2 > y1:
            k = max(3, max(x2 - x1, y2 - y1) | 1)  # odd kernel, full box side
            for _ in range(3):
                out[y1:y2, x1:x2] = cv2.GaussianBlur(out[y1:y2, x1:x2], (k, k), 0)
    return out


def fast_blur(img, xyxy, side=16):
    """Same as blur, computed at low resolution (longest box side shrunk to `side` px), so it costs about 1 ms on the board."""
    out = img.copy()
    for x1, y1, x2, y2 in xyxy:
        if x2 > x1 and y2 > y1:
            r = side / max(x2 - x1, y2 - y1)
            small = cv2.resize(out[y1:y2, x1:x2], (max(1, round((x2 - x1) * r)), max(1, round((y2 - y1) * r))),
                               interpolation=cv2.INTER_AREA)
            k = max(3, max(small.shape[:2]) | 1)
            for _ in range(3):
                small = cv2.GaussianBlur(small, (k, k), 0)
            out[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
    return out


def weak_blur(img, xyxy):
    """The original single-pass blur (kernel = half the box side). Kept only to report that it leaks."""
    out = img.copy()
    for x1, y1, x2, y2 in xyxy:
        if x2 > x1 and y2 > y1:
            k = max(3, min(x2 - x1, y2 - y1) // 2 | 1)
            out[y1:y2, x1:x2] = cv2.GaussianBlur(out[y1:y2, x1:x2], (k, k), 0)
    return out
