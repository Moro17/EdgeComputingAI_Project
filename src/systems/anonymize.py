"""Shared anonymization applied to every system's detections: scale boxes by 1.3, Gaussian-blur them."""
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


def weak_blur(img, xyxy):
    """The original single-pass blur (kernel = half the box side). Kept only to report that it leaks."""
    out = img.copy()
    for x1, y1, x2, y2 in xyxy:
        if x2 > x1 and y2 > y1:
            k = max(3, min(x2 - x1, y2 - y1) // 2 | 1)
            out[y1:y2, x1:x2] = cv2.GaussianBlur(out[y1:y2, x1:x2], (k, k), 0)
    return out
