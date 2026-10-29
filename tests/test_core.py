"""Self-checks for the scoring core. Run: python tests/test_core.py (or pytest)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from judge.arcface import ARC_EYES, eye_align, session  # noqa: E402
from metrics.exposure import coverage, max_run  # noqa: E402
from systems.anonymize import blur, box_mask, fast_blur, pixelate, scale_boxes  # noqa: E402


def test_eye_align_maps_eyes_to_template():
    img = np.zeros((600, 800, 3), np.uint8)
    left, right = (300, 200), (340, 210)  # rolled face
    img[left[1] - 1 : left[1] + 2, left[0] - 1 : left[0] + 2] = 255
    img[right[1] - 1 : right[1] + 2, right[0] - 1 : right[0] + 2] = 255
    crop = eye_align(img, left, right)
    for (x, y) in ARC_EYES.round().astype(int):
        assert crop[y - 2 : y + 3, x - 2 : x + 3].max() > 0


def test_coverage_and_scaling():
    shape = (600, 800)
    mask = box_mask(scale_boxes(np.array([[100, 100, 50, 50, 0.9]]), shape), shape)
    assert coverage(mask, 100, 100, 50, 50) == 1.0  # 1.3x box covers the original
    assert abs(coverage(mask, 125, 100, 50, 50) - (157.5 - 125) / 50) < 0.03  # mask ends at x=157.5: ~65%
    assert coverage(mask, 700, 500, 50, 50) == 0.0
    assert coverage(mask, -20, -20, 10, 10) == 0.0  # fully off-frame
    assert scale_boxes(np.empty((0, 5)), shape).shape == (0, 4)


def test_max_run_and_session():
    assert max_run([True, True, False, True, True, True, False]) == 3
    assert max_run([]) == 0
    assert session("P2E_S1_C2.1") == "P2E_S1.1"
    assert session("P1L_S3_C3") == "P1L_S3"


def test_pixelate_flattens_the_face():
    img = np.zeros((100, 100, 3), np.uint8)
    img[::2] = 255  # high-contrast stripes stand in for facial detail
    out = pixelate(img, np.array([[20, 20, 80, 80]]))
    assert out[30:70, 30:70].std() < 0.05 * img[30:70, 30:70].std()
    assert np.array_equal(out[:20], img[:20])  # outside the box untouched


def test_fast_blur_matches_blur_and_stays_in_the_box():
    img = np.random.default_rng(0).integers(0, 256, (120, 120, 3), dtype=np.uint8)
    box = np.array([[20, 20, 80, 90]])
    a, b = fast_blur(img, box), blur(img, box)
    assert np.abs(a[20:90, 20:80].astype(int) - b[20:90, 20:80]).mean() < 5
    mask = np.ones(img.shape[:2], bool)
    mask[20:90, 20:80] = False
    assert np.array_equal(a[mask], img[mask])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
