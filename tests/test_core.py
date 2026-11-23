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


def test_bootstrap_ci():
    import pandas as pd
    from metrics.exposure import bootstrap_ci

    faces = pd.DataFrame({"seq": "s", "identity": [1, 1, 2, 3, 4], "leaked": [True, False, False, False, True]})
    lo, hi = bootstrap_ci(faces)
    assert 0 <= lo <= 50 <= hi <= 100


def test_leak_vs_length():
    import pandas as pd
    from metrics.exposure import leak_vs_length

    faces = pd.DataFrame({"seq": "s", "identity": 1, "frame": range(10), "leaked": [False] * 4 + [True] + [False] * 5})
    r = leak_vs_length(faces, lengths=(1, 5, 10)).set_index("length")
    assert r.loc[1, "p_leak"] == 0.1 and r.loc[10, "p_leak"] == 1.0
    assert abs(r.loc[5, "p_leak"] - 5 / 6) < 1e-9  # windows starting 0..5, five contain frame 4


def test_enroll_ids_follow_crop_order_when_gt_is_sorted_by_track():
    """Music Video GT comes ordered by track, not by frame: each embedding must keep its own identity."""
    import tempfile

    import cv2
    import pandas as pd

    import benchmark

    tmp = Path(tempfile.mkdtemp())
    img = np.zeros((60, 60, 3), np.uint8)
    cv2.imwrite(str(tmp / "f.png"), img)
    # person A in frames 1-2, then person B in frames 1-2 (track order)
    gt = pd.DataFrame({"frame": [1, 2, 1, 2], "identity": ["A", "A", "B", "B"], "x": [0, 0, 30, 30],
                       "y": 0, "w": 20, "h": 20, "lx": np.nan, "ly": np.nan, "rx": np.nan, "ry": np.nan})

    class DS:
        load_gt = staticmethod(lambda seq: gt)
        frame_path = staticmethod(lambda seq, n: tmp / "f.png")

    class J:
        def embed(self, crops):
            return np.zeros((len(crops), 2))

    seen = []  # identity of each crop, in the order enroll embeds them
    orig = benchmark.align
    benchmark.align = lambda im, r: seen.append(r.identity) or np.zeros((112, 112, 3), np.uint8)
    try:
        ids, _ = benchmark.enroll(DS, J(), "s", [1, 2], tmp)
    finally:
        benchmark.align = orig
    assert list(ids) == seen == ["A", "B", "A", "B"]


def test_recipe_frames_degrade_their_share_only():
    """The dark recipe calibrates on clean + dark:1 + dark:3 frames, never on the dark:2 test condition."""
    from systems import quantize

    orig = quantize.calibration_frames
    quantize.calibration_frames = lambda n: [np.full((8, 8, 3), 200, np.uint8) for _ in range(n)]
    try:
        out = quantize.recipe_frames(quantize.RECIPES["dark"]["calib"], n=10)
    finally:
        quantize.calibration_frames = orig
    assert len(out) == 10
    means = [f.mean() for f in out]
    assert all(m == 200 for m in means[:3]) and all(m < 180 for m in means[3:])
    assert np.mean(means[3:6]) > np.mean(means[6:])  # dark:1 brighter than dark:3
    assert all("dark:2" not in str(r["calib"]) for r in quantize.RECIPES.values())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
