"""Self-checks for the stress-tier degradations. Run: python tests/test_degrade.py (or pytest)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from degrade.degrade import apply, parse  # noqa: E402


def test_parse():
    assert parse("dark:2") == ("dark", 2)
    for bad in ["dark:0", "dark:4", "fog:1", "dark", "dark:", ":2"]:
        try:
            parse(bad)
            assert False, bad
        except ValueError:
            pass


def test_apply_shape_determinism_and_strength():
    img = np.full((60, 80, 3), 100, np.uint8)  # not 128: the occluders are grey 128
    img[20:40, 30:50] = 255
    for kind in ["dark", "blur", "down", "occl", "jpeg", "noise"]:
        out = apply(img, kind, 2, seed=7)
        assert out.shape == img.shape and out.dtype == np.uint8, kind
        assert not np.array_equal(out, img), kind
        assert np.array_equal(out, apply(img, kind, 2, seed=7)), kind
    assert apply(img, "dark", 3, 0).mean() < apply(img, "dark", 1, 0).mean()
    edge = lambda a: np.abs(np.diff(a.astype(int), axis=1)).sum()  # noqa: E731
    assert edge(apply(img, "blur", 3, 0)) <= edge(apply(img, "blur", 1, 0)) <= edge(img)


def test_jpeg_gets_worse_with_severity():
    img = np.random.default_rng(1).integers(0, 256, (120, 160, 3), dtype=np.uint8)
    err = [np.abs(apply(img, "jpeg", k, 0).astype(int) - img).mean() for k in (1, 2, 3)]
    assert err[0] < err[1] < err[2]


def test_noise_keeps_brightness_and_grows_with_severity():
    img = np.full((120, 160, 3), 100, np.uint8)
    out = [apply(img, "noise", k, 0) for k in (1, 2, 3)]
    assert all(abs(o.mean() - 100) < 2 for o in out)  # unlike dark: no gamma, mean stays
    std = [o.astype(float).std() for o in out]
    assert std[0] < std[1] < std[2]


def test_occl_seed_changes_position():
    img = np.zeros((600, 800, 3), np.uint8)
    assert not np.array_equal(apply(img, "occl", 1, 1), apply(img, "occl", 1, 2))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
