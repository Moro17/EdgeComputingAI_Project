"""Music Video loader + box-only judge path on a synthetic 4-frame video. Run: python tests/test_loaders.py (or pytest)."""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import splits  # noqa: E402
from judge.arcface import align, templates  # noqa: E402
from loaders import musicvideo as mv  # noqa: E402


def fixture(root):
    v = root / "tara"
    (v / "frames").mkdir(parents=True)
    for n in range(1, 5):
        cv2.imwrite(str(v / "frames" / f"{n:08d}.jpg"), np.zeros((60, 80, 3), np.uint8))
    (v / "gt.txt").write_text("1,3,10,10,20,20\n2 3 12 10 20 20\n3,5,40,5,20,25,0.9\n4,3,11,10,20,20\n")
    mv.parse_raw(v / "gt.txt", "tara").to_csv(v / "gt_unified.csv", index=False)


def test_musicvideo_halves_are_separate_sessions():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        fixture(root)
        assert mv.sequences(root) == ["tara.1", "tara.2"]
        assert mv.frame_numbers("tara.1", root) == [1, 2] and mv.frame_numbers("tara.2", root) == [3, 4]
        assert mv.session("tara.1") != mv.session("tara.2")
        gt = mv.load_gt("tara.2", root)
        assert list(gt.columns) == mv.COLS
        assert gt.identity.tolist() == ["tara:5", "tara:3"] and gt.lx.isna().all()
        assert mv.frame_path("tara.2", 3, root) == root / "tara" / "frames" / "00000003.jpg"


def test_string_identities_get_templates():
    e = np.eye(3, dtype=np.float32)
    emb = {"tara.1": (np.array(["tara:3", "tara:3"]), e[:2]), "tara.2": (np.array(["tara:3"]), e[2:])}
    t = templates(emb, "tara.2", mv.session)
    assert list(t) == ["tara:3"] and np.allclose(t["tara:3"], (e[0] + e[1]) / np.sqrt(2))


def test_align_falls_back_to_box_without_eyes():
    img = np.zeros((60, 80, 3), np.uint8)
    img[10:30, 10:30] = 200
    r = SimpleNamespace(x=10, y=10, w=20, h=20, lx=np.nan, ly=np.nan, rx=np.nan, ry=np.nan)
    assert align(img, r).shape == (112, 112, 3) and align(img, r).mean() == 200
    assert align(img, SimpleNamespace(x=200, y=200, w=20, h=20, lx=np.nan)).max() == 0  # box off-frame


def test_iou():
    a = np.array([[0, 0, 10, 10]], float)
    assert np.allclose(mv.iou(a, np.array([[0, 0, 10, 10], [5, 0, 10, 10], [50, 50, 5, 5]], float)), [[1, 1 / 3, 0]])


ROOT = Path(__file__).resolve().parents[1]


def test_splits_are_disjoint_and_test_is_the_cached_benchmark():
    sets = [set(s) for s in (splits.TRAIN, splits.VAL, splits.TEST, splits.CALIB)]
    assert sum(map(len, sets)) == len(set().union(*sets))
    assert splits.TEST == json.loads((ROOT / "experiments/board/data_config.json").read_text())["seqs"]
    assert (len(splits.TRAIN), len(splits.VAL), len(splits.TEST)) == (24, 36, 12)
    assert all(s.startswith("P1") for s in splits.TRAIN) and splits.split_of("P2E_S5_C1.1") == "calib"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
