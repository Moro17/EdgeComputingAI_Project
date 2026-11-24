"""Self-checks for the policies after the detector. Run: python tests/test_policies.py (or pytest)."""
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from systems.policies import COLS, budget, bytetrack, coast, hold, keyframes, parse_spec, run_policy, skip, window  # noqa: E402


def dets(rows):  # (frame, x, y, w, h, score)
    return pd.DataFrame([(f, -1, x, y, w, h, s) for f, x, y, w, h, s in rows],
                        columns=["frame", "track_id", "x", "y", "w", "h", "score"])


def tracks(rows):  # (frame, tid, x, y, w, h)
    return pd.DataFrame([(f, t, x, y, w, h, .9, "det") for f, t, x, y, w, h in rows], columns=COLS)


def test_parse_spec():
    assert parse_spec("yunet_int8") == ("yunet_int8", [])
    assert parse_spec("yunet_int8+skip3+bytetrack+coast+pose") == (
        "yunet_int8", [("skip", 3), ("bytetrack", None), ("coast", None), ("pose", None)])
    assert parse_spec("y+skip3+risk_d60_t0.1+lite+hold+back3")[1][0] == ("skip", 3)
    for bad in ["yunet+coast", "yunet+bytetrack+hold+coast", "yunet+pose", "yunet+foo", "yunet+skip0",
                "yunet+bytetrack+skip2", "yunet+skip3", "yunet+skip3+lite+back3"]:
        try:
            parse_spec(bad)
            assert False, bad
        except ValueError:
            pass


def test_skip_and_hold():
    frames = list(range(10, 17))
    keys = keyframes(frames, 3)
    assert keys == [10, 13, 16]
    d = dets([(f, 100, 100, 50, 50, 0.9) for f in frames])
    out = hold(skip(d, keys), frames, keys)
    assert sorted(out.frame) == frames
    assert (out[out.frame == 11].kind == "hold").all()
    assert (out[out.frame == 13].kind == "det").all()
    assert list(out.columns) == COLS


def test_skip_longer_than_sequence():
    frames = [1, 2, 3]
    keys = keyframes(frames, 10)
    assert keys == [1]
    out = hold(skip(dets([(2, 0, 0, 10, 10, .9)]), keys), frames, keys)
    assert out.empty and list(out.columns) == COLS


def test_empty_dets():
    assert list(hold(skip(dets([]), [1]), [1, 2], [1]).columns) == COLS
    assert list(bytetrack(dets([]), [0, 1], 0.9).columns) == COLS
    assert list(coast(bytetrack(dets([]), [0, 1], 0.9), [0, 1], (600, 800)).columns) == COLS


def test_run_policy_b1_filters_threshold():
    d = dets([(1, 0, 0, 10, 10, 0.95), (1, 50, 50, 10, 10, 0.3)])
    out = run_policy([], d, [1], thr=0.9, shape=(600, 800))
    assert len(out) == 1


def test_bytetrack_keeps_identity_and_low_scores():
    rows = [(f, 100 + 5 * f, 100, 50, 50, 0.95) for f in range(6)]
    rows[3] = (3, 115, 100, 50, 50, 0.4)  # a low-score detection that B1 at 0.9 would drop
    out = bytetrack(dets(rows), list(range(6)), thr=0.9)
    assert set(out.frame) == set(range(6)) and out.track_id.nunique() == 1


def test_bytetrack_fresh_per_call():
    d = dets([(0, 100, 100, 50, 50, 0.95)])
    assert bytetrack(d, [0], 0.9).track_id.tolist() == bytetrack(d, [0], 0.9).track_id.tolist()


def test_window_recovers_isolated_miss():
    d = dets([(0, 100, 100, 50, 50, .95), (2, 104, 100, 50, 50, .95), (9, 400, 300, 50, 50, .95)])
    out = run_policy([("window", 2)], d, list(range(10)), thr=0.9, shape=(600, 800))
    assert len(out[out.frame == 1]) == 1 and out[out.frame == 1].kind.iloc[0] == "window"  # 0 and 2 overlap: one box
    assert len(out[out.frame == 5]) == 0 and len(out[out.frame == 7]) == 1
    assert parse_spec("yunet+window3+bytetrack") == ("yunet", [("window", 3), ("bytetrack", None)])
    assert list(window(skip(dets([]), [0]), [0, 1], 2).columns) == COLS


def test_keep_masks_detections_the_tracker_drops():
    d = dets([(0, 100, 100, 50, 50, 0.95), (1, 400, 300, 50, 50, 0.95)])  # second face: a track born at frame 1
    bt = run_policy([("bytetrack", None)], d, [0, 1], thr=0.9, shape=(600, 800))
    kept = run_policy([("bytetrack", None), ("keep", None)], d, [0, 1], thr=0.9, shape=(600, 800))
    assert not ((bt.frame == 1) & (bt.x == 400)).any()  # vanilla ByteTrack: unconfirmed new track, face not masked
    assert ((kept.frame == 1) & (kept.x == 400)).any()


def test_coast_ignores_untracked_keep_rows():
    t = pd.concat([tracks([(0, 1, 100, 100, 40, 40)]),
                   pd.DataFrame([(0, -1, 500, 300, 40, 40, .95, "det"), (3, -1, 600, 300, 40, 40, .95, "det")],
                                columns=COLS)])
    out = coast(t, list(range(5)), (600, 800), max_age=2)
    assert len(out[out.track_id == -1]) == 2  # raw rows passed through, not coasted as one fake track
    assert set(out[out.track_id == 1].frame) == {0, 1, 2}


def test_budget_grows_fragile_tracks_and_coasts_longer():
    steady = tracks([(f, 1, 100, 100, 40, 40) for f in range(30)])
    out = budget(steady, list(range(30)), (600, 800), thr=0.5)
    assert out[out.frame == 0].w.iloc[0] > 40 * 1.4  # new track: r high, big mask
    assert abs(out[out.frame == 29].w.iloc[0] - 40) < 2  # confidently seen for a while: back to ~ the box
    lost = tracks([(0, 1, 100, 100, 40, 40)])
    out = budget(lost, list(range(60)), (600, 800), thr=0.5)
    assert out.frame.max() > 15  # a fragile lost track is coasted longer than coast()'s 15
    for bad in ["yunet+budget", "yunet+bytetrack+coast+budget"]:
        try:
            parse_spec(bad)
            assert False, bad
        except ValueError:
            pass


def test_grow_scales_final_boxes_about_centre():
    d = dets([(0, 100, 100, 40, 20, .95)])
    assert parse_spec("yunet+window3+grow140") == ("yunet", [("window", 3), ("grow", 140)])
    out = run_policy([("grow", 150)], d, [0], thr=0.9, shape=(600, 800)).iloc[0]
    assert (out.x, out.y, out.w, out.h) == (90, 95, 60, 30)
    for bad in ["yunet+grow90", "yunet+grow", "yunet+window", "yunet+grow120+window3"]:
        try:
            parse_spec(bad)
            assert False, bad
        except ValueError:
            pass


def test_budget_knobs_in_spec():
    assert parse_spec("yunet+bytetrack+budget_d0.7_g1.8_a10-60") == (
        "yunet", [("bytetrack", None), ("budget", (("age_max", 60), ("age_min", 10), ("decay", .7), ("grow_max", 1.8)))])
    assert parse_spec("yunet+bytetrack+budget_g2") == ("yunet", [("bytetrack", None), ("budget", (("grow_max", 2.0),))])
    lost = tracks([(0, 1, 100, 100, 40, 40)])
    short = run_policy([("budget", (("age_max", 5), ("age_min", 2)))], lost, list(range(60)), thr=.5, shape=(600, 800))
    assert short.frame.max() <= 5
    for bad in ["yunet+bytetrack+budget_d1.2", "yunet+bytetrack+budget_a50-10", "yunet+bytetrack+budget_x"]:
        try:
            parse_spec(bad)
            assert False, bad
        except ValueError:
            pass


def test_back_covers_frames_before_first_detection():
    d = dets([(5, 100, 100, 40, 40, .95)])
    out = run_policy([("back", 3)], d, list(range(10)), thr=.9, shape=(600, 800))
    assert sorted(out.frame) == [2, 3, 4, 5] and (out[out.frame < 5].kind == "back").all()
    out = run_policy([("back", 3)], dets([(1, 100, 100, 40, 40, .95)]), list(range(10)), thr=.9, shape=(600, 800))
    assert sorted(out.frame) == [0, 1]
    assert parse_spec("y+bytetrack+keep+coast+back3+grow120")[1][-2:] == [("back", 3), ("grow", 120)]


def test_spread_is_a_union_both_ways():
    d = dets([(5, 100, 100, 40, 40, .95), (6, 110, 100, 40, 40, .95)])
    out = run_policy([("spread", 2)], d, list(range(8)), thr=.9, shape=(600, 800))
    assert sorted(out.frame.unique()) == [3, 4, 5, 6, 7] and len(out[out.frame == 5]) == 2  # no NMS


def test_online_lookback_equals_offline_back():
    import numpy as np
    from edge_pipeline import LookBack
    rng = np.random.default_rng(0)
    frames = list(range(40))
    rows = [(f, *rng.integers(0, 500, 2), 40, 40, .95) for f in frames if rng.random() < .6 for _ in range(rng.integers(1, 3))]
    d = dets(rows)
    off = run_policy([("back", 3)], d, frames, thr=.9, shape=(600, 800))
    lb, online = LookBack(3), {}
    items = [{"frame": f, "boxes": d[d.frame == f][["x", "y", "w", "h", "score"]].to_numpy()} for f in frames]
    for it in items:
        for r in lb.push(it):
            online[r["frame"]] = r
    for r in lb.flush():
        online[r["frame"]] = r
    for f in frames:
        a = sorted(map(tuple, np.concatenate(online[f]["mask"])[:, :4].round(2).tolist()))
        b = sorted(map(tuple, off[off.frame == f][["x", "y", "w", "h"]].to_numpy().round(2).tolist()))
        assert a == b, f


def test_lite_keeps_low_score_faces_of_a_track():
    rows = [(f, 100 + 3 * f, 100, 40, 40, .95) for f in range(10)]
    rows[4] = (4, 112, 100, 40, 40, .3)  # dips below threshold: per-frame drops it, the track keeps it
    rows.append((6, 500, 400, 40, 40, .3))  # low score with no track: not masked
    out = run_policy([("lite", None)], dets(rows), list(range(10)), thr=.9, shape=(600, 800))
    assert 4 in set(out.frame) and not (out.x == 500).any()
    assert len(run_policy([], dets(rows), list(range(10)), thr=.9, shape=(600, 800))) == 9
    for bad in ["y+bytetrack+lite", "y+lite+keep", "y+lite+coast"]:
        try:
            parse_spec(bad)
            assert False, bad
        except ValueError:
            pass


def test_online_tracker_and_lookback_equal_offline_lite_back():
    import numpy as np
    from edge_pipeline import LookBack
    from systems.bytelite import LiteTracker
    rng = np.random.default_rng(1)
    frames = list(range(60))
    rows = []
    for f in frames:  # two moving faces whose score dips, plus clutter
        for k in range(2):
            rows.append((f, 100 + 4 * f + 300 * k, 200, 50, 50, float(rng.choice([.95, .95, .4, .05]))))
        if rng.random() < .3:
            rows.append((f, *rng.integers(0, 700, 2), 30, 30, float(rng.random())))
    d = dets(rows)
    off = run_policy([("lite", None), ("back", 3)], d, frames, thr=.9, shape=(600, 800))
    tr, lb, online = LiteTracker(.9), LookBack(3), {}
    for f in frames:
        b = d[(d.frame == f) & (d.score >= .1)][["x", "y", "w", "h", "score"]].to_numpy()
        for r in lb.push({"frame": f, "boxes": tr.update(b)}):
            online[r["frame"]] = r
    for r in lb.flush():
        online[r["frame"]] = r
    for f in frames:
        a = sorted(map(tuple, np.concatenate(online[f]["mask"])[:, :4].round(2).tolist()))
        b = sorted(map(tuple, off[off.frame == f][["x", "y", "w", "h"]].to_numpy().round(2).tolist()))
        assert a == b, f


def test_coast_fills_gap_with_velocity():
    t = tracks([(0, 1, 100, 100, 40, 40), (1, 1, 110, 100, 40, 40), (4, 1, 140, 100, 40, 40)])
    out = coast(t, list(range(5)), (600, 800))
    g = out[out.frame == 2].iloc[0]
    assert g.kind == "coast" and abs((g.x + g.w / 2) - 140) < 1e-6 and abs(g.w - 50) < 1e-6
    assert len(out[out.frame == 4]) == 1


def test_coast_stops_after_max_age():
    out = coast(tracks([(0, 1, 100, 100, 40, 40)]), list(range(30)), (600, 800), max_age=15)
    assert out.frame.max() == 15


def test_coast_drops_offframe():
    t = tracks([(0, 1, 700, 100, 40, 40), (1, 1, 760, 100, 40, 40)])
    assert (coast(t, list(range(10)), (600, 800)).x < 800).all()


def test_risk_gate_promotes_tall_or_persistent_weak_detections():
    assert parse_spec("y+risk_l0.3_h13_p3+lite+back3")[1][0] == ("risk", (("h_min", 13), ("low", .3), ("persist", 3)))
    for bad in ["y+risk", "y+risk_l0.3", "y+risk_p1", "y+lite+risk_h13"]:
        try:
            parse_spec(bad)
            assert False, bad
        except ValueError:
            pass
    fr = list(range(6))
    tall = [(0, 100, 100, 60, 100, .5)]                       # weak but tall: a face you could recognise
    small = [(0, 400, 100, 20, 30, .5)]                       # weak and small, once: texture
    steady = [(f, 600, 300, 30, 40, .4) for f in range(2, 6)]  # weak and small but in the same place 4 frames
    flicker = [(1, 50, 400, 30, 40, .4), (3, 50, 400, 30, 40, .4)]  # weak, same place, but frames 1 and 3 only
    d = dets(tall + small + steady + flicker)
    out = run_policy([("risk", (("h_min", 13), ("low", .3), ("persist", 3)))], d, fr, thr=.9, shape=(600, 800))  # 78 px
    assert (out.x == 100).any() and not (out.x == 400).any() and not (out.x == 50).any()
    assert sorted(out[out.x == 600].frame) == [4, 5]  # promoted from the 3rd consecutive frame; back fills 2-3
    assert not len(run_policy([], d, fr, thr=.9, shape=(600, 800)))
    # dark rule: same weak, small, one-off detection is promoted only in a dark frame
    one = dets([(0, 400, 100, 20, 30, .45), (1, 400, 100, 20, 30, .45)])
    spec = parse_spec("y+risk_d60_t0.4")[1]
    out = run_policy(spec, one, [0, 1], thr=.9, shape=(600, 800), lum={0: 120.0, 1: 30.0})
    assert list(out.frame) == [1]


def test_dark_confirm_needs_the_previous_dark_frame():
    # frame 0 dark with a one-off box elsewhere; the face box is seen dark at frames 1 and 2
    d = dets([(0, 600, 400, 20, 30, .2), (1, 400, 100, 20, 30, .2), (2, 401, 101, 20, 30, .2)])
    lum = {0: 30.0, 1: 30.0, 2: 30.0}
    plain = run_policy(parse_spec("y+risk_d60_t0.1")[1], d, [0, 1, 2], thr=.9, shape=(600, 800), lum=lum)
    assert sorted(plain.frame) == [0, 1, 2]
    conf = run_policy(parse_spec("y+risk_d60_t0.1_c")[1], d, [0, 1, 2], thr=.9, shape=(600, 800), lum=lum)
    assert sorted(conf.frame) == [2]  # frame 0 and 1 have nothing to confirm them


def test_zone_rule_promotes_in_zone_only_with_motion():
    import numpy as np

    zone = np.zeros((30, 40), bool)
    zone[:, :20] = True  # left half of an 800x600 frame
    d = dets([(0, 100, 100, 20, 30, .4), (0, 600, 100, 20, 30, .4), (0, 300, 300, 20, 30, .4),
              (1, 100, 100, 20, 30, .4)])
    moving = np.zeros((60, 80), np.uint8)
    moving[:, :] = 5
    moving[25:40, 25:40] = 0  # the box at (300, 300) is in the zone but static
    spec = parse_spec("y+risk_z0.3_m2")[1]
    out = run_policy(spec, d, [0, 1], thr=.9, shape=(600, 800), zone=zone,
                     motion={0: moving, 1: np.zeros((60, 80), np.uint8)})
    assert list(zip(out.frame, out.x)) == [(0, 100)]  # outside the zone / static box / frame 1 without motion
    try:
        run_policy(spec, d, [0, 1], thr=.9, shape=(600, 800))
        assert False
    except ValueError:
        pass


def test_reflection_zone_spec_names_the_zone_and_runs_like_the_zone_rule():
    import numpy as np

    spec = parse_spec("y+risk_d60_t0.1_zr0.2_m0+lite+back3")[1]
    assert dict(dict(spec)["risk"])["zone_src"] == "reflection"
    zone = np.zeros((30, 40), bool)
    zone[:, 20:] = True  # door glass on the right half
    d = dets([(0, 600, 100, 20, 30, .25), (0, 100, 100, 20, 30, .25)])
    still = np.zeros((60, 80), np.uint8)
    out = run_policy(parse_spec("y+risk_zr0.2_m0")[1], d, [0], thr=.9, shape=(600, 800), zone=zone, motion={0: still})
    assert list(out.x) == [600]  # m0: no motion needed; only the box on the glass is promoted


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
