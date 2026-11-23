"""Coverage, identity exposure and exposed-run metrics."""
import numpy as np
import pandas as pd

PROTECTED_COVERAGE = 0.8


def coverage(mask, x, y, w, h):
    """Fraction of the GT box (clipped to the frame) inside the anonymization mask."""
    x1, y1 = max(int(round(x)), 0), max(int(round(y)), 0)
    x2, y2 = min(int(round(x + w)), mask.shape[1]), min(int(round(y + h)), mask.shape[0])
    return float(mask[y1:y2, x1:x2].mean()) if x2 > x1 and y2 > y1 else 0.0


def max_run(flags):
    """Longest run of consecutive True values (over an identity's annotated frames, in frame order)."""
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def summarise(faces, masked, latency_ms):
    """faces: one row per GT face (seq, frame, identity, protected, leaked, leaked_null).
    masked: per-frame masked-area fractions. latency_ms: per-frame detector latencies."""
    faces = faces.sort_values(["seq", "identity", "frame"])
    units = faces.groupby(["seq", "identity"])  # an appearance = one identity in one sequence
    unprotected = (~faces.protected).sum()
    lat = np.asarray(latency_ms)
    return {
        "frame_recall": faces.protected.mean(),
        "appearances": units.ngroups,
        "pct_appearances_leaked": 100 * units.leaked.any().mean(),
        "pct_appearances_leaked_null": 100 * units.leaked_null.any().mean(),
        "pct_identities_leaked": 100 * faces.groupby("identity").leaked.any().mean(),
        "max_exposed_run": int(units.leaked.apply(max_run).max()),
        "mean_max_exposed_run": units.leaked.apply(max_run).mean(),
        "probe_hit_rate": faces.leaked.sum() / unprotected if unprotected else 0.0,
        "pct_area_masked": 100 * float(np.mean(masked)),
        "fps": 1000 / lat.mean(),
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
    }


def bootstrap_ci(faces, n=1000, seed=0):
    """95% CI of % identities leaked, resampling identities with replacement (29 people: one person = 3.4%)."""
    per_id = faces.groupby("identity").leaked.any().to_numpy(float)
    rng = np.random.default_rng(seed)
    stats = 100 * per_id[rng.integers(0, len(per_id), (n, len(per_id)))].mean(1)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def leak_vs_length(faces, lengths=(1, 5, 10, 25, 50, 100)):
    """Probability that a window of L consecutive annotated frames of one appearance contains a leaked frame.

    p_independent = 1 - (1 - p_frame)^L is what frame-level numbers predict if misses were independent;
    the gap to p_leak says whether misses cluster (p_leak below) or spread (above).
    """
    p_frame = faces.leaked.mean()
    rows = []
    for L in lengths:
        hits = n = 0
        for _, g in faces.sort_values("frame").groupby(["seq", "identity"]):
            v = g.leaked.to_numpy(int)
            if len(v) < L:
                continue
            c = np.convolve(v, np.ones(L, int), "valid")  # leaked frames per window
            hits, n = hits + int((c > 0).sum()), n + len(c)
        rows.append({"length": L, "p_leak": hits / n if n else np.nan, "p_independent": 1 - (1 - p_frame) ** L,
                     "n_windows": n})
    return pd.DataFrame(rows)
