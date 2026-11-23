"""Paired comparison of two systems on the same people: does B expose fewer identities than A?

With ~29 people one identity is 3.4 points, so an unpaired CI per system is wide. Pairing helps: both systems see the
same people, so only the discordant people (exposed by one system, not the other) carry information.

Per condition prints: % exposed for A and B, discordant counts (A only / B only), the exact two-sided sign-test
(McNemar) p-value, and a 95% paired bootstrap CI of B - A (resampling people).

Reads <out>/scores/<system>_t<thr><sfx>/<seq>.csv written by benchmark.py.

  python src/analysis/paired.py --out results/rq1 --a yunet_480x640_npu+window3 \\
      --b yunet_480x640_npu+bytetrack+keep+back3 --runs 0.9 0.6@dark2 0.9@occl2
"""
import argparse
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd


def exposed(out, system, run):
    """Per-identity bool Series: exposed in at least one frame of any sequence of the split."""
    d = out / "scores" / f"{system}_t{run}"
    if not d.is_dir():
        raise SystemExit(f"missing {d}: run benchmark.py for {system} first")
    faces = pd.concat([pd.read_csv(f) for f in sorted(d.glob("*.csv"))], ignore_index=True)
    return faces.groupby("identity").leaked.any()


def sign_test(b, c):
    """Exact two-sided binomial p-value for b vs c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def compare(ea, eb, n=10000, seed=0):
    ids = ea.index.intersection(eb.index)
    a, b = ea[ids].to_numpy(float), eb[ids].to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), (n, len(ids)))
    diff = 100 * (b[idx].mean(1) - a[idx].mean(1))
    return {"people": len(ids), "pct_a": 100 * a.mean(), "pct_b": 100 * b.mean(),
            "a_only": int(((a == 1) & (b == 0)).sum()), "b_only": int(((a == 0) & (b == 1)).sum()),
            "p_sign": sign_test(int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())),
            "diff_lo": float(np.percentile(diff, 2.5)), "diff_hi": float(np.percentile(diff, 97.5))}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--a", required=True, help="reference system spec")
    p.add_argument("--b", required=True, help="system spec compared against A")
    p.add_argument("--runs", nargs="+", required=True, metavar="THR[@KINDSEV]",
                   help="threshold and degradation suffix as in the scores dir names, e.g. 0.9 0.6@dark2")
    p.add_argument("--csv", type=Path, default=None)
    args = p.parse_args()
    rows = [{"run": r, **compare(exposed(args.out, args.a, r), exposed(args.out, args.b, r))} for r in args.runs]
    df = pd.DataFrame(rows)
    print(f"A = {args.a}\nB = {args.b}")
    print(df.round(3).to_string(index=False))
    pooled_a, pooled_b = df.a_only.sum(), df.b_only.sum()
    print(f"pooled over runs: A-only {pooled_a}, B-only {pooled_b}, sign test p = {sign_test(pooled_a, pooled_b):.4g}"
          " (runs share people, so the pooled p is optimistic)")
    if args.csv:
        df.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
