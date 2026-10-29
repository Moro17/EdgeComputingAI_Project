"""Fixed ChokePoint split: fine-tune on train, pick checkpoints on val, report only test.

Disjoint by person-session: train is P1, val and test are P2 (S1 = test, the sequences cached in experiments/board).
P1 files have no .1/.2 suffix, P2 ones do. CALIB (S5) is only used for INT8 calibration and never scored.
"""
from systems.quantize import CALIB_SEQS

_C = ("C1", "C2", "C3")
TRAIN = [f"P1{t}_S{s}_{c}" for t in "EL" for s in (1, 2, 3, 4) for c in _C]
VAL = [f"P2{t}_S{s}_{c}.{h}" for t in "EL" for s in (2, 3, 4) for c in _C for h in (1, 2)]
TEST = [f"P2{t}_S1_{c}.{h}" for t in "EL" for c in _C for h in (1, 2)]
CALIB = list(CALIB_SEQS)


def split_of(seq):
    for name, seqs in (("train", TRAIN), ("val", VAL), ("test", TEST), ("calib", CALIB)):
        if seq in seqs:
            return name
    raise KeyError(seq)
