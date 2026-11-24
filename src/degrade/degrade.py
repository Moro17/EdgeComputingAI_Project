"""Synthetic degradations for the stress tier: the conditions in which detectors miss faces.

A degradation is "KIND:SEV" (e.g. "dark:2"), severity 1-3. Deterministic per frame: seed = frame number, so the
desktop and the board see the same degraded pixels. Only numpy + OpenCV, so the file also runs on the board.

| kind  | what                                                   | 1          | 2          | 3           |
|-------|--------------------------------------------------------|------------|------------|-------------|
| dark  | 255*(img/255)^gamma + Gaussian noise sigma (8-field bank) | g 2, s 3   | g 3, s 6   | g 4, s 10   |
| blur  | horizontal motion kernel, length in px                 | 7          | 15         | 25          |
| down  | INTER_AREA down by f, INTER_LINEAR back up             | 2          | 3          | 4           |
| occl  | grey rectangles, 10% of the frame side each, random    | 2          | 4          | 8           |
| jpeg  | re-encode at JPEG quality q (low-bitrate stream blocks) | 30         | 15         | 8           |
| noise | Gaussian sensor noise sigma at normal brightness (bank) | 10         | 20         | 30          |
"""
import cv2
import numpy as np

KINDS = ("dark", "blur", "down", "occl", "jpeg", "noise")
DARK = {1: (2.0, 3.0), 2: (3.0, 6.0), 3: (4.0, 10.0)}
BLUR = {1: 7, 2: 15, 3: 25}
DOWN = {1: 2, 2: 3, 3: 4}
OCCL = {1: 2, 2: 4, 3: 8}
JPEG = {1: 30, 2: 15, 3: 8}
NOISE = {1: 10.0, 2: 20.0, 3: 30.0}


NOISE_BANK = 8  # frame n uses field n % 8: drawing fresh Gaussian noise per frame cost ~90 ms on the board's A53
_bank = {}


def noise_field(shape, seed):
    """Unit Gaussian noise for a frame, from a fixed bank (same values on every machine)."""
    if shape not in _bank:
        _bank[shape] = np.random.default_rng(0).standard_normal((NOISE_BANK, *shape), dtype=np.float32)
    return _bank[shape][seed % NOISE_BANK]


def parse(s):
    """'dark:2' -> ('dark', 2). Raises ValueError on an unknown kind or a severity outside 1-3."""
    kind, sep, sev = s.partition(":")
    if not sep or kind not in KINDS or sev not in ("1", "2", "3"):
        raise ValueError(f"degradation must be KIND:SEV with KIND in {KINDS} and SEV in 1-3, got {s!r}")
    return kind, int(sev)


def apply(img, kind, sev, seed):
    """Degraded copy of a BGR uint8 frame (same shape and dtype)."""
    if kind == "dark":
        gamma, sigma = DARK[sev]
        lut = np.round(255 * (np.arange(256) / 255) ** gamma).astype(np.uint8)
        out = noise_field(img.shape, seed) * np.float32(sigma)
        out += cv2.LUT(img, lut)
        return np.clip(out, 0, 255, out=out).astype(np.uint8)
    if kind == "blur":
        n = BLUR[sev]
        return cv2.filter2D(img, -1, np.full((1, n), 1 / n, np.float32))
    if kind == "down":
        h, w = img.shape[:2]
        f = DOWN[sev]
        small = cv2.resize(img, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if kind == "occl":
        out = img.copy()
        h, w = img.shape[:2]
        rh, rw = max(1, h // 10), max(1, w // 10)
        rng = np.random.default_rng(seed)
        for _ in range(OCCL[sev]):
            y, x = int(rng.integers(0, h - rh + 1)), int(rng.integers(0, w - rw + 1))
            out[y : y + rh, x : x + rw] = 128
        return out
    if kind == "jpeg":
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG[sev]])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if kind == "noise":  # high-ISO sensor noise without the darkening of "dark"
        out = noise_field(img.shape, seed) * np.float32(NOISE[sev])
        out += img
        return np.clip(out, 0, 255, out=out).astype(np.uint8)
    raise ValueError(f"unknown degradation {kind!r}")


def load(path, n, deg=None):
    """Read frame n from path, degraded when deg = 'KIND:SEV' (None: clean)."""
    img = cv2.imread(str(path))
    return img if deg is None else apply(img, *parse(deg), seed=n)
