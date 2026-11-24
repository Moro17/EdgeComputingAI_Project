"""Side-by-side demo video of the door zone, rendered on the PC from the BOARD's own detections (NPU INT8, cached
dets of src/board_bench.py), so it shows what the board computes without needing the board:

    left   deployed policy (dark rule + tracker + look-back 3)
    right  + door zone rule (risk _zr0.1_m0): a detection >= 0.1 centred in the camera's door zone is masked.
           Green outline = the zone (learnt on val, systems/scene.py reflection_zone); green box = a face masked
           inside it. A red box on the left marks a face the deployed policy left visible but the zone masked.

    python src/render_zone_demo.py --seq P2E_S1_C3.1 --out dist/demo_videos/zone_P2E_S1_C3.1.mp4
Needs ffmpeg, ECA_DATA and the cached dets (ECA_RESULTS/repro_desktop/dets/<model>/). Not for the paper: a demo.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import luma  # noqa: E402
from degrade.degrade import load as load_frame  # noqa: E402
from loaders import chokepoint as ds  # noqa: E402
from systems import scene  # noqa: E402
from systems.anonymize import fast_blur, scale_boxes  # noqa: E402
from systems.policies import parse_spec, run_policy  # noqa: E402

RES = Path(os.environ.get("ECA_RESULTS", "results")) / "repro_desktop"
BASE = "risk_d60_t0.1+lite+back3"
ZONE = "risk_d60_t0.1_zr0.1_m0+lite+back3"
TW, TH = 640, 480


def boxes(spec, d, frames, shape, lum, zone):
    steps = parse_spec("x+" + spec)[1]
    still = {n: np.zeros((60, 80), np.uint8) for n in frames}
    out = run_policy(steps, d, frames, THR, shape, lum=lum, zone=zone, motion=still)
    return dict(list(out.groupby("frame")))


def label(img, text, color=(255, 255, 255)):
    for c, t in (((0, 0, 0), 4), (color, 2)):
        cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, c, t, cv2.LINE_AA)


def main():
    global THR
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq", default="P2E_S1_C3.1")
    p.add_argument("--model", default="yunet_ft_480x640_npu")
    p.add_argument("--thr", type=float, default=0.7)
    p.add_argument("--start", type=int, default=0, help="first frame index")
    p.add_argument("--frames", type=int, default=0, help="0 = whole sequence")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    THR = a.thr
    frames = ds.frame_numbers(a.seq)
    frames = frames[a.start:a.start + a.frames] if a.frames else frames[a.start:]
    shape = cv2.imread(str(ds.frame_path(a.seq, frames[0]))).shape
    d = pd.read_csv(RES / "dets" / a.model / f"{a.seq}.csv")
    d = d[d.frame.isin(frames)]
    lum = luma(ds, a.seq, ds.frame_numbers(a.seq), RES)
    zone = scene.reflection_zone(a.seq, shape)
    left = boxes(BASE, d, frames, shape, lum, None)
    right = boxes(ZONE, d, frames, shape, lum, zone)
    zmask = cv2.resize(zone.astype(np.uint8), (TW, TH), interpolation=cv2.INTER_NEAREST)
    cnts, _ = cv2.findContours(zmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sx, sy = TW / shape[1], TH / shape[0]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{2 * TW}x{TH}",
           "-r", "25", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
           str(a.out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    caught = 0
    for n in frames:
        img = load_frame(ds.frame_path(a.seq, n), n)
        tiles = []
        xy = {}
        for side, bx in (("L", left), ("R", right)):
            g = bx.get(n)
            xy[side] = scale_boxes(g[["x", "y", "w", "h"]].to_numpy(), shape) if g is not None else np.zeros((0, 4), int)
            out = fast_blur(img, xy[side]) if len(xy[side]) else img
            tiles.append(cv2.resize(out, (TW, TH), interpolation=cv2.INTER_AREA))
        L, R = tiles
        # faces the zone added: right boxes with no left box overlapping them
        for x1, y1, x2, y2 in xy["R"]:
            hit = any(min(x2, u2) > max(x1, u1) and min(y2, v2) > max(y1, v1) for u1, v1, u2, v2 in xy["L"])
            cx, cy = (x1 + x2) / 2 / shape[1], (y1 + y2) / 2 / shape[0]
            inz = zone[min(int(cy * zone.shape[0]), zone.shape[0] - 1), min(int(cx * zone.shape[1]), zone.shape[1] - 1)]
            r = (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy))
            if inz:
                cv2.rectangle(R, *r, (0, 255, 0), 2)
            if not hit:
                caught += 1
                cv2.rectangle(L, *r, (0, 0, 255), 2)
        cv2.drawContours(R, cnts, -1, (0, 200, 0), 2)
        label(L, "acum: regula de intuneric + tracker + look-back 3")
        label(R, "+ zona usii (prag 0.1)", (120, 255, 120))
        proc.stdin.write(np.hstack([L, R]).tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"{a.out}: {len(frames)} frames; {caught} face boxes masked only thanks to the zone (red on the left)")


if __name__ == "__main__":
    main()
