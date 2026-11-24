"""H.264 demo videos for the board from the benchmark data (the VPU decodes H.264 mp4; frames stay out of git).

    python src/make_demo_videos.py --out dist/demo_videos       # needs ffmpeg on PATH, ECA_DATA set
    scp dist/demo_videos/*.mp4 imx:/root/edge/                  # or: .\\scripts\\demo.ps1 install-videos

chokepoint.mp4       ChokePoint test P2E_S1_C2.1 (24 people walking through a portal), clean
chokepoint_dark.mp4  the same sequence at dark:2 (src/degrade), where the dark rule matters
musicvideo.mp4       30 s from each of 4 Music Video clips (shot changes, fast motion), centre-cropped to 4:3
reflection.mp4       ChokePoint test P2E_S1_C3.1: faces reflected in the door glass (src/analysis/outside_gt.py)
<name>_zone.npy      next to each ChokePoint video: its camera's door zone (systems/scene.py reflection_zone) for
                     edge_pipeline.py --zone (demo_board.sh door)
"""
import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loaders.chokepoint as cp  # noqa: E402
import loaders.musicvideo as mv  # noqa: E402
from degrade.degrade import load  # noqa: E402
from systems import scene  # noqa: E402

FPS = 25
W, H = 960, 720  # 4:3, like the 640x480 detector input: no aspect distortion in the g2d resize
CP_SEQ = "P2E_S1_C2.1"
REFL_SEQ = "P2E_S1_C3.1"
MV_CLIPS = ["BrunoMars.1", "Westlife.1", "T-ara.1", "GirlsAloud.1"]
MV_SECONDS = 30


def letterbox(img):
    """Fit into W x H. Wider-than-4:3 frames are centre-cropped, not padded: black bars would lower the frame's mean
    grey, which the dark rule reads (79% of Music Video frames under grey 60 with bars, 62% without)."""
    h, w = img.shape[:2]
    if w * 3 > h * 4:
        cw = h * 4 // 3
        img = img[:, (w - cw) // 2:(w - cw) // 2 + cw]
        h, w = img.shape[:2]
    r = min(W / w, H / h)
    nw, nh = round(w * r), round(h * r)
    out = np.zeros((H, W, 3), np.uint8)
    y, x = (H - nh) // 2, (W - nw) // 2
    out[y:y + nh, x:x + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out


def write(path, frames):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p",
           "-crf", "20", "-g", str(2 * FPS), "-movflags", "+faststart", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = 0
    for f in frames:
        proc.stdin.write(letterbox(f).tobytes())
        n += 1
    proc.stdin.close()
    if proc.wait():
        sys.exit(f"ffmpeg failed on {path}")
    print(f"{path}: {n} frames, {n / FPS:.0f} s", flush=True)


def chokepoint(deg=None, seq=CP_SEQ):
    for n in cp.frame_numbers(seq):
        yield load(cp.frame_path(seq, n), n, deg)


def zone(path, seq):
    """The camera's door zone as a bool grid; 4:3 like the video, so it maps onto any 4:3 size."""
    np.save(path, scene.reflection_zone(seq, cv2.imread(str(cp.frame_path(seq, cp.frame_numbers(seq)[0]))).shape))


def musicvideo():
    for clip in MV_CLIPS:
        for n in mv.frame_numbers(clip)[:MV_SECONDS * FPS]:
            yield cv2.imread(str(mv.frame_path(clip, n)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("dist/demo_videos"))
    p.add_argument("--only", nargs="*", default=["chokepoint", "chokepoint_dark", "musicvideo", "reflection"])
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    make = {"chokepoint": chokepoint, "chokepoint_dark": lambda: chokepoint("dark:2"), "musicvideo": musicvideo,
            "reflection": lambda: chokepoint(seq=REFL_SEQ)}
    seq = {"chokepoint": CP_SEQ, "chokepoint_dark": CP_SEQ, "reflection": REFL_SEQ}
    for name in a.only:
        if name in seq:
            zone(a.out / f"{name}_zone.npy", seq[name])
        if name == "musicvideo" or not (a.out / f"{name}.mp4").exists() or name == "reflection":
            write(a.out / f"{name}.mp4", make[name]())


if __name__ == "__main__":
    main()
