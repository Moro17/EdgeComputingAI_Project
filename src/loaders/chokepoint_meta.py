"""Build gt/chokepoint_metadata.csv: one row per ChokePoint camera sequence.

Usage: python src/loaders/chokepoint_meta.py
"""
import json
import os
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("ECA_DATA", ROOT / "data")) / "chokepoint"  # ECA_DATA: data outside the code dir (Colab)
GT_DIR = DATA / "groundtruth"
OUT = ROOT / "gt" / "chokepoint_metadata.csv"

ENV = {"P1E": "indoor", "P1L": "indoor", "P2E": "outdoor", "P2L": "indoor"}
FRONTAL = {  # "most frontal view" list from the dataset README
    "P1E_S1_C1", "P1E_S2_C2", "P1E_S3_C3", "P1E_S4_C1", "P1L_S1_C1", "P1L_S2_C2", "P1L_S3_C3", "P1L_S4_C1",
    "P2E_S1_C3", "P2E_S2_C2", "P2E_S3_C1", "P2E_S4_C2", "P2L_S1_C1", "P2L_S2_C2", "P2L_S3_C3", "P2L_S4_C2",
}
# S5 has no official eye GT; PLUSFiaQ ships face boxes for these (licence request pending)
PLUSFIAQ = ["P2E_S5_C1.1", "P2E_S5_C2.1", "P2E_S5_C3.1", "P2L_S5_C1.1", "P2L_S5_C2.1", "P2L_S5_C3.1"]


def zenodo_files():
    with urllib.request.urlopen("https://zenodo.org/api/records/815657") as r:
        return json.load(r)["files"]


def parse_gt(path):
    frames = ET.parse(path).getroot().findall("frame")
    people = [p for f in frames for p in f.findall("person")]
    per_frame = [len(f.findall("person")) for f in frames]
    return {
        "n_frames": len(frames),
        "n_annotated_frames": sum(n > 0 for n in per_frame),
        "n_face_instances": len(people),
        "max_faces_per_frame": max(per_frame, default=0),
        "n_identities": len({p.get("id") for p in people}),
        "identities": " ".join(sorted({p.get("id") for p in people})),
    }


def row(seq, gt_type, gt_file, files, stats=None):
    base = seq.split("_C")[0]  # e.g. P2E_S1
    portal = seq[:3]
    archive = files.get(base, {})
    frames = DATA / "frames" / seq
    return {
        "dataset": "chokepoint",
        "sequence": seq,
        "portal": portal[:2],
        "direction": {"E": "enter", "L": "leave"}[portal[2]],
        "session": base.split("_")[1],
        "camera": seq.split("_C")[1][0],
        "environment": ENV[portal],
        "frontal_view": seq.split(".")[0] in FRONTAL,
        "resolution": "800x600",
        "fps": 30,
        "gt_type": gt_type,
        "gt_file": gt_file,
        **(stats or {}),
        "archive": f"{base}.tar.xz",
        "archive_mb": round(archive.get("size", 0) / 1e6, 1),
        "archive_md5": archive.get("checksum", "").removeprefix("md5:"),
        "frames_dir": str(frames.relative_to(ROOT)) if frames.exists() else "",
        "n_frames_on_disk": len(list(frames.glob("*.jpg"))) if frames.exists() else 0,
        "license": "CC-BY-NC-4.0",
    }


def main():
    files = {f["key"][: -len(".tar.xz")]: f for f in zenodo_files() if f["key"].endswith(".tar.xz")}
    rows = [row(p.stem, "eyes", str(p.relative_to(ROOT)), files, parse_gt(p)) for p in sorted(GT_DIR.glob("*.xml"))]
    rows += [row(s, "boxes_plusfiaq_pending", "", files) for s in PLUSFIAQ]
    df = pd.DataFrame(rows)
    assert len(df) == 72 + 6 and df["sequence"].is_unique, len(df)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.drop(columns="identities").to_string(max_rows=20))


if __name__ == "__main__":
    main()
