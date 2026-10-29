"""Download ChokePoint from Zenodo and rebuild data/chokepoint/.

Layout produced:
  README.txt
  groundtruth/<seq>.xml               eye-coordinate GT for all sequences
  frames/<seq>/*.jpg                  e.g. frames/P2E_S1_C1.1/00000300.jpg
  crops/<seq>/<identity>/*.pgm        96x96 face crops (portal archives P1E, P1L, P2E, P2L)

Usage:
  python src/loaders/chokepoint_download.py                 # default subset
  python src/loaders/chokepoint_download.py P1E_S1 P1L      # specific archives
  python src/loaders/chokepoint_download.py --all           # everything (~12 GB)
"""
import argparse
import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path

import chokepoint_meta
from chokepoint_meta import zenodo_files

DEFAULT = ["P2E_S1", "P2L_S1", "P2E_S5", "P2L_S5", "P2E"]
URL = "https://zenodo.org/records/815657/files/{}?download=1"


def download(name, md5, dest):
    """Stream name to dest, checking md5. Returns the path."""
    path = dest / name
    h = hashlib.md5()
    with urllib.request.urlopen(URL.format(name)) as r, open(path, "wb") as f:
        while chunk := r.read(1 << 20):
            h.update(chunk)
            f.write(chunk)
    if h.hexdigest() != md5:
        path.unlink()
        raise RuntimeError(f"md5 mismatch for {name}")
    return path


def extract(archive, dest):
    with tarfile.open(archive) as t:
        t.extractall(dest, filter="data")  # rejects absolute paths / .. escapes


def fetch(name, files, out):
    marker = out / f".done_{name}"
    if marker.exists():
        print(f"skip {name} (already done)")
        return
    key = f"{name}.tar.xz"
    print(f"download {key} ({files[key]['size'] / 1e6:.0f} MB)")
    tmp = out / f"_tmp_{name}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    extract(download(key, files[key]["checksum"].removeprefix("md5:"), tmp), tmp)
    (tmp / key).unlink()

    if name == "groundtruth":
        shutil.copytree(tmp / "groundtruth", out / "groundtruth", dirs_exist_ok=True)
    elif "_S" in name:  # session archive -> one nested archive per camera
        for inner in sorted(tmp.rglob("*.tar.xz")):
            extract(inner, out / "frames")
    else:  # portal archive -> face crops per sequence/identity
        for d in tmp.iterdir():
            shutil.copytree(d, out / "crops" / d.name, dirs_exist_ok=True)
    shutil.rmtree(tmp)
    marker.touch()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("archives", nargs="*", default=DEFAULT, help="archive names without .tar.xz")
    p.add_argument("--all", action="store_true", help="download every archive")
    p.add_argument("--out", type=Path, default=chokepoint_meta.DATA)
    args = p.parse_args()

    files = {f["key"]: f for f in zenodo_files()}
    names = sorted(k[: -len(".tar.xz")] for k in files if k.endswith(".tar.xz")) if args.all else args.archives
    unknown = [n for n in names if f"{n}.tar.xz" not in files]
    if unknown:
        p.error(f"unknown archives: {unknown}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "frames").mkdir(exist_ok=True)
    readme = args.out / "README.txt"
    if not readme.exists():
        urllib.request.urlretrieve(URL.format("README.txt"), readme)
    for name in ["groundtruth", *dict.fromkeys(n for n in names if n != "groundtruth")]:
        fetch(name, files, args.out)

    if args.out.resolve() == chokepoint_meta.DATA.resolve():
        chokepoint_meta.main()


if __name__ == "__main__":
    main()
