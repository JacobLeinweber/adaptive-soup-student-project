"""ZIP archives for the hyperparameter search on another machine.

    python pack_hp_search.py inputs    # data and seed-52 checkpoints that are not in git -> hp_search_inputs_cross_dataset_seed52.zip
    python pack_hp_search.py results   # results/hp_search/ including checkpoints -> hp_search_results_<date>.zip

inputs: exactly the files marked "git-lfs" in configs/hp_search_cross_dataset.json (prepared data and the supplied
seed-52 source-only checkpoints of the six cross-dataset transfers), under their repository paths, so the archive
is unpacked into the repository root. Every file is checked against its size and SHA-256 before packing.
results: everything below results/hp_search/ except temporary files; checkpoints are stored, the rest compressed.
Both write the archive under a temporary name first and print its path and size.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "hp_search_cross_dataset.json"
RESULTS = ROOT / "results" / "hp_search"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finish(tmp: Path, target: Path, n_files: int, raw_bytes: int) -> None:
    os.replace(tmp, target)
    size = target.stat().st_size
    print(f"Saved {target}")
    print(f"  {n_files} files, {raw_bytes / 1e6:.1f} MB unpacked, ZIP {size / 1e6:.1f} MB ({size} bytes)")
    print(f"  SHA-256 of the ZIP: {sha256(target)}")


def pack_inputs(output: Path | None) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    entries = [entry for entry in config["inputs"] if entry["stored"] == "git-lfs"]
    problems = []
    for entry in entries:
        path = ROOT / entry["path"]
        if not path.is_file():
            problems.append(f"missing: {entry['path']}")
        elif path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            problems.append(f"differs from the configuration (size or SHA-256): {entry['path']}")
    if problems:
        sys.exit("Not packed:\n  " + "\n  ".join(problems))
    target = output or ROOT / config["inputs_zip"]
    tmp = target.with_name(target.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for entry in entries:
            archive.write(ROOT / entry["path"], arcname=entry["path"])
            print(f"  {entry['bytes'] / 1e6:8.1f} MB  {entry['path']}")
    finish(tmp, target, len(entries), sum(entry["bytes"] for entry in entries))
    print(f"Unpack in the repository root of the other machine: python -m zipfile -e {target.name} .")


def pack_results(source: Path, output: Path | None) -> None:
    if not source.is_dir():
        sys.exit(f"Nothing to pack: {source} does not exist")
    files = sorted(path for path in source.rglob("*") if path.is_file() and ".tmp." not in path.name)
    base = ROOT if source.is_relative_to(ROOT) else source.parent  # inside the repository: paths from its root
    target = output or ROOT / f"hp_search_results_{datetime.now():%Y%m%d_%H%M}.zip"
    tmp = target.with_name(target.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            stored = zipfile.ZIP_STORED if path.suffix == ".pt" else zipfile.ZIP_DEFLATED  # weights hardly compress
            archive.write(path, arcname=path.relative_to(base).as_posix(), compress_type=stored)
    finished = sum(1 for path in files if path.name == "result.json")
    finish(tmp, target, len(files), sum(path.stat().st_size for path in files))
    print(f"  finished runs in the archive: {finished}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["inputs", "results"])
    parser.add_argument("--output", type=Path, help="path of the ZIP (default: repository root)")
    parser.add_argument("--source", type=Path, default=RESULTS, help="results: folder to pack (default: results/hp_search)")
    args = parser.parse_args()
    if args.what == "inputs":
        pack_inputs(args.output)
    else:
        pack_results(args.source.resolve(), args.output)


if __name__ == "__main__":
    main()
