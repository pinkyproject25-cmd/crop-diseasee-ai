"""Verify and install owner-supplied Candidate-v2 runtime artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


MODEL_VERSION = "mixed-pv-plantdoc-mnv3-20261006T194137Z"
EXPECTED_HASHES = {
    "crop_classifier.onnx": "864dd9f77c01e6b4f77acfc1e0446b49c39855f31d5aed65c0d2a9a3b9ad4bb7",
    "labels.json": "16925c0cb74cd5be219eb9e8cf87f0fa4f5c255f49b8db7c865c55050cb7d307",
    "candidate_manifest.json": "357efbe591ca6e960acb5bf79e1211d164ec63bbfbd10244be40924db84d8c15",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Private directory containing the three Candidate-v2 files")
    parser.add_argument("--destination", type=Path, default=Path(__file__).parent / "models")
    args = parser.parse_args()

    for filename, expected in EXPECTED_HASHES.items():
        path = args.source / filename
        if not path.is_file():
            raise SystemExit(f"Missing required artifact: {path}")
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(f"SHA-256 mismatch for {filename}: {actual}")

    labels = json.loads((args.source / "labels.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.source / "candidate_manifest.json").read_text(encoding="utf-8"))
    if len(labels) != 38 or len(set(labels)) != 38:
        raise SystemExit("labels.json must contain the recorded 38 unique ordered labels")
    if manifest.get("model_version") != MODEL_VERSION:
        raise SystemExit("Manifest model version mismatch")
    if manifest.get("production_approved") is not False or manifest.get("candidate_only") is not True:
        raise SystemExit("Manifest must identify an experimental, non-production candidate")

    args.destination.mkdir(parents=True, exist_ok=True)
    for filename in EXPECTED_HASHES:
        shutil.copyfile(args.source / filename, args.destination / filename)
    print(f"Verified and installed Candidate-v2 artifacts in {args.destination.resolve()}")


if __name__ == "__main__":
    main()
