"""Fetch three pinned CC-BY-4.0 PlantDoc images for Stage 1 verification."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen


REVISION = "5467f6012d78d1c446145d5f582da6096f852ae8"
BASE_URL = f"https://raw.githubusercontent.com/pratikkayal/PlantDoc-Dataset/{REVISION}/test"
FIXTURES = {
    "healthy-grape.jpg": (
        "grape leaf/730-grape-leaf-2560x1600-nature-wallpaper.jpg",
        "468f656f6c89d8f241f07246300acb790a11f3e288e9c3750dac1b4da9405577",
    ),
    "diseased-apple-scab.jpg": (
        "Apple Scab Leaf/apples_apple-scab_01_zoom.jpg",
        "b47288fa0e351c4483d62ff1bd6db178e3fba89137407726e6f2890cf7a5d248",
    ),
    "low-confidence.jpg": (
        "Tomato leaf mosaic virus/page_2.jpg",
        "995ada04ba135f89218ddc3f17617e59411b63ae5167c25ded94284aeda9d7e8",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)

    for filename, (relative_path, expected_hash) in FIXTURES.items():
        url = f"{BASE_URL}/{quote(relative_path, safe='/')}"
        with urlopen(url, timeout=60) as response:
            payload = response.read()
        actual_hash = hashlib.sha256(payload).hexdigest()
        if actual_hash != expected_hash:
            raise SystemExit(f"Fixture SHA-256 mismatch for {filename}: {actual_hash}")
        (args.destination / filename).write_bytes(payload)
        print(f"{filename}: {actual_hash}")


if __name__ == "__main__":
    main()
