"""Fetch pinned private Candidate-v2 files before the API starts."""

import hashlib
import os
import re
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from install_candidate_v2 import EXPECTED_HASHES

REPO = "pinkyproject25-cmd/crop-diseasee-ai-artifacts"
MODEL_DIR = Path(__file__).resolve().parent / "models"
MAX_BYTES = 50 * 1024 * 1024


def matches(path: Path, expected: str) -> bool:
    if not path.is_file():
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    missing = [
        name for name, expected in EXPECTED_HASHES.items()
        if not matches(MODEL_DIR / name, expected)
    ]
    if not missing:
        print("Verified all Candidate-v2 model files")
        return

    token = os.environ.get("MODEL_ARTIFACT_TOKEN")
    ref = os.environ.get("MODEL_ARTIFACT_REF", "")
    if not token or not re.fullmatch(r"[0-9a-fA-F]{40}", ref):
        raise SystemExit("Private model token and pinned commit SHA are required")

    for name in missing:
        url = (
            f"https://api.github.com/repos/{REPO}/contents/{quote(name)}?"
            f"{urlencode({'ref': ref})}"
        )
        request = Request(url, headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.raw+json",
        })
        temporary = MODEL_DIR / f".{name}.download"
        try:
            total = 0
            with urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_BYTES:
                        raise ValueError(f"{name} exceeds the download limit")
                    output.write(chunk)
            if not matches(temporary, EXPECTED_HASHES[name]):
                raise ValueError(f"SHA-256 mismatch for {name}")
            temporary.replace(MODEL_DIR / name)
        finally:
            temporary.unlink(missing_ok=True)

    print("Downloaded and verified all Candidate-v2 model files")


if __name__ == "__main__":
    main()