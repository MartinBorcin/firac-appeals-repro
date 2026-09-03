#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify released artifact sizes and SHA-256 hashes.")
    parser.add_argument("--manifest", default="artifacts/MANIFEST.json")
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    errors = []
    expected = set()
    for row in payload["files"]:
        path = root / row["path"]
        expected.add(path.resolve())
        if not path.is_file():
            errors.append(f"missing: {path}")
            continue
        if path.stat().st_size != row["bytes"]:
            errors.append(f"size mismatch: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            errors.append(f"hash mismatch: {path}")
    actual = {
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.resolve() != manifest_path.resolve()
    }
    for path in sorted(actual - expected):
        errors.append(f"unmanifested: {path.relative_to(root.resolve())}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Verified {len(expected)} artifacts ({payload['total_bytes']} bytes).")


if __name__ == "__main__":
    main()
