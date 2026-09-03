#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a checksum manifest for public artifacts.")
    parser.add_argument("--root", default="artifacts")
    parser.add_argument("--output", default="artifacts/MANIFEST.json")
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output)
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.resolve() == output.resolve():
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    payload = {
        "schema_version": 1,
        "root": root.as_posix(),
        "files": rows,
        "total_bytes": sum(row["bytes"] for row in rows),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
