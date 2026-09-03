#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from firac_appeals.utils.io import read_jsonl, write_jsonl


BINARY_LABELS = {"allowed", "dismissed"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a downstream task directory with binary outcome examples only."
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outcome_rows = read_jsonl(source_dir / "outcome_prediction.jsonl")
    retained = [
        row
        for row in outcome_rows
        if str(row.get("target", {}).get("label", "")).lower() in BINARY_LABELS
    ]
    excluded = [row for row in outcome_rows if row not in retained]
    write_jsonl(output_dir / "outcome_prediction.jsonl", retained)

    # Issue and rule tasks are copied unchanged so the directory remains usable by
    # the existing three-task experiment runner.
    for task in ("issue_generation", "rule_recommendation"):
        source = source_dir / f"{task}.jsonl"
        if source.exists():
            write_jsonl(output_dir / f"{task}.jsonl", read_jsonl(source))

    stats = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "outcome_prediction": {
            "original_count": len(outcome_rows),
            "retained_count": len(retained),
            "excluded_count": len(excluded),
            "retained_fraction": len(retained) / len(outcome_rows) if outcome_rows else 0.0,
            "retained_labels": dict(
                Counter(str(row["target"]["label"]).lower() for row in retained)
            ),
            "excluded_labels": dict(
                Counter(str(row.get("target", {}).get("label", "")).lower() for row in excluded)
            ),
        },
    }
    (output_dir / "binary_outcome_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(stats["outcome_prediction"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
