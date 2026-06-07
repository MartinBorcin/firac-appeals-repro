#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

from firac_appeals.utils.io import ensure_parent, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/test temporal splits for JSONL case rows.")
    parser.add_argument("--input", required=True, help="Input JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cutoff", required=True, help="ISO date. Rows on/after this date go to test.")
    parser.add_argument(
        "--date-path",
        default="metadata.decision_date",
        help="Dot path to date field; supports DD-MM-YYYY and ISO dates.",
    )
    parser.add_argument("--dataset-name", default="supreme_court_vanga")
    parser.add_argument("--write-combined", action="store_true", help="Also write all.jsonl with split assigned.")
    args = parser.parse_args()

    cutoff = date.fromisoformat(args.cutoff)
    rows = read_jsonl(args.input)
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    date_values: list[date] = []

    for row in rows:
        raw_date = get_path(row, args.date_path)
        parsed = parse_date(raw_date)
        if parsed is None:
            missing.append(row)
            continue
        date_values.append(parsed)
        split = "test" if parsed >= cutoff else "train"
        updated = dict(row)
        updated["split"] = split
        updated.setdefault("split_metadata", {})
        updated["split_metadata"] = {
            **updated.get("split_metadata", {}),
            "split_strategy": "temporal",
            "cutoff": args.cutoff,
            "date_path": args.date_path,
            "date_value": parsed.isoformat(),
            "test_rule": "date >= cutoff",
        }
        (test if split == "test" else train).append(updated)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "test.jsonl", test)
    if args.write_combined:
        write_jsonl(output_dir / "all.jsonl", sorted(train + test, key=lambda row: row["split_metadata"]["date_value"]))
    write_jsonl(output_dir / "missing_date.jsonl", missing)

    stats = build_stats(args, rows, train, test, missing, date_values)
    ensure_parent(output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest_rows = [
        {
            "case_id": row.get("case_id", ""),
            "split": row.get("split", ""),
            "decision_date": row.get("split_metadata", {}).get("date_value", ""),
            "source_dataset": row.get("source_dataset", args.dataset_name),
        }
        for row in train + test
    ]
    write_jsonl(output_dir / "manifest.jsonl", sorted(manifest_rows, key=lambda row: row["decision_date"]))
    print(
        f"Wrote temporal split to {output_dir}: "
        f"train={len(train)} test={len(test)} missing_date={len(missing)} cutoff={args.cutoff}"
    )


def build_stats(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    missing: list[dict[str, Any]],
    date_values: list[date],
) -> dict[str, Any]:
    total_with_dates = len(train) + len(test)
    test_years = Counter(row["split_metadata"]["date_value"][:4] for row in test)
    train_years = Counter(row["split_metadata"]["date_value"][:4] for row in train)
    return {
        "dataset_name": args.dataset_name,
        "input": args.input,
        "date_path": args.date_path,
        "split_strategy": "temporal",
        "cutoff": args.cutoff,
        "test_rule": "date >= cutoff",
        "rows_total": len(rows),
        "rows_with_dates": total_with_dates,
        "rows_missing_dates": len(missing),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_percent_of_dated": percent(len(train), total_with_dates),
        "test_percent_of_dated": percent(len(test), total_with_dates),
        "min_date": min(date_values).isoformat() if date_values else "",
        "max_date": max(date_values).isoformat() if date_values else "",
        "train_min_date": min_date(train),
        "train_max_date": max_date(train),
        "test_min_date": min_date(test),
        "test_max_date": max_date(test),
        "train_year_counts": dict(sorted(train_years.items())),
        "test_year_counts": dict(sorted(test_years.items())),
    }


def get_path(row: dict[str, Any], path: str) -> Any:
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def parse_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def percent(numerator: int, denominator: int) -> float:
    return round(100 * numerator / denominator, 4) if denominator else 0.0


def min_date(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    return min(row["split_metadata"]["date_value"] for row in rows)


def max_date(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    return max(row["split_metadata"]["date_value"] for row in rows)


if __name__ == "__main__":
    main()
