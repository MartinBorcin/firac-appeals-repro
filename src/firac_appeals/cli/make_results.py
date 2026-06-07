from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble compact result tables from FIRAC metric JSON files.")
    parser.add_argument("--metrics-dir", required=True, help="Directory containing *.json metric files")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pattern", default="*.json")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    rows = [row for path in sorted(metrics_dir.rglob(args.pattern)) if (row := metric_row(path, metrics_dir))]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "results.csv", rows)
    write_markdown(output_dir / "results.md", rows)
    print(f"Wrote {len(rows)} result rows to {output_dir}")


def metric_row(path: Path, root: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, dict):
        return {}
    row: dict[str, Any] = {
        "source_file": str(path.relative_to(root)),
        "task": payload.get("task", payload.get("summary_variant", "")),
        "count": payload.get("count", ""),
    }
    for key, value in sorted(aggregate.items()):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            row[key] = round(float(value), 6)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = fieldnames_for(rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = fieldnames_for(rows)
    lines = []
    if fieldnames:
        lines.append("| " + " | ".join(fieldnames) + " |")
        lines.append("| " + " | ".join("---" for _ in fieldnames) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(field, "")) for field in fieldnames) + " |")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def fieldnames_for(rows: list[dict[str, Any]]) -> list[str]:
    leading = ["source_file", "task", "count"]
    rest = sorted({key for row in rows for key in row if key not in leading})
    return leading + rest


if __name__ == "__main__":
    main()
