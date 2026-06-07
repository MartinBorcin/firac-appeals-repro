from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from firac_appeals.schema.models import CaseRecord
from firac_appeals.utils.io import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize generic JSONL/CSV judgments into FIRAC case JSONL.")
    parser.add_argument("--input", required=True, help="Input JSONL, JSON array, or CSV file")
    parser.add_argument("--output", required=True, help="Output case JSONL")
    parser.add_argument("--source-dataset", default="user_data")
    parser.add_argument("--id-field", default="case_id")
    parser.add_argument("--text-field", default="judgment_text")
    parser.add_argument("--summary-field", default="reference_summary")
    parser.add_argument(
        "--metadata-fields",
        nargs="*",
        default=[],
        help="Optional fields to copy into metadata. Defaults to all non-core fields.",
    )
    args = parser.parse_args()

    rows = read_records(Path(args.input))
    cases = [
        normalize_case(
            row,
            source_dataset=args.source_dataset,
            id_field=args.id_field,
            text_field=args.text_field,
            summary_field=args.summary_field,
            metadata_fields=args.metadata_fields,
        )
        for row in rows
    ]
    write_jsonl(args.output, cases)
    print(f"Wrote {len(cases)} FIRAC case records to {args.output}")


def read_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError("JSON input must be an array of objects")
        return [dict(row) for row in value]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def normalize_case(
    row: dict[str, Any],
    *,
    source_dataset: str,
    id_field: str,
    text_field: str,
    summary_field: str,
    metadata_fields: list[str],
) -> dict[str, Any]:
    remapped = dict(row)
    if id_field in row:
        remapped["case_id"] = row[id_field]
    if text_field in row:
        remapped["judgment_text"] = row[text_field]
    if summary_field in row:
        remapped["reference_summary"] = row[summary_field]
    record = CaseRecord.from_any(remapped, source_dataset=source_dataset)
    out = record.to_dict()
    core_fields = {id_field, text_field, summary_field, "case_id", "judgment_text", "reference_summary"}
    if metadata_fields:
        out["metadata"] = {field: row.get(field, "") for field in metadata_fields}
    else:
        out["metadata"] = {key: value for key, value in row.items() if key not in core_fields}
    return out


if __name__ == "__main__":
    main()
