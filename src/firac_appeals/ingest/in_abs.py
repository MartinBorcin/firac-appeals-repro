from __future__ import annotations

from pathlib import Path
from typing import Any

from firac_appeals.ingest.common import load_tabular
from firac_appeals.schema.models import CaseRecord


def load_in_abs(input_path: str | Path) -> list[dict[str, Any]]:
    """Load IN-Abs from common extracted layouts.

    Supported inputs:
    - JSONL/JSON/CSV tables with judgment and summary fields.
    - Extracted directories containing judgment/document text files and optional
      summary/headnote text files with matching stems.
    """
    path = Path(input_path)
    if path.is_file():
        return [_record(row).to_dict() for row in load_tabular(path)]
    if not path.is_dir():
        raise FileNotFoundError(path)

    tabular = sorted(path.rglob("*.jsonl")) + sorted(path.rglob("*.json")) + sorted(path.rglob("*.csv"))
    if tabular:
        return [_record(row).to_dict() for row in load_tabular(tabular[0])]

    judgment_files = [
        p
        for p in path.rglob("*.txt")
        if any(token in str(p).lower() for token in ["judgement", "judgment", "document"])
    ]
    summary_files = {
        split_stem(p): p
        for p in path.rglob("*.txt")
        if any(token in str(p).lower() for token in ["summary", "summaries", "headnote"])
    }
    rows: list[dict[str, Any]] = []
    for judgment_file in sorted(judgment_files):
        split = infer_split(judgment_file)
        summary_file = summary_files.get(split_stem(judgment_file))
        judgment = judgment_file.read_text(encoding="utf-8", errors="ignore")
        summary = summary_file.read_text(encoding="utf-8", errors="ignore") if summary_file else ""
        rows.append(
            CaseRecord(
                case_id=f"in_abs_{split}_{judgment_file.stem}",
                source_dataset="in_abs",
                judgment_text=judgment,
                reference_summary=summary,
                metadata={
                    "source_path": str(judgment_file),
                    "summary_path": str(summary_file or ""),
                    "original_id": judgment_file.stem,
                },
            ).to_dict()
            | {"split": split}
        )
    if not rows:
        raise ValueError(f"Could not find IN-Abs records in {path}")
    return rows


def _record(row: dict[str, Any]) -> CaseRecord:
    return CaseRecord.from_any(row, source_dataset="in_abs")


def infer_split(path: Path) -> str:
    lowered = str(path).lower()
    if "test-data" in lowered or "/test/" in lowered:
        return "test"
    if "train-data" in lowered or "/train/" in lowered:
        return "train"
    if "dev-data" in lowered or "val-data" in lowered or "/dev/" in lowered or "/val/" in lowered:
        return "dev"
    return "unknown"


def split_stem(path: Path) -> str:
    return f"{infer_split(path)}::{path.stem}"
