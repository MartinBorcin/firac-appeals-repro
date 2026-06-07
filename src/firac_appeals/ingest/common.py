from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_tabular(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [dict(row) for row in data]
        if isinstance(data, dict):
            for key in ["data", "rows", "examples", "records"]:
                if isinstance(data.get(key), list):
                    return [dict(row) for row in data[key]]
        raise ValueError(f"Unsupported JSON layout: {p}")
    if suffix == ".csv":
        with p.open("r", encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "Parquet input requires pandas and a parquet engine such as pyarrow. "
                "Install them separately or export metadata as JSONL/CSV."
            ) from exc
        return pd.read_parquet(p).fillna("").to_dict(orient="records")
    raise ValueError(f"Unsupported table format: {p}")


def first_present(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return ""
