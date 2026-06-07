from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from firac_appeals.ingest.common import first_present, load_tabular
from firac_appeals.schema.models import CaseRecord, stable_case_id


TEXT_SUFFIXES = {".txt", ".html", ".htm"}
PDF_SUFFIXES = {".pdf"}


def load_vanga_supreme_court(
    input_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    source_dataset: str = "supreme_court_vanga",
) -> list[dict[str, Any]]:
    """Load vanga Supreme Court judgments from local extracted files.

    `input_path` may be an extracted text/PDF directory or a tabular file with
    text fields. `metadata_path` may be JSONL/JSON/CSV/parquet and is joined by
    case id, filename, or path when possible.
    """
    path = Path(input_path)
    metadata_rows = load_metadata_by_key(metadata_path) if metadata_path else {}
    if path.is_file() and path.suffix.lower() in {".jsonl", ".json", ".csv", ".parquet"}:
        return [record_from_row(row, source_dataset) for row in load_tabular(path)]
    if path.is_file():
        text = extract_text_from_file(path)
        metadata = metadata_rows.get(path.stem, {})
        return [record_from_vanga(metadata, text, source_dataset, source_path=path)]
    if not path.is_dir():
        raise FileNotFoundError(path)

    rows: list[dict[str, Any]] = []
    for file_path in sorted(iter_judgment_files(path)):
        text = extract_text_from_file(file_path)
        metadata = metadata_for_file(metadata_rows, file_path)
        rows.append(record_from_vanga(metadata, text, source_dataset, source_path=file_path))
    if not rows:
        raise ValueError(f"Could not find extracted judgment text/PDF files in {path}")
    return rows


def load_metadata_by_key(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    rows = load_tabular(path)
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        for key in candidate_keys(row):
            by_key.setdefault(key, row)
    return by_key


def record_from_row(row: dict[str, Any], source_dataset: str) -> dict[str, Any]:
    record = CaseRecord.from_any(row, source_dataset=source_dataset)
    return record.to_dict()


def record_from_vanga(
    metadata: dict[str, Any],
    judgment_text: str,
    source_dataset: str,
    *,
    source_path: Path,
) -> dict[str, Any]:
    source_id = str(first_present(metadata, ["case_id", "doc_id", "filename", "path", "title"]))
    case_id = source_id or stable_case_id(source_dataset, source_path.stem, judgment_text[:500])
    record = CaseRecord(
        case_id=case_id,
        source_dataset=source_dataset,
        judgment_text=judgment_text,
        reference_summary="",
        metadata={**metadata, "source_path": str(source_path)},
    )
    return record.to_dict()


def iter_judgment_files(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES | PDF_SUFFIXES
    ]


def extract_text_from_file(path: str | Path) -> str:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".txt", ".html", ".htm"}:
        return strip_html(p.read_text(encoding="utf-8", errors="ignore"))
    if suffix == ".pdf":
        return extract_pdf_text(p)
    raise ValueError(f"Unsupported judgment file: {p}")


def extract_pdf_text(path: Path) -> str:
    try:
        import fitz

        with fitz.open(path) as doc:
            return "\n".join(page.get_text("text") for page in doc)
    except ImportError:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "PDF extraction requires pymupdf or pypdf. Install one of them, "
                "or provide extracted .txt/.html judgment files."
            ) from exc
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)


def strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()


def metadata_for_file(metadata_rows: dict[str, dict[str, Any]], path: Path) -> dict[str, Any]:
    for key in [path.stem, path.name, str(path), path.as_posix()]:
        if key in metadata_rows:
            return metadata_rows[key]
    return {}


def candidate_keys(row: dict[str, Any]) -> list[str]:
    keys = []
    for field in ["case_id", "doc_id", "filename", "file_name", "path", "source_path"]:
        value = str(row.get(field, "") or "").strip()
        if value:
            keys.extend([value, Path(value).name, Path(value).stem])
    return [key for key in keys if key]
