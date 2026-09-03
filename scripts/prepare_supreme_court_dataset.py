#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

import fitz
import pandas as pd

from firac_appeals.schema.models import normalize_space
from firac_appeals.utils.io import ensure_parent, write_jsonl


BASE_URL = "https://indian-supreme-court-judgments.s3.amazonaws.com"
WORD_RE = re.compile(r"\w+")
SPACE_RE = re.compile(r"\s+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Supreme Court judgments from vanga public S3 data.")
    parser.add_argument("--raw-dir", default="data/raw/supreme_court_vanga")
    parser.add_argument("--output", default="data/interim/supreme_court_vanga.jsonl")
    parser.add_argument("--years", nargs="*", type=int, default=[])
    parser.add_argument("--year-from", type=int, default=1950)
    parser.add_argument("--year-to", type=int, default=2026)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-per-year", type=int, default=None)
    parser.add_argument("--min-body-words", type=int, default=100)
    args = parser.parse_args()

    years = args.years or list(range(args.year_from, args.year_to + 1))
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for year in years:
        rows.extend(prepare_year(year, raw_dir, download=args.download, args=args))
        if args.limit is not None and len(rows) >= args.limit:
            rows = rows[: args.limit]
            break

    write_jsonl(args.output, rows)
    write_stats(Path(args.output).with_suffix(".stats.json"), rows)
    print(f"Wrote {len(rows)} Supreme Court records to {args.output}")


def prepare_year(year: int, raw_dir: Path, *, download: bool, args: argparse.Namespace) -> list[dict[str, Any]]:
    year_dir = raw_dir / f"year={year}"
    year_dir.mkdir(parents=True, exist_ok=True)
    index_path = year_dir / "english.index.json"
    metadata_path = year_dir / "metadata.parquet"
    if download:
        download_if_missing(f"{BASE_URL}/data/tar/year={year}/english/english.index.json", index_path)
        download_if_missing(f"{BASE_URL}/metadata/parquet/year={year}/metadata.parquet", metadata_path)
    if not index_path.exists() or not metadata_path.exists():
        print(f"Skipping {year}: missing index or metadata. Use --download to fetch.")
        return []

    index = json.loads(index_path.read_text(encoding="utf-8"))
    metadata = pd.read_parquet(metadata_path).fillna("")
    metadata_by_pdf = {}
    for record in metadata.to_dict(orient="records"):
        path = str(record.get("path", "")).strip()
        if path:
            metadata_by_pdf[f"{path}_EN.pdf"] = record

    target_files = set(metadata_by_pdf)
    if args.sample_per_year:
        target_files = set(sorted(target_files)[: args.sample_per_year])

    rows: list[dict[str, Any]] = []
    for part in index.get("parts", []):
        part_name = part["name"]
        tar_path = year_dir / part_name
        if download:
            download_if_missing(f"{BASE_URL}/data/tar/year={year}/english/{part_name}", tar_path)
        if not tar_path.exists():
            print(f"Skipping {year}/{part_name}: tar not found. Use --download to fetch.")
            continue
        rows.extend(read_tar_part(tar_path, metadata_by_pdf, target_files, args.min_body_words))
    print(f"Prepared {len(rows)} records for year={year}")
    return rows


def download_if_missing(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    ensure_parent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    print(f"Downloading {url}")
    request = urllib.request.Request(url, method="GET")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=120) as response, tmp.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(path)


def read_tar_part(
    tar_path: Path,
    metadata_by_pdf: dict[str, dict[str, Any]],
    target_files: set[str],
    min_body_words: int,
) -> list[dict[str, Any]]:
    rows = []
    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith("_EN.pdf"):
                continue
            name = Path(member.name).name
            if name not in target_files:
                continue
            meta = metadata_by_pdf.get(name, {})
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            pdf_bytes = extracted.read()
            extraction = extract_pdf_text(pdf_bytes, meta)
            body_words = len(words(extraction["judgment_text"]))
            rows.append(
                {
                    "case_id": case_id(meta, name),
                    "source_dataset": "supreme_court_vanga",
                    "judgment_text": extraction["judgment_text"],
                    "reference_summary": "",
                    "metadata": clean_metadata(meta)
                    | {
                        "pdf_file": name,
                        "pdf_tar": str(tar_path),
                        "pdf_size_bytes": len(pdf_bytes),
                        "pdf_pages": extraction["pages"],
                        "pdf_text_chars": extraction["full_text_chars"],
                        "body_text_chars": len(extraction["judgment_text"]),
                        "body_words": body_words,
                        "body_start_strategy": extraction["start_strategy"],
                        "body_start_offset": extraction["start_offset"],
                        "body_text_ratio": extraction["body_text_ratio"],
                        "possible_ocr_needed": extraction["possible_ocr_needed"],
                        "possible_headnote_or_front_matter_retained": extraction[
                            "possible_headnote_or_front_matter_retained"
                        ],
                        "usable_for_annotation": body_words >= min_body_words
                        and not extraction["possible_ocr_needed"],
                    },
                    "split": "unassigned",
                }
            )
    return rows


def extract_pdf_text(pdf_bytes: bytes, meta: dict[str, Any]) -> dict[str, Any]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_texts = [page.get_text("text") for page in doc]
    full_text = "\n".join(page_texts)
    start_offset, strategy = find_body_start(full_text)
    if start_offset is None:
        start_offset, strategy = find_title_start(full_text, str(meta.get("title", "")))
    if start_offset is None:
        start_offset, strategy = 0, "full_text_fallback"
    body = clean_pdf_text(full_text[start_offset:])
    full_words = len(words(full_text))
    body_words = len(words(body))
    printable_ratio = sum(c.isprintable() or c.isspace() for c in full_text) / max(len(full_text), 1)
    possible_ocr_needed = full_words < max(50, doc.page_count * 20) or printable_ratio < 0.95
    return {
        "judgment_text": body,
        "pages": doc.page_count,
        "full_text_chars": len(full_text),
        "start_offset": start_offset,
        "start_strategy": strategy,
        "body_text_ratio": body_words / max(full_words, 1),
        "possible_ocr_needed": possible_ocr_needed,
        "possible_headnote_or_front_matter_retained": strategy in {"title_start", "full_text_fallback"},
    }


def find_body_start(text: str) -> tuple[int | None, str]:
    patterns = [
        (
            "scr_judgment_order",
            re.compile(
                r"Judgment\s*/\s*Order\s+of\s+the\s+Supreme\s+Court\s+(?:Judgment|Order)?",
                re.I,
            ),
        ),
        (
            "delivered_by",
            re.compile(r"(?:The\s+)?(?:Judg\w*|Order)[\s\S]{0,180}?delivered\s+by", re.I),
        ),
        ("was_delivered_by", re.compile(r"was\s+delivered\s+by", re.I)),
        (
            "standalone_judgment",
            re.compile(r"(?:^|\n)\s*J\s*U\s*D\s*G\s*M\s*E\s*N\s*T\s*(?:\n|$|[A-Z][A-Za-z .]*,?\s*J\.)", re.I),
        ),
        ("standalone_order", re.compile(r"(?:^|\n)\s*O\s*R\s*D\s*E\s*R\s*[.:]?\s*(?:\n|$)", re.I)),
    ]
    matches: list[tuple[int, str]] = []
    for name, pattern in patterns:
        for match in pattern.finditer(text):
            # Standalone ORDER at the beginning of an extracted SCR page is often
            # the previous case tail. Prefer later judgment-delivery markers.
            if name == "standalone_order" and match.start() < 1500:
                continue
            start = match.start()
            if name == "was_delivered_by":
                start = max(0, start - 220)
            matches.append((start, name))
            break
    if not matches:
        return None, ""
    return sorted(matches, key=lambda item: start_priority(item[1], item[0]))[0]


def start_priority(strategy: str, offset: int) -> tuple[int, int]:
    priority = {
        "scr_judgment_order": 0,
        "delivered_by": 1,
        "was_delivered_by": 2,
        "standalone_judgment": 3,
        "standalone_order": 4,
    }
    return priority.get(strategy, 99), offset


def find_title_start(text: str, title: str) -> tuple[int | None, str]:
    title = normalize_space(title)
    if not title:
        return None, ""
    variants = {title, title.replace(" versus ", " v. "), title.replace(" versus ", " v ")}
    compact_text = SPACE_RE.sub(" ", text)
    for variant in variants:
        idx = compact_text.lower().find(variant.lower())
        if idx >= 0:
            return idx, "title_start"
    return None, ""


def clean_pdf_text(text: str) -> str:
    text = CONTROL_RE.sub(" ", text)
    text = re.sub(r"(?<=[a-z])-\s+(?=[a-z])", "", text)
    text = text.replace("\u00ad", "")
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def clean_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "title",
        "petitioner",
        "respondent",
        "description",
        "judge",
        "author_judge",
        "citation",
        "case_id",
        "cnr",
        "decision_date",
        "disposal_nature",
        "court",
        "available_languages",
        "path",
        "nc_display",
        "scraped_at",
        "year",
    ]
    return {key: normalize_space(meta.get(key, "")) for key in keys}


def case_id(meta: dict[str, Any], pdf_name: str) -> str:
    ident = normalize_space(meta.get("case_id", "")) or normalize_space(meta.get("nc_display", ""))
    if ident:
        ident = re.sub(r"[^A-Za-z0-9]+", "_", ident).strip("_").lower()
    else:
        ident = Path(pdf_name).stem.lower()
    return f"sc_{ident}"


def words(text: str) -> list[str]:
    return WORD_RE.findall(text or "")


def write_stats(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        ensure_parent(path).write_text("{}", encoding="utf-8")
        return
    body_words = [row["metadata"]["body_words"] for row in rows]
    usable = sum(1 for row in rows if row["metadata"]["usable_for_annotation"])
    stats = {
        "records": len(rows),
        "usable_for_annotation": usable,
        "min_body_words": min(body_words),
        "max_body_words": max(body_words),
        "mean_body_words": sum(body_words) / len(body_words),
        "possible_ocr_needed": sum(1 for row in rows if row["metadata"]["possible_ocr_needed"]),
        "possible_headnote_or_front_matter_retained": sum(
            1 for row in rows if row["metadata"]["possible_headnote_or_front_matter_retained"]
        ),
    }
    ensure_parent(path).write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
