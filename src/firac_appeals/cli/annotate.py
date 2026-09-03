#!/usr/bin/env python
from __future__ import annotations

import argparse
import random

from firac_appeals.annotation.pipeline import FiracPipeline, run_pipeline_sync
from firac_appeals.utils.io import read_jsonl, read_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/annotation/firac_pipeline.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--sample", choices=["first", "random", "test_split"], default="first")
    parser.add_argument("--sort-by", choices=["input", "judgment_length"], default="input")
    parser.add_argument("--max-judgment-chars", type=int, default=None)
    parser.add_argument("--require-usable-for-annotation", action="store_true")
    parser.add_argument("--deduplicate-case-id", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    rows = select_rows(
        rows,
        args.sample,
        args.limit,
        args.seed,
        args.sort_by,
        args.max_judgment_chars,
        args.require_usable_for_annotation,
        args.deduplicate_case_id,
    )
    config = read_yaml(args.config)
    if args.overwrite:
        config.setdefault("run", {})["overwrite"] = True
    pipeline = FiracPipeline(config, args.output_dir)
    results = run_pipeline_sync(pipeline, rows)
    ok = sum(1 for row in results if row.get("annotation_status") == "ok")
    print(f"Annotated {ok}/{len(results)} cases into {args.output_dir}")


def select_rows(
    rows: list[dict],
    sample: str,
    limit: int | None,
    seed: int,
    sort_by: str,
    max_judgment_chars: int | None,
    require_usable_for_annotation: bool,
    deduplicate_case_id: bool,
) -> list[dict]:
    if require_usable_for_annotation:
        rows = [
            row
            for row in rows
            if row.get("metadata", {}).get("usable_for_annotation", True)
        ]
    if sample == "test_split":
        test_rows = [row for row in rows if str(row.get("split", "")).lower() == "test"]
        rows = test_rows or rows
    elif sample == "random":
        rng = random.Random(seed)
        rows = rows[:]
        rng.shuffle(rows)
    if max_judgment_chars is not None:
        rows = [row for row in rows if len(row.get("judgment_text", "")) <= max_judgment_chars]
    if deduplicate_case_id:
        rows = deduplicate_rows_by_case_id(rows)
    if sort_by == "judgment_length":
        rows = sorted(rows, key=lambda row: len(row.get("judgment_text", "")))
    if limit is not None:
        rows = rows[:limit]
    return rows


def deduplicate_rows_by_case_id(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    deduplicated: list[dict] = []
    for row in rows:
        case_id = str(row.get("case_id", ""))
        if case_id in seen:
            continue
        seen.add(case_id)
        deduplicated.append(row)
    return deduplicated


if __name__ == "__main__":
    main()
