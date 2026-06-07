#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from firac_appeals.evaluation.downstream import build_task_examples
from firac_appeals.utils.io import ensure_parent, read_jsonl, write_jsonl


DEFAULT_TASKS = ["outcome_prediction", "issue_generation", "rule_recommendation"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export FIRAC-derived downstream benchmark tasks.")
    parser.add_argument("--input", required=True, help="Annotated final.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS, choices=DEFAULT_TASKS)
    parser.add_argument("--split", default="", help="Optional split filter, e.g. train or test")
    parser.add_argument("--source-dataset", default="", help="Optional source_dataset filter")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-judgment", action="store_true", help="Include original judgment text")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    rows = [row for row in rows if row.get("annotation_status", "ok") == "ok"]
    if args.split:
        rows = [row for row in rows if str(row.get("split", "")).lower() == args.split.lower()]
    if args.source_dataset:
        rows = [row for row in rows if str(row.get("source_dataset", "")) == args.source_dataset]
    if args.limit is not None:
        rows = rows[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {
        "input": args.input,
        "rows_seen": len(rows),
        "split_filter": args.split,
        "source_dataset_filter": args.source_dataset,
        "tasks": {},
    }
    for task in args.tasks:
        examples = build_task_examples(rows, task)
        if args.include_judgment:
            judgment_by_id = {str(row.get("case_id", "")): str(row.get("judgment_text", "")) for row in rows}
            for example in examples:
                example["input"]["judgment_text"] = judgment_by_id.get(str(example.get("case_id", "")), "")
        path = output_dir / f"{task}.jsonl"
        write_jsonl(path, examples)
        stats["tasks"][task] = task_stats(examples, str(path))

    stats_path = ensure_parent(output_dir / "stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {len(args.tasks)} downstream task files to {output_dir}")


def task_stats(examples: list[dict[str, Any]], path: str) -> dict[str, Any]:
    split_counts = Counter(str(example.get("split", "")) for example in examples)
    source_counts = Counter(str(example.get("source_dataset", "")) for example in examples)
    target_counts = Counter()
    for example in examples:
        target = example.get("target") or {}
        if "label" in target:
            target_counts[str(target["label"])] += 1
    return {
        "path": path,
        "count": len(examples),
        "splits": dict(split_counts),
        "source_datasets": dict(source_counts),
        "target_labels": dict(target_counts),
    }


if __name__ == "__main__":
    main()
