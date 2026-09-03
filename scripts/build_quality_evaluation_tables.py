#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from firac_appeals.evaluation.human_rubric import (
    load_annotator_aliases,
    load_tasks,
    parse_tasks,
)
from firac_appeals.evaluation.quality_tables import (
    alignment_table,
    annotator_summary_table,
    human_likert_table,
    llm_likert_table,
    load_llm_scores,
    render_quality_tables_latex,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final human and LLM quality-evaluation tables.")
    parser.add_argument("--human-export", required=True)
    parser.add_argument("--human-summary", required=True)
    parser.add_argument("--assignment-manifest", required=True)
    parser.add_argument("--llm-jsonl", action="append", required=True)
    parser.add_argument("--expected-llm-cases", type=int, default=45020)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    aliases = load_annotator_aliases(args.assignment_manifest)
    human_rows, human_annotations, issues = parse_tasks(load_tasks(args.human_export), aliases)
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        raise SystemExit(f"Human export has {len(errors)} validation errors; regenerate the final report first")
    human_summary = json.loads(Path(args.human_summary).read_text(encoding="utf-8"))
    llm_scores, llm_metadata = load_llm_scores(args.llm_jsonl)

    human_table = human_likert_table(
        human_rows, bootstrap_replicates=args.bootstrap_replicates, seed=args.seed
    )
    llm_table = llm_likert_table(llm_scores)
    annotator_table = annotator_summary_table(human_rows)
    alignment, alignment_metadata = alignment_table(human_rows, llm_scores, human_summary)
    llm_metadata["expected_cases"] = args.expected_llm_cases
    llm_metadata["coverage"] = llm_metadata["valid_cases"] / args.expected_llm_cases

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "human_likert": human_table,
        "llm_likert": llm_table,
        "alignment": alignment,
        "annotator_summary": annotator_table,
        "coverage": {
            "human_cases": len({row.case_id for row in human_rows}),
            "human_annotations": len(human_annotations),
            "llm": llm_metadata,
            "alignment": alignment_metadata,
        },
        "provenance": {
            "human_export": _file_provenance(args.human_export),
            "human_summary": _file_provenance(args.human_summary),
            "assignment_manifest": _file_provenance(args.assignment_manifest),
            "llm_jsonl": [_file_provenance(path) for path in args.llm_jsonl],
            "bootstrap_replicates": args.bootstrap_replicates,
            "seed": args.seed,
        },
    }
    (output / "quality_tables.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output / "human_likert.csv", human_table)
    _write_csv(output / "llm_likert.csv", llm_table)
    _write_csv(output / "alignment.csv", alignment)
    _write_csv(output / "annotator_summary.csv", annotator_table)
    latex = render_quality_tables_latex(
        human_table,
        llm_table,
        alignment,
        annotator_table,
        human_cases=payload["coverage"]["human_cases"],
        human_annotations=payload["coverage"]["human_annotations"],
        llm_cases=llm_metadata["valid_cases"],
        llm_expected_cases=args.expected_llm_cases,
        alignment_cases=alignment_metadata["matched_cases"],
    )
    (output / "quality_tables.tex").write_text(latex, encoding="utf-8")
    print(
        json.dumps(
            {
                "human_cases": payload["coverage"]["human_cases"],
                "human_annotations": payload["coverage"]["human_annotations"],
                "llm_valid_cases": llm_metadata["valid_cases"],
                "llm_coverage": llm_metadata["coverage"],
                **alignment_metadata,
                "output_dir": str(output),
            },
            indent=2,
        )
    )


def _file_provenance(path: str | Path) -> dict[str, Any]:
    item = Path(path)
    return {
        "path": str(item),
        "size": item.stat().st_size,
        "sha256": hashlib.sha256(item.read_bytes()).hexdigest(),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


if __name__ == "__main__":
    main()
