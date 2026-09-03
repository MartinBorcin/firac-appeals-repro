#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from firac_appeals.evaluation.human_rubric import (
    analyze,
    assignment_design_summary,
    load_annotator_aliases,
    load_tasks,
    parse_tasks,
    validate_against_manifest,
    write_report_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and report the round-2 Label Studio human evaluation."
    )
    parser.add_argument("--input", required=True, help="Full Label Studio project export JSON")
    parser.add_argument(
        "--assignment-manifest",
        default=(
            "runs/evaluations/human/label_studio/round2_assignments_20260813/"
            "assignment_manifest.json"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument(
        "--strict-final",
        action="store_true",
        help="Exit non-zero for validation errors or fewer than 30 overlap cases.",
    )
    args = parser.parse_args()

    manifest = Path(args.assignment_manifest) if args.assignment_manifest else None
    aliases = load_annotator_aliases(manifest if manifest and manifest.exists() else None)
    tasks = load_tasks(args.input)
    rows, annotations, issues = parse_tasks(tasks, aliases)
    if manifest and manifest.exists():
        issues.extend(validate_against_manifest(annotations, manifest))
    summary = analyze(
        rows,
        annotations,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    if manifest and manifest.exists():
        summary["assignment_design"] = assignment_design_summary(annotations, manifest)
    input_path = Path(args.input)
    input_payload = json.loads(input_path.read_text(encoding="utf-8"))
    summary["provenance"] = {
        "input_file": str(input_path),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "export_metadata": input_payload.get("export_metadata", {})
        if isinstance(input_payload, dict)
        else {},
        "assignment_manifest": str(manifest) if manifest else "",
        "assignment_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()
        if manifest and manifest.exists()
        else "",
    }
    write_report_artifacts(args.output_dir, rows, annotations, issues, summary)

    severity = Counter(issue.severity for issue in issues)
    coverage = summary["coverage"]
    overall = summary["quality"]["overall"][0]
    alpha = summary["agreement"]["groups"][0]["krippendorff_alpha_ordinal"]
    print(
        json.dumps(
            {
                "annotations": coverage["annotations"],
                "unique_cases": coverage["unique_cases"],
                "overlap_cases": coverage["overlap_cases"],
                "case_level_mean": overall["mean"],
                "ordinal_alpha": alpha,
                "validation": dict(severity),
                "output_dir": args.output_dir,
            },
            indent=2,
        )
    )
    design = summary.get("assignment_design", {})
    final_incomplete = design and not (design.get("core_complete") and design.get("overlap_complete"))
    if args.strict_final and (severity["error"] or coverage["overlap_cases"] < 30 or final_incomplete):
        raise SystemExit("Final-report validation failed; inspect validation.json and coverage.")


if __name__ == "__main__":
    main()
