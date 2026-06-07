#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from firac_appeals.evaluation.downstream import evaluate_predictions
from firac_appeals.utils.io import ensure_parent, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Score predictions for FIRAC downstream tasks.")
    parser.add_argument(
        "--task",
        required=True,
        choices=["outcome_prediction", "issue_generation", "rule_recommendation"],
    )
    parser.add_argument("--references", required=True, help="Task JSONL produced by build_downstream_tasks.py")
    parser.add_argument("--predictions", required=True, help="Prediction JSONL with case_id and task output")
    parser.add_argument("--output", required=True, help="Aggregate metrics JSON")
    parser.add_argument("--per-case-output", default="", help="Optional per-case metrics JSONL")
    parser.add_argument(
        "--semantic-issue-model",
        default="",
        help="Optional sentence-transformers model for issue semantic cosine metrics",
    )
    parser.add_argument("--semantic-device", default="", help="Optional device for semantic metric model")
    parser.add_argument("--semantic-batch-size", type=int, default=32)
    args = parser.parse_args()

    references = read_jsonl(args.references)
    predictions = read_jsonl(args.predictions)
    result = evaluate_predictions(
        args.task,
        references,
        predictions,
        semantic_model_name=args.semantic_issue_model or None,
        semantic_device=args.semantic_device or None,
        semantic_batch_size=args.semantic_batch_size,
    )
    aggregate = {
        "task": result["task"],
        "count": result["count"],
        "aggregate": result["aggregate"],
        "references": args.references,
        "predictions": args.predictions,
    }
    ensure_parent(args.output).write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if args.per_case_output:
        write_jsonl(args.per_case_output, result["per_case"])
    print(f"Wrote {args.task} metrics for {result['count']} cases to {args.output}")


if __name__ == "__main__":
    main()
