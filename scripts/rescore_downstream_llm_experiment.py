#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from firac_appeals.evaluation.downstream import evaluate_predictions
from firac_appeals.utils.io import ensure_parent, read_jsonl, write_jsonl


DEFAULT_TASKS = ["outcome_prediction", "issue_generation", "rule_recommendation"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute metrics for a saved downstream LLM run.")
    parser.add_argument("--task-dir", required=True, help="Directory containing downstream task JSONL files")
    parser.add_argument("--experiment-dir", required=True, help="Saved model run directory")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS, choices=DEFAULT_TASKS)
    parser.add_argument(
        "--semantic-issue-model",
        default="",
        help="Optional sentence-transformers model for issue semantic cosine metrics",
    )
    parser.add_argument("--semantic-device", default="", help="Optional device for semantic metric model")
    parser.add_argument("--semantic-batch-size", type=int, default=32)
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    summary_path = experiment_dir / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.setdefault("model_slug", experiment_dir.name)
    summary.setdefault("task_dir", args.task_dir)
    summary.setdefault("tasks", {})

    for task in args.tasks:
        reference_path = Path(args.task_dir) / f"{task}.jsonl"
        prediction_path = experiment_dir / task / "predictions.jsonl"
        references = read_jsonl(reference_path)
        predictions = read_jsonl(prediction_path) if prediction_path.exists() else []
        metrics = evaluate_predictions(
            task,
            references,
            predictions,
            semantic_model_name=args.semantic_issue_model if task == "issue_generation" else None,
            semantic_device=args.semantic_device or None,
            semantic_batch_size=args.semantic_batch_size,
        )

        metrics_path = experiment_dir / task / "metrics.json"
        ensure_parent(metrics_path).write_text(
            json.dumps(
                {
                    "task": metrics["task"],
                    "count": metrics["count"],
                    "aggregate": metrics["aggregate"],
                    "references": str(reference_path),
                    "predictions": str(prediction_path),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        write_jsonl(experiment_dir / task / "per_case_metrics.jsonl", metrics["per_case"])
        summary["tasks"][task] = {
            "ok": sum(1 for row in predictions if row.get("prediction_status") == "ok"),
            "total": len(predictions),
            "predictions": str(prediction_path),
            "metrics": str(metrics_path),
            "aggregate": metrics["aggregate"],
        }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Rescored downstream experiment summary at {summary_path}")


if __name__ == "__main__":
    main()
