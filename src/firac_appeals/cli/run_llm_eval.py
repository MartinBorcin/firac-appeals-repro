#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from firac_appeals.evaluation.downstream import evaluate_predictions
from firac_appeals.evaluation.downstream_llm import DownstreamLLMRunner
from firac_appeals.utils.io import ensure_parent, read_jsonl, read_yaml, write_jsonl


DEFAULT_TASKS = ["outcome_prediction", "issue_generation", "rule_recommendation"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a full downstream zero-shot LLM experiment.")
    parser.add_argument("--task-dir", required=True, help="Directory containing downstream task JSONL files")
    parser.add_argument("--output-dir", required=True, help="Experiment output root")
    parser.add_argument("--config", required=True, help="Downstream LLM config")
    parser.add_argument("--model-slug", required=True, help="Stable slug used in output paths")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS, choices=DEFAULT_TASKS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = read_yaml(args.config)
    if args.overwrite:
        config["overwrite"] = True
    experiment_dir = Path(args.output_dir) / args.model_slug
    experiment_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "model_slug": args.model_slug,
        "model": config.get("backend", {}).get("model", ""),
        "config": args.config,
        "task_dir": args.task_dir,
        "tasks": {},
    }
    for task in args.tasks:
        reference_path = Path(args.task_dir) / f"{task}.jsonl"
        rows = read_jsonl(reference_path)
        if args.limit is not None:
            rows = rows[: args.limit]
        task_dir = experiment_dir / task
        runner = DownstreamLLMRunner(task, dict(config), task_dir)
        results = asyncio.run(runner.run_many(rows))
        ok = sum(1 for row in results if row.get("prediction_status") == "ok")
        prediction_path = task_dir / "predictions.jsonl"
        predictions = read_jsonl(prediction_path) if prediction_path.exists() else []
        evaluation_config = config.get("evaluation", {}) or {}
        semantic_issue_model = evaluation_config.get("semantic_issue_model")
        metrics = evaluate_predictions(
            task,
            rows,
            predictions,
            semantic_model_name=semantic_issue_model if task == "issue_generation" else None,
            semantic_device=evaluation_config.get("semantic_device"),
            semantic_batch_size=int(evaluation_config.get("semantic_batch_size", 32)),
        )
        metrics_path = task_dir / "metrics.json"
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
        write_jsonl(task_dir / "per_case_metrics.jsonl", metrics["per_case"])
        summary["tasks"][task] = {
            "ok": ok,
            "total": len(results),
            "predictions": str(prediction_path),
            "metrics": str(metrics_path),
            "aggregate": metrics["aggregate"],
        }
    summary_path = experiment_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote downstream experiment summary to {summary_path}")


if __name__ == "__main__":
    main()
