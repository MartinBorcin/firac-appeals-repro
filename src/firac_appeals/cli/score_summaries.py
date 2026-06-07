#!/usr/bin/env python
from __future__ import annotations

import argparse

from firac_appeals.evaluation.summarization import evaluate_rows, write_metrics, write_predictions
from firac_appeals.utils.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions-output", default="")
    parser.add_argument(
        "--summary-variant",
        default="firac_inabs_headnote",
        choices=["firac_inabs_headnote", "firac_inabs_headnote_l_capped"],
    )
    parser.add_argument("--split", default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bertscore", action="store_true")
    parser.add_argument("--bertscore-batch-size", type=int, default=4)
    parser.add_argument("--bertscore-model-type", default="")
    args = parser.parse_args()

    rows = read_jsonl(args.predictions)
    if args.split:
        rows = [row for row in rows if str(row.get("split", "")).lower() == args.split.lower()]
    rows = [row for row in rows if row.get("annotation_status", "ok") == "ok" and row.get("reference_summary")]
    if args.limit is not None:
        rows = rows[: args.limit]
    result = evaluate_rows(
        rows,
        summary_variant=args.summary_variant,
        compute_bertscore=args.bertscore,
        bertscore_batch_size=args.bertscore_batch_size,
        bertscore_model_type=args.bertscore_model_type or None,
    )
    write_metrics(args.output, result)
    if args.predictions_output:
        write_predictions(args.predictions_output, rows, summary_variant=args.summary_variant)
    print(f"Wrote metrics for {len(rows)} cases to {args.output}")


if __name__ == "__main__":
    main()
