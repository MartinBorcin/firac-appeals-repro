#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from firac_appeals.utils.io import read_jsonl


def prediction_map(path: str) -> dict[str, str]:
    output: dict[str, str] = {}
    for row in read_jsonl(path):
        case_id = str(row["case_id"])
        label = row.get("label", row.get("predicted"))
        if label is not None and row.get("prediction_status", "ok") == "ok":
            output[case_id] = str(label)
    return output


def score(gold: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(f1_score(gold, predicted, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(gold, predicted)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired case-bootstrap outcome comparisons.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--system", action="append", required=True, help="NAME=PREDICTIONS_JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260604)
    args = parser.parse_args()

    systems = dict(item.split("=", 1) for item in args.system)
    references = {
        str(row["case_id"]): str(row["target"]["label"])
        for row in read_jsonl(args.reference)
    }
    predictions = {name: prediction_map(path) for name, path in systems.items()}
    common = sorted(set(references).intersection(*(set(rows) for rows in predictions.values())))
    if not common:
        raise SystemExit("No case IDs are shared by the reference and all systems")

    gold = np.asarray([references[case_id] for case_id in common])
    arrays = {
        name: np.asarray([rows[case_id] for case_id in common])
        for name, rows in predictions.items()
    }
    rng = np.random.default_rng(args.seed)
    bootstrap = {name: {metric: [] for metric in ("accuracy", "macro_f1", "mcc")} for name in systems}
    for _ in range(args.replicates):
        indices = rng.integers(0, len(common), size=len(common))
        sampled_gold = gold[indices]
        for name, values in arrays.items():
            for metric, value in score(sampled_gold, values[indices]).items():
                bootstrap[name][metric].append(value)

    payload: dict[str, object] = {
        "reference": args.reference,
        "n_common": len(common),
        "replicates": args.replicates,
        "seed": args.seed,
        "systems": {},
        "pairwise": [],
    }
    for name, values in arrays.items():
        point = score(gold, values)
        payload["systems"][name] = {
            metric: {
                "point": point[metric],
                "ci_95": [float(x) for x in np.quantile(bootstrap[name][metric], [0.025, 0.975])],
            }
            for metric in point
        }

    names = list(systems)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            comparison = {"left": left, "right": right, "metrics": {}}
            for metric in ("accuracy", "macro_f1", "mcc"):
                differences = np.asarray(bootstrap[left][metric]) - np.asarray(bootstrap[right][metric])
                lower_tail = (int(np.count_nonzero(differences <= 0)) + 1) / (args.replicates + 1)
                upper_tail = (int(np.count_nonzero(differences >= 0)) + 1) / (args.replicates + 1)
                p_value = min(1.0, 2 * min(lower_tail, upper_tail))
                comparison["metrics"][metric] = {
                    "difference": score(gold, arrays[left])[metric] - score(gold, arrays[right])[metric],
                    "ci_95": [float(x) for x in np.quantile(differences, [0.025, 0.975])],
                    "p_two_sided": p_value,
                }
            payload["pairwise"].append(comparison)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
