#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from firac_appeals.utils.io import read_jsonl, write_csv


SCORE_FIELDS = [
    "facts_faithfulness",
    "facts_coverage",
    "issues_faithfulness",
    "issues_coverage",
    "rules_faithfulness",
    "rules_coverage",
    "analysis_faithfulness",
    "analysis_coverage",
    "conclusion_faithfulness",
    "conclusion_coverage",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Label Studio human FIRAC scores to LLM rubric scores.")
    parser.add_argument("--human-label-studio-json", required=True)
    parser.add_argument("--llm-rubric-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()

    human = load_human_label_studio(Path(args.human_label_studio_json))
    llm = load_llm_scores(Path(args.llm_rubric_jsonl))
    case_ids = sorted(set(human) & set(llm))
    rows = []
    metrics: dict[str, Any] = {"n_cases": len(case_ids), "fields": {}}
    for field in SCORE_FIELDS:
        pairs = [(human[c][field], llm[c][field]) for c in case_ids if field in human[c] and field in llm[c]]
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        metrics["fields"][field] = {
            "n": len(pairs),
            "human_mean": mean(xs),
            "llm_mean": mean(ys),
            "pearson": pearson(xs, ys),
            "spearman": spearman(xs, ys),
            "quadratic_weighted_kappa": weighted_kappa(xs, ys, min_rating=1, max_rating=5),
        }
    for case_id in case_ids:
        row = {"case_id": case_id}
        for field in SCORE_FIELDS:
            row[f"human_{field}"] = human[case_id].get(field, "")
            row[f"llm_{field}"] = llm[case_id].get(field, "")
        rows.append(row)

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_csv:
        fieldnames = ["case_id"] + [f"{side}_{field}" for field in SCORE_FIELDS for side in ("human", "llm")]
        write_csv(args.output_csv, rows, fieldnames)
    print(json.dumps(metrics, indent=2, sort_keys=True))


def load_human_label_studio(path: Path) -> dict[str, dict[str, float]]:
    tasks = json.loads(path.read_text(encoding="utf-8"))
    per_case: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for task in tasks:
        case_id = str(task.get("data", {}).get("case_id", ""))
        if not case_id:
            continue
        annotations = task.get("annotations") or task.get("completions") or []
        for annotation in annotations:
            for result in annotation.get("result", []):
                field = result.get("from_name", "")
                if field not in SCORE_FIELDS:
                    continue
                choices = result.get("value", {}).get("choices", [])
                if not choices:
                    continue
                try:
                    per_case[case_id][field].append(float(choices[0]))
                except ValueError:
                    continue
    return {
        case_id: {field: mean(values) for field, values in fields.items()}
        for case_id, fields in per_case.items()
    }


def load_llm_scores(path: Path) -> dict[str, dict[str, float]]:
    per_case: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in read_jsonl(path):
        case_id = str(row.get("case_id", ""))
        scores = row.get("llm_rubric", {}).get("scores", {})
        for field in SCORE_FIELDS:
            if field in scores:
                per_case[case_id][field].append(float(scores[field]))
    return {
        case_id: {field: mean(values) for field, values in fields.items()}
        for case_id, fields in per_case.items()
    }


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    return pearson(ranks(xs), ranks(ys))


def ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    output = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        rank = (i + j + 2) / 2
        for k in range(i, j + 1):
            output[indexed[k][0]] = rank
        i = j + 1
    return output


def weighted_kappa(xs: list[float], ys: list[float], min_rating: int, max_rating: int) -> float | None:
    if len(xs) < 2:
        return None
    labels = list(range(min_rating, max_rating + 1))
    n = len(labels)
    observed = [[0.0 for _ in labels] for _ in labels]
    for x, y in zip(xs, ys):
        xi = clamp_round(x, min_rating, max_rating) - min_rating
        yi = clamp_round(y, min_rating, max_rating) - min_rating
        observed[xi][yi] += 1
    row_sums = [sum(row) for row in observed]
    col_sums = [sum(observed[i][j] for i in range(n)) for j in range(n)]
    total = sum(row_sums)
    if total == 0:
        return None
    weighted_observed = 0.0
    weighted_expected = 0.0
    for i in range(n):
        for j in range(n):
            weight = ((i - j) ** 2) / ((n - 1) ** 2)
            weighted_observed += weight * observed[i][j] / total
            weighted_expected += weight * (row_sums[i] * col_sums[j] / total) / total
    if weighted_expected == 0:
        return None
    return 1 - weighted_observed / weighted_expected


def clamp_round(value: float, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, int(round(value))))


if __name__ == "__main__":
    main()
