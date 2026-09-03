#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from export_label_studio_firac_eval import build_task


OUTCOMES = ("allowed", "dismissed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build balanced FIRAC human-evaluation round-2 tasks.")
    parser.add_argument("--in-abs", required=True, help="Annotated IN-Abs JSONL")
    parser.add_argument("--sc-test", required=True, help="Supreme Court test JSONL")
    parser.add_argument("--sc-train", required=True, help="Supreme Court train JSONL")
    parser.add_argument("--main-output", default="outputs/human_eval/round2_main_tasks.json")
    parser.add_argument(
        "--calibration-output",
        default="outputs/human_eval/round2_calibration_tasks.json",
    )
    parser.add_argument(
        "--selection-manifest",
        default="outputs/human_eval/round2_selection_manifest.json",
    )
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()

    in_abs = load_jsonl(Path(args.in_abs))
    sc_test = load_jsonl(Path(args.sc_test))
    sc_train = load_jsonl(Path(args.sc_train))

    in_abs_main = select_in_abs_main(in_abs, args.seed)
    sc_main = select_sc_main(sc_test, args.seed)
    main_rows = tag_and_interleave(in_abs_main, sc_main, "main")

    main_ids = {case_id(row) for row in main_rows}
    in_abs_calibration = select_calibration(
        [row for row in in_abs if row.get("split") == "train" and case_id(row) not in main_ids],
        {"allowed": 3, "dismissed": 2},
        args.seed + 101,
    )
    recent_sc_train = [
        row
        for row in sc_train
        if decision_year(row) in {"2020", "2021", "2022", "2023"} and case_id(row) not in main_ids
    ]
    sc_calibration = select_calibration(
        recent_sc_train,
        {"allowed": 2, "dismissed": 3},
        args.seed + 202,
    )
    calibration_rows = tag_and_interleave(in_abs_calibration, sc_calibration, "calibration")

    validate_selection(main_rows, calibration_rows)
    write_tasks(Path(args.main_output), main_rows)
    write_tasks(Path(args.calibration_output), calibration_rows)
    write_manifest(Path(args.selection_manifest), args.seed, main_rows, calibration_rows)

    print_summary("main", main_rows)
    print_summary("calibration", calibration_rows)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_in_abs_main(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    test = [row for row in rows if row.get("split") == "test" and row.get("outcome_label") in OUTCOMES]
    selected = list(test)
    counts = Counter(str(row.get("outcome_label")) for row in selected)
    for outcome in OUTCOMES:
        needed = 50 - counts[outcome]
        if needed < 0:
            selected = remove_to_target(selected, outcome, 50, seed)
        elif needed:
            candidates = [
                row for row in rows if row.get("split") == "train" and row.get("outcome_label") == outcome
            ]
            selected.extend(spread_sample(candidates, needed, random.Random(f"{seed}:in_abs:{outcome}")))
    return sorted(selected, key=lambda row: (str(row.get("outcome_label")), case_id(row)))


def select_sc_main(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for outcome in OUTCOMES:
        candidates = [row for row in rows if row.get("outcome_label") == outcome]
        by_year: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in candidates:
            by_year[decision_year(row)].append(row)
        quotas = proportional_quotas({year: len(items) for year, items in by_year.items()}, 50)
        for year, quota in sorted(quotas.items()):
            rng = random.Random(f"{seed}:sc:{outcome}:{year}")
            selected.extend(spread_sample(by_year[year], quota, rng))
    return sorted(selected, key=lambda row: (str(row.get("outcome_label")), decision_year(row), case_id(row)))


def select_calibration(
    rows: list[dict[str, Any]], quotas: dict[str, int], seed: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for outcome, quota in quotas.items():
        candidates = [row for row in rows if row.get("outcome_label") == outcome]
        selected.extend(spread_sample(candidates, quota, random.Random(f"{seed}:{outcome}")))
    return sorted(selected, key=lambda row: (str(row.get("outcome_label")), case_id(row)))


def proportional_quotas(counts: dict[str, int], total: int) -> dict[str, int]:
    population = sum(counts.values())
    if population < total:
        raise ValueError(f"Cannot select {total} rows from population of {population}")
    raw = {key: total * count / population for key, count in counts.items()}
    quotas = {key: int(value) for key, value in raw.items()}
    remainder = total - sum(quotas.values())
    order = sorted(raw, key=lambda key: (raw[key] - quotas[key], counts[key], key), reverse=True)
    for key in order[:remainder]:
        quotas[key] += 1
    return quotas


def spread_sample(
    rows: list[dict[str, Any]], count: int, rng: random.Random
) -> list[dict[str, Any]]:
    if count > len(rows):
        raise ValueError(f"Cannot select {count} rows from {len(rows)} candidates")
    if count == 0:
        return []
    ordered = sorted(rows, key=lambda row: (judgment_words(row), case_id(row)))
    chosen = []
    for index in range(count):
        start = index * len(ordered) // count
        end = (index + 1) * len(ordered) // count
        chosen.append(ordered[rng.randrange(start, end)])
    return chosen


def remove_to_target(
    rows: list[dict[str, Any]], outcome: str, target: int, seed: int
) -> list[dict[str, Any]]:
    matching = [row for row in rows if row.get("outcome_label") == outcome]
    keep = {case_id(row) for row in spread_sample(matching, target, random.Random(f"{seed}:trim:{outcome}"))}
    return [row for row in rows if row.get("outcome_label") != outcome or case_id(row) in keep]


def tag_and_interleave(
    left: list[dict[str, Any]], right: list[dict[str, Any]], role: str
) -> list[dict[str, Any]]:
    tagged_left = [{**row, "human_eval_sample_role": role} for row in left]
    tagged_right = [{**row, "human_eval_sample_role": role} for row in right]
    output: list[dict[str, Any]] = []
    for index in range(max(len(tagged_left), len(tagged_right))):
        if index < len(tagged_left):
            output.append(tagged_left[index])
        if index < len(tagged_right):
            output.append(tagged_right[index])
    return output


def validate_selection(main_rows: list[dict[str, Any]], calibration_rows: list[dict[str, Any]]) -> None:
    if len(main_rows) != 200:
        raise ValueError(f"Expected 200 main rows, got {len(main_rows)}")
    if len(calibration_rows) != 10:
        raise ValueError(f"Expected 10 calibration rows, got {len(calibration_rows)}")
    main_ids = [case_id(row) for row in main_rows]
    calibration_ids = [case_id(row) for row in calibration_rows]
    if len(main_ids) != len(set(main_ids)):
        raise ValueError("Duplicate case IDs in main selection")
    if len(calibration_ids) != len(set(calibration_ids)):
        raise ValueError("Duplicate case IDs in calibration selection")
    if set(main_ids) & set(calibration_ids):
        raise ValueError("Calibration cases overlap the main selection")
    for source in ("in_abs", "supreme_court_vanga"):
        source_rows = [row for row in main_rows if row.get("source_dataset") == source]
        if len(source_rows) != 100:
            raise ValueError(f"Expected 100 {source} main rows, got {len(source_rows)}")
        counts = Counter(row.get("outcome_label") for row in source_rows)
        if counts != Counter({"allowed": 50, "dismissed": 50}):
            raise ValueError(f"Unexpected {source} outcome counts: {counts}")


def write_tasks(path: Path, rows: list[dict[str, Any]]) -> None:
    tasks = [build_task(index + 1, row, False) for index, row in enumerate(rows)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_manifest(
    path: Path, seed: int, main_rows: list[dict[str, Any]], calibration_rows: list[dict[str, Any]]
) -> None:
    def compact(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "case_id": case_id(row),
            "source_dataset": row.get("source_dataset", ""),
            "split": row.get("split", ""),
            "outcome_label": row.get("outcome_label", ""),
            "decision_year": decision_year(row),
            "judgment_words": judgment_words(row),
        }

    payload = {
        "seed": seed,
        "policy": {
            "main": "100 per corpus; 50 allowed and 50 dismissed; test-first; SC stratified by year and judgment length",
            "calibration": "5 non-evaluation cases per corpus; allowed/dismissed only; no overlap with main",
        },
        "main": [compact(row) for row in main_rows],
        "calibration": [compact(row) for row in calibration_rows],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def print_summary(label: str, rows: list[dict[str, Any]]) -> None:
    counts = Counter(
        (str(row.get("source_dataset")), str(row.get("split")), str(row.get("outcome_label")))
        for row in rows
    )
    print(label, len(rows))
    for key, count in sorted(counts.items()):
        print("  ", key, count)


def case_id(row: dict[str, Any]) -> str:
    return str(row.get("case_id", ""))


def decision_year(row: dict[str, Any]) -> str:
    value = str((row.get("metadata") or {}).get("decision_date", ""))
    return value[-4:] if len(value) >= 4 else "unknown"


def judgment_words(row: dict[str, Any]) -> int:
    return len(str(row.get("judgment_text", "")).split())


if __name__ == "__main__":
    main()
