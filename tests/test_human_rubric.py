from __future__ import annotations

import json

import pytest

from firac_appeals.evaluation.human_rubric import (
    SCORE_FIELDS,
    analyze,
    assignment_design_summary,
    krippendorff_alpha_ordinal,
    parse_tasks,
    write_report_artifacts,
)


def annotation(annotation_id: int, email: str, scores: list[int]) -> dict:
    result = []
    for field, score in zip(SCORE_FIELDS, scores, strict=True):
        result.append(
            {
                "from_name": field,
                "value": {"choices": [str(score)]},
            }
        )
        if score <= 3:
            result.append(
                {
                    "from_name": f"{field}_reason",
                    "value": {"text": [f"Reason for {field}"]},
                }
            )
    return {
        "id": annotation_id,
        "completed_by": {"id": annotation_id, "email": email},
        "result": result,
    }


def task(task_id: int, case_id: str, *annotations: dict, outcome: str = "allowed") -> dict:
    return {
        "id": task_id,
        "data": {
            "case_id": case_id,
            "source_dataset": "in_abs",
            "outcome_label": outcome,
            "split": "test",
        },
        "annotations": list(annotations),
    }


def test_parse_requires_reason_for_low_score() -> None:
    item = annotation(1, "a@example.com", [5] * 10)
    item["result"][0]["value"]["choices"] = ["3"]
    rows, annotations, issues = parse_tasks([task(1, "c1", item)])

    assert len(rows) == 10
    assert annotations[0]["scores_complete"] is True
    assert [issue.code for issue in issues] == ["missing_low_score_reason"]


def test_case_level_estimate_does_not_overweight_overlap_case() -> None:
    tasks = [
        task(
            1,
            "low_double",
            annotation(1, "a@example.com", [1] * 10),
            annotation(2, "b@example.com", [1] * 10),
        ),
        task(2, "high_single", annotation(3, "a@example.com", [5] * 10)),
    ]
    rows, annotations, issues = parse_tasks(tasks)
    summary = analyze(rows, annotations, bootstrap_replicates=50, seed=7)

    assert not issues
    # Case means are 1 and 5, so the primary estimate is 3. Rating-level pooling
    # would incorrectly produce 2.333 because the overlap case has two raters.
    assert summary["quality"]["overall"][0]["mean"] == pytest.approx(3.0)


def test_ordinal_alpha_extremes() -> None:
    assert krippendorff_alpha_ordinal([[5, 5], [4, 4], [1, 1]]) == pytest.approx(1.0)
    value = krippendorff_alpha_ordinal([[1, 5], [5, 1], [1, 5]])
    assert value is not None and value < 0


def test_report_artifacts_are_complete(tmp_path) -> None:
    tasks = [
        task(
            1,
            "c1",
            annotation(1, "a@example.com", [5, 4, 5, 4, 3, 3, 4, 4, 5, 5]),
            annotation(2, "b@example.com", [5, 4, 5, 4, 4, 3, 4, 4, 5, 4]),
        )
    ]
    rows, annotations, issues = parse_tasks(tasks)
    summary = analyze(rows, annotations, bootstrap_replicates=50, seed=11)
    write_report_artifacts(tmp_path, rows, annotations, issues, summary)

    expected = {
        "summary.json",
        "validation.json",
        "scores_long.csv",
        "annotations.csv",
        "case_scores.csv",
        "quality_table.csv",
        "agreement_table.csv",
        "annotator_severity.csv",
        "low_score_explanations.csv",
        "paper_tables.tex",
        "results_summary.md",
    }
    assert expected == {path.name for path in tmp_path.iterdir()}
    assert json.loads((tmp_path / "summary.json").read_text())["coverage"]["overlap_cases"] == 1


def test_assignment_design_tracks_core_and_overlap_completion(tmp_path) -> None:
    tasks = [
        task(
            1,
            "c1",
            annotation(1, "a@example.com", [5] * 10),
            annotation(2, "b@example.com", [5] * 10),
        )
    ]
    _rows, annotations, _issues = parse_tasks(
        tasks, {"a@example.com": "A1", "b@example.com": "A2"}
    )
    manifest = {
        "annotators": {"A1": "a@example.com", "A2": "b@example.com", "A3": "c@example.com"},
        "assignments": {
            alias: [
                {
                    "task_id": 1,
                    "case_id": "c1",
                    "source_dataset": "in_abs",
                    "outcome": "allowed",
                    "assignment_type": "A1-A2 overlap" if alias != "A3" else "individual",
                }
            ]
            for alias in ("A1", "A2", "A3")
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    design = assignment_design_summary(annotations, path)

    assert design["planned_core_annotations"] == 3
    assert design["completed_core_annotations"] == 2
    assert design["completed_overlap_cases"] == 1
    assert design["core_complete"] is False
