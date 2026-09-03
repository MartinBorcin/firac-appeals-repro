from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


SECTIONS = ("facts", "issues", "rules", "analysis", "conclusion")
DIMENSIONS = ("faithfulness", "coverage")
SCORE_FIELDS = tuple(f"{section}_{dimension}" for section in SECTIONS for dimension in DIMENSIONS)
VALID_SCORES = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class ScoreRow:
    task_id: int
    case_id: str
    source_dataset: str
    outcome: str
    split: str
    annotator: str
    annotator_email: str
    annotation_id: int | str
    field: str
    section: str
    dimension: str
    score: int
    reason: str = ""


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str


def load_tasks(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("tasks") or payload.get("results") or []
    if not isinstance(payload, list):
        raise ValueError("Label Studio export must be a task list or an object containing 'tasks'")
    return payload


def load_annotator_aliases(manifest_path: str | Path | None) -> dict[str, str]:
    if not manifest_path:
        return {}
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return {email.lower(): alias for alias, email in payload.get("annotators", {}).items()}


def parse_tasks(
    tasks: Sequence[dict[str, Any]], annotator_aliases: dict[str, str] | None = None
) -> tuple[list[ScoreRow], list[dict[str, Any]], list[ValidationIssue]]:
    aliases = {key.lower(): value for key, value in (annotator_aliases or {}).items()}
    rows: list[ScoreRow] = []
    annotations: list[dict[str, Any]] = []
    issues: list[ValidationIssue] = []
    seen_annotation_keys: set[tuple[int, str]] = set()

    for task in tasks:
        data = task.get("data") or {}
        task_id = _int_or_zero(task.get("id"))
        case_id = str(data.get("case_id") or "")
        if not task_id or not case_id:
            issues.append(ValidationIssue("error", "missing_task_identity", f"Task {task_id} lacks an ID or case_id"))
            continue
        active = [
            annotation
            for annotation in (task.get("annotations") or task.get("completions") or [])
            if not annotation.get("was_cancelled")
        ]
        task_annotators: list[str] = []
        for annotation in active:
            annotation_id: int | str = annotation.get("id", "")
            email = _annotator_email(annotation)
            alias = aliases.get(email.lower(), email or f"user_{_annotator_id(annotation)}")
            task_annotators.append(alias)
            duplicate_key = (task_id, alias)
            if duplicate_key in seen_annotation_keys:
                issues.append(
                    ValidationIssue(
                        "error", "duplicate_annotator_task", f"{alias} annotated task {task_id} more than once"
                    )
                )
            seen_annotation_keys.add(duplicate_key)

            results = annotation.get("result") or []
            by_name = {str(result.get("from_name") or ""): result.get("value") or {} for result in results}
            scores: dict[str, int] = {}
            reasons: dict[str, str] = {}
            annotation_rows: list[ScoreRow] = []
            for field in SCORE_FIELDS:
                choices = by_name.get(field, {}).get("choices") or []
                if not choices:
                    issues.append(
                        ValidationIssue("error", "missing_score", f"Annotation {annotation_id} lacks {field}")
                    )
                    continue
                try:
                    score = int(choices[0])
                except (TypeError, ValueError):
                    issues.append(
                        ValidationIssue("error", "invalid_score", f"Annotation {annotation_id} has invalid {field}")
                    )
                    continue
                if score not in VALID_SCORES:
                    issues.append(
                        ValidationIssue(
                            "error", "score_out_of_range", f"Annotation {annotation_id}: {field}={score}"
                        )
                    )
                    continue
                reason = _text_value(by_name.get(f"{field}_reason", {}))
                scores[field] = score
                reasons[field] = reason
                if score <= 3 and not reason:
                    issues.append(
                        ValidationIssue(
                            "error",
                            "missing_low_score_reason",
                            f"Annotation {annotation_id} has {field}={score} without a reason",
                        )
                    )
                annotation_rows.append(
                    ScoreRow(
                        task_id=task_id,
                        case_id=case_id,
                        source_dataset=str(data.get("source_dataset") or ""),
                        outcome=str(data.get("outcome_label") or ""),
                        split=str(data.get("split") or ""),
                        annotator=alias,
                        annotator_email=email,
                        annotation_id=annotation_id,
                        field=field,
                        section=field.rsplit("_", 1)[0],
                        dimension=field.rsplit("_", 1)[1],
                        score=score,
                        reason=reason,
                    )
                )
            scores_complete = len(scores) == len(SCORE_FIELDS)
            if scores_complete:
                rows.extend(annotation_rows)
            annotations.append(
                {
                    "task_id": task_id,
                    "case_id": case_id,
                    "source_dataset": str(data.get("source_dataset") or ""),
                    "outcome": str(data.get("outcome_label") or ""),
                    "split": str(data.get("split") or ""),
                    "annotator": alias,
                    "annotator_email": email,
                    "annotation_id": annotation_id,
                    "scores_complete": scores_complete,
                    "reviewer_notes": _text_value(by_name.get("reviewer_notes", {})),
                }
            )
        if len(active) > 2:
            issues.append(
                ValidationIssue("error", "too_many_annotations", f"Task {task_id} has {len(active)} annotations")
            )
        if len(task_annotators) != len(set(task_annotators)):
            issues.append(
                ValidationIssue("error", "duplicate_task_annotator", f"Task {task_id} repeats an annotator")
            )
    return rows, annotations, issues


def validate_against_manifest(
    annotations: Sequence[dict[str, Any]], manifest_path: str | Path
) -> list[ValidationIssue]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    allowed = {
        (alias, int(item["task_id"]))
        for alias, items in manifest.get("assignments", {}).items()
        for item in items
    }
    issues: list[ValidationIssue] = []
    for annotation in annotations:
        key = (str(annotation["annotator"]), int(annotation["task_id"]))
        if key not in allowed:
            issues.append(
                ValidationIssue(
                    "error", "outside_assignment", f"{key[0]} annotated unassigned task {key[1]}"
                )
            )
    return issues


def assignment_design_summary(
    annotations: Sequence[dict[str, Any]], manifest_path: str | Path
) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    assignments = manifest.get("assignments", {})
    core_aliases = [alias for alias in ("A1", "A2", "A3") if alias in assignments]
    expected: dict[tuple[str, int], dict[str, Any]] = {}
    for alias in core_aliases:
        for item in assignments[alias]:
            expected[(alias, int(item["task_id"]))] = item
    completed = {
        (str(item["annotator"]), int(item["task_id"]))
        for item in annotations
        if item.get("scores_complete")
    }
    missing = sorted(set(expected) - completed)
    planned_overlap_tasks = sorted(
        {
            task_id
            for (_alias, task_id), item in expected.items()
            if "overlap" in str(item.get("assignment_type", ""))
        }
    )
    completed_by_task = Counter(task_id for alias, task_id in completed if alias in core_aliases)
    complete_overlap_tasks = [task_id for task_id in planned_overlap_tasks if completed_by_task[task_id] >= 2]
    planned_cases: dict[int, dict[str, Any]] = {}
    for (_alias, task_id), item in expected.items():
        planned_cases[task_id] = item
    completed_core_tasks = {task_id for alias, task_id in completed if alias in core_aliases}
    completed_strata = Counter(
        (item.get("source_dataset", ""), item.get("outcome", ""))
        for task_id, item in planned_cases.items()
        if task_id in completed_core_tasks
    )
    planned_strata = Counter(
        (item.get("source_dataset", ""), item.get("outcome", "")) for item in planned_cases.values()
    )
    return {
        "core_annotators": core_aliases,
        "planned_core_annotations": len(expected),
        "completed_core_annotations": len(set(expected) & completed),
        "planned_core_cases": len(planned_cases),
        "covered_core_cases": len(set(planned_cases) & completed_core_tasks),
        "planned_overlap_cases": len(planned_overlap_tasks),
        "completed_overlap_cases": len(complete_overlap_tasks),
        "missing_core_assignments": [
            {"annotator": alias, "task_id": task_id, "case_id": expected[(alias, task_id)].get("case_id", "")}
            for alias, task_id in missing
        ],
        "planned_cases_by_stratum": _stratum_counter_rows(planned_strata),
        "covered_cases_by_stratum": _stratum_counter_rows(completed_strata),
        "core_complete": not missing,
        "overlap_complete": len(complete_overlap_tasks) == len(planned_overlap_tasks),
    }


def analyze(
    rows: Sequence[ScoreRow],
    annotations: Sequence[dict[str, Any]],
    *,
    bootstrap_replicates: int = 5000,
    seed: int = 20260817,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("No valid score rows found")
    cases = sorted({row.case_id for row in rows})
    overlap_cases = sorted(
        case_id
        for case_id in cases
        if len({row.annotator for row in rows if row.case_id == case_id}) >= 2
    )
    result: dict[str, Any] = {
        "analysis": {
            "unit_of_analysis": "case (annotators averaged within case)",
            "confidence_interval": "case-clustered percentile bootstrap",
            "bootstrap_replicates": bootstrap_replicates,
            "seed": seed,
            "score_scale": [1, 5],
        },
        "coverage": coverage_summary(rows, annotations),
        "quality": quality_summary(rows, bootstrap_replicates, seed),
        "agreement": agreement_summary(rows, overlap_cases, bootstrap_replicates, seed + 1),
        "annotator_severity": annotator_severity(rows, overlap_cases, bootstrap_replicates, seed + 2),
        "explanations": explanation_summary(rows, annotations),
    }
    return result


def coverage_summary(
    rows: Sequence[ScoreRow], annotations: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        if annotation["scores_complete"]:
            by_case[str(annotation["case_id"])].append(annotation)
    strata = Counter(
        (values[0]["source_dataset"], values[0]["outcome"])
        for values in by_case.values()
        if values
    )
    multiplicity = Counter(len(values) for values in by_case.values())
    return {
        "annotations": sum(len(values) for values in by_case.values()),
        "unique_cases": len(by_case),
        "overlap_cases": sum(count for multiplicity_value, count in multiplicity.items() if multiplicity_value >= 2),
        "case_annotation_multiplicity": {str(key): value for key, value in sorted(multiplicity.items())},
        "annotations_by_annotator": dict(sorted(Counter(a["annotator"] for a in annotations).items())),
        "cases_by_stratum": [
            {"source_dataset": source, "outcome": outcome, "n_cases": count}
            for (source, outcome), count in sorted(strata.items())
        ],
    }


def quality_summary(
    rows: Sequence[ScoreRow], bootstrap_replicates: int, seed: int
) -> dict[str, Any]:
    group_specs: list[tuple[str, Callable[[ScoreRow], str]]] = [
        ("overall", lambda _row: "overall"),
        ("section", lambda row: row.section),
        ("dimension", lambda row: row.dimension),
        ("criterion", lambda row: row.field),
        ("source_dataset", lambda row: row.source_dataset),
        ("outcome", lambda row: row.outcome),
        ("split", lambda row: row.split),
        ("stratum", lambda row: f"{row.source_dataset}|{row.outcome}"),
        ("annotator", lambda row: row.annotator),
    ]
    output: dict[str, list[dict[str, Any]]] = {}
    for offset, (name, grouper) in enumerate(group_specs):
        keys = sorted({grouper(row) for row in rows})
        output[name] = [
            _quality_group(
                [row for row in rows if grouper(row) == key],
                key,
                bootstrap_replicates,
                seed + offset * 1000 + index,
            )
            for index, key in enumerate(keys)
        ]
    output["balanced_strata"] = [
        _balanced_strata_estimate(rows, bootstrap_replicates, seed + 9000)
    ]
    return output


def _quality_group(
    rows: Sequence[ScoreRow], label: str, bootstrap_replicates: int, seed: int
) -> dict[str, Any]:
    case_values = _case_means(rows)
    annotation_scores = [row.score for row in rows]
    estimate = statistics.mean(case_values.values())
    low_count = sum(score <= 3 for score in annotation_scores)
    return {
        "group": label,
        "n_cases": len(case_values),
        "n_ratings": len(annotation_scores),
        "mean": estimate,
        "ci95": _cluster_bootstrap_mean(rows, bootstrap_replicates, seed),
        "sd_between_cases": statistics.stdev(case_values.values()) if len(case_values) > 1 else None,
        "median_case_score": statistics.median(case_values.values()),
        "score_distribution": {str(score): annotation_scores.count(score) for score in VALID_SCORES},
        "ceiling_rate": annotation_scores.count(5) / len(annotation_scores),
        "low_score_rate": low_count / len(annotation_scores),
    }


def _balanced_strata_estimate(
    rows: Sequence[ScoreRow], bootstrap_replicates: int, seed: int
) -> dict[str, Any]:
    strata: dict[tuple[str, str], list[ScoreRow]] = defaultdict(list)
    for row in rows:
        strata[(row.source_dataset, row.outcome)].append(row)
    observed = statistics.mean(statistics.mean(_case_means(values).values()) for values in strata.values())
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(bootstrap_replicates):
        means: list[float] = []
        for values in strata.values():
            per_case_rows = _rows_by_case(values)
            ids = list(per_case_rows)
            sampled = [rng.choice(ids) for _ in ids]
            means.append(statistics.mean(_mean_scores(per_case_rows[case_id]) for case_id in sampled))
        samples.append(statistics.mean(means))
    return {
        "group": "equal_weight_four_source_outcome_strata",
        "n_cases": len({row.case_id for row in rows}),
        "n_strata": len(strata),
        "mean": observed,
        "ci95": percentile_interval(samples),
    }


def agreement_summary(
    rows: Sequence[ScoreRow],
    overlap_cases: Sequence[str],
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    overlap = [row for row in rows if row.case_id in overlap_cases]
    groups: list[tuple[str, list[ScoreRow]]] = [("overall", overlap)]
    groups.extend((section, [row for row in overlap if row.section == section]) for section in SECTIONS)
    groups.extend((field, [row for row in overlap if row.field == field]) for field in SCORE_FIELDS)
    estimates = [
        _agreement_group(group_rows, label, bootstrap_replicates, seed + index)
        for index, (label, group_rows) in enumerate(groups)
    ]
    pairwise: list[dict[str, Any]] = []
    pair_names = sorted(
        {
            tuple(sorted(annotators))
            for annotators in _annotators_by_case(overlap).values()
            if len(annotators) == 2
        }
    )
    for index, pair in enumerate(pair_names):
        pair_cases = [
            case_id
            for case_id, annotators in _annotators_by_case(overlap).items()
            if tuple(sorted(annotators)) == pair
        ]
        pair_rows = [row for row in overlap if row.case_id in pair_cases]
        item = _agreement_group(pair_rows, "-".join(pair), bootstrap_replicates, seed + 100 + index)
        item["annotator_pair"] = list(pair)
        pairwise.append(item)
    return {
        "notes": [
            "Reliability is computed only on independently double-annotated cases.",
            "Krippendorff alpha uses the ordinal distance derived from pooled category frequencies.",
            "Intervals resample cases, preserving the ten correlated criterion ratings within each case.",
            "Raw agreement is descriptive and can be inflated by the score-5 ceiling effect.",
        ],
        "groups": estimates,
        "annotator_pairs": pairwise,
    }


def _agreement_group(
    rows: Sequence[ScoreRow], label: str, bootstrap_replicates: int, seed: int
) -> dict[str, Any]:
    units = _reliability_units(rows)
    pairs = [pair for ratings in units.values() for pair in _unordered_pairs(ratings)]
    differences = [abs(left - right) for left, right in pairs]
    alpha = krippendorff_alpha_ordinal(units.values())
    alpha_samples = _bootstrap_reliability(rows, bootstrap_replicates, seed)
    return {
        "group": label,
        "n_cases": len({case_id for case_id, _field in units}),
        "n_units": len(units),
        "n_pairs": len(pairs),
        "exact_agreement": _safe_ratio(sum(value == 0 for value in differences), len(differences)),
        "within_one_agreement": _safe_ratio(sum(value <= 1 for value in differences), len(differences)),
        "mean_absolute_difference": statistics.mean(differences) if differences else None,
        "krippendorff_alpha_ordinal": alpha,
        "krippendorff_alpha_ci95": percentile_interval(alpha_samples),
    }


def krippendorff_alpha_ordinal(units: Iterable[Sequence[int]]) -> float | None:
    ratings_by_unit = [list(values) for values in units if len(values) >= 2]
    all_values = [value for values in ratings_by_unit for value in values]
    if len(all_values) < 2:
        return None
    marginals = Counter(all_values)

    def distance(left: int, right: int) -> float:
        low, high = sorted((left, right))
        if low == high:
            return 0.0
        mass = sum(marginals[value] for value in range(low, high + 1))
        mass -= (marginals[low] + marginals[high]) / 2
        return mass * mass

    observed_numerator = 0.0
    observed_denominator = 0.0
    for values in ratings_by_unit:
        weight = 1 / (len(values) - 1)
        for left_index, left in enumerate(values):
            for right_index, right in enumerate(values):
                if left_index != right_index:
                    observed_numerator += weight * distance(left, right)
                    observed_denominator += weight
    observed = observed_numerator / observed_denominator
    expected_numerator = 0.0
    expected_denominator = len(all_values) * (len(all_values) - 1)
    for left, left_count in marginals.items():
        for right, right_count in marginals.items():
            count = left_count * right_count - (left_count if left == right else 0)
            expected_numerator += count * distance(left, right)
    expected = expected_numerator / expected_denominator
    if expected == 0:
        return None
    return 1 - observed / expected


def annotator_severity(
    rows: Sequence[ScoreRow],
    overlap_cases: Sequence[str],
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    overlap = [row for row in rows if row.case_id in overlap_cases]
    annotators_by_case = _annotators_by_case(overlap)
    pairs = sorted(
        {tuple(sorted(values)) for values in annotators_by_case.values() if len(values) == 2}
    )
    output: list[dict[str, Any]] = []
    for index, (first, second) in enumerate(pairs):
        case_differences: dict[str, float] = {}
        for case_id, annotators in annotators_by_case.items():
            if tuple(sorted(annotators)) != (first, second):
                continue
            first_scores = [row.score for row in overlap if row.case_id == case_id and row.annotator == first]
            second_scores = [row.score for row in overlap if row.case_id == case_id and row.annotator == second]
            if first_scores and second_scores:
                case_differences[case_id] = statistics.mean(first_scores) - statistics.mean(second_scores)
        values = list(case_differences.values())
        rng = random.Random(seed + index)
        samples = [
            statistics.mean(rng.choice(values) for _ in values)
            for _ in range(bootstrap_replicates)
        ] if values else []
        output.append(
            {
                "first_annotator": first,
                "second_annotator": second,
                "n_cases": len(values),
                "mean_paired_difference_first_minus_second": statistics.mean(values) if values else None,
                "ci95": percentile_interval(samples),
            }
        )
    return {
        "notes": "Paired differences use only shared cases and are preferable to confounded marginal annotator means.",
        "pairs": output,
    }


def explanation_summary(
    rows: Sequence[ScoreRow], annotations: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    low = [row for row in rows if row.score <= 3]
    return {
        "low_scores": len(low),
        "low_scores_with_reason": sum(bool(row.reason.strip()) for row in low),
        "reason_completion_rate": _safe_ratio(sum(bool(row.reason.strip()) for row in low), len(low)),
        "annotations_with_additional_notes": sum(bool(a["reviewer_notes"].strip()) for a in annotations),
        "low_scores_by_criterion": dict(sorted(Counter(row.field for row in low).items())),
    }


def write_report_artifacts(
    output_dir: str | Path,
    rows: Sequence[ScoreRow],
    annotations: Sequence[dict[str, Any]],
    issues: Sequence[ValidationIssue],
    summary: dict[str, Any],
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "validation.json").write_text(
        json.dumps([issue.__dict__ for issue in issues], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "scores_long.csv", [_without_email(row.__dict__) for row in rows])
    _write_csv(output / "annotations.csv", [_without_email(row) for row in annotations])
    _write_csv(output / "case_scores.csv", _case_score_rows(rows))
    _write_csv(output / "quality_table.csv", _flatten_quality(summary["quality"]))
    _write_csv(
        output / "agreement_table.csv",
        summary["agreement"]["groups"] + summary["agreement"]["annotator_pairs"],
    )
    _write_csv(output / "annotator_severity.csv", summary["annotator_severity"]["pairs"])
    _write_csv(
        output / "low_score_explanations.csv",
        [_without_email(row.__dict__) for row in rows if row.score <= 3],
    )
    (output / "paper_tables.tex").write_text(render_latex(summary), encoding="utf-8")
    (output / "results_summary.md").write_text(render_markdown(summary, issues), encoding="utf-8")


def render_markdown(summary: dict[str, Any], issues: Sequence[ValidationIssue]) -> str:
    coverage = summary["coverage"]
    overall = summary["quality"]["overall"][0]
    agreement = summary["agreement"]["groups"][0]
    error_count = sum(issue.severity == "error" for issue in issues)
    design = summary.get("assignment_design", {})
    design_text = ""
    if design:
        design_text = (
            f"Planned core design progress: {design['completed_core_annotations']}/"
            f"{design['planned_core_annotations']} annotations and {design['completed_overlap_cases']}/"
            f"{design['planned_overlap_cases']} double-annotated overlap cases.\n\n"
        )
    return (
        "# Round-2 human evaluation\n\n"
        f"Validated complete annotations: {coverage['annotations']} across "
        f"{coverage['unique_cases']} cases; {coverage['overlap_cases']} cases are double-annotated.\n\n"
        + design_text
        + f"The case-level mean rubric score was {overall['mean']:.3f} "
        f"(95% case-clustered bootstrap CI {_format_ci(overall['ci95'])}). "
        f"The ceiling rate was {overall['ceiling_rate']:.1%}.\n\n"
        f"Across double-annotated case–criterion units, exact agreement was "
        f"{agreement['exact_agreement']:.1%}, agreement within one point was "
        f"{agreement['within_one_agreement']:.1%}, and ordinal Krippendorff's alpha was "
        f"{_format_number(agreement['krippendorff_alpha_ordinal'])} "
        f"(95% case-clustered bootstrap CI {_format_ci(agreement['krippendorff_alpha_ci95'])}).\n\n"
        f"Validation errors: {error_count}. This text is generated from `summary.json`; "
        "retain the tables and confidence intervals for the final manuscript.\n"
    )


def render_latex(summary: dict[str, Any]) -> str:
    quality_rows = summary["quality"]["section"]
    agreement_rows = {row["group"]: row for row in summary["agreement"]["groups"]}
    lines = [
        "% Generated by scripts/report_human_eval_round2.py; do not edit manually.",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Section & Cases & Mean & 95\\% CI & $\\leq 3$ (\\%) \\\\",
        "\\midrule",
    ]
    for row in quality_rows:
        label = row["group"].title().replace("Rules", "Rules/Authorities")
        lines.append(
            f"{label} & {row['n_cases']} & {row['mean']:.2f} & {_format_ci(row['ci95'])} & "
            f"{100 * row['low_score_rate']:.1f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "", "\\begin{tabular}{lrrrrr}", "\\toprule"])
    lines.append(
        "Scope & Cases & Exact (\\%) & Within 1 (\\%) & Ordinal $\\alpha$ & 95\\% CI " + "\\\\"
    )
    lines.append("\\midrule")
    for label in ("overall", *SECTIONS):
        row = agreement_rows[label]
        lines.append(
            f"{label.title()} & {row['n_cases']} & {100 * row['exact_agreement']:.1f} & "
            f"{100 * row['within_one_agreement']:.1f} & "
            f"{_format_number(row['krippendorff_alpha_ordinal'])} & "
            f"{_format_ci(row['krippendorff_alpha_ci95'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def percentile_interval(values: Sequence[float | None], confidence: float = 0.95) -> list[float] | None:
    clean = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not clean:
        return None
    alpha = (1 - confidence) / 2
    return [_quantile(clean, alpha), _quantile(clean, 1 - alpha)]


def _cluster_bootstrap_mean(
    rows: Sequence[ScoreRow], bootstrap_replicates: int, seed: int
) -> list[float] | None:
    by_case = _rows_by_case(rows)
    case_ids = list(by_case)
    if not case_ids:
        return None
    rng = random.Random(seed)
    samples = [
        statistics.mean(_mean_scores(by_case[rng.choice(case_ids)]) for _ in case_ids)
        for _ in range(bootstrap_replicates)
    ]
    return percentile_interval(samples)


def _bootstrap_reliability(
    rows: Sequence[ScoreRow], bootstrap_replicates: int, seed: int
) -> list[float]:
    by_case = _rows_by_case(rows)
    case_ids = list(by_case)
    if not case_ids:
        return []
    rng = random.Random(seed)
    output: list[float] = []
    for _ in range(bootstrap_replicates):
        units: dict[tuple[int, str], list[int]] = defaultdict(list)
        for replicate_index in range(len(case_ids)):
            sampled_id = rng.choice(case_ids)
            for row in by_case[sampled_id]:
                units[(replicate_index, row.field)].append(row.score)
        value = krippendorff_alpha_ordinal(units.values())
        if value is not None:
            output.append(value)
    return output


def _reliability_units(rows: Sequence[ScoreRow]) -> dict[tuple[str, str], list[int]]:
    units: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        units[(row.case_id, row.field)].append(row.score)
    return {key: values for key, values in units.items() if len(values) >= 2}


def _case_means(rows: Sequence[ScoreRow]) -> dict[str, float]:
    return {case_id: _mean_scores(values) for case_id, values in _rows_by_case(rows).items()}


def _rows_by_case(rows: Sequence[ScoreRow]) -> dict[str, list[ScoreRow]]:
    output: dict[str, list[ScoreRow]] = defaultdict(list)
    for row in rows:
        output[row.case_id].append(row)
    return output


def _mean_scores(rows: Sequence[ScoreRow]) -> float:
    return statistics.mean(row.score for row in rows)


def _annotators_by_case(rows: Sequence[ScoreRow]) -> dict[str, set[str]]:
    output: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        output[row.case_id].add(row.annotator)
    return output


def _unordered_pairs(values: Sequence[int]) -> list[tuple[int, int]]:
    return [(values[left], values[right]) for left in range(len(values)) for right in range(left + 1, len(values))]


def _case_score_rows(rows: Sequence[ScoreRow]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[ScoreRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.case_id, row.field)].append(row)
    for (case_id, field), values in sorted(grouped.items()):
        first = values[0]
        output.append(
            {
                "case_id": case_id,
                "task_id": first.task_id,
                "source_dataset": first.source_dataset,
                "outcome": first.outcome,
                "field": field,
                "n_annotators": len({row.annotator for row in values}),
                "case_mean": statistics.mean(row.score for row in values),
            }
        )
    return output


def _flatten_quality(quality: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scope, rows in quality.items():
        for row in rows:
            output.append({"scope": scope, **row})
    return output


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
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
                {key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()}
            )


def _without_email(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "annotator_email"}


def _annotator_email(annotation: dict[str, Any]) -> str:
    completed_by = annotation.get("completed_by")
    if isinstance(completed_by, dict):
        return str(completed_by.get("email") or "")
    return str(annotation.get("created_username") or annotation.get("updated_username") or "")


def _annotator_id(annotation: dict[str, Any]) -> int | str:
    completed_by = annotation.get("completed_by")
    return completed_by.get("id", "unknown") if isinstance(completed_by, dict) else completed_by or "unknown"


def _text_value(value: dict[str, Any]) -> str:
    text = value.get("text") or []
    return " ".join(str(item) for item in text).strip()


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _stratum_counter_rows(counter: Counter[tuple[str, str]]) -> list[dict[str, Any]]:
    return [
        {"source_dataset": source, "outcome": outcome, "n_cases": count}
        for (source, outcome), count in sorted(counter.items())
    ]


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _format_ci(interval: Sequence[float] | None) -> str:
    return "NA" if not interval else f"[{interval[0]:.3f}, {interval[1]:.3f}]"


def _format_number(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}"
