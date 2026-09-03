from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from firac_appeals.evaluation.human_rubric import SCORE_FIELDS, ScoreRow, percentile_interval


SECTION_ORDER = ("facts", "issues", "rules", "analysis", "conclusion")
FIELD_ORDER = tuple(
    f"{section}_{dimension}"
    for section in SECTION_ORDER
    for dimension in ("faithfulness", "coverage")
)
SECTION_LABELS = {
    "facts": "Facts",
    "issues": "Issues",
    "rules": "Rules/Authorities",
    "analysis": "Analysis",
    "conclusion": "Conclusion",
}
DIMENSION_LABELS = {"faithfulness": "Correctness", "coverage": "Coverage"}


def load_llm_scores(paths: Sequence[str | Path]) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    scores_by_case: dict[str, dict[str, int]] = {}
    invalid_rows = 0
    duplicate_rows = 0
    source_counts: Counter[str] = Counter()
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                case_id = str(row.get("case_id") or "")
                raw_scores = (row.get("llm_rubric") or {}).get("scores") or {}
                try:
                    scores = {field: int(raw_scores[field]) for field in SCORE_FIELDS}
                except (KeyError, TypeError, ValueError):
                    invalid_rows += 1
                    continue
                if not case_id or any(score not in range(1, 6) for score in scores.values()):
                    invalid_rows += 1
                    continue
                if case_id in scores_by_case:
                    duplicate_rows += 1
                    continue
                scores_by_case[case_id] = scores
                source_counts[str(row.get("source_dataset") or "unknown")] += 1
    return scores_by_case, {
        "valid_cases": len(scores_by_case),
        "invalid_rows_ignored": invalid_rows,
        "duplicate_rows_ignored": duplicate_rows,
        "valid_cases_by_source": dict(sorted(source_counts.items())),
    }


def human_likert_table(
    rows: Sequence[ScoreRow], *, bootstrap_replicates: int = 5000, seed: int = 20260818
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        grouped[(row.case_id, row.field)].append(row.score)
    output: list[dict[str, Any]] = []
    for index, field in enumerate(FIELD_ORDER):
        units = {case_id: values for (case_id, item_field), values in grouped.items() if item_field == field}
        output.append(
            _weighted_human_row(
                field, units, bootstrap_replicates=bootstrap_replicates, seed=seed + index
            )
        )
    case_unit_means: dict[str, list[float]] = defaultdict(list)
    for (case_id, _field), ratings in grouped.items():
        case_unit_means[case_id].append(statistics.mean(ratings))
    overall_case_means = [statistics.mean(values) for values in case_unit_means.values()]
    output.append(
        _weighted_human_row(
            "overall",
            grouped,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed + 100,
            bootstrap_values=overall_case_means,
        )
    )
    return output


def llm_likert_table(scores_by_case: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for field in FIELD_ORDER:
        values = [scores[field] for scores in scores_by_case.values()]
        output.append(_unweighted_likert_row(field, values))
    case_means = [statistics.mean(scores[field] for field in FIELD_ORDER) for scores in scores_by_case.values()]
    all_scores = [scores[field] for scores in scores_by_case.values() for field in FIELD_ORDER]
    output.append(_unweighted_likert_row("overall", all_scores, case_values=case_means))
    return output


def annotator_summary_table(rows: Sequence[ScoreRow]) -> list[dict[str, Any]]:
    ratings_by_annotator: dict[str, list[int]] = defaultdict(list)
    ratings_by_case: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        ratings_by_annotator[row.annotator].append(row.score)
        ratings_by_case[row.annotator][row.case_id].append(row.score)
    output: list[dict[str, Any]] = []
    for annotator in sorted(ratings_by_annotator):
        ratings = ratings_by_annotator[annotator]
        case_means = [statistics.mean(values) for values in ratings_by_case[annotator].values()]
        output.append(
            {
                "annotator": annotator,
                "n_cases": len(case_means),
                "n_ratings": len(ratings),
                "mean_case_score": statistics.mean(case_means),
                "case_score_sample_sd": statistics.stdev(case_means),
                "case_score_sample_variance": statistics.variance(case_means),
                "pct_score_5": 100 * ratings.count(5) / len(ratings),
                "pct_below_4": 100 * sum(score < 4 for score in ratings) / len(ratings),
            }
        )
    return output


def alignment_table(
    human_rows: Sequence[ScoreRow],
    llm_scores: dict[str, dict[str, int]],
    human_summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    human_units: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for row in human_rows:
        human_units[(row.case_id, row.field)][row.annotator] = row.score

    output: list[dict[str, Any]] = []
    overlap_pairs: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for ratings in human_units.values():
        if len(ratings) != 2:
            continue
        first, second = sorted(ratings)
        overlap_pairs[(first, second)].append((ratings[first], ratings[second]))
    pooled = [pair for values in overlap_pairs.values() for pair in values]
    alpha_groups = {row["group"]: row for row in human_summary["agreement"]["groups"]}
    alpha_pairs = {row["group"]: row for row in human_summary["agreement"]["annotator_pairs"]}
    output.append(_human_pair_row("Human--human overall", pooled, alpha_groups["overall"]))
    for pair, values in sorted(overlap_pairs.items()):
        key = f"{pair[0]}-{pair[1]}"
        label = f"{pair[0]}--{pair[1]}"
        output.append(_human_pair_row(label, values, alpha_pairs[key]))

    matched: list[tuple[str, str, list[int], int]] = []
    missing_human_cases: set[str] = set()
    for (case_id, field), ratings in sorted(human_units.items()):
        llm_case = llm_scores.get(case_id)
        if llm_case is None or field not in llm_case:
            missing_human_cases.add(case_id)
            continue
        matched.append((case_id, field, list(ratings.values()), llm_case[field]))
    mean_pairs = [(statistics.mean(ratings), llm) for _case, _field, ratings, llm in matched]
    minimum_pairs = [(min(ratings), llm) for _case, _field, ratings, llm in matched]
    output.append(_human_llm_row("LLM--human mean", mean_pairs))
    output.append(_human_llm_row("LLM--human minimum", minimum_pairs))
    metadata = {
        "matched_cases": len({case_id for case_id, _field, _ratings, _llm in matched}),
        "matched_case_criteria": len(matched),
        "human_cases_without_llm_score": sorted(missing_human_cases),
    }
    return output, metadata


def render_quality_tables_latex(
    human_rows: Sequence[dict[str, Any]],
    llm_rows: Sequence[dict[str, Any]],
    alignment_rows: Sequence[dict[str, Any]],
    annotator_rows: Sequence[dict[str, Any]],
    *,
    human_cases: int,
    human_annotations: int,
    llm_cases: int,
    llm_expected_cases: int,
    alignment_cases: int,
) -> str:
    blocks = [
        _render_likert_latex(
            human_rows,
            caption=(
                "Human rubric-score distribution. Percentages are case-weighted so each of the "
                f"{human_cases} cases contributes equally; {human_annotations} annotations were collected. "
                "CI denotes a case-bootstrap 95\\% confidence interval for the mean."
            ),
            label="tab:human_likert",
            mean_decimals=2,
        ),
        _render_likert_latex(
            llm_rows,
            caption=(
                "LLM-judge rubric-score distribution over successfully evaluated cases "
                f"({llm_cases:,}/{llm_expected_cases:,}; incomplete cases excluded). CI denotes a "
                "case-clustered normal 95\\% confidence interval for the mean."
            ),
            label="tab:llm_likert",
            mean_decimals=3,
        ),
        _render_alignment_latex(
            alignment_rows,
            caption=(
                "Human inter-annotator agreement and human--LLM alignment. Human-pair rows use "
                "case--criterion ratings from 30 double-annotated cases (10 per pair); human--LLM "
                f"rows compare the LLM with the available human ratings over {alignment_cases} cases. "
                "$F_1(<4)$ measures overlap/detection of scores below 4; AUPRC is reported only when "
                "human ratings serve as the screening reference."
            ),
            label="tab:annotation_alignment",
        ),
        render_annotator_summary_latex(annotator_rows),
    ]
    return "\n\n".join(blocks) + "\n"


def render_annotator_summary_latex(rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Annotator score distributions. Mean, SD, and variance are computed across "
        "the 50 case-level means (each averaging ten rubric scores); Score 5 and $<4$ are "
        "percentages of the 500 individual rubric scores.}",
        "\\label{tab:annotator_distributions}",
        "\\begin{tabular}{@{}lrrrrr@{}}",
        "\\toprule",
        "Annotator & Mean & SD & Variance & Score 5 (\\%) & $<4$ (\\%) \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['annotator']} & {row['mean_case_score']:.3f} & "
            f"{row['case_score_sample_sd']:.3f} & {row['case_score_sample_variance']:.3f} & "
            f"{row['pct_score_5']:.1f} & {row['pct_below_4']:.1f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def _weighted_human_row(
    field: str,
    units: dict[Any, list[int]],
    *,
    bootstrap_replicates: int,
    seed: int,
    bootstrap_values: Sequence[float] | None = None,
) -> dict[str, Any]:
    counts = {score: 0.0 for score in range(1, 6)}
    unit_means: list[float] = []
    for ratings in units.values():
        weight = 1 / len(ratings)
        for score in ratings:
            counts[score] += weight
        unit_means.append(statistics.mean(ratings))
    total = sum(counts.values())
    return {
        **_row_identity(field),
        "n_units": len(units),
        **{f"pct_{score}": 100 * counts[score] / total for score in range(1, 6)},
        "pct_below_4": 100 * sum(counts[score] for score in (1, 2, 3)) / total,
        "mean": statistics.mean(unit_means),
        "ci95": _bootstrap_values(
            list(bootstrap_values) if bootstrap_values is not None else unit_means,
            bootstrap_replicates,
            seed,
        ),
    }


def _unweighted_likert_row(
    field: str, values: Sequence[int], *, case_values: Sequence[float] | None = None
) -> dict[str, Any]:
    counts = Counter(values)
    clustered_values = list(case_values) if case_values is not None else list(values)
    return {
        **_row_identity(field),
        "n_units": len(clustered_values),
        **{f"pct_{score}": 100 * counts[score] / len(values) for score in range(1, 6)},
        "pct_below_4": 100 * sum(counts[score] for score in (1, 2, 3)) / len(values),
        "mean": statistics.mean(values),
        "ci95": _normal_mean_ci(clustered_values),
    }


def _human_pair_row(
    label: str, pairs: Sequence[tuple[float, float]], alpha_row: dict[str, Any]
) -> dict[str, Any]:
    base = _pair_metrics(pairs)
    return {
        "comparison": label,
        **base,
        "ordinal_alpha": alpha_row["krippendorff_alpha_ordinal"],
        "ordinal_alpha_ci95": alpha_row["krippendorff_alpha_ci95"],
        "auprc": None,
    }


def _human_llm_row(label: str, pairs: Sequence[tuple[float, float]]) -> dict[str, Any]:
    base = _pair_metrics(pairs)
    references = [left < 4 for left, _right in pairs]
    llm_scores = [right for _left, right in pairs]
    return {
        "comparison": label,
        **base,
        "ordinal_alpha": None,
        "ordinal_alpha_ci95": None,
        "auprc": average_precision_from_ordinal_scores(references, llm_scores),
    }


def _pair_metrics(pairs: Sequence[tuple[float, float]]) -> dict[str, Any]:
    differences = [abs(left - right) for left, right in pairs]
    left_low = [left < 4 for left, _right in pairs]
    right_low = [right < 4 for _left, right in pairs]
    true_positive = sum(left and right for left, right in zip(left_low, right_low))
    positive_total = sum(left_low) + sum(right_low)
    return {
        "n_units": len(pairs),
        "exact_agreement": sum(value == 0 for value in differences) / len(differences),
        "within_one_agreement": sum(value <= 1 for value in differences) / len(differences),
        "mean_absolute_difference": statistics.mean(differences),
        "spearman_rho": spearman([left for left, _right in pairs], [right for _left, right in pairs]),
        "f1_below_4": 2 * true_positive / positive_total if positive_total else None,
        "reference_below_4": sum(left_low),
        "comparison_below_4": sum(right_low),
        "both_below_4": true_positive,
    }


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(right) != len(left):
        return None
    left_ranks = _ranks(left)
    right_ranks = _ranks(right)
    left_mean = statistics.mean(left_ranks)
    right_mean = statistics.mean(right_ranks)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left_ranks, right_ranks)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left_ranks)
        * sum((value - right_mean) ** 2 for value in right_ranks)
    )
    return numerator / denominator if denominator else None


def average_precision_from_ordinal_scores(
    references: Sequence[bool], predicted_scores: Sequence[int]
) -> float | None:
    positives = sum(references)
    if not positives:
        return None
    previous_recall = 0.0
    average_precision = 0.0
    # Low rubric scores are stronger predictions of the positive (<4) class.
    for threshold in sorted(set(predicted_scores)):
        predicted = [score <= threshold for score in predicted_scores]
        true_positive = sum(reference and prediction for reference, prediction in zip(references, predicted))
        predicted_positive = sum(predicted)
        recall = true_positive / positives
        precision = true_positive / predicted_positive if predicted_positive else 1.0
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
    return average_precision


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        rank = (start + end + 2) / 2
        for position in range(start, end + 1):
            ranks[order[position]] = rank
        start = end + 1
    return ranks


def _bootstrap_values(values: Sequence[float], replicates: int, seed: int) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    samples = [
        statistics.mean(rng.choice(values) for _ in values)
        for _ in range(replicates)
    ]
    return percentile_interval(samples)


def _normal_mean_ci(values: Sequence[float]) -> list[float] | None:
    if not values:
        return None
    mean = statistics.mean(values)
    if len(values) == 1:
        return [mean, mean]
    margin = 1.959963984540054 * statistics.stdev(values) / math.sqrt(len(values))
    return [max(1.0, mean - margin), min(5.0, mean + margin)]


def _row_identity(field: str) -> dict[str, str]:
    if field == "overall":
        return {"field": field, "component": "Overall", "dimension": ""}
    section, dimension = field.rsplit("_", 1)
    return {
        "field": field,
        "component": SECTION_LABELS[section],
        "dimension": DIMENSION_LABELS[dimension],
    }


def _render_likert_latex(
    rows: Sequence[dict[str, Any]], *, caption: str, label: str, mean_decimals: int
) -> str:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{@{}llrrrrrrr@{}}",
        "\\toprule",
        " & & \\multicolumn{6}{c}{Ratings (\\%)} & \\\\",
        "\\cmidrule(lr){3-8}",
        "Component & Dimension & 1 & 2 & 3 & 4 & 5 & $<4$ & Mean [95\\% CI] \\\\",
        "\\midrule",
    ]
    previous_component = ""
    for row in rows:
        if row["field"] == "overall":
            lines.append("\\midrule")
        component = row["component"] if row["component"] != previous_component else ""
        previous_component = row["component"]
        lines.append(
            f"{component} & {row['dimension']} & "
            + " & ".join(f"{row[f'pct_{score}']:.1f}" for score in range(1, 6))
            + f" & {row['pct_below_4']:.1f} & {row['mean']:.{mean_decimals}f} "
            + f"[{row['ci95'][0]:.{mean_decimals}f}, "
            + f"{row['ci95'][1]:.{mean_decimals}f}] \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
        ]
    )
    return "\n".join(lines)


def _render_alignment_latex(
    rows: Sequence[dict[str, Any]], *, caption: str, label: str
) -> str:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{@{}lrrrrrrr@{}}",
        "\\toprule",
        "Comparison & Exact (\\%) & Within 1 (\\%) & MAE & $\\rho$ & Ordinal $\\alpha$ [95\\% CI] & $F_1(<4)$ & AUPRC \\\\",
        "\\midrule",
    ]
    for index, row in enumerate(rows):
        if index == 4:
            lines.append("\\midrule")
        alpha = "--"
        if row["ordinal_alpha"] is not None:
            interval = row["ordinal_alpha_ci95"]
            alpha = f"{row['ordinal_alpha']:.3f} [{interval[0]:.3f}, {interval[1]:.3f}]"
        auprc = "--" if row["auprc"] is None else f"{row['auprc']:.3f}"
        lines.append(
            f"{row['comparison']} & {100 * row['exact_agreement']:.1f} & "
            f"{100 * row['within_one_agreement']:.1f} & {row['mean_absolute_difference']:.3f} & "
            f"{_format_optional(row['spearman_rho'])} & {alpha} & "
            f"{_format_optional(row['f1_below_4'])} & {auprc} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    return "\n".join(lines)


def _format_optional(value: float | None) -> str:
    return "--" if value is None else f"{value:.3f}"
