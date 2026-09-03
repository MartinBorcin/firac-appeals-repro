from __future__ import annotations

import pytest

from firac_appeals.evaluation.human_rubric import ScoreRow
from firac_appeals.evaluation.quality_tables import (
    alignment_table,
    annotator_summary_table,
    average_precision_from_ordinal_scores,
    human_likert_table,
    render_quality_tables_latex,
)


def score_row(case_id: str, annotator: str, field: str, score: int) -> ScoreRow:
    section, dimension = field.rsplit("_", 1)
    return ScoreRow(
        task_id=1,
        case_id=case_id,
        source_dataset="in_abs",
        outcome="allowed",
        split="test",
        annotator=annotator,
        annotator_email="",
        annotation_id=f"{case_id}-{annotator}",
        field=field,
        section=section,
        dimension=dimension,
        score=score,
    )


def test_human_likert_weights_each_case_equally() -> None:
    rows = []
    for field in (
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
    ):
        rows.extend(
            [
                score_row("double", "A1", field, 3),
                score_row("double", "A2", field, 5),
                score_row("single", "A1", field, 5),
            ]
        )

    table = human_likert_table(rows, bootstrap_replicates=20, seed=3)
    facts = table[0]
    overall = table[-1]

    # The double-annotated case has total weight one, not two: score 3 gets
    # 0.5/2 total case weight and score 5 gets 1.5/2.
    assert facts["pct_3"] == pytest.approx(25.0)
    assert facts["pct_5"] == pytest.approx(75.0)
    assert facts["mean"] == pytest.approx(4.5)
    assert overall["mean"] == pytest.approx(4.5)


def test_average_precision_respects_ordinal_ties_and_thresholds() -> None:
    value = average_precision_from_ordinal_scores(
        [True, False, True, False],
        [1, 2, 3, 4],
    )
    assert value == pytest.approx(5 / 6)


def test_alignment_mean_and_minimum_have_distinct_low_references() -> None:
    rows = [
        score_row("c1", "A1", "facts_faithfulness", 5),
        score_row("c1", "A2", "facts_faithfulness", 3),
    ]
    alpha = {
        "krippendorff_alpha_ordinal": 0.2,
        "krippendorff_alpha_ci95": [0.0, 0.4],
    }
    summary = {
        "agreement": {
            "groups": [{"group": "overall", **alpha}],
            "annotator_pairs": [{"group": "A1-A2", **alpha}],
        }
    }
    table, metadata = alignment_table(
        rows,
        {"c1": {"facts_faithfulness": 3}},
        summary,
    )

    mean_row = next(row for row in table if row["comparison"] == "LLM--human mean")
    minimum_row = next(row for row in table if row["comparison"] == "LLM--human minimum")
    assert mean_row["reference_below_4"] == 0
    assert mean_row["f1_below_4"] == 0
    assert minimum_row["reference_below_4"] == 1
    assert minimum_row["f1_below_4"] == 1
    assert metadata["matched_case_criteria"] == 1


def test_latex_uses_booktabs_and_percentage_headers() -> None:
    likert = [
        {
            "field": "overall",
            "component": "Overall",
            "dimension": "",
            **{f"pct_{score}": 20.0 for score in range(1, 6)},
            "pct_below_4": 60.0,
            "mean": 3.0,
            "ci95": [2.9, 3.1],
        }
    ]
    alignment = [
        {
            "comparison": "Human--human overall",
            "exact_agreement": 0.8,
            "within_one_agreement": 0.95,
            "mean_absolute_difference": 0.2,
            "spearman_rho": 0.4,
            "ordinal_alpha": 0.3,
            "ordinal_alpha_ci95": [0.1, 0.5],
            "f1_below_4": 0.5,
            "auprc": None,
        }
    ]
    latex = render_quality_tables_latex(
        likert,
        likert,
        alignment,
        [
            {
                "annotator": "A1",
                "mean_case_score": 4.5,
                "case_score_sample_sd": 0.2,
                "case_score_sample_variance": 0.04,
                "pct_score_5": 75.0,
                "pct_below_4": 5.0,
            }
        ],
        human_cases=120,
        human_annotations=150,
        llm_cases=45003,
        llm_expected_cases=45020,
        alignment_cases=120,
    )

    assert "\\toprule" in latex
    assert "\\cmidrule" in latex
    assert "Exact (\\%)" in latex
    assert latex.count("\\begin{table*}") == 3
    assert "\\label{tab:annotator_distributions}" in latex


def test_annotator_summary_uses_case_means_for_dispersion() -> None:
    rows = [
        score_row("c1", "A1", "facts_faithfulness", 3),
        score_row("c1", "A1", "facts_coverage", 5),
        score_row("c2", "A1", "facts_faithfulness", 5),
        score_row("c2", "A1", "facts_coverage", 5),
    ]
    result = annotator_summary_table(rows)[0]

    # Case means are 4 and 5; the pooled mean is identical because both cases
    # contain the same number of rubric scores.
    assert result["mean_case_score"] == pytest.approx(4.5)
    assert result["case_score_sample_variance"] == pytest.approx(0.5)
    assert result["pct_score_5"] == pytest.approx(75.0)
    assert result["pct_below_4"] == pytest.approx(25.0)
