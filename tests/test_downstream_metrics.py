from pathlib import Path

from firac_appeals.evaluation.downstream import build_task_examples, evaluate_predictions
from firac_appeals.utils.io import read_jsonl


FIXTURES = Path(__file__).resolve().parents[1] / "data" / "fixtures"


def test_downstream_metrics_for_fixture_predictions() -> None:
    annotations = read_jsonl(FIXTURES / "annotations.jsonl")
    for task in ["outcome_prediction", "issue_generation", "rule_recommendation"]:
        refs = build_task_examples(annotations, task)
        preds = read_jsonl(FIXTURES / "downstream_predictions" / f"{task}.jsonl")
        result = evaluate_predictions(task, refs, preds)
        assert result["count"] == 2
        assert result["aggregate"]
