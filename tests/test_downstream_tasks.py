from pathlib import Path

from firac_appeals.evaluation.downstream import build_task_examples
from firac_appeals.utils.io import read_jsonl


FIXTURES = Path(__file__).resolve().parents[1] / "data" / "fixtures"


def test_build_all_downstream_tasks() -> None:
    rows = read_jsonl(FIXTURES / "annotations.jsonl")
    assert len(build_task_examples(rows, "outcome_prediction")) == 2
    assert len(build_task_examples(rows, "issue_generation")) == 2
    assert len(build_task_examples(rows, "rule_recommendation")) == 2
