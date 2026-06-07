from pathlib import Path

from firac_appeals.evaluation.summarization import evaluate_rows
from firac_appeals.utils.io import read_jsonl


FIXTURES = Path(__file__).resolve().parents[1] / "data" / "fixtures"


def test_summarization_metrics() -> None:
    rows = read_jsonl(FIXTURES / "annotations.jsonl")
    result = evaluate_rows(rows)
    assert result["count"] == 2
    assert "rouge1_f1" in result["aggregate"]
