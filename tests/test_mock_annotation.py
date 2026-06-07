from pathlib import Path

from firac_appeals.annotation.pipeline import FiracPipeline, run_pipeline_sync
from firac_appeals.utils.io import read_jsonl, read_yaml


ROOT = Path(__file__).resolve().parents[1]


def test_mock_annotation_end_to_end(tmp_path: Path) -> None:
    config = read_yaml(ROOT / "configs" / "annotation" / "mock.yaml")
    rows = read_jsonl(ROOT / "data" / "fixtures" / "cases.jsonl")
    pipeline = FiracPipeline(config, tmp_path, root=ROOT)
    results = run_pipeline_sync(pipeline, rows)
    assert len(results) == 2
    assert all(row["annotation_status"] == "ok" for row in results)
    assert (tmp_path / "final.jsonl").exists()
