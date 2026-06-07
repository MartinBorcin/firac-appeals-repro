import json
from pathlib import Path

from firac_appeals.ingest.in_abs import load_in_abs
from firac_appeals.ingest.vanga import load_vanga_supreme_court


def test_load_in_abs_extracted_layout(tmp_path: Path) -> None:
    judgment_dir = tmp_path / "train-data" / "judgement"
    summary_dir = tmp_path / "train-data" / "summary"
    judgment_dir.mkdir(parents=True)
    summary_dir.mkdir(parents=True)
    (judgment_dir / "case1.txt").write_text("Judgment text", encoding="utf-8")
    (summary_dir / "case1.txt").write_text("Reference summary", encoding="utf-8")

    rows = load_in_abs(tmp_path)

    assert len(rows) == 1
    assert rows[0]["case_id"] == "in_abs_train_case1"
    assert rows[0]["reference_summary"] == "Reference summary"
    assert rows[0]["split"] == "train"


def test_load_vanga_extracted_text_with_metadata(tmp_path: Path) -> None:
    text_dir = tmp_path / "judgments"
    text_dir.mkdir()
    (text_dir / "case-a.txt").write_text("Supreme Court judgment text", encoding="utf-8")
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(
        json.dumps({"case_id": "sc_case_a", "filename": "case-a.txt", "decision_date": "2024-07-02"}) + "\n",
        encoding="utf-8",
    )

    rows = load_vanga_supreme_court(text_dir, metadata_path=metadata)

    assert len(rows) == 1
    assert rows[0]["case_id"] == "sc_case_a"
    assert rows[0]["metadata"]["decision_date"] == "2024-07-02"
    assert rows[0]["judgment_text"] == "Supreme Court judgment text"
