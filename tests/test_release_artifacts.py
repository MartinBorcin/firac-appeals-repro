from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def test_human_export_is_anonymous_and_complete() -> None:
    payload = json.loads(
        (ARTIFACTS / "human/label_studio_round2_sanitized.json").read_text(encoding="utf-8")
    )
    assert len(payload["tasks"]) == 200
    annotations = [item for task in payload["tasks"] for item in task["annotations"]]
    assert len(annotations) == 150
    assert set(payload["tasks"][0]["data"]) == {
        "case_id",
        "source_dataset",
        "split",
        "outcome_label",
    }
    assert all(
        item["completed_by"]["email"].endswith("@redacted.invalid")
        for item in annotations
    )
    manifest = json.loads(
        (ARTIFACTS / "human/assignment_manifest_sanitized.json").read_text(encoding="utf-8")
    )
    assignment_rows = [row for rows in manifest["assignments"].values() for row in rows]
    assert assignment_rows
    assert all("email" not in row and "task_url" not in row for row in assignment_rows)


def test_full_corpus_llm_coverage() -> None:
    in_abs = sum(1 for _ in read_jsonl(ARTIFACTS / "llm_judge/in_abs.jsonl"))
    supreme_court = sum(1 for _ in read_jsonl(ARTIFACTS / "llm_judge/supreme_court.jsonl"))
    assert (in_abs, supreme_court, in_abs + supreme_court) == (7121, 37882, 45003)


def test_authority_patch_counts() -> None:
    rows = list(read_jsonl(ARTIFACTS / "authority_patch/strict_authorities_patch.jsonl"))
    assert len(rows) == 12476
    operations = [operation for row in rows for operation in row["operations"]]
    assert len(operations) == 26012
    assert sum(operation["value"] == "" for operation in operations) == 25979


def test_representative_reported_metrics() -> None:
    in_ljp = json.loads(
        (ARTIFACTS / "downstream/ljp/in_abs/tfidf_lr/metrics.json").read_text(encoding="utf-8")
    )
    sc_ljp = json.loads(
        (ARTIFACTS / "downstream/ljp/supreme_court/inlegalbert/summary.json").read_text(
            encoding="utf-8"
        )
    )
    issue = json.loads(
        (ARTIFACTS / "downstream/issue/metrics/sc_qwen36.metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert round(in_ljp["macro_f1"], 3) == 0.663
    assert round(sc_ljp["aggregate"]["macro_f1"]["mean"], 3) == 0.700
    assert round(issue["aggregate"]["issue_semantic_f1"], 3) == 0.738
