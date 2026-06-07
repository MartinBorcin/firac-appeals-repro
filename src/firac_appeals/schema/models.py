from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any


CONCLUSION_TO_OUTCOME = {
    "appeal allowed": "allowed",
    "appeals allowed": "allowed",
    "appeal dismissed": "dismissed",
    "appeals dismissed": "dismissed",
    "appeal partly allowed": "partly_allowed",
    "appeals partly allowed": "partly_allowed",
    "appeal disposed of": "disposed",
    "appeals disposed of": "disposed",
    "petition allowed": "allowed",
    "petitions allowed": "allowed",
    "petition dismissed": "dismissed",
    "petitions dismissed": "dismissed",
    "petition partly allowed": "partly_allowed",
    "petitions partly allowed": "partly_allowed",
    "petition disposed of": "disposed",
    "petitions disposed of": "disposed",
    "application allowed": "allowed",
    "application dismissed": "dismissed",
    "application disposed of": "disposed",
    "matter remitted to high court": "remitted",
    "matter remitted to lower court": "remitted",
    "matter referred to a larger bench": "referred",
    "reference answered": "answered",
    "references answered": "answered",
    "order accordingly": "other",
}


def stable_case_id(*parts: str) -> str:
    material = "::".join(p.strip() for p in parts if p)
    return hashlib.sha1(material.encode("utf-8")).hexdigest()[:16]


def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


@dataclass
class CaseRecord:
    case_id: str
    judgment_text: str
    source_dataset: str = ""
    reference_summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_any(cls, row: dict[str, Any], source_dataset: str) -> "CaseRecord":
        text = first_present(
            row,
            ["judgment_text", "judgement_text", "judgment", "judgement", "document", "text", "full_text"],
        )
        summary = first_present(
            row,
            ["reference_summary", "summary", "abstractive_summary", "headnote", "target"],
        )
        existing_id = normalize_space(first_present(row, ["case_id", "id", "doc_id", "filename"]))
        case_id = existing_id or stable_case_id(source_dataset, text[:500], summary[:200])
        return cls(
            case_id=case_id,
            judgment_text=normalize_space(text),
            source_dataset=source_dataset,
            reference_summary=normalize_space(summary),
            metadata={k: v for k, v in row.items() if k not in {"judgment_text", "reference_summary"}},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "source_dataset": self.source_dataset,
            "judgment_text": self.judgment_text,
            "reference_summary": self.reference_summary,
            "metadata": self.metadata,
        }


def first_present(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return str(row[key])
    return ""


def canonical_outcome(conclusion: str) -> str:
    key = normalize_space(conclusion).lower().strip(".")
    if key in CONCLUSION_TO_OUTCOME:
        return CONCLUSION_TO_OUTCOME[key]
    for phrase, label in CONCLUSION_TO_OUTCOME.items():
        if phrase in key:
            return label
    return "unknown"
