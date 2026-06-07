from __future__ import annotations

from typing import Any

try:
    from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
except ImportError:  # pragma: no cover - exercised only on bare cluster envs
    BaseModel = None  # type: ignore[assignment]
    ValidationError = ValueError  # type: ignore[assignment]


if BaseModel is not None:

    class LooseModel(BaseModel):
        model_config = ConfigDict(extra="allow")


    class MetadataLawyers(LooseModel):
        for_petitioner_or_appellant: list[str] = Field(default_factory=list)
        for_respondent: list[str] = Field(default_factory=list)


    class MetadataCaseDetails(LooseModel):
        case_no: list[str] = Field(default_factory=list)
        petitioner_or_appellant: list[str] = Field(default_factory=list)
        respondent: list[str] = Field(default_factory=list)


    class MetadataOutput(LooseModel):
        neutral_citation: str = ""
        court: str = ""
        district: str = ""
        impugned_court: str = ""
        bench_coram: list[str] = Field(default_factory=list)
        judgment_by: list[str] = Field(default_factory=list)
        judgment_date: str = ""
        lawyers: MetadataLawyers = Field(default_factory=MetadataLawyers)
        case_details: list[MetadataCaseDetails] = Field(default_factory=list)


    class IssueItem(LooseModel):
        question: str
        why_decisive: str = ""
        source_basis: str = ""


    class IssuesOutput(LooseModel):
        issues: list[IssueItem]


    class RuleItem(LooseModel):
        type: str = "other"
        reference: str
        provision: str = ""
        case_name: str = ""
        citation: str = ""
        treatment_type: str = "Not applicable to authority type"
        relevance: str = ""
        used_by_court: bool = True

        @field_validator("type")
        @classmethod
        def valid_type(cls, value: str) -> str:
            allowed = {
                "statute",
                "rule",
                "regulation",
                "constitutional_provision",
                "precedent",
                "legal_principle",
                "other",
            }
            return value if value in allowed else "other"

        @field_validator("treatment_type")
        @classmethod
        def valid_treatment(cls, value: str) -> str:
            normalized = str(value or "").strip()
            allowed = {
                "Relied on",
                "Referred",
                "Distinguished",
                "Overruled",
                "Not applicable",
                "Not applicable to authority type",
            }
            if normalized == "Not applicable":
                return "Not applicable to authority type"
            return normalized if normalized in allowed else "Not applicable to authority type"


    class RulesOutput(LooseModel):
        rules: list[RuleItem]


    class AnalysisItem(LooseModel):
        issue: str
        steps: list[str]
        conclusion: str = ""


    class AnalysisOutput(LooseModel):
        analysis: list[AnalysisItem]


    class ConclusionOutput(LooseModel):
        label: str
        text: str
        simplified_label: str

        @field_validator("simplified_label")
        @classmethod
        def valid_simple_label(cls, value: str) -> str:
            return value if value in {"allowed", "dismissed", "other"} else "other"


    class FiracOutput(LooseModel):
        facts: list[str]
        issues: list[str]
        rules: list[RuleItem]
        analysis: list[AnalysisItem]
        conclusion: ConclusionOutput

        @field_validator("facts", "issues")
        @classmethod
        def no_empty_strings(cls, value: list[str]) -> list[str]:
            return [item.strip() for item in value if item and item.strip()]


    class FactsLeakage(LooseModel):
        has_leakage: bool = False
        problem_sentences: list[str] = Field(default_factory=list)
        explanation: str = ""


    class VerifierOutput(LooseModel):
        is_usable: bool = True
        facts_leakage: FactsLeakage = Field(default_factory=FactsLeakage)
        faithfulness_errors: list[str] = Field(default_factory=list)
        missing_material_points: list[str] = Field(default_factory=list)
        rules_issues: list[str] = Field(default_factory=list)
        analysis_issues: list[str] = Field(default_factory=list)
        conclusion_issue: str = ""
        recommended_quality_flags: list[str] = Field(default_factory=list)


    STAGE_MODELS: dict[str, type[BaseModel]] = {
        "metadata_ner": MetadataOutput,
        "issue_analyst": IssuesOutput,
        "legal_researcher": RulesOutput,
        "logician": AnalysisOutput,
        "firac_judge": FiracOutput,
        "firac_verifier": VerifierOutput,
    }
else:
    STAGE_MODELS = {}


REQUIRED_KEYS = {
    "metadata_ner": ["neutral_citation", "court", "lawyers", "case_details"],
    "issue_analyst": ["issues"],
    "legal_researcher": ["rules"],
    "logician": ["analysis"],
    "firac_judge": ["facts", "issues", "rules", "analysis", "conclusion"],
    "firac_verifier": ["is_usable", "facts_leakage"],
}


def validate_stage(stage: str, value: dict[str, Any]) -> dict[str, Any]:
    model = STAGE_MODELS.get(stage)
    if model is not None:
        return model.model_validate(value).model_dump(mode="json")
    required = REQUIRED_KEYS.get(stage, [])
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"Missing required keys for {stage}: {missing}")
    return normalize_without_pydantic(stage, value)


def normalize_without_pydantic(stage: str, value: dict[str, Any]) -> dict[str, Any]:
    if stage == "firac_judge":
        value["facts"] = ensure_list(value.get("facts"))
        value["issues"] = ensure_list(value.get("issues"))
        value["rules"] = ensure_list(value.get("rules"))
        value["analysis"] = ensure_list(value.get("analysis"))
        conclusion = value.get("conclusion")
        if not isinstance(conclusion, dict):
            conclusion = {"label": str(conclusion or ""), "text": str(conclusion or ""), "simplified_label": "other"}
        conclusion["simplified_label"] = normalize_simple_label(conclusion.get("simplified_label", "other"))
        value["conclusion"] = conclusion
    elif stage in {"issue_analyst", "legal_researcher", "logician"}:
        key = {"issue_analyst": "issues", "legal_researcher": "rules", "logician": "analysis"}[stage]
        value[key] = ensure_list(value.get(key))
    return value


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_simple_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in {"allowed", "dismissed", "other"} else "other"
