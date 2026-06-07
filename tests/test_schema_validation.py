from firac_appeals.schema.models import canonical_outcome
from firac_appeals.schema.validation import validate_stage


def test_validate_firac_stage_and_outcome() -> None:
    value = validate_stage(
        "firac_judge",
        {
            "facts": ["A fact."],
            "issues": ["An issue?"],
            "rules": [{"type": "unknown", "reference": "Some principle"}],
            "analysis": [{"issue": "An issue?", "steps": ["A step."], "conclusion": "Done."}],
            "conclusion": {
                "label": "Appeal allowed",
                "text": "The appeal is allowed.",
                "simplified_label": "allowed",
            },
        },
    )
    assert value["rules"][0]["type"] == "other"
    assert canonical_outcome(value["conclusion"]["label"]) == "allowed"
