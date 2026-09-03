from firac_appeals.evaluation.authority_provenance import JudgmentIndex, audit_precedent_rule


def test_locates_case_name_across_versus_variants() -> None:
    judgment = JudgmentIndex.build("The Court followed State of Bombay vs Atma Ram Sridhar Vaidya.")

    result = judgment.locate("State of Bombay v. Atma Ram Sridhar Vaidya")

    assert result["status"] == "present"
    assert result["method"] == "normalized"
    assert "State of Bombay vs Atma Ram" in result["evidence"]


def test_does_not_infer_citation_from_case_name() -> None:
    judgment = JudgmentIndex.build("The Court followed State of Bombay v. Atma Ram Sridhar Vaidya.")

    result = judgment.locate("AIR 1951 SC 184")

    assert result["status"] == "absent"


def test_proposes_non_destructive_review_patch() -> None:
    judgment = JudgmentIndex.build("The Court followed State of Bombay v. Atma Ram Sridhar Vaidya.")
    rule = {
        "type": "precedent",
        "case_name": "State of Bombay v. Atma Ram Sridhar Vaidya",
        "citation": "AIR 1951 SC 184",
        "reference": "State of Bombay v. Atma Ram Sridhar Vaidya",
    }

    audit = audit_precedent_rule(case_id="case-1", rule_index=2, rule=rule, judgment=judgment)

    assert audit["audit_status"] == "partial"
    assert audit["requires_review"] is True
    assert audit["proposed_patch"]["status"] == "pending_review"
    assert audit["proposed_patch"]["operations"] == [
        {
            "op": "replace",
            "path": "/firac/rules/2/citation",
            "old_value": "AIR 1951 SC 184",
            "value": "",
            "reason": "Identifier not found in the source judgment after conservative normalization.",
            "confidence": "pending_review",
        }
    ]
