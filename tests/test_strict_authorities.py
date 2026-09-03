from firac_appeals.evaluation.strict_authorities import (
    materialize_line_evidence,
    patch_operations,
    validate_identifier_field,
    validate_rule_output,
)


JUDGMENT = "The Court followed State of Bombay vs Atma Ram (1), reported at AIR 1951 SC 157."


def test_verbatim_evidence_allows_cosmetic_case_name_normalization() -> None:
    field, errors = validate_identifier_field(
        "case_name",
        {
            "decision": "correct",
            "value": "State of Bombay v. Atma Ram",
            "evidence": "State of Bombay vs Atma Ram (1)",
        },
        JUDGMENT,
    )
    assert not errors
    assert field["value"] == "State of Bombay v. Atma Ram"


def test_citation_numbers_must_appear_in_evidence() -> None:
    _, errors = validate_identifier_field(
        "citation",
        {"decision": "correct", "value": "AIR 1951 SC 184", "evidence": "AIR 1951 SC 157"},
        JUDGMENT,
    )
    assert errors == ["citation numbers absent from evidence: ['184']"]


def test_single_digit_ocr_and_line_wrap_normalization_are_allowed() -> None:
    field, errors = validate_identifier_field(
        "citation",
        {
            "decision": "correct",
            "value": "1 Hale P.C. 509",
            "evidence": "according to I Hale P.C. 509",
        },
        "The judgment states according to I Hale P.C.\n509 in the discussion.",
    )
    assert not errors
    assert field["value"] == "1 Hale P.C. 509"


def test_decorative_edge_ellipses_are_removed_before_validation() -> None:
    field, errors = validate_identifier_field(
        "case_name",
        {
            "decision": "correct",
            "value": "Atma Ram",
            "evidence": "... State of Bombay vs Atma Ram ...",
        },
        JUDGMENT,
    )
    assert not errors
    assert field["evidence"] == "State of Bombay vs Atma Ram"


def test_evidence_matching_ignores_only_punctuation_and_whitespace() -> None:
    field, errors = validate_identifier_field(
        "case_name",
        {
            "decision": "keep",
            "value": "Commissioner of Income-tax v. Teja Singh",
            "evidence": "Commissioner of Income tax, vs Teja Singh",
        },
        "The Commissioner of Income-tax vs Teja Singh was considered.",
        "Commissioner of Income-tax v. Teja Singh",
    )
    assert not errors
    assert field["decision"] == "keep"


def test_line_evidence_is_copied_from_judgment() -> None:
    item, errors = materialize_line_evidence(
        {
            "decision": "keep",
            "value": "AIR 1951 SC 157",
            "evidence_line_start": 2,
            "evidence_line_end": 3,
        },
        ["First line", "AIR 1951", "SC 157"],
    )
    assert not errors
    assert item["evidence"] == "AIR 1951\nSC 157"


def test_rule_output_requires_verbatim_evidence() -> None:
    _, errors = validate_rule_output(
        {
            "rule_index": 4,
            "reference": {"decision": "keep", "value": "Atma Ram", "evidence": "Atma Ram"},
            "case_name": {"decision": "keep", "value": "Atma Ram", "evidence": "not copied"},
            "citation": {"decision": "remove", "value": "", "evidence": ""},
        },
        judgment=JUDGMENT,
        expected_rule_index=4,
        original_rule={"reference": "Atma Ram", "case_name": "Atma Ram", "citation": ""},
    )
    assert errors == [
        "case_name: evidence is not a judgment substring after whitespace normalization"
    ]


def test_patch_operations_only_include_changed_identifier_fields() -> None:
    operations = patch_operations(
        rule_index=2,
        original_rule={"reference": "Case A", "case_name": "Case A", "citation": "AIR 2000 SC 1"},
        corrected_fields={
            "reference": {"value": "Case A", "evidence": "Case A"},
            "case_name": {"decision": "keep", "value": "Case A", "evidence": "Case A"},
            "citation": {"decision": "remove", "value": "", "evidence": ""},
        },
    )
    assert operations == [
        {
            "op": "replace",
            "path": "/firac/rules/2/citation",
            "old_value": "AIR 2000 SC 1",
            "value": "",
        }
    ]
