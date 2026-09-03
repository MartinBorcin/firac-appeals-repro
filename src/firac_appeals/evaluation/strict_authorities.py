from __future__ import annotations

import re
from typing import Any


IDENTIFIER_FIELDS = ("reference", "case_name", "citation")
NUMBER_RE = re.compile(r"\d+")


def validate_identifier_field(
    field: str, item: Any, judgment: str, original_value: str = ""
) -> tuple[dict[str, Any], list[str]]:
    errors = []
    if not isinstance(item, dict):
        item = {}
        errors.append("field output is not an object")
    decision = str(item.get("decision") or "").casefold().strip()
    if decision not in {"keep", "remove", "correct"}:
        errors.append("decision must be keep, remove, or correct")
        decision = "remove" if not item.get("value") else "correct"
    proposed_value = " ".join(str(item.get("value") or "").split())
    if decision == "keep":
        value = original_value
    elif decision == "remove":
        value = ""
    else:
        value = proposed_value
    evidence = clean_evidence(item.get("evidence"))

    if decision == "keep" and not original_value:
        errors.append("cannot keep an originally empty field")
    if decision == "remove" and (proposed_value or evidence):
        errors.append("remove decision must have empty value and evidence")
    if decision == "correct" and not proposed_value:
        errors.append("correct decision has empty value")
    if not value and evidence:
        errors.append("empty value has non-empty evidence")
    if value and not evidence:
        errors.append("non-empty value has no evidence")
    if evidence and not evidence_in_judgment(evidence, judgment):
        errors.append("evidence is not a judgment substring after whitespace normalization")
    if field == "citation" and value:
        missing_numbers = [
            number for number in NUMBER_RE.findall(value) if len(number) > 1 and number not in evidence
        ]
        if missing_numbers:
            errors.append(f"citation numbers absent from evidence: {missing_numbers}")

    return {"decision": decision, "value": value, "evidence": evidence}, errors


def evidence_in_judgment(evidence: str, judgment: str) -> bool:
    if evidence in judgment:
        return True
    if " ".join(evidence.split()) in " ".join(judgment.split()):
        return True
    compact_evidence = "".join(character.casefold() for character in evidence if character.isalnum())
    compact_judgment = "".join(character.casefold() for character in judgment if character.isalnum())
    return bool(compact_evidence and compact_evidence in compact_judgment)


def clean_evidence(value: Any) -> str:
    evidence = str(value or "").strip()
    while evidence.startswith(("...", "…")):
        evidence = evidence[1:] if evidence.startswith("…") else evidence[3:]
        evidence = evidence.lstrip()
    while evidence.endswith(("...", "…")):
        evidence = evidence[:-1] if evidence.endswith("…") else evidence[:-3]
        evidence = evidence.rstrip()
    return evidence


def validate_rule_output(
    parsed_rule: Any,
    *,
    judgment: str,
    expected_rule_index: int,
    original_rule: dict[str, Any] | None = None,
    judgment_lines: list[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    errors = []
    if not isinstance(parsed_rule, dict):
        parsed_rule = {}
        errors.append("rule output is not an object")
    try:
        returned_index = int(parsed_rule.get("rule_index"))
    except (TypeError, ValueError):
        returned_index = -1
    if returned_index != expected_rule_index:
        errors.append(f"wrong rule_index: expected {expected_rule_index}, got {returned_index}")

    fields = {}
    original_rule = original_rule or {}
    for field in IDENTIFIER_FIELDS:
        item = parsed_rule.get(field)
        line_errors = []
        if judgment_lines is not None:
            item, line_errors = materialize_line_evidence(item, judgment_lines)
        fields[field], field_errors = validate_identifier_field(
            field,
            item,
            judgment,
            str(original_rule.get(field) or ""),
        )
        fields[field]["evidence_line_start"] = (item or {}).get("evidence_line_start", 0)
        fields[field]["evidence_line_end"] = (item or {}).get("evidence_line_end", 0)
        errors.extend(f"{field}: {error}" for error in line_errors)
        errors.extend(f"{field}: {error}" for error in field_errors)
    return fields, errors


def materialize_line_evidence(
    item: Any, judgment_lines: list[str], max_span: int = 10
) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(item, dict):
        return {}, ["field output is not an object"]
    output = dict(item)
    decision = str(output.get("decision") or "").casefold().strip()
    if decision == "remove":
        output.update({"evidence": "", "evidence_line_start": 0, "evidence_line_end": 0})
        return output, []
    try:
        start = int(output.get("evidence_line_start"))
        end = int(output.get("evidence_line_end", start))
    except (TypeError, ValueError):
        output["evidence"] = ""
        return output, ["invalid evidence line number"]
    if start < 1 or end < start or end > len(judgment_lines):
        output["evidence"] = ""
        return output, [f"evidence line range outside judgment: {start}-{end}"]
    if end - start + 1 > max_span:
        output["evidence"] = ""
        return output, [f"evidence line range exceeds {max_span} lines"]
    output["evidence"] = "\n".join(judgment_lines[start - 1 : end])
    output["evidence_line_start"] = start
    output["evidence_line_end"] = end
    return output, []


def patch_operations(
    *, rule_index: int, original_rule: dict[str, Any], corrected_fields: dict[str, Any]
) -> list[dict[str, Any]]:
    operations = []
    for field in IDENTIFIER_FIELDS:
        old_value = str(original_rule.get(field) or "")
        new_value = str(corrected_fields[field]["value"] or "")
        if old_value == new_value:
            continue
        operations.append(
            {
                "op": "replace",
                "path": f"/firac/rules/{rule_index}/{field}",
                "old_value": old_value,
                "value": new_value,
            }
        )
    return operations
