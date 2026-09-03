from __future__ import annotations

import bisect
import re
import unicodedata
from dataclasses import dataclass
from typing import Any


WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
PLACEHOLDERS = {"", "n/a", "na", "none", "not available", "not applicable", "unknown"}
VERSUS_TOKENS = {"v", "vs", "versus"}


@dataclass(frozen=True)
class JudgmentIndex:
    text: str
    folded_text: str
    normalized_text: str
    normalized_starts: tuple[int, ...]
    raw_spans: tuple[tuple[int, int], ...]

    @classmethod
    def build(cls, text: str) -> JudgmentIndex:
        tokens: list[str] = []
        starts: list[int] = []
        spans: list[tuple[int, int]] = []
        normalized_length = 0
        for match in WORD_RE.finditer(unicodedata.normalize("NFKC", text)):
            token = canonical_token(match.group(0))
            if not token:
                continue
            if tokens:
                normalized_length += 1
            starts.append(normalized_length)
            tokens.append(token)
            spans.append(match.span())
            normalized_length += len(token)
        return cls(
            text=text,
            folded_text=text.casefold(),
            normalized_text=" ".join(tokens),
            normalized_starts=tuple(starts),
            raw_spans=tuple(spans),
        )

    def locate(self, identifier: str) -> dict[str, Any]:
        value = clean_identifier(identifier)
        if not value:
            return empty_match(value)

        literal_start = self.folded_text.find(value.casefold())
        if literal_start >= 0:
            literal_end = literal_start + len(value)
            return match_payload(
                value,
                "literal",
                literal_start,
                literal_end,
                evidence_window(self.text, literal_start, literal_end),
            )

        query_tokens = canonical_tokens(value)
        if not query_tokens or sum(len(token) for token in query_tokens) < 4:
            return {**empty_match(value), "status": "ambiguous_short"}
        query = " ".join(query_tokens)
        normalized_start = bounded_find(self.normalized_text, query)
        if normalized_start < 0:
            return empty_match(value)

        first_token = max(0, bisect.bisect_right(self.normalized_starts, normalized_start) - 1)
        last_position = normalized_start + len(query) - 1
        last_token = max(first_token, bisect.bisect_right(self.normalized_starts, last_position) - 1)
        raw_start = self.raw_spans[first_token][0]
        raw_end = self.raw_spans[min(last_token, len(self.raw_spans) - 1)][1]
        return match_payload(
            value,
            "normalized",
            raw_start,
            raw_end,
            evidence_window(self.text, raw_start, raw_end),
        )


def canonical_token(token: str) -> str:
    value = unicodedata.normalize("NFKC", token).casefold().strip("_")
    if value in VERSUS_TOKENS:
        return "v"
    return value


def canonical_tokens(text: str) -> list[str]:
    return [canonical_token(match.group(0)) for match in WORD_RE.finditer(text) if match.group(0)]


def clean_identifier(value: Any) -> str:
    text = " ".join(str(value or "").split()).strip()
    return "" if text.casefold() in PLACEHOLDERS else text


def bounded_find(haystack: str, needle: str) -> int:
    padded_haystack = f" {haystack} "
    position = padded_haystack.find(f" {needle} ")
    return position - 1 if position >= 0 else -1


def evidence_window(text: str, start: int, end: int, radius: int = 140) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return " ".join(text[left:right].split())


def empty_match(value: str) -> dict[str, Any]:
    return {
        "value": value,
        "status": "empty" if not value else "absent",
        "method": "none",
        "start": None,
        "end": None,
        "evidence": "",
    }


def match_payload(
    value: str, method: str, start: int, end: int, evidence: str
) -> dict[str, Any]:
    return {
        "value": value,
        "status": "present",
        "method": method,
        "start": start,
        "end": end,
        "evidence": evidence,
    }


def audit_precedent_rule(
    *, case_id: str, rule_index: int, rule: dict[str, Any], judgment: JudgmentIndex
) -> dict[str, Any]:
    identifiers = {
        field: judgment.locate(rule.get(field, ""))
        for field in ("case_name", "citation", "reference")
    }
    populated = [field for field, result in identifiers.items() if result["status"] != "empty"]
    absent = [field for field in populated if identifiers[field]["status"] == "absent"]
    ambiguous = [field for field in populated if identifiers[field]["status"] == "ambiguous_short"]
    present = [field for field in populated if identifiers[field]["status"] == "present"]

    if not populated:
        status = "no_identifier"
    elif absent or ambiguous:
        status = "partial" if present else "unsupported"
    else:
        status = "supported"

    operations = []
    for field in absent:
        operations.append(
            {
                "op": "replace",
                "path": f"/firac/rules/{rule_index}/{field}",
                "old_value": identifiers[field]["value"],
                "value": "",
                "reason": "Identifier not found in the source judgment after conservative normalization.",
                "confidence": "pending_review",
            }
        )

    return {
        "audit_id": f"{case_id}::rule::{rule_index}",
        "case_id": case_id,
        "rule_index": rule_index,
        "rule_type": str(rule.get("type") or ""),
        "original_rule": rule,
        "identifiers": identifiers,
        "audit_status": status,
        "requires_review": bool(absent or ambiguous),
        "review_reasons": [
            *[f"{field}:absent" for field in absent],
            *[f"{field}:ambiguous_short" for field in ambiguous],
        ],
        "proposed_patch": {
            "status": "pending_review",
            "operations": operations,
        },
    }
