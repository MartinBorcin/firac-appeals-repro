from __future__ import annotations

import ast
import json
import re
from typing import Any


def parse_jsonish(text: str) -> dict[str, Any]:
    """Parse model output that should be a JSON/Python-dict object."""
    cleaned = strip_code_fences(text).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        candidate = match.group(0)
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            value = ast.literal_eval(candidate)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|python)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text
