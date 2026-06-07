from firac_appeals.annotation.json_parse import parse_jsonish


def test_parse_fenced_json() -> None:
    assert parse_jsonish('```json\n{"ok": true}\n```') == {"ok": True}


def test_parse_json_with_extra_text() -> None:
    assert parse_jsonish('Here is the object: {"value": 3}') == {"value": 3}
