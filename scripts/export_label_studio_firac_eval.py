#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Export FIRAC annotations as Label Studio tasks.")
    parser.add_argument("--annotations", required=True, help="final.jsonl or directory containing case JSON files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-reference-summary", action="store_true")
    args = parser.parse_args()

    rows = load_annotations(Path(args.annotations))
    if args.limit is not None:
        rows = rows[: args.limit]
    tasks = [build_task(idx + 1, row, args.include_reference_summary) for idx, row in enumerate(rows)]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(tasks)} Label Studio tasks to {out}")


def load_annotations(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        files = sorted((path / "cases").glob("*.json")) if (path / "cases").exists() else sorted(path.glob("*.json"))
        return [json.loads(file.read_text(encoding="utf-8")) for file in files]
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_task(task_id: int, row: dict[str, Any], include_reference_summary: bool) -> dict[str, Any]:
    firac = row.get("firac") or {}
    data = {
        "case_id": row.get("case_id", ""),
        "source_dataset": row.get("source_dataset", ""),
        "split": row.get("split", ""),
        "sample_role": row.get("human_eval_sample_role", ""),
        "outcome_label": row.get("outcome_label", ""),
        "judgment_html": render_judgment(row, include_reference_summary),
        "structured_html": render_structured_firac(firac),
    }
    return {"id": task_id, "data": data}


def render_judgment(row: dict[str, Any], include_reference_summary: bool) -> str:
    parts = [
        f"<div class='case-id'><strong>Case ID:</strong> {escape(row.get('case_id', ''))}</div>",
        paragraphs(row.get("judgment_text", "")),
    ]
    if include_reference_summary and row.get("reference_summary"):
        parts.append("<hr/><h3>Reference Summary</h3>")
        parts.append(paragraphs(row.get("reference_summary", "")))
    return "\n".join(parts)


def render_structured_firac(firac: dict[str, Any]) -> str:
    return "\n".join(
        [
            "<h3>Facts</h3>",
            render_list(firac.get("facts", [])),
            "<h3>Issues</h3>",
            render_issues(firac.get("issues", [])),
            "<h3>Rules / Authorities</h3>",
            render_rules(firac.get("rules", [])),
            "<h3>Analysis</h3>",
            render_analysis(firac.get("analysis", [])),
            "<h3>Conclusion</h3>",
            render_conclusion(firac.get("conclusion", {})),
        ]
    )


def render_issues(value: Any) -> str:
    if not isinstance(value, list):
        return paragraphs(value)
    items = []
    for item in value:
        if isinstance(item, dict):
            question = item.get("question") or item.get("issue") or item
            detail = item.get("why_decisive") or item.get("source_basis") or ""
            text = f"<strong>{escape(question)}</strong>"
            if detail:
                text += f"<br/><span>{escape(detail)}</span>"
            items.append(text)
        else:
            items.append(escape(item))
    return "<ol>" + "".join(f"<li>{item}</li>" for item in items) + "</ol>"


def render_rules(value: Any) -> str:
    if not isinstance(value, list):
        return paragraphs(value)
    authorities = []
    for item in value:
        if not isinstance(item, dict):
            authorities.append(f"<li>{escape(item)}</li>")
            continue
        authority_type = str(item.get("type", "") or "").strip()
        reference = str(item.get("reference", item.get("name", "")) or "").strip()
        case_name = str(item.get("case_name", "") or "").strip()
        title = reference or case_name or "Unnamed authority"
        heading = f"<strong>{escape(humanize(authority_type))}: {escape(title)}</strong>"

        attributes: list[tuple[str, Any]] = []
        if authority_type == "precedent":
            if case_name and case_name != reference:
                attributes.append(("Case", case_name))
            attributes.append(("Citation", item.get("citation", "")))
            treatment = item.get("treatment", item.get("treatment_type", ""))
            if str(treatment or "").strip().casefold() not in {"", "not applicable", "n/a"}:
                attributes.append(("Treatment", treatment))
        elif authority_type in {"statute", "rule", "regulation", "constitutional_provision"}:
            attributes.append(("Provision", item.get("provision", "")))
        else:
            # Unknown/other authority types can still carry a useful provision or citation.
            attributes.append(("Provision", item.get("provision", "")))
            attributes.append(("Citation", item.get("citation", "")))
        attributes.append(("Relevance", item.get("relevance", "")))

        details = "".join(
            f"<li><strong>{escape(label)}:</strong> {escape(content)}</li>"
            for label, content in attributes
            if str(content or "").strip()
        )
        nested = f"<ul>{details}</ul>" if details else ""
        authorities.append(f"<li>{heading}{nested}</li>")
    return f"<ol class='authority-list'>{''.join(authorities)}</ol>"


def humanize(value: str) -> str:
    return value.replace("_", " ").strip().title() or "Authority"


def render_analysis(value: Any) -> str:
    if not isinstance(value, list):
        return paragraphs(value)
    sections = []
    for item in value:
        if not isinstance(item, dict):
            sections.append(f"<li>{escape(item)}</li>")
            continue
        issue = item.get("issue", "")
        steps = item.get("steps", [])
        conclusion = item.get("conclusion", "")
        block = []
        if issue:
            block.append(f"<h4>{escape(issue)}</h4>")
        block.append(render_list(steps))
        if conclusion:
            block.append(f"<p><strong>Issue conclusion:</strong> {escape(conclusion)}</p>")
        sections.append("<section>" + "\n".join(block) + "</section>")
    return "\n".join(sections)


def render_conclusion(value: Any) -> str:
    if isinstance(value, dict):
        lines = []
        for key in ("label", "simplified_label", "text"):
            if value.get(key):
                lines.append(f"<p><strong>{escape(key)}:</strong> {escape(value.get(key))}</p>")
        return "\n".join(lines)
    return paragraphs(value)


def render_list(value: Any) -> str:
    if not isinstance(value, list):
        return paragraphs(value)
    return "<ol>" + "".join(f"<li>{escape(item)}</li>" for item in value) + "</ol>"


def paragraphs(value: Any) -> str:
    text = str(value or "")
    chunks = [chunk.strip() for chunk in text.splitlines() if chunk.strip()]
    if not chunks:
        return "<p></p>"
    return "\n".join(f"<p>{escape(chunk)}</p>" for chunk in chunks)


def escape(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


if __name__ == "__main__":
    main()
