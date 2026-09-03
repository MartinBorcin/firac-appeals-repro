#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from firac_appeals.evaluation.downstream import rule_items, text_items
from firac_appeals.utils.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build predicted-issue and retrieved-candidate authority tasks.")
    parser.add_argument("--base", required=True, help="Gold rule_recommendation task JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--predicted-issues", default="")
    parser.add_argument("--retrieved-candidates", default="")
    args = parser.parse_args()

    rows = read_jsonl(args.base)
    issues = by_case(args.predicted_issues)
    candidates = by_case(args.retrieved_candidates)
    output = []
    for row in rows:
        case_id = str(row.get("case_id", ""))
        rebuilt = json.loads(json.dumps(row))
        facts = text_items((rebuilt.get("input") or {}).get("facts"))
        issue_values = text_items(issues.get(case_id, {}).get("issues")) if issues else text_items(
            (rebuilt.get("input") or {}).get("issues")
        )
        input_text = "Facts:\n" + "\n".join(f"- {fact}" for fact in facts)
        if issue_values:
            input_text += "\n\nIssues:\n" + "\n".join(f"- {issue}" for issue in issue_values)
        rebuilt["input"]["issues"] = issue_values
        if candidates:
            rule_values = rule_items(candidates.get(case_id, {}).get("rules"))
            rebuilt["input"]["candidate_rules"] = rule_values
            input_text += "\n\nCandidate authorities (select only from this list):\n"
            input_text += json.dumps(rule_values, ensure_ascii=False)
        rebuilt["input"]["input_text"] = input_text
        output.append(rebuilt)
    write_jsonl(args.output, output)
    print(f"Wrote {len(output)} authority ablation tasks to {args.output}")


def by_case(path: str) -> dict[str, dict]:
    if not path:
        return {}
    return {str(row.get("case_id", "")): row for row in read_jsonl(path)}


if __name__ == "__main__":
    main()
