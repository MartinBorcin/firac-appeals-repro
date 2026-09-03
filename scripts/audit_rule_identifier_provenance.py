#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from firac_appeals.evaluation.authority_provenance import JudgmentIndex, audit_precedent_rule
from firac_appeals.utils.io import ensure_parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether precedent identifiers are explicitly present in source judgments."
    )
    parser.add_argument("--input", action="append", required=True, help="Input annotation JSONL; repeatable")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=0, help="Optional total-case smoke-test limit")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = ensure_parent(output_dir / "rule_identifier_audit.jsonl")
    issue_path = ensure_parent(output_dir / "potential_issues.jsonl")
    patch_path = ensure_parent(output_dir / "proposed_patches.jsonl")
    summary_path = ensure_parent(output_dir / "summary.json")

    cases = 0
    precedent_rules = 0
    issues = 0
    counters: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    started = time.monotonic()

    with (
        audit_path.open("w", encoding="utf-8") as audit_file,
        issue_path.open("w", encoding="utf-8") as issue_file,
        patch_path.open("w", encoding="utf-8") as patch_file,
    ):
        stop = False
        for input_name in args.input:
            input_path = Path(input_name)
            with input_path.open("r", encoding="utf-8") as source_file:
                for line_number, line in enumerate(source_file, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    cases += 1
                    case_id = str(row.get("case_id") or f"{input_path.stem}:{line_number}")
                    source_dataset = str(row.get("source_dataset") or "")
                    judgment = JudgmentIndex.build(str(row.get("judgment_text") or ""))
                    for rule_index, rule in enumerate((row.get("firac") or {}).get("rules") or []):
                        if str(rule.get("type") or "").casefold() != "precedent":
                            continue
                        precedent_rules += 1
                        audit = audit_precedent_rule(
                            case_id=case_id,
                            rule_index=rule_index,
                            rule=rule,
                            judgment=judgment,
                        )
                        audit.update(
                            {
                                "source_dataset": source_dataset,
                                "split": row.get("split", ""),
                                "source_file": str(input_path),
                                "source_line": line_number,
                            }
                        )
                        write_row(audit_file, audit)
                        counters[f"audit_status:{audit['audit_status']}"] += 1
                        by_source[f"{source_dataset}:{audit['audit_status']}"] += 1
                        for field, result in audit["identifiers"].items():
                            counters[f"{field}:{result['status']}:{result['method']}"] += 1
                        if audit["requires_review"]:
                            issues += 1
                            write_row(issue_file, audit)
                            if audit["proposed_patch"]["operations"]:
                                write_row(
                                    patch_file,
                                    {
                                        "audit_id": audit["audit_id"],
                                        "case_id": case_id,
                                        "source_dataset": source_dataset,
                                        "source_file": str(input_path),
                                        "source_line": line_number,
                                        "rule_index": rule_index,
                                        **audit["proposed_patch"],
                                    },
                                )
                    if args.limit and cases >= args.limit:
                        stop = True
                        break
            if stop:
                break

    elapsed = time.monotonic() - started
    summary: dict[str, Any] = {
        "inputs": args.input,
        "cases": cases,
        "precedent_rules": precedent_rules,
        "potential_issue_rules": issues,
        "potential_issue_rate": issues / precedent_rules if precedent_rules else 0.0,
        "elapsed_seconds": elapsed,
        "rules_per_second": precedent_rules / elapsed if elapsed else 0.0,
        "counts": dict(sorted(counters.items())),
        "counts_by_source": dict(sorted(by_source.items())),
        "artifacts": {
            "audit": str(audit_path),
            "potential_issues": str(issue_path),
            "proposed_patches": str(patch_path),
        },
        "policy": {
            "source_of_truth": "judgment_text only",
            "normalization": "Unicode NFKC, case-folding, punctuation/whitespace, and v/vs/versus",
            "patches_applied": False,
            "absence_interpretation": "review flag, not proof of hallucination",
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def write_row(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
