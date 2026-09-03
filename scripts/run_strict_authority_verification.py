#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI

from firac_appeals.annotation.json_parse import parse_jsonish
from firac_appeals.evaluation.strict_authorities import (
    IDENTIFIER_FIELDS,
    patch_operations,
    validate_rule_output,
)
from firac_appeals.utils.io import append_jsonl, ensure_parent, write_jsonl


SYSTEM_PROMPT = (
    "Verify precedent identifiers using only the supplied judgment. "
    "Return one compact JSON object and no commentary."
)
CANDIDATE_STATUSES = {"absent", "ambiguous_short"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a source-grounded strict-authority audit and optional patch."
    )
    parser.add_argument("--audit-input", required=True, help="Full deterministic Rule audit JSONL")
    parser.add_argument("--audit-output", required=True, help="Full verifier ledger JSONL")
    parser.add_argument("--patch-output", required=True, help="Actionable identifier patch JSONL")
    parser.add_argument("--model", default="gpt-oss-120b")
    parser.add_argument(
        "--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-judgment-chars", type=int, default=450_000)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--include-case-id", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-unresolved", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args))


async def run(args: argparse.Namespace) -> None:
    groups = load_candidate_groups(args.audit_input)
    groups = select_groups(groups, args.include_case_id, args.max_cases)
    output_path = ensure_parent(args.audit_output)
    if args.overwrite:
        output_path.write_text("", encoding="utf-8")
    completed = completed_audit_ids(output_path, args.retry_unresolved)

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise ValueError(f"Missing API key environment variable: {args.api_key_env}")
    http_client = httpx.AsyncClient(timeout=240, trust_env=False)
    client = AsyncOpenAI(base_url=args.base_url, api_key=api_key, http_client=http_client, timeout=240)
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    counters = defaultdict(int)
    started = time.monotonic()

    async def process(group: dict[str, Any], judgment: str) -> None:
        needed = [row for row in group["rules"] if row["audit_id"] not in completed]
        if not needed:
            counters["skipped_rules"] += len(group["rules"])
            return
        if len(judgment) > args.max_judgment_chars:
            results = [unresolved_result(row, args.model, "judgment_exceeds_context_limit") for row in needed]
        else:
            async with semaphore:
                results = await verify_group(client, needed, judgment, args.model)
        async with write_lock:
            for result in results:
                append_jsonl(output_path, result)
                counters[result["verification_status"]] += 1
            counters["processed_cases"] += 1
            if counters["processed_cases"] % 100 == 0:
                elapsed = max(time.monotonic() - started, 0.001)
                print(
                    json.dumps(
                        {
                            "processed_cases": counters["processed_cases"],
                            "verified_rules": counters["verified"],
                            "unresolved_rules": counters["unresolved"],
                            "cases_per_second": round(counters["processed_cases"] / elapsed, 3),
                        }
                    ),
                    flush=True,
                )

    pending: set[asyncio.Task[None]] = set()
    for source_file, line_groups in groups_by_source(groups).items():
        remaining_lines = set(line_groups)
        with Path(source_file).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line_number not in remaining_lines:
                    continue
                row = json.loads(line)
                judgment = str(row.get("judgment_text") or "")
                pending.add(asyncio.create_task(process(line_groups[line_number], judgment)))
                remaining_lines.remove(line_number)
                if len(pending) >= args.concurrency * 2:
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        task.result()
                if not remaining_lines:
                    break
        if remaining_lines:
            raise ValueError(f"Missing {len(remaining_lines)} source rows in {source_file}")
    if pending:
        await asyncio.gather(*pending)
    await http_client.aclose()

    compact_audit(output_path)
    build_patch(output_path, args.patch_output)
    elapsed = time.monotonic() - started
    print(
        json.dumps(
            {
                "candidate_cases": len(groups),
                "candidate_rules": sum(len(group["rules"]) for group in groups.values()),
                "elapsed_seconds": elapsed,
                "audit_output": str(output_path),
                "patch_output": args.patch_output,
                "counters": dict(counters),
            },
            indent=2,
        )
    )


def load_candidate_groups(path: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for line in Path(path).open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        identifiers = row["identifiers"]
        if not any(
            identifiers[field]["status"] in CANDIDATE_STATUSES
            for field in ("case_name", "citation")
        ):
            continue
        case_id = row["case_id"]
        group = groups.setdefault(
            case_id,
            {
                "case_id": case_id,
                "source_file": row["source_file"],
                "source_line": int(row["source_line"]),
                "rules": [],
            },
        )
        if (group["source_file"], group["source_line"]) != (
            row["source_file"],
            int(row["source_line"]),
        ):
            raise ValueError(f"Inconsistent source location for {case_id}")
        group["rules"].append(row)
    for group in groups.values():
        group["rules"].sort(key=lambda row: int(row["rule_index"]))
    return groups


def select_groups(
    groups: dict[str, dict[str, Any]], include_case_ids: list[str], max_cases: int | None
) -> dict[str, dict[str, Any]]:
    if include_case_ids:
        missing = sorted(set(include_case_ids) - set(groups))
        if missing:
            raise ValueError(f"Requested case IDs are not candidates: {missing}")
        selected = {case_id: groups[case_id] for case_id in include_case_ids}
    else:
        selected = dict(sorted(groups.items()))
    if max_cases is not None:
        selected = dict(list(selected.items())[:max_cases])
    return selected


def groups_by_source(groups: dict[str, dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    output: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for group in groups.values():
        source_file = group["source_file"]
        source_line = group["source_line"]
        if source_line in output[source_file]:
            raise ValueError(f"Duplicate source line {source_file}:{source_line}")
        output[source_file][source_line] = group
    return output


def completed_audit_ids(path: Path, retry_unresolved: bool) -> set[str]:
    if not path.exists():
        return set()
    latest = {
        row["audit_id"]: row
        for row in (json.loads(line) for line in path.open() if line.strip())
    }
    return {
        audit_id
        for audit_id, row in latest.items()
        if not (retry_unresolved and row.get("verification_status") == "unresolved")
    }


async def verify_group(
    client: AsyncOpenAI, rules: list[dict[str, Any]], judgment: str, model: str
) -> list[dict[str, Any]]:
    judgment_lines = judgment.splitlines() or [judgment]
    prompt = build_prompt(rules, judgment_lines)
    previous_response = ""
    validation_error = ""
    for attempt in range(2):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        if attempt and previous_response:
            messages.extend(
                [
                    {"role": "assistant", "content": previous_response},
                    {
                        "role": "user",
                        "content": (
                            "The previous JSON failed validation: "
                            f"{validation_error}. Return corrected JSON. Use valid JUDGMENT line "
                            "numbers for supported fields; otherwise remove the field."
                        ),
                    },
                ]
            )
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                seed=20260817,
                max_tokens=min(7000, 500 + 450 * len(rules)),
                response_format={"type": "json_object"},
                extra_body=(
                    {"reasoning_effort": "low"}
                    if model == "gpt-oss-120b"
                    else {"chat_template_kwargs": {"enable_thinking": False}}
                ),
            )
            previous_response = response.choices[0].message.content or ""
            parsed = parse_jsonish(previous_response)
            parsed_rules = parsed.get("rules") if isinstance(parsed, dict) else None
            if not isinstance(parsed_rules, list):
                raise ValueError("top-level rules is not a list")
            by_index = {}
            for item in parsed_rules:
                if isinstance(item, dict):
                    try:
                        by_index[int(item.get("rule_index"))] = item
                    except (TypeError, ValueError):
                        continue
            results = []
            all_errors = []
            for row in rules:
                rule_index = int(row["rule_index"])
                fields, errors = validate_rule_output(
                    by_index.get(rule_index),
                    judgment=judgment,
                    expected_rule_index=rule_index,
                    original_rule=row["original_rule"],
                    judgment_lines=judgment_lines,
                )
                if errors:
                    all_errors.extend(f"rule {rule_index}: {error}" for error in errors)
                results.append(verified_result(row, model, fields, errors, attempt + 1))
            if not all_errors:
                return results
            validation_error = "; ".join(all_errors[:8])
            if attempt == 1:
                return results
        except Exception as exc:  # noqa: BLE001
            validation_error = repr(exc)
            if attempt == 1:
                return [unresolved_result(row, model, validation_error, attempts=2) for row in rules]
            await asyncio.sleep(1)
    raise AssertionError("unreachable")


def build_prompt(rules: list[dict[str, Any]], judgment_lines: list[str]) -> str:
    candidates = [
        {
            "rule_index": row["rule_index"],
            **{field: str(row["original_rule"].get(field) or "") for field in IDENTIFIER_FIELDS},
        }
        for row in rules
    ]
    numbered_judgment = "\n".join(
        f"[L{line_number:06d}] {line}" for line_number, line in enumerate(judgment_lines, 1)
    )
    return f"""Create a source-grounded version of each supplied precedent identifier.

For reference, case_name, and citation independently:
- Choose keep when the supplied value is supported modulo punctuation, shorthand, abbreviation, or OCR noise.
- For keep, repeat the supplied value unchanged and provide source evidence.
- Choose remove when the field is not explicitly supported; return empty value and evidence.
- Choose correct only for a genuine material misrepresentation, not cosmetic or OCR differences.
- Do not replace a clean normalized value with an OCR-corrupted source spelling.
- OCR variants such as Madhya/Madbya, Bom./Born., SC/S.C./&C., and 1/I are KEEP decisions.
- A missing or damaged reporter abbreviation is also KEEP when the surrounding title and citation numbers match.
- A distinctive party-name shorthand such as "Barendra Kumar Ghosh's case" may KEEP the supplied full case title when it clearly denotes that precedent.
- Prefer keep whenever the source plausibly contains the same identifier with only cosmetic or OCR noise.
- For keep or correct, return the inclusive source line range containing the identifier.
- Use at most ten lines. For remove, return line numbers 0 and 0.
- Cosmetic differences may include punctuation, whitespace, v/vs/versus, abbreviations, and obvious OCR noise.
- Never infer a citation from a case name or a case name from a citation.
- Never add or change material identifiers such as years, volumes, page numbers, docket numbers, or party identities.
- If a field is not explicitly supported, return an empty value and empty evidence.
- Do not add an identifier where the supplied field is already empty.
- Do not change any fields other than reference, case_name, and citation.

Field meanings:
- reference: generic human-readable authority label; for a precedent it normally repeats or abbreviates the case name.
- case_name: structured precedent name.
- citation: reporter, docket, or formal case identifier.

Return only compact JSON in this form:
{{"rules":[{{"rule_index":0,"reference":{{"decision":"keep|remove|correct","value":"","evidence_line_start":1,"evidence_line_end":1}},"case_name":{{"decision":"keep|remove|correct","value":"","evidence_line_start":1,"evidence_line_end":1}},"citation":{{"decision":"keep|remove|correct","value":"","evidence_line_start":0,"evidence_line_end":0}}}}]}}

PRECEDENT RULES:
{json.dumps(candidates, ensure_ascii=False)}

JUDGMENT WITH SOURCE LINE IDS:
{numbered_judgment}
"""


def verified_result(
    row: dict[str, Any],
    model: str,
    fields: dict[str, Any],
    errors: list[str],
    attempts: int,
) -> dict[str, Any]:
    status = "unresolved" if errors else "verified"
    operations = [] if errors else patch_operations(
        rule_index=int(row["rule_index"]),
        original_rule=row["original_rule"],
        corrected_fields=fields,
    )
    return {
        **result_identity(row),
        "model": model,
        "verification_status": status,
        "attempts": attempts,
        "original_identifiers": {
            field: str(row["original_rule"].get(field) or "") for field in IDENTIFIER_FIELDS
        },
        "corrected_identifiers": fields,
        "validation_errors": errors,
        "operations": operations,
    }


def unresolved_result(
    row: dict[str, Any], model: str, error: str, attempts: int = 0
) -> dict[str, Any]:
    return {
        **result_identity(row),
        "model": model,
        "verification_status": "unresolved",
        "attempts": attempts,
        "original_identifiers": {
            field: str(row["original_rule"].get(field) or "") for field in IDENTIFIER_FIELDS
        },
        "corrected_identifiers": {},
        "validation_errors": [error],
        "operations": [],
    }


def result_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "audit_id": row["audit_id"],
        "case_id": row["case_id"],
        "source_dataset": row.get("source_dataset", ""),
        "source_file": row["source_file"],
        "source_line": int(row["source_line"]),
        "rule_index": int(row["rule_index"]),
    }


def compact_audit(path: Path) -> None:
    latest = {
        row["audit_id"]: row
        for row in (json.loads(line) for line in path.open() if line.strip())
    }
    write_jsonl(path, [latest[audit_id] for audit_id in sorted(latest)])


def build_patch(audit_path: Path, patch_path: str) -> None:
    patches = []
    for line in audit_path.open(encoding="utf-8"):
        row = json.loads(line)
        if row["verification_status"] != "verified" or not row["operations"]:
            continue
        patches.append(
            {
                **{key: row[key] for key in (
                    "audit_id",
                    "case_id",
                    "source_dataset",
                    "source_file",
                    "source_line",
                    "rule_index",
                    "model",
                )},
                "operations": row["operations"],
            }
        )
    write_jsonl(patch_path, patches)


if __name__ == "__main__":
    main()
