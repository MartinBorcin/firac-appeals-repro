#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from firac_appeals.annotation.client import AsyncScheduler, LLMRequest, build_client
from firac_appeals.annotation.json_parse import parse_jsonish
from firac_appeals.annotation.pipeline import build_retry_prompt, configure_logger
from firac_appeals.annotation.prompts import PromptStore
from firac_appeals.utils.io import append_jsonl, ensure_parent, read_jsonl, read_yaml


SCORE_FIELDS = [
    "facts_faithfulness",
    "facts_coverage",
    "issues_faithfulness",
    "issues_coverage",
    "rules_faithfulness",
    "rules_coverage",
    "analysis_faithfulness",
    "analysis_coverage",
    "conclusion_faithfulness",
    "conclusion_coverage",
]

ALLOWED_FLAGS = {
    "Critical hallucination",
    "Outcome leakage in facts",
    "Wrong outcome label",
    "Missing decisive issue",
    "Missing material authority",
    "Poor source readability",
}


class ScoreModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facts_faithfulness: int
    facts_coverage: int
    issues_faithfulness: int
    issues_coverage: int
    rules_faithfulness: int
    rules_coverage: int
    analysis_faithfulness: int
    analysis_coverage: int
    conclusion_faithfulness: int
    conclusion_coverage: int

    @field_validator("*")
    @classmethod
    def score_range(cls, value: int) -> int:
        if value < 1 or value > 5:
            raise ValueError("scores must be integers in [1, 5]")
        return value


class RationaleModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    facts: str = ""
    issues: str = ""
    rules: str = ""
    analysis: str = ""
    conclusion: str = ""


class RubricJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scores: ScoreModel
    critical_flags: list[str] = Field(default_factory=list)
    rationales: RationaleModel = Field(default_factory=RationaleModel)
    reviewer_notes: str = ""

    @field_validator("critical_flags")
    @classmethod
    def valid_flags(cls, value: list[str]) -> list[str]:
        return [flag for flag in value if flag in ALLOWED_FLAGS]


class RubricEvalRunner:
    def __init__(self, config: dict[str, Any], output_dir: str | Path, prompt_path: str) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = configure_logger(self.output_dir / "llm_rubric.log", config.get("log_level", "INFO"))
        self.write_case_files = bool(config.get("output", {}).get("write_case_files", True))
        self.save_intermediates = bool(config.get("output", {}).get("save_intermediates", False))
        self.prompt_store = PromptStore({"firac_rubric_judge": prompt_path})
        self.client = build_client(config["backend"])
        self.scheduler = AsyncScheduler(self.client, config.get("scheduler", {}), logger=self.logger)
        self.max_validation_retries = int(config.get("scheduler", {}).get("max_validation_retries", 1))
        self.max_tokens = int(config.get("max_tokens", 2048))
        self.overwrite = bool(config.get("overwrite", False))

    def done_path(self, case_id: str, replicate: int) -> Path:
        suffix = f".r{replicate}" if replicate else ""
        return self.output_dir / "cases" / f"{case_id}{suffix}.json"

    def is_done(self, case_id: str, replicate: int) -> bool:
        if self.overwrite:
            return False
        return self.done_path(case_id, replicate).exists()

    async def run_one(self, row: dict[str, Any], replicate: int) -> dict[str, Any]:
        case_id = str(row.get("case_id", ""))
        if self.is_done(case_id, replicate):
            self.logger.info("case_skip_done case_id=%s replicate=%d", case_id, replicate)
            return json.loads(self.done_path(case_id, replicate).read_text(encoding="utf-8")) | {
                "_skipped_existing": True
            }
        firac_json = json.dumps(row.get("firac") or {}, ensure_ascii=False)
        prompt = self.prompt_store.render(
            "firac_rubric_judge",
            judgment=str(row.get("judgment_text", "")),
            firac_json=firac_json,
        )
        judgment = await self.generate_validated(case_id, replicate, prompt)
        result = {
            "case_id": case_id,
            "source_dataset": row.get("source_dataset", ""),
            "annotation_run_id": row.get("run_id", ""),
            "outcome_label": row.get("outcome_label", ""),
            "judge_model": self.config["backend"].get("model", ""),
            "judge_replicate": replicate,
            "llm_rubric": judgment,
        }
        if self.write_case_files:
            path = self.done_path(case_id, replicate)
            ensure_parent(path)
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.info("case_done case_id=%s replicate=%d", case_id, replicate)
        return result

    async def generate_validated(self, case_id: str, replicate: int, prompt: str) -> dict[str, Any]:
        last_error = ""
        request_stage = f"firac_rubric_judge.r{replicate}"
        for attempt in range(self.max_validation_retries + 1):
            stage = request_stage if attempt == 0 else f"{request_stage}.retry{attempt}"
            self.logger.info(
                "stage_request case_id=%s stage=%s attempt=%d prompt_chars=%d max_tokens=%d",
                case_id,
                stage,
                attempt,
                len(prompt),
                self.max_tokens,
            )
            started = time.monotonic()
            response = await self.scheduler.run_one(LLMRequest(case_id, stage, prompt, max_tokens=self.max_tokens))
            elapsed = time.monotonic() - started
            self.logger.info(
                "stage_response case_id=%s stage=%s attempt=%d response_chars=%d reasoning_chars=%d elapsed_sec=%.2f",
                case_id,
                stage,
                attempt,
                len(response.text),
                len(response.reasoning),
                elapsed,
            )
            self.save_raw(case_id, stage, response.text)
            try:
                parsed = parse_jsonish(response.text)
                return RubricJudgment.model_validate(parsed).model_dump(mode="json")
            except Exception as exc:  # noqa: BLE001
                last_error = repr(exc)
                self.logger.warning(
                    "stage_validation_error case_id=%s stage=%s attempt=%d error=%s",
                    case_id,
                    stage,
                    attempt,
                    last_error,
                )
                prompt = build_retry_prompt(prompt, response.text, last_error)
        raise ValueError(f"rubric judge did not produce valid structured output after retries: {last_error}")

    async def run_many(self, rows: list[dict[str, Any]], replicates: int) -> list[dict[str, Any]]:
        max_case_concurrency = int(
            self.config.get("scheduler", {}).get(
                "max_case_concurrency",
                self.config.get("scheduler", {}).get("max_in_flight_requests", 4),
            )
        )
        semaphore = asyncio.Semaphore(max_case_concurrency)
        final_path = self.output_dir / "final.jsonl"
        results: list[dict[str, Any]] = []

        async def guarded(row: dict[str, Any], replicate: int) -> dict[str, Any]:
            try:
                async with semaphore:
                    return await self.run_one(row, replicate)
            except Exception as exc:  # noqa: BLE001
                case_id = str(row.get("case_id", ""))
                self.logger.exception("case_error case_id=%s replicate=%d error=%r", case_id, replicate, exc)
                result = {
                    "case_id": case_id,
                    "judge_replicate": replicate,
                    "rubric_status": "error",
                    "rubric_error": repr(exc),
                }
                path = self.output_dir / "errors" / f"{case_id}.r{replicate}.json"
                ensure_parent(path)
                path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                if not self.config.get("continue_on_error", True):
                    raise
                return result

        tasks = [
            asyncio.create_task(guarded(row, replicate))
            for row in rows
            for replicate in range(replicates)
        ]
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            if result.get("llm_rubric") and not result.get("_skipped_existing"):
                append_jsonl(final_path, result)
            result.pop("_skipped_existing", None)
        return results

    def save_raw(self, case_id: str, stage: str, text: str) -> None:
        if not self.save_intermediates:
            return
        path = self.output_dir / "intermediates" / case_id / f"{stage}.txt"
        ensure_parent(path)
        path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM-as-judge FIRAC rubric scoring.")
    parser.add_argument("--input", required=True, help="Annotated final.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/evaluation/rubric_openai_compatible.yaml")
    parser.add_argument("--prompt", default="prompts/rubric/v1/firac_rubric_judge.md")
    parser.add_argument("--case-ids", default="", help="Text file with one case_id per line")
    parser.add_argument("--label-studio-tasks", default="", help="Label Studio task JSON; uses data.case_id")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    selected_ids = load_selected_case_ids(args.case_ids, args.label_studio_tasks)
    if selected_ids:
        rows_by_id = {str(row.get("case_id", "")): row for row in rows}
        missing = [case_id for case_id in selected_ids if case_id not in rows_by_id]
        if missing:
            raise ValueError(f"{len(missing)} requested case IDs are missing from input; first missing: {missing[:5]}")
        rows = [rows_by_id[case_id] for case_id in selected_ids]
    if args.limit is not None:
        rows = rows[: args.limit]
    config = read_yaml(args.config)
    if args.overwrite:
        config["overwrite"] = True
    runner = RubricEvalRunner(config, args.output_dir, args.prompt)
    results = run_sync(runner, rows, args.replicates)
    ok = sum(1 for row in results if row.get("llm_rubric"))
    print(f"LLM rubric scored {ok}/{len(results)} outputs into {args.output_dir}")


def load_selected_case_ids(case_ids_path: str, label_studio_tasks_path: str) -> list[str]:
    if case_ids_path and label_studio_tasks_path:
        raise ValueError("Use either --case-ids or --label-studio-tasks, not both")
    if case_ids_path:
        return [
            line.strip()
            for line in Path(case_ids_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if label_studio_tasks_path:
        tasks = json.loads(Path(label_studio_tasks_path).read_text(encoding="utf-8"))
        return [
            str(task.get("data", {}).get("case_id", "")).strip()
            for task in tasks
            if str(task.get("data", {}).get("case_id", "")).strip()
        ]
    return []


def run_sync(runner: RubricEvalRunner, rows: list[dict[str, Any]], replicates: int) -> list[dict[str, Any]]:
    workers = int(os.environ.get("FIRAC_LLM_THREAD_POOL_WORKERS", "").strip() or 0)
    if workers <= 0:
        workers = int(runner.config.get("scheduler", {}).get("thread_pool_workers", 0) or 0)
    if workers <= 0:
        return asyncio.run(runner.run_many(rows, replicates))

    async def run_with_executor() -> list[dict[str, Any]]:
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="firac-rubric")
        loop.set_default_executor(executor)
        try:
            return await runner.run_many(rows, replicates)
        finally:
            executor.shutdown(wait=True, cancel_futures=False)

    return asyncio.run(run_with_executor())


if __name__ == "__main__":
    main()
