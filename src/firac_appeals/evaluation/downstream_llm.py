from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from firac_appeals.annotation.client import AsyncScheduler, LLMRequest, build_client
from firac_appeals.annotation.json_parse import parse_jsonish
from firac_appeals.annotation.prompts import PromptStore
from firac_appeals.annotation.pipeline import build_retry_prompt, configure_logger
from firac_appeals.utils.io import append_jsonl, ensure_parent


class OutcomePrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    rationale: str = ""

    @field_validator("label")
    @classmethod
    def valid_label(cls, value: str) -> str:
        value = value.lower().strip()
        if value not in {"allowed", "dismissed", "other"}:
            raise ValueError("label must be one of allowed, dismissed, other")
        return value


class IssuePrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issues: list[str] = Field(min_length=1, max_length=5)

    @field_validator("issues")
    @classmethod
    def clean_issues(cls, value: list[str]) -> list[str]:
        cleaned = [issue.strip() for issue in value if str(issue or "").strip()]
        if not cleaned:
            raise ValueError("issues must contain at least one non-empty item")
        return cleaned[:5]


RuleType = Literal[
    "statute",
    "rule",
    "regulation",
    "constitutional_provision",
    "precedent",
    "legal_principle",
    "other",
]


RULE_TYPE_ALIASES = {
    "act": "statute",
    "acts": "statute",
    "constitutional": "constitutional_provision",
    "constitutional_rule": "constitutional_provision",
    "constitutional_provisions": "constitutional_provision",
    "judgment": "precedent",
    "case": "precedent",
    "case_law": "precedent",
    "legal principle": "legal_principle",
    "legal_principles": "legal_principle",
    "principle": "legal_principle",
    "procedural": "rule",
    "procedural_rule": "rule",
    "procedural_rules": "rule",
}


class RuleItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: RuleType
    reference: str
    provision: str = ""
    case_name: str = ""
    citation: str = ""
    relevance: str = ""

    @model_validator(mode="before")
    @classmethod
    def populate_reference(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            for key in ("reference", "provision", "case_name", "citation", "relevance"):
                if data.get(key) is None:
                    data[key] = ""
            rule_type = str(data.get("type", "") or "").strip().lower().replace("-", "_")
            rule_type = "_".join(rule_type.split())
            if rule_type:
                data["type"] = RULE_TYPE_ALIASES.get(rule_type, rule_type)
        if isinstance(data, dict) and not str(data.get("reference", "") or "").strip():
            reference = (
                data.get("case_name")
                or data.get("citation")
                or data.get("provision")
                or data.get("name")
                or data.get("authority")
                or ""
            )
            if reference:
                data = dict(data)
                data["reference"] = reference
        return data

    @field_validator("reference")
    @classmethod
    def reference_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("reference must not be empty")
        return value


class RulePrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rules: list[RuleItem] = Field(default_factory=list, max_length=10)

    @field_validator("rules")
    @classmethod
    def require_rules(cls, value: list[RuleItem]) -> list[RuleItem]:
        if not value:
            raise ValueError("rules must contain at least one item")
        return value


SCHEMAS = {
    "outcome_prediction": OutcomePrediction,
    "issue_generation": IssuePrediction,
    "rule_recommendation": RulePrediction,
}


class DownstreamLLMRunner:
    def __init__(self, task: str, config: dict[str, Any], output_dir: str | Path) -> None:
        self.task = task
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = configure_logger(
            self.output_dir / f"{task}.log",
            config.get("log_level", "INFO"),
        )
        self.write_case_files = bool(config.get("output", {}).get("write_case_files", True))
        self.save_intermediates = bool(config.get("output", {}).get("save_intermediates", False))
        self.client = build_client(config["backend"])
        self.scheduler = AsyncScheduler(self.client, config.get("scheduler", {}), logger=self.logger)
        self.max_validation_retries = int(config.get("scheduler", {}).get("max_validation_retries", 1))
        self.max_tokens = int(config.get("max_tokens", config["backend"].get("max_tokens", 2048)))
        self.task_max_tokens = {
            str(key): int(value)
            for key, value in (config.get("task_max_tokens") or {}).items()
        }
        self.overwrite = bool(config.get("overwrite", False))
        self.prompt_store = PromptStore(config["prompts"])

    async def run_many(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        max_case_concurrency = int(
            self.config.get("scheduler", {}).get(
                "max_case_concurrency",
                self.config.get("scheduler", {}).get("max_in_flight_requests", 4),
            )
        )
        semaphore = asyncio.Semaphore(max_case_concurrency)
        final_path = self.output_dir / "predictions.jsonl"
        results: list[dict[str, Any]] = []

        async def guarded(row: dict[str, Any]) -> dict[str, Any]:
            try:
                async with semaphore:
                    return await self.run_one(row)
            except Exception as exc:  # noqa: BLE001
                case_id = str(row.get("case_id", ""))
                self.logger.exception("case_error case_id=%s task=%s error=%r", case_id, self.task, exc)
                result = {
                    "case_id": case_id,
                    "task": self.task,
                    "prediction_status": "error",
                    "prediction_error": repr(exc),
                }
                ensure_parent(self.output_dir / "errors" / f"{case_id}.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                if not self.config.get("continue_on_error", True):
                    raise
                return result

        tasks = [asyncio.create_task(guarded(row)) for row in rows]
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            if result.get("prediction_status") == "ok" and not result.pop("_skipped_existing", False):
                append_jsonl(final_path, result)
        return results

    async def run_one(self, row: dict[str, Any]) -> dict[str, Any]:
        case_id = str(row.get("case_id", ""))
        done_path = self.output_dir / "cases" / f"{case_id}.json"
        if done_path.exists() and not self.overwrite:
            self.logger.info("case_skip_done case_id=%s", case_id)
            return json.loads(done_path.read_text(encoding="utf-8")) | {"_skipped_existing": True}
        prompt = build_task_prompt(self.prompt_store, self.task, row)
        prediction = await self.generate_validated(case_id, prompt)
        result = {
            "case_id": case_id,
            "task": self.task,
            "prediction_status": "ok",
            "model": self.config["backend"].get("model", ""),
            **prediction,
        }
        if self.write_case_files:
            ensure_parent(done_path).write_text(
                json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        self.logger.info("case_done case_id=%s", case_id)
        return result

    async def generate_validated(self, case_id: str, prompt: str) -> dict[str, Any]:
        schema = SCHEMAS[self.task]
        last_error = ""
        for attempt in range(self.max_validation_retries + 1):
            stage = self.task if attempt == 0 else f"{self.task}.retry{attempt}"
            self.logger.info(
                "stage_request case_id=%s stage=%s attempt=%d prompt_chars=%d",
                case_id,
                stage,
                attempt,
                len(prompt),
            )
            started = time.monotonic()
            max_tokens = self.task_max_tokens.get(self.task, self.max_tokens)
            response = await self.scheduler.run_one(LLMRequest(case_id, stage, prompt, max_tokens=max_tokens))
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
            save_raw(self.output_dir, case_id, stage, response.text, self.save_intermediates)
            try:
                parsed = parse_jsonish(response.text)
                return schema.model_validate(parsed).model_dump(mode="json")
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
        raise ValueError(f"{self.task} did not produce valid structured output after retries: {last_error}")


def build_task_prompt(prompt_store: PromptStore, task: str, row: dict[str, Any]) -> str:
    input_text = str((row.get("input") or {}).get("input_text", ""))
    return prompt_store.render(task, input_text=input_text)


def save_raw(output_dir: Path, case_id: str, stage: str, text: str, enabled: bool) -> None:
    if not enabled:
        return
    ensure_parent(output_dir / "intermediates" / case_id / f"{stage}.txt").write_text(text, encoding="utf-8")
