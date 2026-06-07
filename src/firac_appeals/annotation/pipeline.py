from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from firac_appeals.annotation.client import AsyncScheduler, LLMRequest, build_client
from firac_appeals.annotation.json_parse import parse_jsonish
from firac_appeals.annotation.prompts import PromptStore
from firac_appeals.schema.models import canonical_outcome
from firac_appeals.schema.validation import validate_stage
from firac_appeals.utils.io import append_jsonl, ensure_parent


class FiracPipeline:
    def __init__(self, config: dict[str, Any], output_dir: str | Path, root: str | Path = ".") -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = configure_logger(
            self.output_dir / "annotation.log",
            self.config.get("output", {}).get("log_level", "INFO"),
        )
        self.prompt_store = PromptStore(config["prompts"], root=root)
        self.client = build_client(config["backend"])
        self.scheduler = AsyncScheduler(self.client, config.get("scheduler", {}), logger=self.logger)
        self.overwrite = bool(config.get("run", {}).get("overwrite", False))
        output_config = config.setdefault("output", {})
        self.save_intermediates = bool(output_config.get("save_intermediates", False))
        self.save_prompts = bool(output_config.get("save_prompts", False))
        self.write_case_files = bool(output_config.get("write_case_files", True))
        self.max_validation_retries = int(config.get("scheduler", {}).get("max_validation_retries", 1))
        self.stage_max_tokens = config.get("stage_max_tokens", {})
        self.stages = config.get("stages", {})
        self.logger.info(
            "Initialized FIRAC pipeline output_dir=%s overwrite=%s write_case_files=%s",
            self.output_dir,
            self.overwrite,
            self.write_case_files,
        )

    def done_path(self, case_id: str) -> Path:
        return self.output_dir / "cases" / f"{case_id}.json"

    def is_done(self, case_id: str) -> bool:
        if self.overwrite:
            return False
        path = self.done_path(case_id)
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return bool(data.get("firac"))
        except Exception:  # noqa: BLE001
            return False

    async def annotate_case(self, case: dict[str, Any]) -> dict[str, Any]:
        case_id = case["case_id"]
        judgment = case.get("judgment_text", "")
        self.logger.info("case_start case_id=%s judgment_chars=%d", case_id, len(judgment))

        metadata = {}
        if self.stage_enabled("metadata_ner"):
            metadata = await self._json_stage("metadata_ner", case_id, judgment=metadata_view(judgment))
        issues = await self._json_stage("issue_analyst", case_id, judgment=judgment)
        researcher = await self._json_stage("legal_researcher", case_id, judgment=judgment)
        logic = await self._json_stage(
            "logician",
            case_id,
            judgment=judgment,
            issues=json.dumps(issues, ensure_ascii=False),
            rules=json.dumps(researcher, ensure_ascii=False),
        )
        firac = await self._json_stage(
            "firac_judge",
            case_id,
            judgment=judgment,
            issue_analysis=json.dumps(issues, ensure_ascii=False),
            researcher_analysis=json.dumps(researcher, ensure_ascii=False),
            logic_analysis=json.dumps(logic, ensure_ascii=False),
        )
        if "citations_extraction" in self.config.get("prompts", {}):
            citations = await self._json_stage("citations_extraction", case_id, text=researcher)
        else:
            citations = extract_citations_from_firac(firac)
        verifier = {}
        if "firac_verifier" in self.config.get("prompts", {}) and self.stage_enabled("firac_verifier"):
            verifier = await self._json_stage(
                "firac_verifier",
                case_id,
                judgment=judgment,
                firac_json=json.dumps(firac, ensure_ascii=False),
            )
        conclusion = get_conclusion_label(firac)
        outcome_label = get_simplified_label(firac) or canonical_outcome(str(conclusion))

        result = {
            **case,
            "metadata_llm": metadata,
            "firac": normalize_firac_keys(firac),
            "citations": citations,
            "outcome_label": outcome_label,
            "verification": verifier,
            "intermediates": {
                "issue_analysis": issues,
                "researcher_analysis": researcher,
                "logic_analysis": logic,
            },
            "annotation_status": "ok",
        }
        self._save_case_result(case_id, result)
        self.logger.info("case_done case_id=%s outcome=%s", case_id, result["outcome_label"])
        return result

    def stage_enabled(self, stage: str) -> bool:
        return bool(self.stages.get(stage, True))

    async def _text_stage(self, stage: str, case_id: str, **values: str) -> str:
        prompt = self.prompt_store.render(stage, **values)
        self._save_prompt(case_id, stage, prompt)
        self.logger.debug(
            "stage_request case_id=%s stage=%s prompt_chars=%d max_tokens=%s",
            case_id,
            stage,
            len(prompt),
            self.stage_max_tokens.get(stage),
        )
        started = time.monotonic()
        response = await self.scheduler.run_one(
            LLMRequest(case_id, stage, prompt, max_tokens=self.stage_max_tokens.get(stage))
        )
        elapsed = time.monotonic() - started
        self.logger.debug(
            "stage_response case_id=%s stage=%s response_chars=%d elapsed_sec=%.2f",
            case_id,
            stage,
            len(response.text),
            elapsed,
        )
        self._save_raw(case_id, stage, response.text)
        return response.text

    async def _json_stage(self, stage: str, case_id: str, **values: str) -> dict[str, Any]:
        prompt = self.prompt_store.render(stage, **values)
        self._save_prompt(case_id, stage, prompt)
        last_error = ""
        for attempt in range(self.max_validation_retries + 1):
            stage_name = stage if attempt == 0 else f"{stage}.retry{attempt}"
            max_tokens = self.stage_max_tokens.get(stage)
            self.logger.info(
                "stage_request case_id=%s stage=%s attempt=%d prompt_chars=%d max_tokens=%s",
                case_id,
                stage,
                attempt,
                len(prompt),
                max_tokens,
            )
            started = time.monotonic()
            try:
                response = await self.scheduler.run_one(LLMRequest(case_id, stage_name, prompt, max_tokens=max_tokens))
            except Exception as exc:  # noqa: BLE001
                elapsed = time.monotonic() - started
                self.logger.exception(
                    "stage_transport_error case_id=%s stage=%s attempt=%d elapsed_sec=%.2f error=%r",
                    case_id,
                    stage,
                    attempt,
                    elapsed,
                    exc,
                )
                raise
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
            self._save_raw(case_id, stage_name, response.text)
            self._save_reasoning(case_id, stage_name, response.reasoning)
            try:
                parsed = parse_jsonish(response.text)
                validated = validate_stage(stage, parsed)
                self.logger.info("stage_valid case_id=%s stage=%s attempt=%d", case_id, stage, attempt)
                return validated
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
                self._save_prompt(case_id, f"{stage}.retry{attempt + 1}", prompt)
        raise ValueError(f"{stage} did not produce valid structured output after retries: {last_error}")

    def _save_raw(self, case_id: str, stage: str, text: str) -> None:
        if not self.save_intermediates:
            return
        path = self.output_dir / "intermediates" / case_id / f"{stage}.txt"
        ensure_parent(path)
        path.write_text(text, encoding="utf-8")

    def _save_reasoning(self, case_id: str, stage: str, text: str) -> None:
        if not (self.save_intermediates and text):
            return
        path = self.output_dir / "intermediates" / case_id / f"{stage}.reasoning.txt"
        ensure_parent(path)
        path.write_text(text, encoding="utf-8")

    def _save_prompt(self, case_id: str, stage: str, prompt: str) -> None:
        if not (self.save_intermediates and self.save_prompts):
            return
        path = self.output_dir / "intermediates" / case_id / f"{stage}.prompt.txt"
        ensure_parent(path)
        path.write_text(prompt, encoding="utf-8")

    def _save_case_result(self, case_id: str, result: dict[str, Any]) -> None:
        if self.write_case_files:
            ensure_parent(self.done_path(case_id))
            self.done_path(case_id).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_case_result(self, case_id: str) -> dict[str, Any]:
        return json.loads(self.done_path(case_id).read_text(encoding="utf-8"))

    async def annotate_many(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        final_path = self.output_dir / self.config.get("run", {}).get("final_filename", "final.jsonl")
        scheduler_config = self.config.get("scheduler", {})
        max_case_concurrency = int(
            scheduler_config.get("max_case_concurrency", scheduler_config.get("max_in_flight_requests", 4))
        )
        semaphore = asyncio.Semaphore(max_case_concurrency)

        async def run_case(case: dict[str, Any]) -> dict[str, Any]:
            case_id = case["case_id"]
            if self.is_done(case_id):
                self.logger.info("case_skip_done case_id=%s", case_id)
                return {**self._load_case_result(case_id), "_skipped_existing": True}
            try:
                async with semaphore:
                    return await self.annotate_case(case)
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("case_error case_id=%s error=%r", case_id, exc)
                result = {**case, "annotation_status": "error", "error": repr(exc)}
                path = self.output_dir / "errors" / f"{case_id}.json"
                ensure_parent(path)
                path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                if not self.config.get("run", {}).get("continue_on_error", True):
                    raise
                return result

        tasks = [asyncio.create_task(run_case(case)) for case in cases]
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            if result.get("annotation_status") == "ok" and not result.get("_skipped_existing"):
                append_jsonl(final_path, result)
            result.pop("_skipped_existing", None)
        return results


def normalize_firac_keys(value: dict[str, Any]) -> dict[str, Any]:
    conclusion = value.get("Conclusion", value.get("conclusion", {}))
    if isinstance(conclusion, str):
        conclusion = {"label": conclusion, "text": conclusion, "simplified_label": ""}
    return {
        "facts": value.get("Facts", value.get("facts", [])),
        "issues": value.get("Issues", value.get("issues", [])),
        "rules": value.get("Rules", value.get("rules", [])),
        "analysis": value.get("Analysis", value.get("analysis", [])),
        "conclusion": conclusion,
    }


def get_conclusion_label(value: dict[str, Any]) -> str:
    conclusion = value.get("Conclusion", value.get("conclusion", ""))
    if isinstance(conclusion, dict):
        return str(conclusion.get("label") or conclusion.get("text") or "")
    return str(conclusion)


def get_simplified_label(value: dict[str, Any]) -> str:
    conclusion = value.get("Conclusion", value.get("conclusion", ""))
    if isinstance(conclusion, dict):
        label = str(conclusion.get("simplified_label", "")).strip().lower()
        if label in {"allowed", "dismissed", "other"}:
            return label
    return ""


def extract_citations_from_firac(value: dict[str, Any]) -> dict[str, Any]:
    rules = value.get("Rules", value.get("rules", []))
    cases = []
    legislations = []
    if isinstance(rules, list):
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            rule_type = str(rule.get("type", "")).lower()
            reference = str(rule.get("reference", ""))
            treatment = str(rule.get("treatment_type", ""))
            if rule_type == "precedent":
                cases.append(
                    {
                        "CaseName": rule.get("case_name", reference),
                        "CaseRef": rule.get("citation", ""),
                        "TreatmentType": treatment,
                        "Relevance": rule.get("relevance", ""),
                    }
                )
            elif rule_type in {"statute", "rule", "regulation", "legislation"}:
                legislations.append(
                    {
                        "LegName": reference,
                        "LegNumber": rule.get("provision", ""),
                        "Relevance": rule.get("relevance", ""),
                    }
                )
    return {"CasesReferred": cases, "LegislationsReferred": legislations}


def build_retry_prompt(original_prompt: str, previous_response: str, error: str) -> str:
    return (
        f"{original_prompt}\n\n"
        "Your previous response could not be parsed or validated against the required schema.\n"
        "Return the corrected response only. Do not include markdown, commentary, or code fences.\n\n"
        f"Validation error:\n{error}\n\n"
        f"Previous response:\n{previous_response}"
    )


def metadata_view(judgment: str, head_chars: int = 12000, tail_chars: int = 8000) -> str:
    if len(judgment) <= head_chars + tail_chars:
        return judgment
    return (
        judgment[:head_chars]
        + "\n\n[... middle of judgment omitted for metadata extraction ...]\n\n"
        + judgment[-tail_chars:]
    )


def configure_logger(path: Path, level_name: str) -> logging.Logger:
    logger = logging.getLogger(f"firac_appeals.annotation.{path}")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, str(level_name).upper(), logging.INFO))
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logger.level)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logger.level)
    logger.addHandler(stream_handler)
    return logger


def run_pipeline_sync(pipeline: FiracPipeline, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    thread_pool_workers = configured_thread_pool_workers(pipeline.config.get("scheduler", {}))
    if thread_pool_workers <= 0:
        return asyncio.run(pipeline.annotate_many(cases))

    async def run_with_executor() -> list[dict[str, Any]]:
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=thread_pool_workers, thread_name_prefix="firac-llm")
        loop.set_default_executor(executor)
        try:
            return await pipeline.annotate_many(cases)
        finally:
            executor.shutdown(wait=True, cancel_futures=False)

    return asyncio.run(run_with_executor())


def configured_thread_pool_workers(scheduler_config: dict[str, Any]) -> int:
    env_value = os.environ.get("FIRAC_LLM_THREAD_POOL_WORKERS", "").strip()
    if env_value:
        return int(env_value)
    return int(scheduler_config.get("thread_pool_workers", 0) or 0)
