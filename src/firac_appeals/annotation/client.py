from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMRequest:
    case_id: str
    stage: str
    prompt: str
    max_tokens: int | None = None


@dataclass
class LLMResponse:
    case_id: str
    stage: str
    text: str
    raw: dict[str, Any]
    reasoning: str = ""


class OpenAICompatibleClient:
    """Small client for OpenAI-compatible chat-completions endpoints."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.model = str(config.get("model", "") or "")
        self.base_urls = resolve_base_urls(config)
        self._endpoint_index = 0
        self._endpoint_lock = asyncio.Lock()
        self.api_key = os.environ.get(str(config.get("api_key_env", "") or ""), "") or "EMPTY"
        self.timeout = float(config.get("timeout_seconds", 180))
        self.temperature = float(config.get("temperature", 0))
        self.top_p = config.get("top_p")
        self.seed = config.get("seed")
        self.max_tokens = int(config.get("max_tokens", 4096))
        self.system_prompt = str(config.get("system_prompt", "") or "")
        self.extra_body = dict(config.get("extra_body") or {})
        self.stream = bool(config.get("stream", False))
        self.no_proxy = bool(config.get("no_proxy", False))
        if not self.base_urls:
            raise ValueError("backend.base_url or backend.base_urls is required")
        if not self.model:
            raise ValueError("backend.model is required")

    def _base_payload(self, request: LLMRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": build_messages(self.system_prompt, request.prompt),
            "temperature": self.temperature,
            "max_tokens": request.max_tokens or self.max_tokens,
        }
        if self.top_p is not None:
            payload["top_p"] = float(self.top_p)
        if self.seed is not None:
            payload["seed"] = int(self.seed)
        return payload

    async def generate(self, request: LLMRequest) -> LLMResponse:
        base_url = await self.select_base_url()
        try:
            return await self._generate_with_openai(request, base_url)
        except ImportError:
            return await asyncio.to_thread(self._generate_with_urllib, request, base_url)

    async def select_base_url(self) -> str:
        async with self._endpoint_lock:
            url = self.base_urls[self._endpoint_index % len(self.base_urls)]
            self._endpoint_index += 1
            return url

    async def _generate_with_openai(self, request: LLMRequest, base_url: str) -> LLMResponse:
        if self.stream:
            return await asyncio.to_thread(self._stream_with_sync_openai, request, base_url)

        import httpx
        from openai import AsyncOpenAI

        http_client = httpx.AsyncClient(timeout=self.timeout, trust_env=not self.no_proxy)
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            http_client=http_client,
        )
        kwargs = self._base_payload(request)
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        try:
            response = await client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            reasoning = str(
                getattr(message, "reasoning", None)
                or getattr(message, "reasoning_content", None)
                or ""
            )
            text = message.content or ""
            raw = response.model_dump(mode="json")
            raw["base_url"] = base_url
            return LLMResponse(request.case_id, request.stage, text, raw, reasoning)
        finally:
            await http_client.aclose()

    def _stream_with_sync_openai(self, request: LLMRequest, base_url: str) -> LLMResponse:
        import httpx
        from openai import OpenAI

        http_client = httpx.Client(timeout=self.timeout, trust_env=not self.no_proxy)
        client = OpenAI(
            base_url=base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            http_client=http_client,
        )
        kwargs = self._base_payload(request)
        kwargs["stream"] = True
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        finish_reason = None
        try:
            for chunk in client.chat.completions.create(**kwargs):
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                finish_reason = choice.finish_reason or finish_reason
                delta = choice.delta
                reasoning_delta = getattr(delta, "reasoning", None)
                if reasoning_delta is None:
                    reasoning_delta = getattr(delta, "reasoning_content", None)
                content_delta = getattr(delta, "content", None)
                if reasoning_delta:
                    reasoning_parts.append(reasoning_delta)
                if content_delta:
                    content_parts.append(content_delta)
        finally:
            http_client.close()
        content = "".join(content_parts)
        reasoning = "".join(reasoning_parts)
        return LLMResponse(
            request.case_id,
            request.stage,
            content,
            {
                "stream": True,
                "finish_reason": finish_reason,
                "base_url": base_url,
                "content_chars": len(content),
                "reasoning_chars": len(reasoning),
            },
            reasoning,
        )

    def _generate_with_urllib(self, request: LLMRequest, base_url: str) -> LLMResponse:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = self._base_payload(request)
        payload.update(self.extra_body)
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP error {exc.code}: {body}") from exc
        message = raw["choices"][0]["message"]
        text = message.get("content") or ""
        reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
        return LLMResponse(request.case_id, request.stage, text, raw, reasoning)


class MockClient:
    async def generate(self, request: LLMRequest) -> LLMResponse:
        stage = request.stage.split(".retry", 1)[0]
        outputs = {
            "metadata_ner": {
                "neutral_citation": "",
                "court": "Mock Court",
                "district": "",
                "impugned_court": "",
                "bench_coram": [],
                "judgment_by": [],
                "judgment_date": "",
                "lawyers": {"for_petitioner_or_appellant": [], "for_respondent": []},
                "case_details": [],
            },
            "issue_analyst": {
                "issues": [
                    {
                        "question": "What decisive legal question does the appeal raise?",
                        "why_decisive": "This is the mock deciding issue.",
                        "source_basis": "mock",
                    }
                ]
            },
            "legal_researcher": {
                "rules": [
                    {
                        "type": "legal_principle",
                        "reference": "Mock legal principle",
                        "provision": "",
                        "case_name": "",
                        "citation": "",
                        "treatment_type": "Not applicable to authority type",
                        "relevance": "A mock rule used for smoke tests.",
                        "used_by_court": True,
                    }
                ]
            },
            "logician": {
                "analysis": [
                    {
                        "issue": "What decisive legal question does the appeal raise?",
                        "steps": ["The relevant rule applies to the material facts."],
                        "conclusion": "The mock issue is resolved.",
                    }
                ]
            },
            "firac_judge": {
                "facts": ["This is a mock fact."],
                "issues": ["What decisive legal question does the appeal raise?"],
                "rules": [
                    {
                        "type": "legal_principle",
                        "reference": "Mock legal principle",
                        "provision": "",
                        "case_name": "",
                        "citation": "",
                        "treatment_type": "Not applicable to authority type",
                        "relevance": "A mock rule used for smoke tests.",
                        "used_by_court": True,
                    }
                ],
                "analysis": [
                    {
                        "issue": "What decisive legal question does the appeal raise?",
                        "steps": ["The relevant rule applies to the material facts."],
                        "conclusion": "The mock issue is resolved.",
                    }
                ],
                "conclusion": {
                    "label": "Order accordingly",
                    "text": "The matter is ordered accordingly.",
                    "simplified_label": "other",
                },
            },
            "firac_verifier": {
                "is_usable": True,
                "facts_leakage": {"has_leakage": False, "problem_sentences": [], "explanation": ""},
                "faithfulness_errors": [],
                "missing_material_points": [],
                "rules_issues": [],
                "analysis_issues": [],
                "conclusion_issue": "",
                "recommended_quality_flags": [],
            },
            "outcome_prediction": {"label": "other", "rationale": "Mock prediction."},
            "issue_generation": {"issues": ["What decisive legal question does the appeal raise?"]},
            "rule_recommendation": {
                "rules": [
                    {
                        "type": "legal_principle",
                        "reference": "Mock legal principle",
                        "provision": "",
                        "case_name": "",
                        "citation": "",
                        "relevance": "Mock relevance.",
                    }
                ]
            },
            "firac_rubric_judge.r0": {
                "scores": {
                    "facts_faithfulness": 4,
                    "facts_coverage": 4,
                    "issues_faithfulness": 4,
                    "issues_coverage": 4,
                    "rules_faithfulness": 4,
                    "rules_coverage": 4,
                    "analysis_faithfulness": 4,
                    "analysis_coverage": 4,
                    "conclusion_faithfulness": 4,
                    "conclusion_coverage": 4,
                },
                "critical_flags": [],
                "rationales": {
                    "facts": "Mock rationale.",
                    "issues": "Mock rationale.",
                    "rules": "Mock rationale.",
                    "analysis": "Mock rationale.",
                    "conclusion": "Mock rationale.",
                },
                "reviewer_notes": "",
            },
        }
        text = json.dumps(outputs.get(stage, outputs.get(stage.split(".r", 1)[0], {})))
        return LLMResponse(request.case_id, request.stage, text, {"mock": True})


def build_client(config: dict[str, Any]) -> OpenAICompatibleClient | MockClient:
    if config.get("name", "openai_compatible") == "mock":
        return MockClient()
    return OpenAICompatibleClient(config)


def resolve_base_urls(config: dict[str, Any]) -> list[str]:
    base_urls = config.get("base_urls") or []
    if isinstance(base_urls, str):
        base_urls = [base_urls]
    if base_urls:
        return [str(url).rstrip("/") for url in base_urls if str(url).strip()]
    base_url = str(config.get("base_url", "") or "").rstrip("/")
    return [base_url] if base_url else []


def build_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


class AsyncScheduler:
    def __init__(
        self,
        client: OpenAICompatibleClient | MockClient,
        config: dict[str, Any],
        logger: logging.Logger | None = None,
    ) -> None:
        self.client = client
        self.max_in_flight = int(config.get("max_in_flight_requests", 8))
        self.max_retries = int(config.get("max_retries", 2))
        self.retry_backoff = float(config.get("retry_backoff_seconds", 3))
        self.logger = logger or logging.getLogger(__name__)
        self._semaphore = asyncio.Semaphore(self.max_in_flight)

    async def run_one(self, request: LLMRequest) -> LLMResponse:
        last_error: Exception | None = None
        async with self._semaphore:
            for attempt in range(self.max_retries + 1):
                try:
                    return await self.client.generate(request)
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    self.logger.warning(
                        "llm_attempt_error case_id=%s stage=%s provider_attempt=%d/%d error=%r",
                        request.case_id,
                        request.stage,
                        attempt,
                        self.max_retries,
                        exc,
                    )
                    if attempt < self.max_retries:
                        await asyncio.sleep(self.retry_backoff * (attempt + 1))
        assert last_error is not None
        raise last_error

    async def run_many(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        return await asyncio.gather(*(self.run_one(req) for req in requests))
