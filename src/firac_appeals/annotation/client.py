from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

@dataclass
class LLMRequest:
    case_id: str
    stage: str
    prompt: str
    max_tokens: int | None = None
    extra_payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    case_id: str
    stage: str
    text: str
    raw: dict[str, Any]
    reasoning: str = ""


class OpenAICompatibleClient:
    def __init__(self, config: dict[str, Any]) -> None:
        self.model = config.get("model")
        self.base_urls = resolve_base_urls(config)
        self._endpoint_index = 0
        self._endpoint_lock = asyncio.Lock()
        self.base_url = self.base_urls[0] if self.base_urls else ""
        self.api_key = os.environ.get(config.get("api_key_env", ""), "") or "EMPTY"
        self.timeout = float(config.get("timeout_seconds", 180))
        temperature = config.get("temperature", 0)
        self.temperature = None if temperature is None else float(temperature)
        self.top_p = config.get("top_p")
        self.top_k = config.get("top_k")
        self.min_p = config.get("min_p")
        self.seed = config.get("seed")
        self.max_tokens = int(config.get("max_tokens", 4096))
        self.system_prompt = config.get("system_prompt", "")
        self.extra_body = config.get("extra_body") or {}
        self.stream = bool(config.get("stream", False))
        self.no_proxy = bool(config.get("no_proxy", False))
        if not self.base_url:
            raise ValueError("backend.base_url is required")
        if not self.model:
            raise ValueError("backend.model is required")

    def _base_payload(self, request: LLMRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": build_messages(self.system_prompt, request.prompt),
            "max_tokens": request.max_tokens or self.max_tokens,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = float(self.top_p)
        if self.seed is not None:
            payload["seed"] = int(self.seed)
        payload.update(request.extra_payload)
        return payload

    def _openai_extra_body(self) -> dict[str, Any]:
        extra_body = dict(self.extra_body)
        if self.top_k is not None:
            extra_body["top_k"] = int(self.top_k)
        if self.min_p is not None:
            extra_body["min_p"] = float(self.min_p)
        return extra_body

    async def generate(self, request: LLMRequest) -> LLMResponse:
        base_url = await self.select_base_url()
        try:
            return await self._generate_with_openai(request, base_url)
        except ImportError:
            pass

        try:
            import httpx
        except ImportError:
            return await asyncio.to_thread(self._generate_with_urllib, request)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = self._base_payload(request)
        payload.update(self._openai_extra_body())
        async with httpx.AsyncClient(timeout=self.timeout, trust_env=not self.no_proxy) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            raw = response.json()
        text = raw["choices"][0]["message"]["content"]
        return LLMResponse(request.case_id, request.stage, text, raw)

    async def select_base_url(self) -> str:
        if not self.base_urls:
            raise ValueError("No base URLs available for OpenAI-compatible client")
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
        extra_body = self._openai_extra_body()
        if extra_body:
            kwargs["extra_body"] = extra_body
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
        extra_body = self._openai_extra_body()
        if extra_body:
            kwargs["extra_body"] = extra_body
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
        raw = {
            "stream": True,
            "finish_reason": finish_reason,
            "content_chars": len(content),
            "reasoning_chars": len(reasoning),
            "base_url": base_url,
        }
        return LLMResponse(request.case_id, request.stage, content, raw, reasoning)

    def _generate_with_urllib(self, request: LLMRequest) -> LLMResponse:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = self._base_payload(request)
        payload.update(self._openai_extra_body())
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
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
        if stage.startswith("ljp."):
            _, condition, stage_name = stage.split(".", 2)
            output = mock_ljp_output(condition, stage_name)
            text = json.dumps(output)
            return LLMResponse(request.case_id, request.stage, text, {"mock": True})
        outputs = {
            "metadata_ner": {
                "neutral_citation": "",
                "court": "",
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
                        "question": "What decisive legal question does the judgment resolve?",
                        "why_decisive": "This is a mock issue.",
                        "source_basis": "mock",
                    }
                ]
            },
            "legal_researcher": {"rules": []},
            "logician": {
                "analysis": [
                    {
                        "issue": "What decisive legal question does the judgment resolve?",
                        "steps": ["The relevant rule applies to the material facts."],
                        "conclusion": "The mock issue is resolved.",
                    }
                ]
            },
            "firac_judge": {
                "facts": ["This is a mock fact."],
                "issues": ["What decisive legal question does the judgment resolve?"],
                "rules": [],
                "analysis": [
                    {
                        "issue": "What decisive legal question does the judgment resolve?",
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
            "outcome_prediction": {
                "label": "other",
                "rationale": "Mock prediction.",
            },
            "issue_generation": {
                "issues": ["What decisive legal question does the appeal raise?"],
            },
            "rule_recommendation": {
                "rules": [
                    {
                        "type": "other",
                        "reference": "Mock authority",
                        "provision": "",
                        "case_name": "",
                        "citation": "",
                        "relevance": "Mock relevance.",
                    }
                ]
            },
        }
        text = json.dumps(outputs.get(stage, {}))
        return LLMResponse(request.case_id, request.stage, text, {"mock": True})


def mock_ljp_output(condition: str, stage: str) -> dict[str, Any]:
    if stage == "procedural_analysis":
        return {"posture": "Mock appeal", "procedural_history": [], "standards_or_constraints": []}
    if stage == "issue_analysis":
        return {"issues": ["Whether the appeal should be allowed?"]}
    if stage == "legal_research":
        return {
            "rules": [
                {
                    "type": "legal_principle",
                    "reference": "Mock legal principle",
                    "provision": "",
                    "case_name": "",
                    "citation": "",
                    "relevance": "Mock relevance",
                }
            ]
        }
    if stage == "logical_analysis":
        return {
            "analysis": [
                {
                    "issue": "Whether the appeal should be allowed?",
                    "steps": ["Apply the mock principle to the facts."],
                    "conclusion": "The issue favors dismissal.",
                }
            ]
        }
    if stage in {"appellant_argument", "respondent_argument"}:
        position = "appellant" if stage == "appellant_argument" else "respondent"
        return {"position": position, "arguments": [f"Mock {position} argument."], "requested_outcome": "dismissed"}
    if stage == "judgment" and condition == "zero_shot":
        return {"label": "dismissed", "rationale": "Mock rationale."}
    if stage == "judgment" and condition == "cot":
        return {"reasoning": ["Mock reasoning step."], "label": "dismissed", "rationale": "Mock rationale."}
    return {
        "facts": ["Mock fact."],
        "issues": ["Whether the appeal should be allowed?"],
        "rules": [],
        "analysis": [
            {
                "issue": "Whether the appeal should be allowed?",
                "steps": ["Apply the law to the facts."],
                "conclusion": "The issue favors dismissal.",
            }
        ],
        "conclusion": {"label": "dismissed", "text": "The appeal is dismissed."},
    }


def build_client(config: dict[str, Any]) -> OpenAICompatibleClient | MockClient:
    name = config.get("name", "openai_compatible")
    if name == "mock":
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
        self.limiter = AdaptiveRequestLimiter(config, self.logger)

    async def run_one(self, request: LLMRequest) -> LLMResponse:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            await self.limiter.acquire(request)
            try:
                self.logger.debug(
                    "llm_attempt_start case_id=%s stage=%s provider_attempt=%d concurrency=%d in_flight=%d",
                    request.case_id,
                    request.stage,
                    attempt,
                    self.limiter.capacity,
                    self.limiter.in_flight,
                )
                response = await self.client.generate(request)
                self.logger.debug(
                    "llm_attempt_done case_id=%s stage=%s provider_attempt=%d",
                    request.case_id,
                    request.stage,
                    attempt,
                )
                return response
            except Exception as exc:  # noqa: BLE001 - preserve provider errors in run logs
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
            finally:
                await self.limiter.release()
        assert last_error is not None
        raise last_error

    async def run_many(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        semaphore = asyncio.Semaphore(self.max_in_flight)

        async def guarded(req: LLMRequest) -> LLMResponse:
            async with semaphore:
                return await self.run_one(req)

        return await asyncio.gather(*(guarded(req) for req in requests))


class AdaptiveRequestLimiter:
    def __init__(self, scheduler_config: dict[str, Any], logger: logging.Logger) -> None:
        throttle = scheduler_config.get("adaptive_throttle") or {}
        self.metrics_urls = resolve_metrics_urls(throttle)
        metrics_url = str(throttle.get("metrics_url", "") or "")
        self.enabled = bool(throttle.get("enabled", bool(self.metrics_urls)) and self.metrics_urls)
        self.metrics_url = metrics_url
        self.metric_name = str(throttle.get("metric_name", "vllm:num_requests_waiting"))
        self.model = str(throttle.get("model", "") or "")
        self.no_proxy = bool(throttle.get("no_proxy", True))
        self.min_waiting = float(throttle.get("min_waiting_requests", 5))
        self.max_waiting = float(throttle.get("max_waiting_requests", 25))
        default_capacity = int(throttle.get("initial_concurrency", scheduler_config.get("max_in_flight_requests", 50)))
        if not self.enabled:
            default_capacity = int(scheduler_config.get("max_in_flight_requests", default_capacity or 50))
        self.capacity = max(1, default_capacity)
        self.min_capacity = int(throttle.get("min_concurrency", 1))
        self.max_capacity = int(throttle.get("max_concurrency", max(self.capacity, 200)))
        self.additive_increase = int(throttle.get("additive_increase", 5))
        self.multiplicative_decrease = float(throttle.get("multiplicative_decrease", 0.7))
        self.poll_interval = float(throttle.get("poll_interval_seconds", 5))
        self.in_flight = 0
        self.last_poll = 0.0
        self._poll_lock = asyncio.Lock()
        self._condition = asyncio.Condition()
        self.logger = logger
        self.logger.info(
            "adaptive_limiter initialized enabled=%s capacity=%d metrics_url=%s min_waiting=%s max_waiting=%s",
            self.enabled,
            self.capacity,
            ",".join(self.metrics_urls) or "<none>",
            self.min_waiting,
            self.max_waiting,
        )

    async def acquire(self, request: LLMRequest) -> None:
        while True:
            await self.maybe_refresh()
            async with self._condition:
                if self.in_flight < self.capacity:
                    self.in_flight += 1
                    return
                self.logger.debug(
                    "adaptive_limiter_wait case_id=%s stage=%s in_flight=%d capacity=%d",
                    request.case_id,
                    request.stage,
                    self.in_flight,
                    self.capacity,
                )
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=max(self.poll_interval, 1.0))
                except asyncio.TimeoutError:
                    pass

    async def release(self) -> None:
        async with self._condition:
            self.in_flight = max(0, self.in_flight - 1)
            self._condition.notify_all()

    async def maybe_refresh(self) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self.last_poll < self.poll_interval:
            return
        async with self._poll_lock:
            now = time.monotonic()
            if now - self.last_poll < self.poll_interval:
                return
            self.last_poll = now
            try:
                waiting_stats = await asyncio.to_thread(
                    fetch_waiting_request_stats,
                    self.metrics_urls,
                    self.metric_name,
                    self.model,
                    self.no_proxy,
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("adaptive_limiter_metrics_error urls=%s error=%r", self.metrics_urls, exc)
                return
            old_capacity = self.capacity
            if waiting_stats["min"] < self.min_waiting:
                self.capacity = min(self.max_capacity, self.capacity + self.additive_increase)
            elif waiting_stats["max"] > self.max_waiting:
                self.capacity = max(
                    self.min_capacity,
                    int(math.floor(max(1, self.capacity) * self.multiplicative_decrease)),
                )
            if self.capacity != old_capacity:
                self.logger.info(
                    "adaptive_limiter_adjust waiting_min=%s waiting_max=%s capacity=%d->%d in_flight=%d",
                    waiting_stats["min"],
                    waiting_stats["max"],
                    old_capacity,
                    self.capacity,
                    self.in_flight,
                )
            else:
                self.logger.debug(
                    "adaptive_limiter_keep waiting_min=%s waiting_max=%s capacity=%d in_flight=%d",
                    waiting_stats["min"],
                    waiting_stats["max"],
                    self.capacity,
                    self.in_flight,
                )
            async with self._condition:
                self._condition.notify_all()


def fetch_waiting_requests(url: str, metric_name: str, model: str = "", no_proxy: bool = True) -> float:
    request = urllib.request.Request(url, method="GET")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if no_proxy else urllib.request.build_opener()
    with opener.open(request, timeout=10) as response:
        text = response.read().decode("utf-8", errors="replace")
    total = 0.0
    found = False
    pattern = re.compile(rf"^{re.escape(metric_name)}(?:\{{[^}}]*\}})?\s+([-+0-9.eE]+)$")
    for line in text.splitlines():
        if not line or line.startswith("#") or metric_name not in line:
            continue
        if model and model not in line:
            continue
        match = pattern.match(line.strip())
        if not match:
            continue
        total += float(match.group(1))
        found = True
    if not found:
        raise ValueError(f"Metric {metric_name!r} for model {model!r} not found")
    return total


def resolve_metrics_urls(throttle: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    raw_urls = throttle.get("metrics_urls") or []
    if isinstance(raw_urls, str):
        raw_urls = [raw_urls]
    urls.extend(str(url).rstrip("/") for url in raw_urls if str(url).strip())
    single = str(throttle.get("metrics_url", "") or "").rstrip("/")
    if single:
        urls.append(single)
    return sorted(set(urls))


def fetch_waiting_request_stats(
    urls: list[str],
    metric_name: str,
    model: str = "",
    no_proxy: bool = True,
) -> dict[str, float]:
    if not urls:
        raise ValueError("No metrics URLs configured")
    values = [fetch_waiting_requests(url, metric_name, model=model, no_proxy=no_proxy) for url in urls]
    return {
        "min": min(values),
        "max": max(values),
        "total": sum(values),
        "count": float(len(values)),
    }
