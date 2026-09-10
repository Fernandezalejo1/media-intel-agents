"""OpenAI-compatible provider (OpenAI, Azure OpenAI, vLLM, Ollama, LM Studio...).

Adds production behavior on top of the raw HTTP call: retry w/ backoff,
circuit breaker, per-run budget enforcement and tracing/token accounting.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from media_intel.core.governance import RunBudget
from media_intel.core.observability import Span, Tracer
from media_intel.core.resilience import CircuitBreaker, RetryPolicy, awith_retry
from media_intel.llm.base import LLMResponse, Message, ToolCall, ToolSpec, estimate_tokens


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 45.0,
    ):
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.timeout_s = timeout_s
        self._breaker = CircuitBreaker(name="llm", failure_threshold=5, reset_timeout_s=30.0)
        self._policy = RetryPolicy(max_attempts=3, base_delay_s=0.3)

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        budget: RunBudget | None = None,
        tracer: Tracer | None = None,
        parent_span: Span | None = None,
    ) -> LLMResponse:
        budget = budget or RunBudget.from_limits(10**9, 10**9, 10**9)

        def _call() -> dict[str, Any]:
            budget.check_llm_allowed()  # pre-call governance gate
            self._breaker.before()
            return _post_chat(  # async HTTP in the retry loop
                self.base_url,
                self.api_key,
                self._payload(messages, model, tools, temperature, max_tokens),
                self.timeout_s,
            )

        def _on_retry(attempt: int, exc: BaseException) -> None:
            if tracer:
                with tracer.span("llm.retry", {"attempt": attempt, "error": str(exc)}, parent=parent_span):
                    pass

        started = time.perf_counter()
        data = await awith_retry(_call, self._policy, _on_retry)
        self._breaker.success()

        usage = data.get("usage", {})
        input_tokens = int(usage.get("prompt_tokens", 0)) or sum(estimate_tokens(m.content) for m in messages)
        output_tokens = int(usage.get("completion_tokens", 0))
        tool_calls = [
            ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=_parse_args(tc["function"].get("arguments", "{}")),
            )
            for tc in (data.get("choices", [{}])[0].get("message", {}).get("tool_calls") or [])
        ]
        response = LLMResponse(
            content=data.get("choices", [{}])[0].get("message", {}).get("content") or "",
            tool_calls=tool_calls,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            finish_reason=data.get("choices", [{}])[0].get("finish_reason", "stop"),
        )

        budget.record_llm_usage(input_tokens, output_tokens)
        if tracer:
            with tracer.span(
                "llm.completion",
                {
                    "model": model,
                    "latency_ms": response.latency_ms,
                    "finish_reason": response.finish_reason,
                },
                parent=parent_span,
            ) as span:
                tracer.record_tokens(span, model, input_tokens, output_tokens, response.cost_usd)
        return response

    @staticmethod
    def _payload(
        messages: list[Message],
        model: str,
        tools: list[ToolSpec] | None,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = [t.to_openai() for t in tools]
        return payload


async def _post_chat(base_url: str, api_key: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()


def _parse_args(raw: str) -> dict[str, Any]:
    import json

    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        return {"_raw": raw}
