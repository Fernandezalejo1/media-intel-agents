"""Tool registry: Python callables <-> OpenAI-style tool specs, with resilience."""

from __future__ import annotations

import json
from typing import Any, Callable

from media_intel.core.observability import Span, Tracer
from media_intel.core.resilience import CircuitBreaker, RetryPolicy, awith_retry
from media_intel.llm.base import ToolResult, ToolSpec


class Tool:
    def __init__(self, name: str, description: str, parameters: dict[str, Any], handler: Callable[..., Any]):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)


class ToolRegistry:
    """Holds tools, exposes specs for the LLM, executes calls with tracing."""

    def __init__(self, *, retry_policy: RetryPolicy | None = None):
        self._tools: dict[str, Tool] = {}
        self._policy = retry_policy or RetryPolicy(max_attempts=3, base_delay_s=0.15)
        self._breakers: dict[str, CircuitBreaker] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        if name in self._tools:
            raise ValueError(f"tool already registered: {name}")
        self._tools[name] = Tool(name=name, description=description, parameters=parameters, handler=handler)

    def spec_for(self, names: list[str] | None = None) -> list[ToolSpec]:
        wanted = names or list(self._tools)
        return [self._tools[n].spec for n in wanted if n in self._tools]

    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> Tool:
        return self._tools[name]

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        tracer: Tracer | None = None,
        parent: Span | None = None,
    ) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(call_id=name, content=f"unknown tool: {name}", is_error=True)

        breaker = self._breakers.setdefault(name, CircuitBreaker(name=f"tool:{name}"))

        def _invoke() -> str:
            breaker.before()
            return tool.handler(**arguments)

        def _on_retry(attempt: int, exc: BaseException) -> None:
            if tracer:
                with tracer.span(f"tool.{name}.retry", {"attempt": attempt, "error": str(exc)}, parent=parent):
                    pass

        with (tracer.span(f"tool.{name}", {"arguments": arguments}, parent=parent) if tracer else _NullCtx()) as span:
            try:
                result = await awith_retry(_invoke, self._policy, _on_retry)
                breaker.success()
            except Exception as exc:  # noqa: BLE001 - tool errors go back to the LLM
                breaker.failure()
                if tracer and span is not None:
                    span.status = "ERROR"
                return ToolResult(call_id=name, content=f"{type(exc).__name__}: {exc}", is_error=True)
            if hasattr(result, "__await__"):
                # handler was async: re-run once awaiting properly
                result = await result  # pragma: no cover
            content = result if isinstance(result, str) else json.dumps(result, default=str, ensure_ascii=False)
            return ToolResult(call_id=name, content=content)


def schema_object(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required or []}


def json_dumps(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)


class _NullCtx:
    def __enter__(self) -> Any:
        return None

    def __exit__(self, *exc: Any) -> None:
        return None
