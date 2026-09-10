"""Agent base class: LLM call + typed tool loop with budget & tracing."""

from __future__ import annotations

from typing import Any

from media_intel.core.governance import RunBudget
from media_intel.core.observability import Span, Tracer
from media_intel.llm.base import LLMClient, LLMResponse, Message, ToolResult, ToolSpec


class AgentError(RuntimeError):
    pass


class BaseAgent:
    """One agent = role prompt + model choice + optional tool loop."""

    def __init__(
        self,
        name: str,
        role: str,
        llm: LLMClient,
        model: str,
        registry: Any | None = None,
        allowed_tools: list[str] | None = None,
        max_tool_loops: int = 3,
        temperature: float = 0.2,
        max_tokens: int = 800,
    ):
        self.name = name
        self.role = role
        self.llm = llm
        self.model = model
        self.registry = registry
        self.allowed_tools = allowed_tools
        self.max_tool_loops = max_tool_loops
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def run(
        self,
        state: dict[str, Any],
        task: str,
        payload: str,
        *,
        budget: RunBudget | None = None,
        tracer: Tracer | None = None,
        parent_span: Span | None = None,
        expect_json: bool = False,
    ) -> Any:
        with (tracer.span(f"agent.{self.name}", {"task": task}, parent=parent_span) if tracer else _NoopCtx()) as span:
            messages: list[Message] = [
                Message(role="system", content=self.role),
                Message(role="user", content=f"TASK:{task}"),
                Message(role="user", content=payload),
            ]
            specs: list[ToolSpec] | None = None
            if self.registry and self.allowed_tools:
                specs = self.registry.spec_for(self.allowed_tools)

            for _loop in range(self.max_tool_loops + 1):
                response = await self.llm.complete(
                    messages,
                    model=self.model,
                    tools=specs,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    budget=budget,
                    tracer=tracer,
                    parent_span=span,
                )
                if not response.tool_calls:
                    return self._finalize(response, expect_json)
                messages.append(_assistant_with_calls(response))
                for call in response.tool_calls:
                    if self.registry is None:
                        raise AgentError(f"{self.name}: tool call but no registry attached")
                    result: ToolResult = await self.registry.execute(
                        call.name, call.arguments, tracer=tracer, parent=span
                    )
                    messages.append(
                        Message(role="tool", name=call.name, tool_call_id=call.call_id or call.name, content=result.content)
                    )
            raise AgentError(f"{self.name}: exceeded tool loop budget ({self.max_tool_loops})")

    def _finalize(self, response: LLMResponse, expect_json: bool) -> Any:
        if not expect_json:
            return response.content
        from media_intel.llm.base import extract_json

        try:
            return extract_json(response.content)
        except Exception as exc:
            raise AgentError(f"{self.name}: invalid JSON in response: {exc}") from exc


class _NoopCtx:
    def __enter__(self) -> Any:
        return None

    def __exit__(self, *exc: Any) -> None:
        return None


def _assistant_with_calls(response: LLMResponse) -> Message:
    import json

    calls_json = json.dumps(
        [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in response.tool_calls],
        ensure_ascii=False,
    )
    return Message(role="assistant", content=response.content or "", name=f"tool_calls:{calls_json}")
