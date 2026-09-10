"""LLM provider protocol and shared message types."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from media_intel.core.governance import RunBudget, estimate_cost_usd
from media_intel.core.observability import Span, Tracer


@dataclass
class Message:
    role: str  # system | user | assistant | tool
    content: str
    name: str | None = None  # tool name for role="tool"
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            out["name"] = self.name
        if self.tool_call_id is not None:
            out["tool_call_id"] = self.tool_call_id
        return out


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall]
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    finish_reason: str = "stop"
    cost_usd: float = 0.0


@dataclass
class ToolSpec:
    """OpenAI-style JSON-schema tool definition."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class LLMClient(Protocol):
    """Uniform interface implemented by every provider."""

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
    ) -> LLMResponse: ...


def estimate_tokens(text: str) -> int:
    """Cheap deterministic token estimate (~4 chars/token heuristic)."""
    return max(1, len(text) // 4)


def extract_json(text: str) -> Any:
    """Parse the first JSON object/array embedded in an LLM response."""
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"[\[{].*[\]}]", candidate, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def cost_of(response: LLMResponse) -> float:
    return estimate_cost_usd(response.model, response.input_tokens, response.output_tokens)
