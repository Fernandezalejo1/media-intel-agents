"""Deterministic scripted LLM provider.

Produces believable outputs for the media-intelligence flows (narratives,
metadata, summaries, routing, JSON repair) without any network calls, so the
whole platform is demonstrable and testable offline. Set ``LLM_PROVIDER=openai``
to swap in a real model - everything else stays identical.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

from media_intel.llm.base import LLMResponse, Message, ToolCall, ToolSpec, estimate_tokens
from media_intel.core.observability import CURRENT_SPAN, CURRENT_TRACER


def _task_of(messages: list[Message]) -> str:
    for msg in reversed(messages):
        if msg.role == "user" and msg.content.startswith("TASK:"):
            return msg.content[5:].strip()
    return ""


def _payload_of(messages: list[Message]) -> str:
    for msg in reversed(messages):
        if msg.role == "user" and not msg.content.startswith("TASK:"):
            return msg.content
    return ""


class MockLLMClient:
    """Scripted, latency-simulating provider (no network, fully deterministic)."""

    def __init__(self, latency_ms: int = 0):
        self.latency_ms = int(os.environ.get("MOCK_LLM_LATENCY_MS", latency_ms))

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        budget: "RunBudget | None" = None,
        tracer: Any = None,
        parent_span: Any = None,
        **_: Any,
    ) -> LLMResponse:
        from media_intel.core.governance import RunBudget, estimate_cost_usd

        if budget is None:
            budget = RunBudget.from_limits(10**9, 10**9, 10**9)

        started = time.perf_counter()
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000)
        task = _task_of(messages)
        payload = _payload_of(messages)

        if tools and task in {"route", "decide-next"}:
            content, tool_calls = self._router_decision(task, payload)
        elif task == "narrative":
            content = self._narrative(payload)
            tool_calls = []
        elif task == "metadata":
            content = self._metadata(payload)
            tool_calls = []
        elif task == "summary":
            content = self._summary(payload)
            tool_calls = []
        elif task == "review":
            content = self._review(payload)
            tool_calls = []
        elif task == "repair-json":
            content = self._repair(payload)
            tool_calls = []
        else:
            content = f"[mock:{model}] acknowledged. {len(messages)} message(s) received."
            tool_calls = []

        it = estimate_tokens("".join(m.content for m in messages))
        ot = estimate_tokens(content)
        budget.record_llm_usage(it, ot)
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        cost_usd = estimate_cost_usd(model, it, ot)
        tracer = tracer or CURRENT_TRACER.get()
        parent = parent_span or CURRENT_SPAN.get()
        if tracer is not None:
            with tracer.span(
                "llm.completion",
                {"model": model, "latency_ms": latency_ms, "provider": "mock"},
                parent=parent,
            ) as span:
                tracer.record_tokens(span, model, it, ot, cost_usd)
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            model=model,
            input_tokens=it,
            output_tokens=ot,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
        )

    # ------------------------------------------------------------------ tasks

    def _router_decision(self, task: str, payload: str) -> tuple[str, list[ToolCall]]:
        """Supervisor decisions keyed explicitly off the TASK marker."""
        if task == "route":
            route = (
                "matchday"
                if any(k in payload.lower() for k in ("match", "goal", "derby", "fixture", " vs "))
                else "analysis"
            )
            call = ToolCall(
                id=f"call_route_{int(time.time() * 1000) % 10_000_000}",
                name="route_request",
                arguments={"route": route, "reasoning": "matched media-intent keywords in objective"},
            )
            return "", [call]
        call = ToolCall(
            id=f"call_next_{int(time.time() * 1000) % 10_000_000}",
            name="decide_next",
            arguments={"next": "quality_gate", "reasoning": "draft complete; send to quality gate"},
        )
        return "", [call]

    def _narrative(self, payload: str) -> str:
        facts = _try_parse(payload) or {}
        event = facts.get("event_label", "the match")
        home = facts.get("home", "Home")
        away = facts.get("away", "Away")
        score = facts.get("score", "0-0")
        audience = facts.get("audience_index", 1.0)
        momentum = "surging" if float(audience) >= 1.0 else "cooling"
        return (
            f"{event}: {home} vs {away} closed at {score}. "
            f"Audience momentum is {momentum} (index {audience}). "
            f"Editors should lead with the turning point and second-screen engagement spikes."
        )

    def _metadata(self, payload: str) -> str:
        facts = _try_parse(payload) or {}
        return json.dumps(
            {
                "title": f"{facts.get('event_label', 'Match Highlights')}",
                "tags": ["sports", "highlights", facts.get("competition", "league").lower()],
                "sport": facts.get("sport", "football"),
                "language": "en",
                "seo_keywords": [facts.get("home", "home").lower(), facts.get("away", "away").lower(), "highlights"],
                "safety": "clean",
            },
            indent=2,
        )

    def _summary(self, payload: str) -> str:
        body = payload.strip().replace("\n", " ")
        words = body.split()
        return " ".join(words[:40]) + ("..." if len(words) > 40 else "")

    def _review(self, payload: str) -> str:
        facts = _try_parse(payload) or {}
        issues: list[str] = []
        draft = str(facts.get("draft", ""))
        if not draft.strip():
            issues.append("draft is empty")
        if "TODO" in draft:
            issues.append("draft contains unresolved TODO")
        return json.dumps({"approved": not issues, "issues": issues})

    def _repair(self, payload: str) -> str:
        # The mock 'model' always outputs valid JSON; echo back a parsed re-dump.
        try:
            return json.dumps(json.loads(payload), indent=2)
        except json.JSONDecodeError:
            return json.dumps({"repaired": payload[:200]})


def _try_parse(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
