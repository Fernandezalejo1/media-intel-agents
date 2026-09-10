"""Concrete agents: supervisor router + media specialists + quality gate.

Every specialist is prompt+model+tool-policy wired through :class:`BaseAgent`,
so the mock provider and any OpenAI-compatible model are interchangeable.
"""

from __future__ import annotations

import json
from typing import Any

from media_intel.agents.base import AgentError, BaseAgent
from media_intel.core.governance import RunBudget
from media_intel.core.observability import Span, Tracer
from media_intel.llm.base import LLMClient, Message, ToolSpec

ROUTE_TOOL = ToolSpec(
    name="route_request",
    description="Route the objective to exactly one specialist route.",
    parameters={
        "type": "object",
        "properties": {
            "route": {"type": "string", "enum": ["matchday", "analysis"]},
            "reasoning": {"type": "string"},
        },
        "required": ["route"],
    },
)

NEXT_TOOL = ToolSpec(
    name="decide_next",
    description="Choose the next graph node after a specialist finished.",
    parameters={
        "type": "object",
        "properties": {
            "next": {"type": "string", "enum": ["quality_gate", "publish", "done"]},
            "reasoning": {"type": "string"},
        },
        "required": ["next"],
    },
)

SUPERVISOR_ROLE = """\
You are the supervisor of a media-intelligence editorial team.
Classify the objective and pick exactly one route:
- "matchday": live match coverage, goals, moments, fixtures.
- "analysis": audience trends, editorial research, knowledge work.
Respond with the route_request tool call only."""


class SupervisorAgent(BaseAgent):
    """Routes an objective and sequences the workflow via tool-call decisions."""

    def __init__(self, llm: LLMClient, model: str):
        super().__init__(
            name="supervisor",
            role=SUPERVISOR_ROLE,
            llm=llm,
            model=model,
            max_tokens=200,
        )

    async def route(
        self,
        objective: str,
        *,
        budget: RunBudget | None = None,
        tracer: Tracer | None = None,
        parent_span: Span | None = None,
    ) -> dict[str, Any]:
        messages = [
            Message(role="system", content=self.role),
            Message(role="user", content="TASK:route"),
            Message(role="user", content=json.dumps({"objective": objective})),
        ]
        response = await self.llm.complete(
            messages,
            model=self.model,
            tools=[ROUTE_TOOL],
            budget=budget,
            tracer=tracer,
            parent_span=parent_span,
        )
        if response.tool_calls:
            args = response.tool_calls[0].arguments
            return {
                "route": args.get("route", "analysis"),
                "reasoning": args.get("reasoning", "tool-call decision"),
            }
        # Fallback: plain-text models get a conservative default route.
        return {"route": "analysis", "reasoning": (response.content or "")[:200]}

    async def decide_next(
        self,
        state_summary: dict[str, Any],
        *,
        budget: RunBudget | None = None,
        tracer: Tracer | None = None,
        parent_span: Span | None = None,
    ) -> dict[str, Any]:
        messages = [
            Message(role="system", content=self.role),
            Message(role="user", content="TASK:decide-next"),
            Message(role="user", content=json.dumps(state_summary, ensure_ascii=False)),
        ]
        response = await self.llm.complete(
            messages,
            model=self.model,
            tools=[NEXT_TOOL],
            budget=budget,
            tracer=tracer,
            parent_span=parent_span,
        )
        if response.tool_calls:
            args = response.tool_calls[0].arguments
            return {"next": args.get("next", "done"), "reasoning": args.get("reasoning", "")}
        return {"next": "done", "reasoning": "no tool call; defaulting to done"}


class NarrativeAgent(BaseAgent):
    """Generates editorial narrative insight from match/audience facts."""

    def __init__(self, llm: LLMClient, model: str, registry: Any, tools: list[str]):
        super().__init__(
            name="narrative",
            role=(
                "You are a senior sports editor. Using ONLY the provided facts and "
                "retrieved context, write a 2-3 sentence editorial insight. No invention."
            ),
            llm=llm,
            model=model,
            registry=registry,
            allowed_tools=tools,
            max_tokens=600,
        )


class MetadataAgent(BaseAgent):
    """Produces structured metadata JSON for CMS ingestion."""

    def __init__(self, llm: LLMClient, model: str):
        super().__init__(
            name="metadata",
            role=(
                "You generate strict JSON metadata for video/CMS systems: "
                "title, tags, sport, language, seo_keywords, safety. JSON only."
            ),
            llm=llm,
            model=model,
            max_tokens=400,
        )


class SummaryAgent(BaseAgent):
    """Condenses long transcripts/briefs into low-latency summaries."""

    def __init__(self, llm: LLMClient, model: str):
        super().__init__(
            name="summarizer",
            role="You summarize media content in at most 40 words, keeping names and scores.",
            llm=llm,
            model=model,
            max_tokens=200,
        )


class AnalystAgent(BaseAgent):
    """RAG analyst: retrieves audience/KB context, then reasons over it."""

    def __init__(self, llm: LLMClient, model: str, registry: Any, tools: list[str]):
        super().__init__(
            name="analyst",
            role=(
                "You are a media data analyst. Retrieve the context you need with the "
                "available tools, then answer with numbers where possible."
            ),
            llm=llm,
            model=model,
            registry=registry,
            allowed_tools=tools,
            max_tokens=600,
        )


class QualityGateAgent(BaseAgent):
    """Self-check pass: flags unsupported claims before publication (HITL precursor)."""

    def __init__(self, llm: LLMClient, model: str):
        super().__init__(
            name="quality_gate",
            role=(
                "You are the editorial standards editor. Given a draft and the facts, "
                'output strict JSON: {"approved": bool, "issues": [..]}.'
            ),
            llm=llm,
            model=model,
            max_tokens=300,
        )

    async def review(
        self,
        draft: str,
        facts: dict[str, Any],
        *,
        budget: RunBudget | None = None,
        tracer: Tracer | None = None,
        parent_span: Span | None = None,
    ) -> dict[str, Any]:
        result = await self.run(
            state={},
            task="review",
            payload=json.dumps({"draft": draft, "facts": facts}, ensure_ascii=False),
            budget=budget,
            tracer=tracer,
            parent_span=parent_span,
            expect_json=True,
        )
        if not isinstance(result, dict) or "approved" not in result:
            raise AgentError("quality gate returned malformed verdict")
        return result
