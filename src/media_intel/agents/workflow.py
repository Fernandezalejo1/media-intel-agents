"""Workflows: supervisor + specialists graph with HITL publication gate.

Implements the JD's key agentic patterns:
- supervisor routes objectives to sub-agents (matchday / analysis lanes);
- specialists execute with tools and share a typed state;
- the quality gate can fail -> bounded rewrite loop;
- publication raises a graph *interrupt*: the run suspends, persists a
  checkpoint, and waits for an operator decision (approve / reject);
- unhandled failures become structured escalations instead of silent drops.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from media_intel.agents.base import AgentError
from media_intel.agents.graph import Command, GraphInterrupted, StateGraph, interrupt
from media_intel.agents.specialists import (
    AnalystAgent,
    MetadataAgent,
    NarrativeAgent,
    QualityGateAgent,
    SummaryAgent,
    SupervisorAgent,
)
from media_intel.core.governance import BudgetExceeded, RunBudget
from media_intel.llm.base import LLMClient
from media_intel.tools.media_tools import PublishQueue
from media_intel.tools.registry import ToolRegistry

MAX_REWRITES = 1


@dataclass
class WorkflowServices:
    """Dependency bundle assembled by the bootstrap layer."""

    llm: LLMClient
    model_large: str
    model_small: str
    registry: ToolRegistry
    publish_queue: PublishQueue
    trace_store: Any = None
    settings: Any = None


def build_editorial_graph(services: WorkflowServices) -> StateGraph:
    """Supervisor + sub-agents + quality gate + HITL publish, as a stateful graph."""
    graph = StateGraph()
    supervisor = SupervisorAgent(services.llm, services.model_small)
    narrative = NarrativeAgent(services.llm, services.model_large, services.registry, _tools("narrative"))
    analyst = AnalystAgent(services.llm, services.model_large, services.registry, _tools("analyst"))
    metadata = MetadataAgent(services.llm, services.model_small)
    summarizer = SummaryAgent(services.llm, services.model_small)
    quality = QualityGateAgent(services.llm, services.model_small)

    # ---------------------------------------------------------------- nodes

    def _budget_of(state: dict[str, Any]) -> RunBudget | None:
        budget = state.get("budget")
        return budget if isinstance(budget, RunBudget) else None

    async def route(state: dict[str, Any]) -> dict[str, Any]:
        decision = await supervisor.route(state["objective"], budget=_budget_of(state))
        return {"route": decision["route"], "route_reasoning": decision["reasoning"]}

    async def matchday_lane(state: dict[str, Any]) -> dict[str, Any]:
        facts = _gather_facts(state, services.registry)
        draft = await narrative.run(state, task="narrative", payload=_facts_payload(facts), budget=_budget_of(state))
        return {"facts": facts, "outputs": {**state.get("outputs", {}), "narrative": draft}}

    async def analysis_lane(state: dict[str, Any]) -> dict[str, Any]:
        facts = _gather_facts(state, services.registry)
        answer = await analyst.run(state, task="analysis", payload=_facts_payload(facts), budget=_budget_of(state))
        summary = await summarizer.run(state, task="summary", payload=str(answer)[:4000], budget=_budget_of(state))
        return {"facts": facts, "outputs": {**state.get("outputs", {}), "analysis": answer, "summary": summary}}

    async def metadata_node(state: dict[str, Any]) -> dict[str, Any]:
        outputs = state.get("outputs", {})
        source = outputs.get("narrative") or outputs.get("analysis") or ""
        meta = await metadata.run(state, task="metadata", payload=source, budget=_budget_of(state))
        return {"outputs": {**outputs, "metadata": meta}}

    async def quality_gate(state: dict[str, Any]) -> dict[str, Any] | Command:
        outputs = state.get("outputs", {})
        draft = outputs.get("narrative") or outputs.get("analysis") or ""
        verdict = await quality.review(draft, state.get("facts", {}), budget=_budget_of(state))
        rewrites = state.get("rewrites", 0)
        if not verdict.get("approved") and rewrites < MAX_REWRITES:
            return Command(update={"rewrites": rewrites + 1, "quality": verdict}, goto="route")
        return {"quality": verdict}

    async def publish_request(state: dict[str, Any]) -> dict[str, Any]:
        outputs = state.get("outputs", {})
        title = _title_of(outputs)
        body = outputs.get("narrative") or outputs.get("analysis") or ""
        queued = services.publish_queue.file_request(title=title, body=body, channel="web")
        # HITL: suspend the graph, persist checkpoint, wait for the operator.
        return interrupt(
            {
                "type": "approval_required",
                "tool": "publish_editorial",
                "request": queued,
                "draft_title": title,
                "preview": body[:280],
                "quality": state.get("quality", {}),
            }
        )

    async def finalize(state: dict[str, Any]) -> dict[str, Any]:
        outputs = state.get("outputs", {})
        decision = (state.get("escalation") or {}).get("decision") or {}
        pending = outputs.get("pending_publication") or (
            ((state.get("pending_approval") or {}).get("payload") or {}).get("request") or {}
        )
        position = pending.get("position")
        if decision.get("action") == "approve" and position:
            published = services.publish_queue.approve(position)
            return {"outputs": {**outputs, "publication": published, "status": "published"}}
        if decision.get("action") == "reject" and position:
            rejected = services.publish_queue.reject(position)
            return {
                "outputs": {
                    **outputs,
                    "publication": rejected,
                    "status": "rejected",
                    "rejection_reason": decision.get("reason", "operator rejected"),
                }
            }
        # Reached finalize without a human decision: quality gate blocked it.
        issues = (state.get("quality") or {}).get("issues", [])
        return {"outputs": {**outputs, "status": "quality_rejected", "issues": issues}}

    graph.add_node("route", route)
    graph.add_node("matchday_lane", matchday_lane)
    graph.add_node("analysis_lane", analysis_lane)
    graph.add_node("metadata_node", metadata_node)
    graph.add_node("quality_gate", quality_gate)
    graph.add_node("publish_request", publish_request, human_in_the_loop=True)
    graph.add_node("finalize", finalize)

    # ---------------------------------------------------------------- edges

    graph.set_entry("route")
    graph.add_conditional_edges("route", lambda s: "matchday_lane" if s.get("route") == "matchday" else "analysis_lane")
    graph.add_edge("matchday_lane", "metadata_node")
    graph.add_edge("analysis_lane", "metadata_node")
    graph.add_edge("metadata_node", "quality_gate")
    graph.add_conditional_edges("quality_gate", lambda s: "publish_request" if s.get("quality", {}).get("approved") else "finalize")
    graph.add_edge("publish_request", "finalize")
    return graph


class EditorialOrchestrator:
    """Runs the editorial graph with budgets, tracing and escalation handling."""

    def __init__(self, graph: StateGraph, services: WorkflowServices):
        self.graph = graph.with_trace_store(services.trace_store).compile()
        self.services = services
        self.settings = services.settings

    async def run(self, objective: str, run_id: str | None = None) -> dict[str, Any]:
        settings = self.settings
        budget = RunBudget.from_limits(
            max_llm_calls=getattr(settings, "max_llm_calls_per_run", 24),
            max_total_tokens=getattr(settings, "max_total_tokens_per_run", 60_000),
            deadline_ms=getattr(settings, "run_deadline_ms", 20_000),
        )
        run_id = run_id or f"ed-{uuid.uuid4().hex[:8]}"
        try:
            state = await self.graph.run(
                {"objective": objective, "run_id": run_id, "budget": budget}, run_id=run_id
            )
            return self._enrich(state, budget)
        except GraphInterrupted as exc:
            state = dict(exc.state_snapshot)
            request = (exc.payload or {}).get("request") or {}
            state["pending_approval"] = {"node": exc.node, "payload": exc.payload}
            outputs = {**state.get("outputs", {}), "status": "awaiting_approval"}
            if request:
                outputs["pending_publication"] = request
            state["outputs"] = outputs
            return self._enrich(state, budget)
        except BudgetExceeded as exc:
            return self._escalation(run_id, f"budget: {exc.reason}", budget)
        except AgentError as exc:
            return self._escalation(run_id, f"agent failure: {exc}", budget)
        except Exception as exc:  # noqa: BLE001 - last-resort escalation
            return self._escalation(run_id, f"unexpected: {type(exc).__name__}: {exc}", budget)

    async def decide(self, run_id: str, action: str, reason: str = "") -> dict[str, Any]:
        """Operator decision for a suspended HITL run (approve / reject)."""
        if action not in {"approve", "reject"}:
            raise ValueError("action must be approve or reject")
        return await self.graph.resume(run_id, {"action": action, "reason": reason})

    def pending(self) -> list[dict[str, Any]]:
        return self.graph.pending_interruptions()

    # ---------------------------------------------------------------- helpers

    def _escalation(self, run_id: str, reason: str, budget: RunBudget) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "budget_summary": budget.summary(),
            "escalation": {"level": "operator", "reason": reason},
            "outputs": {"status": "escalated", "reason": reason},
        }

    @staticmethod
    def _enrich(state: dict[str, Any], budget: RunBudget) -> dict[str, Any]:
        state = dict(state)
        state["budget_summary"] = budget.summary()
        return state


def _gather_facts(state: dict[str, Any], registry: ToolRegistry) -> dict[str, Any]:
    """Assemble facts via the platform tools (fixtures, events, audience, RAG).

    In production the specialist agent would issue these tool calls itself via
    the LLM tool loop; the lane performs the same calls deterministically here
    so the demo (and tests) are reproducible without a live model.
    """
    objective = state.get("objective", "")
    facts: dict[str, Any] = {"objective": objective, "route": state.get("route", "")}
    try:
        fixtures = registry.get("get_fixtures").handler(competition="")
        needle = objective.lower()
        fixture = next(
            (
                fx
                for fx in fixtures
                if any(
                    alias in needle or alias in str(fx.get("aliases", "")).lower()
                    for alias in [str(fx.get("home", "")).lower(), str(fx.get("away", "")).lower()]
                    + [a.lower() for a in fx.get("aliases", [])]
                )
            ),
            None,
        )
        if fixture is None and fixtures:
            fixture = fixtures[0]
        if fixture:
            match_id = fixture.get("match_id", "")
            events = registry.get("get_match_events").handler(match_id=match_id)
            audience = registry.get("get_audience_metrics").handler(match_id=match_id)
            goals = [ev for ev in events if ev.get("event") == "goal"]
            facts.update(
                {
                    "event_label": fixture.get("competition", "Match"),
                    "home": fixture.get("home", "Home"),
                    "away": fixture.get("away", "Away"),
                    "venue": fixture.get("venue", ""),
                    "score": (goals[-1].get("score") if goals else "0-0"),
                    "goals": [
                        {"minute": g.get("minute"), "player": g.get("player"), "team": g.get("team")}
                        for g in goals
                    ],
                    "audience_index": (audience[0].get("audience_index") if audience else 1.0),
                    "viewers_peak": (audience[0].get("viewers_peak") if audience else None),
                }
            )
        kb_hits = registry.get("search_editorial_kb").handler(query=objective, k=2)
        facts["editorial_guidance"] = [h.get("text") for h in kb_hits]
    except Exception:  # noqa: BLE001 - facts are best-effort; never block the run
        pass
    return facts


def _title_of(outputs: dict[str, Any]) -> str:
    meta = outputs.get("metadata", "")
    if isinstance(meta, str):
        try:
            parsed = json.loads(meta)
            return str(parsed.get("title", "Editorial update"))
        except Exception:  # noqa: BLE001
            return "Editorial update"
    if isinstance(meta, dict):
        return str(meta.get("title", "Editorial update"))
    return "Editorial update"


def _tools(capability: str) -> list[str]:
    """Tool allow-lists per capability (least privilege)."""
    if capability == "narrative":
        return ["get_fixtures", "get_match_events", "get_audience_metrics", "search_transcripts"]
    if capability == "analyst":
        return ["get_audience_metrics", "search_editorial_kb", "search_transcripts"]
    return []


def _facts_payload(facts: dict[str, Any]) -> str:
    return json.dumps(facts, ensure_ascii=False, default=str)
