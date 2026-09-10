"""End-to-end workflow tests: routing, lanes, HITL approval, escalation."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_matchday_run_reaches_hitl(orchestrator):
    result = await orchestrator.run("Cover the El Clasico match: goals, turning point, audience reaction")
    assert result["route"] == "matchday"
    assert result["outputs"]["status"] == "awaiting_approval"
    assert result["pending_approval"]["payload"]["type"] == "approval_required"
    assert result["outputs"]["pending_publication"]["queued"] is True
    # LLM calls were governed
    assert result["budget_summary"]["llm_calls"] >= 3


@pytest.mark.asyncio
async def test_approve_publishes(orchestrator):
    result = await orchestrator.run("Cover the El Clasico match: goals and audience reaction")
    run_id = result["run_id"]
    final = await orchestrator.decide(run_id, "approve", "demo operator")
    assert final["outputs"]["status"] == "published"
    assert final["outputs"]["publication"]["status"] == "published"


@pytest.mark.asyncio
async def test_reject_blocks_publication(orchestrator):
    result = await orchestrator.run("Cover the El Clasico match")
    final = await orchestrator.decide(result["run_id"], "reject", "needs editor review")
    assert final["outputs"]["status"] == "rejected"
    assert final["outputs"]["rejection_reason"] == "needs editor review"


@pytest.mark.asyncio
async def test_analysis_lane_produces_summary(orchestrator):
    result = await orchestrator.run("Analyze audience trends for recent broadcasts")
    assert result["route"] == "analysis"
    outputs = result["outputs"]
    assert outputs["status"] == "awaiting_approval"
    assert outputs.get("summary")


@pytest.mark.asyncio
async def test_pending_interruptions_exposed(orchestrator):
    await orchestrator.run("Cover the derby match")
    pending = orchestrator.pending()
    assert pending and pending[0]["payload"]["type"] == "approval_required"


@pytest.mark.asyncio
async def test_traces_recorded_for_run(orchestrator):
    result = await orchestrator.run("Cover the match: goals")
    traces = orchestrator.services.trace_store.for_run(result["run_id"])
    assert traces
    trace = traces[0]
    names = [s.name for s in trace.spans]
    assert any(n.startswith("node.") for n in names)
    assert any(n.startswith("llm.") or n.startswith("agent.") for n in names)
    assert trace.token_usage["total"] > 0
