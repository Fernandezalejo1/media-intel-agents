"""Agent + tool tests: media tools, registry resilience, routing, quality gate."""

from __future__ import annotations

import json

import pytest

from media_intel.agents.specialists import QualityGateAgent, SupervisorAgent
from media_intel.llm.mock_provider import MockLLMClient
from media_intel.tools.registry import ToolRegistry


# ----------------------------------------------------------------- media tools


def test_fixtures_tool_filters_by_competition(platform):
    tools = platform.registry
    tool = tools.get("get_fixtures")
    rows = tool.handler(competition="LaLiga")
    assert rows and rows[0]["home"] == "Real Madrid"


def test_match_events_tool_returns_timeline(platform):
    tool = platform.registry.get("get_match_events")
    rows = tool.handler(match_id="m-101")
    assert len(rows) == 4
    assert any(ev["player"] == "Bellingham" for ev in rows)


def test_audience_tool_returns_metrics(platform):
    platform.audience_table.append(
        [{"id": "aud-1", "match_id": "m-101", "viewers_peak": 8_400_000, "audience_index": 1.42}]
    )
    tool = platform.registry.get("get_audience_metrics")
    rows = tool.handler(match_id="m-101")
    assert rows and rows[0]["audience_index"] == 1.42


def test_transcript_search_returns_relevant(platform):
    from media_intel.data.seed import _upsert_index
    from media_intel.data.seed import TRANSCRIPTS

    _upsert_index(platform, "transcripts", TRANSCRIPTS)
    tool = platform.registry.get("search_transcripts")
    hits = tool.handler(query="who scored the winner", k=1)
    assert hits and "Bellingham" in hits[0]["text"]


def test_publish_tool_files_request_without_publishing(platform):
    tool = platform.registry.get("publish_editorial")
    result = tool.handler(title="t", body="b", channel="web")
    assert result["queued"] is True
    assert platform.publish_queue.pending[0]["status"] == "pending"


def test_publish_queue_approve_reject_flow(platform):
    q = platform.publish_queue
    q.file_request(title="a", body="b")
    out = q.approve(1)
    assert out["status"] == "published"
    q.file_request(title="c", body="d")
    out = q.reject(2)
    assert out["status"] == "rejected"


# ------------------------------------------------------------ registry resilience


def test_unknown_tool_returns_error_result(platform):
    import asyncio

    result = asyncio.run(platform.registry.execute("nope", {}))
    assert result.is_error and "unknown tool" in result.content


def test_registry_retries_transient_failures(platform):
    import asyncio

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("transient")
        return "ok"

    platform.registry.register("flaky", "test tool", {"type": "object", "properties": {}}, flaky)
    result = asyncio.run(platform.registry.execute("flaky", {}))
    assert result.content == "ok" and calls["n"] == 3


# ------------------------------------------------------------------- agents


def test_supervisor_routes_matchday(platform):
    supervisor = SupervisorAgent(platform.llm, platform.settings.model_small)
    import asyncio

    decision = asyncio.run(supervisor.route("Cover the El Clasico match: goals and audience reaction"))
    assert decision["route"] == "matchday"


def test_supervisor_routes_analysis(platform):
    supervisor = SupervisorAgent(platform.llm, platform.settings.model_small)
    import asyncio

    decision = asyncio.run(supervisor.route("Analyze audience trends across the last month"))
    assert decision["route"] == "analysis"


def test_quality_gate_approves_clean_draft(platform):
    gate = QualityGateAgent(platform.llm, platform.settings.model_small)
    import asyncio

    verdict = asyncio.run(gate.review(draft="A clean draft.", facts={}))
    assert verdict["approved"] is True and verdict["issues"] == []


def test_quality_gate_flags_empty_draft(platform):
    gate = QualityGateAgent(platform.llm, platform.settings.model_small)
    import asyncio

    verdict = asyncio.run(gate.review(draft="", facts={}))
    assert verdict["approved"] is False and verdict["issues"]
