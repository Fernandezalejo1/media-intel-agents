"""Media-domain tools the agents can call.

Includes a deliberately human-gated tool (``publish_editorial``) used by the
HITL workflow: it never executes directly - it files a request that waits in
the graph interrupt until an operator approves or rejects it.
"""

from __future__ import annotations

import json
from typing import Any

from media_intel.core.resilience import RetryPolicy, with_retry
from media_intel.data.lakehouse import LakehouseTable
from media_intel.data.retrieval import Retriever
from media_intel.tools.registry import ToolRegistry, schema_object


def build_media_tools(
    events_table: LakehouseTable,
    audience_table: LakehouseTable,
    retriever: Retriever,
    kb_index: str = "editorial_kb",
    transcript_index: str = "transcripts",
    publish_topic: str = "editorial.publication",
) -> tuple[ToolRegistry, "PublishQueue"]:
    """Build and register the standard toolset; returns (registry, publish_queue)."""
    registry = ToolRegistry()
    queue = PublishQueue(topic=publish_topic)

    registry.register(
        name="get_fixtures",
        description="Look up upcoming or recent matches/fixter info by competition.",
        parameters=schema_object(
            {
                "competition": {"type": "string", "description": "e.g. 'UCL', 'LaLiga'"},
                "team": {"type": "string", "description": "optional team filter"},
            },
            ["competition"],
        ),
        handler=lambda competition, team=None: _retry_call(
            lambda: _find(events_table, {"type": "fixture", "competition": competition, "team": team})
        ),
    )

    registry.register(
        name="get_match_events",
        description="Timeline of match events (goals, cards, subs) for one match id.",
        parameters=schema_object({"match_id": {"type": "string"}}, ["match_id"]),
        handler=lambda match_id: _retry_call(lambda: _find(events_table, {"type": "match_event", "match_id": match_id})),
    )

    registry.register(
        name="get_audience_metrics",
        description="Audience/tuning metrics (viewers, index) for a match or program.",
        parameters=schema_object({"match_id": {"type": "string"}}, ["match_id"]),
        handler=lambda match_id: _retry_call(lambda: _find(audience_table, {"match_id": match_id})),
    )

    registry.register(
        name="search_transcripts",
        description="Semantic search over commentary/video transcripts (vector search).",
        parameters=schema_object({"query": {"type": "string"}, "k": {"type": "integer"}}, ["query"]),
        handler=lambda query, k=4: retriever.retrieve(transcript_index, query, k=k),
    )

    registry.register(
        name="search_editorial_kb",
        description="Search the editorial knowledge base (style rules, franchise facts).",
        parameters=schema_object({"query": {"type": "string"}, "k": {"type": "integer"}}, ["query"]),
        handler=lambda query, k=3: retriever.retrieve(kb_index, query, k=k),
    )

    registry.register(
        name="publish_editorial",
        description="Request publication of an editorial package. Requires human approval.",
        parameters=schema_object(
            {"title": {"type": "string"}, "body": {"type": "string"}, "channel": {"type": "string"}},
            ["title", "body"],
        ),
        handler=queue.file_request,  # never publishes; queues a HITL request
    )

    return registry, queue


class PublishQueue:
    """Collects publication requests awaiting human approval (HITL)."""

    def __init__(self, topic: str):
        self.topic = topic
        self.pending: list[dict[str, Any]] = []

    def file_request(self, title: str, body: str, channel: str = "web") -> dict[str, Any]:
        request = {"title": title, "body": body, "channel": channel, "status": "pending"}
        self.pending.append(request)
        return {"queued": True, "position": len(self.pending), "topic": self.topic}

    def approve(self, position: int) -> dict[str, Any]:
        return self._resolve(position, "published")

    def reject(self, position: int) -> dict[str, Any]:
        return self._resolve(position, "rejected")

    def _resolve(self, position: int, status: str) -> dict[str, Any]:
        if not 1 <= position <= len(self.pending):
            raise ValueError(f"no pending request at position {position}")
        self.pending[position - 1]["status"] = status
        return {**self.pending[position - 1], "topic": self.topic}


def _find(table: LakehouseTable, criteria: dict[str, Any]) -> list[dict[str, Any]]:
    rows = table.scan(limit=10_000)
    out = []
    for row in rows:
        if all(_matches(row.get(k), v) for k, v in criteria.items() if v is not None):
            out.append({k: v for k, v in row.items() if k != "_ingested_at"})
    return out


def _matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.lower() in actual.lower()
    return actual == expected


def _retry_call(fn: Any) -> Any:
    return with_retry(fn, RetryPolicy(max_attempts=2, base_delay_s=0.05))


def dumps(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)
