"""Mock media datasets: fixtures, match events, audience metrics, transcripts, KB.

Seeds the local lakehouse tables and vector indexes so the entire demo runs
offline. In a real deployment this data arrives via streaming/batch pipelines
(same table schemas, Databricks-hosted).
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from media_intel.bootstrap import Platform

FIXTURES: list[dict[str, Any]] = [
    {"id": "fx-1", "type": "fixture", "competition": "LaLiga", "home": "Real Madrid", "away": "FC Barcelona", "aliases": ["el clasico", "clasico"], "match_id": "m-101", "kickoff": "2026-09-12T20:00:00Z", "venue": "Santiago Bernabeu"},
    {"id": "fx-2", "type": "fixture", "competition": "UCL", "home": "Man City", "away": "Inter", "aliases": ["champions league", "city inter"], "match_id": "m-102", "kickoff": "2026-09-15T19:45:00Z", "venue": "Etihad Stadium"},
]

MATCH_EVENTS: list[dict[str, Any]] = [
    {"id": "ev-1", "type": "match_event", "match_id": "m-101", "minute": 12, "event": "goal", "team": "home", "player": "Vinicius Jr", "score": "1-0"},
    {"id": "ev-2", "type": "match_event", "match_id": "m-101", "minute": 55, "event": "goal", "team": "away", "player": "Lamine Yamal", "score": "1-1"},
    {"id": "ev-3", "type": "match_event", "match_id": "m-101", "minute": 88, "event": "goal", "team": "home", "player": "Bellingham", "score": "2-1"},
    {"id": "ev-4", "type": "match_event", "match_id": "m-101", "minute": 90, "event": "yellow_card", "team": "away", "player": "Gavi", "score": "2-1"},
]

AUDIENCE: list[dict[str, Any]] = [
    {"id": "aud-1", "match_id": "m-101", "viewers_peak": 8_400_000, "viewers_avg": 6_900_000, "audience_index": 1.42, "second_screen_rate": 0.38, "region": "ES"},
    {"id": "aud-2", "match_id": "m-102", "viewers_peak": 4_100_000, "viewers_avg": 3_300_000, "audience_index": 1.05, "second_screen_rate": 0.27, "region": "UK"},
]

TRANSCRIPTS: list[dict[str, str]] = [
    {"id": "tr-1", "text": "Vinicius opens the scoring with a blistering run down the left, cutting inside and finishing low at the near post."},
    {"id": "tr-2", "text": "Yamal equalizes for Barcelona with a composed finish after a lightning counter-attack down the right flank."},
    {"id": "tr-3", "text": "Bellingham scores the winner in the 88th minute, arriving late in the box and heading the cross past the keeper."},
    {"id": "tr-4", "text": "The second-screen audience spiked 40 percent during the final ten minutes, the highest of the season so far."},
]

EDITORIAL_KB: list[dict[str, str]] = [
    {"id": "kb-1", "text": "House style: lead with the turning point; keep headline under 70 characters; always name the scorer and minute."},
    {"id": "kb-2", "text": "El Clasico is the franchise fixture: expected audience index above 1.3; prioritize Spanish-language social cuts."},
    {"id": "kb-3", "text": "Publication policy: any editorial package must pass the standards review before going live on owned channels."},
]


def ensure_seeded(platform: "Platform") -> None:
    """Idempotent seeding of tables and vector indexes."""
    if not platform.events_table.scan(limit=1):
        platform.events_table.append([*FIXTURES, *MATCH_EVENTS])
    if not platform.audience_table.scan(limit=1):
        platform.audience_table.append(AUDIENCE)

    _upsert_index(platform, "transcripts", TRANSCRIPTS)
    _upsert_index(platform, "editorial_kb", EDITORIAL_KB)


def _upsert_index(platform: "Platform", index_id: str, docs: list[dict[str, str]]) -> None:
    vectors = platform.embeddings.embed([d["text"] for d in docs])
    records = [
        {"id": doc["id"], "text": doc["text"], "vector": vec, "metadata": {"source": index_id}}
        for doc, vec in zip(docs, vectors)
    ]
    platform.vector_backend.upsert(index_id, records)


def seed_all(platform: "Platform") -> None:
    ensure_seeded(platform)
