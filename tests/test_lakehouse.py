"""Lakehouse local table tests."""

from __future__ import annotations

from media_intel.data.lakehouse import LocalDeltaTable


def test_append_and_scan(tmp_path):
    table = LocalDeltaTable(str(tmp_path), "events")
    n = table.append([{"id": 1, "type": "fixture"}, {"id": 2, "type": "match_event"}])
    assert n == 2
    rows = table.scan(limit=10)
    assert len(rows) == 2
    assert rows[0]["id"] == 1
    assert "_ingested_at" in rows[0]


def test_query_predicate(tmp_path):
    table = LocalDeltaTable(str(tmp_path), "events")
    table.append([{"id": 1, "type": "fixture"}, {"id": 2, "type": "match_event"}])
    hits = table.query(lambda r: r["type"] == "match_event")
    assert len(hits) == 1 and hits[0]["id"] == 2
