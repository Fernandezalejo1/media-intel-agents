"""Lakehouse table abstraction (Delta / Lakebase style).

- :class:`LakehouseTable` protocol: ``append`` / ``scan`` / ``query``.
- :class:`LocalDeltaTable` stores JSONL partitions under ``.lakehouse/``.
- :class:`DatabricksLakehouse` executes statements against a SQL warehouse
  (Statement Execution API), i.e. Delta tables / Lakebase in a real workspace.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Protocol


class LakehouseTable(Protocol):
    def append(self, rows: list[dict[str, Any]]) -> int: ...

    def scan(self, limit: int = 100) -> list[dict[str, Any]]: ...

    def query(self, predicate: Any) -> list[dict[str, Any]]: ...


class LocalDeltaTable:
    """Append-only JSONL table with manifest - good-enough Delta simulation."""

    def __init__(self, root: str, name: str):
        self.dir = os.path.join(root, name)
        self.data_path = os.path.join(self.dir, "part-00000.jsonl")
        os.makedirs(self.dir, exist_ok=True)

    def append(self, rows: list[dict[str, Any]]) -> int:
        with open(self.data_path, "a", encoding="utf-8") as fh:
            for row in rows:
                row = {**row, "_ingested_at": time.time()}
                fh.write(json.dumps(row, default=str) + "\n")
        self._manifest(len(rows))
        return len(rows)

    def scan(self, limit: int = 100) -> list[dict[str, Any]]:
        if not os.path.exists(self.data_path):
            return []
        rows: list[dict[str, Any]] = []
        with open(self.data_path, encoding="utf-8") as fh:
            for line in fh:
                if len(rows) >= limit:
                    break
                rows.append(json.loads(line))
        return rows

    def query(self, predicate: Any) -> list[dict[str, Any]]:
        return [row for row in self.scan(limit=10_000) if predicate(row)]

    def _manifest(self, added: int) -> None:
        entry = {"file": "part-00000.jsonl", "added_rows": added, "ts": time.time()}
        with open(os.path.join(self.dir, "_last_commit.json"), "w", encoding="utf-8") as fh:
            json.dump(entry, fh)


class DatabricksLakehouse:
    """Runs SQL against a Databricks SQL warehouse (Statement Execution API)."""

    def __init__(self, host: str, token: str, warehouse_id: str, catalog: str = "media_intel", schema: str = "default"):
        self.host = host.rstrip("/")
        self.token = token
        self.warehouse_id = warehouse_id
        self.catalog = catalog
        self.schema = schema

    def append(self, rows: list[dict[str, Any]]) -> int:  # pragma: no cover - needs live workspace
        if not rows:
            return 0
        cols = sorted(rows[0].keys())
        values = ", ".join(
            "(" + ", ".join(_sql_literal(row.get(c)) for c in cols) + ")" for row in rows
        )
        table = f"`{self.catalog}`.`{self.schema}`.events"
        self._execute(f"INSERT INTO {table} ({', '.join(cols)}) VALUES {values}")
        return len(rows)

    def scan(self, limit: int = 100) -> list[dict[str, Any]]:  # pragma: no cover
        table = f"`{self.catalog}`.`{self.schema}`.events"
        return self._execute(f"SELECT * FROM {table} LIMIT {limit}")

    def query(self, predicate: Any) -> list[dict[str, Any]]:  # pragma: no cover
        table = f"`{self.catalog}`.`{self.schema}`.events"
        return self._execute(f"SELECT * FROM {table} WHERE {predicate}")

    def _execute(self, sql: str) -> list[dict[str, Any]]:
        import httpx

        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                f"https://{self.host}/api/2.0/sql/statements",
                json={"warehouse_id": self.warehouse_id, "statement": sql, "wait_timeout": "30s", "format": "JSON_ARRAY"},
                headers=headers,
            )
            resp.raise_for_status()
            result = resp.json()
        return result.get("result", {}).get("data_array", [])


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def make_table(backend: str, root: str = ".lakehouse", name: str = "events", **kwargs: Any) -> LakehouseTable:
    if backend == "databricks":
        return DatabricksLakehouse(**kwargs)
    return LocalDeltaTable(root=root, name=name)


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]
