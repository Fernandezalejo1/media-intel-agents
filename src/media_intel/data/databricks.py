"""Databricks adapters - swap local simulators for managed services via env vars.

Backends are selected in ``media_intel.bootstrap`` using:
``DATABRICKS_HOST`` + ``DATABRICKS_TOKEN`` (+ optional endpoints). Without
them, the platform runs fully offline on the local simulators.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from media_intel.data.retrieval import VectorSearchBackend


class DatabricksVectorSearch(VectorSearchBackend):
    """Adapter for Databricks Vector Search (query + upsert via REST)."""

    def __init__(self, host: str, token: str, endpoint: str):
        self.host = host.rstrip("/")
        self.token = token
        self.endpoint = endpoint

    def upsert(self, index_id: str, records: list[dict[str, Any]]) -> int:  # pragma: no cover
        client = self._client()
        for chunk in _chunks(records, 250):
            client.post_json(
                f"/api/2.0/vector-search/indexes/{index_id}/upsert-data",
                {"inputs_json": [r.get("vector") for r in chunk]},
            )
        return len(records)

    def query(  # pragma: no cover
        self,
        index_id: str,
        vector: list[float],
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        client = self._client()
        payload: dict[str, Any] = {
            "columns": ["id", "text", "metadata"],
            "query_vector": vector,
            "num_results": k,
        }
        if filter:
            payload["filters"] = filter
        result = client.post_json(f"/api/2.0/vector-search/indexes/{index_id}/query", payload)
        return [
            {
                "id": row.get("id"),
                "text": row.get("text"),
                "metadata": row.get("metadata", {}),
                "score": row.get("score", 0.0),
            }
            for row in result.get("result", {}).get("data_array", [])
        ]

    def _client(self) -> "_DbxRest":
        return _DbxRest(self.host, self.token)


class _DbxRest:
    """Tiny REST helper to avoid a hard SDK dependency in tests."""

    def __init__(self, host: str, token: str):
        self.host = host
        self.token = token

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        resp = httpx.post(
            f"https://{self.host}{path}",
            json=payload,
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()


class MosaicModelServingClient:
    """Calls a Mosaic AI Model Serving endpoint (OpenAI-compatible chat API)."""

    def __init__(self, host: str, token: str, endpoint: str):
        self.rest = _DbxRest(host, token)
        self.endpoint = endpoint

    def complete(self, messages: list[dict[str, Any]], max_tokens: int = 1024) -> str:  # pragma: no cover
        result = self.rest.post_json(
            f"/serving-endpoints/{self.endpoint}/invocations",
            {"messages": messages, "max_tokens": max_tokens},
        )
        return result.get("choices", [{}])[0].get("message", {}).get("content", "")


def databricks_configured() -> bool:
    return bool(os.environ.get("DATABRICKS_HOST") and os.environ.get("DATABRICKS_TOKEN"))


# ---------------------------------------------------------------------------
# Databricks Asset Bundle (DAB) reference - deployable via `databricks bundle deploy`
# ---------------------------------------------------------------------------

DAB_SAMPLE = """\
# databricks.yml - Databricks Asset Bundle for the media-intel agents
bundle:
  name: media-intel-agents

include:
  - resources/*.yml

targets:
  dev:
    default: true
    workspace:
      host: https://YOUR-WORKSPACE.cloud.databricks.com

resources:
  jobs:
    media_intel_batch:
      name: media-intel-batch
      tasks:
        - task_key: run_agent_batch
          python_wheel_task:
            package_name: media_intel
            entry_point: demo
          job_cluster_key: agent_cluster
      job_clusters:
        - job_cluster_key: agent_cluster
          new_cluster:
            spark_version: 15.4.x-scala2.12
            node_type_id: i3.xlarge
            num_workers: 1
"""
