"""Runtime composition root.

Selects concrete implementations for every port based on settings/env:
- LLM: mock | openai-compatible
- Vector search: local simulator | Databricks Vector Search
- Lakehouse tables: local JSONL | Databricks SQL warehouse
- Tracing: in-memory store (+ optional MLflow sink)

Everything downstream depends only on the protocols, so this is the single
place a real Databricks workspace gets plugged in.
"""

from __future__ import annotations

from dataclasses import dataclass

from media_intel.core.observability import InMemoryTraceStore, MlflowTraceSink, TraceSink
from media_intel.core.settings import Settings
from media_intel.data.lakehouse import DatabricksLakehouse, LakehouseTable, LocalDeltaTable
from media_intel.data.retrieval import EmbeddingProvider, LocalVectorSearch, MockEmbeddings, OpenAIEmbeddings, Retriever, VectorSearchBackend
from media_intel.data.databricks import DatabricksVectorSearch, databricks_configured
from media_intel.llm.base import LLMClient
from media_intel.llm.mock_provider import MockLLMClient
from media_intel.llm.openai_provider import OpenAICompatibleClient
from media_intel.tools.media_tools import PublishQueue, build_media_tools
from media_intel.tools.registry import ToolRegistry


@dataclass
class Platform:
    """Fully wired application dependencies."""

    settings: Settings
    llm: LLMClient
    registry: ToolRegistry
    publish_queue: PublishQueue
    retriever: Retriever
    events_table: LakehouseTable
    audience_table: LakehouseTable
    trace_store: TraceSink
    embeddings: EmbeddingProvider
    vector_backend: VectorSearchBackend


def build_platform(settings: Settings | None = None) -> Platform:
    settings = settings or Settings.from_env()

    # --- LLM ------------------------------------------------------------
    if settings.llm_provider == "openai":
        llm: LLMClient = OpenAICompatibleClient(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            timeout_s=settings.llm_timeout_s,
        )
    else:
        import os

        llm = MockLLMClient(latency_ms=int(os.environ.get("MOCK_LLM_LATENCY_MS", "0")))

    # --- Embeddings + vector search --------------------------------------
    if settings.embedding_provider == "openai":
        embeddings: EmbeddingProvider = OpenAIEmbeddings(model=settings.embedding_model)
    else:
        embeddings = MockEmbeddings(dim=settings.embedding_dim)

    if databricks_configured() and settings.databricks_vector_endpoint:
        vector_backend: VectorSearchBackend = DatabricksVectorSearch(
            host=settings.databricks_host,
            token=settings.databricks_token,
            endpoint=settings.databricks_vector_endpoint,
        )
    else:
        vector_backend = LocalVectorSearch(embedding_dim=settings.embedding_dim)

    retriever = Retriever(embeddings, vector_backend)

    # --- Lakehouse tables -------------------------------------------------
    if databricks_configured() and settings.databricks_warehouse_id:
        kwargs = dict(
            host=settings.databricks_host,
            token=settings.databricks_token,
            warehouse_id=settings.databricks_warehouse_id,
            catalog=settings.databricks_catalog,
            schema=settings.databricks_schema,
        )
        events_table: LakehouseTable = DatabricksLakehouse(**kwargs)
        audience_table: LakehouseTable = DatabricksLakehouse(**kwargs)
    else:
        events_table = LocalDeltaTable(root=".lakehouse", name="events")
        audience_table = LocalDeltaTable(root=".lakehouse", name="audience")

    # --- Tools -------------------------------------------------------------
    registry, publish_queue = build_media_tools(
        events_table=events_table,
        audience_table=audience_table,
        retriever=retriever,
    )

    # --- Tracing -------------------------------------------------------------
    if settings.mlflow_enabled:
        trace_store: TraceSink = MlflowTraceSink(tracking_uri=settings.mlflow_tracking_uri)
    else:
        trace_store = InMemoryTraceStore()

    return Platform(
        settings=settings,
        llm=llm,
        registry=registry,
        publish_queue=publish_queue,
        retriever=retriever,
        events_table=events_table,
        audience_table=audience_table,
        trace_store=trace_store,
        embeddings=embeddings,
        vector_backend=vector_backend,
    )
