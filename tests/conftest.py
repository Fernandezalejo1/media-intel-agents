"""Shared fixtures: isolated settings/platform and editorial orchestrator."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator

import pytest

os.environ["AUTO_APPROVE"] = "false"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from media_intel.bootstrap import Platform  # noqa: E402
from media_intel.core.observability import InMemoryTraceStore  # noqa: E402
from media_intel.core.settings import Settings  # noqa: E402
from media_intel.data.lakehouse import LocalDeltaTable  # noqa: E402
from media_intel.data.retrieval import LocalVectorSearch, MockEmbeddings, Retriever  # noqa: E402
from media_intel.llm.mock_provider import MockLLMClient  # noqa: E402
from media_intel.tools.media_tools import build_media_tools  # noqa: E402


def make_settings(**overrides) -> Settings:
    return Settings(
        llm_provider="mock",
        embedding_provider="mock",
        embedding_dim=128,
        auto_approve=False,
        **overrides,
    )


@pytest.fixture()
def platform(tmp_path) -> Iterator[Platform]:
    settings = make_settings()
    embeddings = MockEmbeddings(dim=128)
    vectors = LocalVectorSearch(root=str(tmp_path / "vectors"), embedding_dim=128)
    retriever = Retriever(embeddings, vectors)
    events = LocalDeltaTable(str(tmp_path / "lakehouse"), "events")
    audience = LocalDeltaTable(str(tmp_path / "lakehouse"), "audience")
    registry, queue = build_media_tools(events_table=events, audience_table=audience, retriever=retriever)

    from media_intel.data.seed import ensure_seeded

    platform = Platform(
        settings=settings,
        llm=MockLLMClient(latency_ms=0),
        registry=registry,
        publish_queue=queue,
        retriever=retriever,
        events_table=events,
        audience_table=audience,
        trace_store=InMemoryTraceStore(),
        embeddings=embeddings,
        vector_backend=vectors,
    )
    ensure_seeded(platform)
    yield platform


@pytest.fixture()
def orchestrator(platform: Platform):
    from media_intel.agents.workflow import EditorialOrchestrator, WorkflowServices, build_editorial_graph
    services = WorkflowServices(
        llm=platform.llm,
        model_large=platform.settings.model_large,
        model_small=platform.settings.model_small,
        registry=platform.registry,
        publish_queue=platform.publish_queue,
        trace_store=platform.trace_store,
        settings=platform.settings,
    )
    graph = build_editorial_graph(services).with_checkpointing(str(tmp_checkpoints()))
    return EditorialOrchestrator(graph, services=services)


def tmp_checkpoints() -> str:
    import tempfile

    return tempfile.mkdtemp(prefix="media-intel-ckpt-")
