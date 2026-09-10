"""AI service API: run workflows, approve HITL requests, inspect traces & retrieval.

Endpoints
---------
GET  /healthz                 liveness
POST /v1/runs                 start an editorial run (async task)
GET  /v1/runs/{id}            run status/result
POST /v1/approvals/{run_id}   operator approve/reject for suspended runs
GET  /v1/approvals            pending HITL approvals
GET  /v1/traces/{run_id}      trace spans + token usage (MLflow-style)
POST /v1/search               semantic search (vector search backend)
"""

from __future__ import annotations

import asyncio
import hmac
import uuid
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from media_intel.agents.workflow import EditorialOrchestrator, WorkflowServices, build_editorial_graph
from media_intel.bootstrap import Platform, build_platform
from media_intel.core.observability import InMemoryTraceStore
from media_intel.core.settings import Settings
from media_intel.runtime.events import EventBus

_TASKS: dict[str, asyncio.Task] = {}


# ---------------------------------------------------------------- schemas
# Defined at module scope so FastAPI (with postponed annotations) resolves them.


class RunRequest(BaseModel):
    objective: str = Field(min_length=1, max_length=2000)


class ApprovalRequest(BaseModel):
    action: str = Field(pattern="^(approve|reject)$")
    reason: str = ""


class SearchRequest(BaseModel):
    query: str
    k: int = Field(default=4, ge=1, le=20)
    index: str = "editorial_kb"


def create_default_app(settings: Settings | None = None) -> FastAPI:
    """App factory used by uvicorn ``--factory``."""
    return create_app(settings or Settings.from_env())


def create_app(settings: Settings | None = None, platform: Platform | None = None) -> FastAPI:
    if settings is None:
        settings = platform.settings if platform is not None else Settings.from_env()
    elif platform is not None and platform.settings != settings:
        import dataclasses

        platform = dataclasses.replace(platform, settings=settings)
    platform = platform or build_platform(settings)
    graph = build_editorial_graph(
        WorkflowServices(
            llm=platform.llm,
            model_large=platform.settings.model_large,
            model_small=platform.settings.model_small,
            registry=platform.registry,
            publish_queue=platform.publish_queue,
            trace_store=platform.trace_store,
            settings=platform.settings,
        )
    )
    orchestrator = EditorialOrchestrator(graph, services=_services(platform))
    bus = EventBus()

    app = FastAPI(
        title="Media Intelligence Agent Platform",
        version="0.1.0",
        description="Supervisor + specialist agents over media data, with HITL approvals, "
        "RAG, cost/latency governance and MLflow-style tracing.",
    )

    # ------------------------------------------------------------ auth
    def require_key(x_api_key: str = Header(default="")) -> None:
        expected = platform.settings.api_key
        if expected and not hmac.compare_digest(x_api_key.encode(), expected.encode()):
            raise HTTPException(status_code=401, detail="invalid API key")

    # ------------------------------------------------------------ routes
    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/runs", dependencies=[Depends(require_key)])
    async def create_run(body: RunRequest) -> dict[str, Any]:
        run_id = f"ed-{uuid.uuid4().hex[:8]}"
        task = asyncio.create_task(orchestrator.run(objective=body.objective, run_id=run_id))
        _TASKS[run_id] = task
        await asyncio.sleep(0)
        return {"run_id": run_id, "status": "started"}

    @app.get("/v1/runs/{run_id}", dependencies=[Depends(require_key)])
    async def get_run(run_id: str) -> dict[str, Any]:
        task = _TASKS.get(run_id)
        if task is None:
            raise HTTPException(404, "unknown run")
        if not task.done():
            return {"run_id": run_id, "status": "running"}
        try:
            result = task.result()
        except Exception as exc:
            raise HTTPException(502, {"error": "run failed", "detail": str(exc)}) from exc
        return {
            "run_id": run_id,
            "status": result.get("outputs", {}).get("status", "completed"),
            "result": _strip(result),
        }

    @app.get("/v1/approvals", dependencies=[Depends(require_key)])
    async def list_approvals() -> dict[str, Any]:
        return {"pending": orchestrator.pending()}

    @app.post("/v1/approvals/{run_id}", dependencies=[Depends(require_key)])
    async def decide(run_id: str, body: ApprovalRequest) -> dict[str, Any]:
        result = await orchestrator.decide(run_id, body.action, body.reason)
        return {"run_id": run_id, "decision": body.action, "result": _strip(result)}

    @app.get("/v1/traces/{run_id}", dependencies=[Depends(require_key)])
    async def get_trace(run_id: str) -> dict[str, Any]:
        if isinstance(platform.trace_store, InMemoryTraceStore):
            traces = platform.trace_store.for_run(run_id)
            if not traces:
                raise HTTPException(404, "no trace for run")
            return {"run_id": run_id, "trace": traces[0].to_dict()}
        raise HTTPException(501, "trace store is external (MLflow)")

    @app.post("/v1/search", dependencies=[Depends(require_key)])
    async def search(body: SearchRequest) -> dict[str, Any]:
        results = platform.retriever.retrieve(body.index, body.query, k=body.k)
        return {"index": body.index, "results": results}

    return app


def _services(platform: Platform) -> WorkflowServices:
    return WorkflowServices(
        llm=platform.llm,
        model_large=platform.settings.model_large,
        model_small=platform.settings.model_small,
        registry=platform.registry,
        publish_queue=platform.publish_queue,
        trace_store=platform.trace_store,
        settings=platform.settings,
    )


def _strip(state: dict[str, Any]) -> dict[str, Any]:
    out = dict(state)
    out.pop("messages", None)
    return out
