"""Run/step tracing compatible with MLflow Tracing concepts.

Every run produces a trace made of spans (steps), events and token usage.
The :class:`TraceStore` acts as a local sink; ``MlflowTraceSink`` forwards
traces to MLflow when ``MLFLOW_ENABLED=true``.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

# Active tracer/span for the current async context; lets graph nodes and
# agents emit spans without threading the tracer through every call.
CURRENT_TRACER: ContextVar["Tracer | None"] = ContextVar("current_tracer", default=None)
CURRENT_SPAN: ContextVar["Span | None"] = ContextVar("current_span", default=None)


@dataclass
class SpanEvent:
    ts: float
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    span_id: str
    trace_id: str
    parent_id: str | None
    name: str
    started_at: float
    ended_at: float | None = None
    status: str = "OK"  # OK | ERROR
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    token_usage: dict[str, int] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.ended_at is None:
            return 0.0
        return round((self.ended_at - self.started_at) * 1000, 3)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TraceSink(Protocol):
    def write_trace(self, trace: "Trace") -> None: ...


@dataclass
class Trace:
    trace_id: str
    run_id: str
    started_at: float
    ended_at: float | None = None
    spans: list[Span] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        if self.ended_at is None:
            return 0.0
        return round((self.ended_at - self.started_at) * 1000, 3)

    @property
    def token_usage(self) -> dict[str, int]:
        usage = {"input": 0, "output": 0}
        for span in self.spans:
            for key in usage:
                usage[key] += span.token_usage.get(key, 0)
        usage["total"] = usage["input"] + usage["output"]
        return usage

    @property
    def cost_usd(self) -> float:
        return sum(span.attributes.get("cost_usd", 0.0) for span in self.spans)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "token_usage": self.token_usage,
            "cost_usd": round(self.cost_usd, 6),
            "spans": [s.to_dict() for s in self.spans],
        }


class InMemoryTraceStore:
    """Local sink + queryable store (swap for MLflow backend in prod)."""

    def __init__(self) -> None:
        self._traces: dict[str, Trace] = {}

    def write_trace(self, trace: Trace) -> None:
        self._traces[trace.trace_id] = trace

    def get(self, trace_id: str) -> Trace | None:
        return self._traces.get(trace_id)

    def for_run(self, run_id: str) -> list[Trace]:
        return [t for t in self._traces.values() if t.run_id == run_id]

    def all(self) -> list[Trace]:
        return list(self._traces.values())


class _NoopSink:
    def write_trace(self, trace: Trace) -> None:  # pragma: no cover
        pass


class Tracer:
    """Context-manager tracer producing one :class:`Trace` per run."""

    def __init__(self, store: TraceSink | None = None, run_id: str | None = None):
        self.store = store or _NoopSink()
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.trace = Trace(
            trace_id=uuid.uuid4().hex,
            run_id=self.run_id,
            started_at=time.time(),
        )

    @contextmanager
    def span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        parent: Span | None = None,
    ) -> Iterator[Span]:
        span = Span(
            span_id=uuid.uuid4().hex[:12],
            trace_id=self.trace.trace_id,
            parent_id=parent.span_id if parent else None,
            name=name,
            started_at=time.time(),
            attributes=dict(attributes or {}),
        )
        self.trace.spans.append(span)
        span_token = CURRENT_SPAN.set(span)
        try:
            yield span
        except Exception as exc:
            if span.status == "OK":
                span.status = "ERROR"
            span.events.append(
                SpanEvent(ts=time.time(), name="exception", attributes={"type": type(exc).__name__, "message": str(exc)})
            )
            raise
        finally:
            span.ended_at = time.time()
            CURRENT_SPAN.reset(span_token)

    def record_tokens(self, span: Span, model: str, input_tokens: int, output_tokens: int, cost_usd: float) -> None:
        span.token_usage = {"input": input_tokens, "output": output_tokens}
        span.attributes.setdefault("model", model)
        span.attributes["input_tokens"] = input_tokens
        span.attributes["output_tokens"] = output_tokens
        span.attributes["cost_usd"] = round(cost_usd, 6)

    def finish(self) -> Trace:
        if self.trace.ended_at is None:
            self.trace.ended_at = time.time()
            self.store.write_trace(self.trace)
        return self.trace


class MlflowTraceSink:
    """Forwards completed traces to an MLflow tracking server (optional)."""

    def __init__(self, tracking_uri: str, experiment: str = "media-intel-agents"):
        self.tracking_uri = tracking_uri
        self.experiment = experiment
        self._client: Any = None

    def _lazy_client(self) -> Any:
        if self._client is None:
            import mlflow  # optional dependency

            mlflow.set_tracking_uri(self.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", ""))
            try:
                mlflow.set_experiment(self.experiment)
            except Exception:  # noqa: BLE001
                pass
            self._client = mlflow
        return self._client

    def write_trace(self, trace: Trace) -> None:
        try:
            mlflow = self._lazy_client()
            with mlflow.start_run(run_name=f"agent-run-{trace.run_id}"):
                for span in trace.spans:
                    if span.token_usage:
                        mlflow.log_metrics(
                            {
                                f"tokens.{span.name}.input": span.token_usage.get("input", 0),
                                f"tokens.{span.name}.output": span.token_usage.get("output", 0),
                            }
                        )
                mlflow.log_dict(trace.to_dict(), "trace.json")
        except Exception:  # noqa: BLE001 - tracing must never break the run
            pass


def export_trace_jsonl(trace: Trace, path: str) -> None:
    """Persist a trace as JSON Lines (OTel-flavored export for offline debugging)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for span in trace.spans:
            fh.write(json.dumps(span.to_dict(), default=str) + "\n")
