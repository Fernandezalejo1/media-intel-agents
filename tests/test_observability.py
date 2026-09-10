"""Tracing tests: spans, token accounting, error recording."""

from __future__ import annotations

import pytest

from media_intel.core.observability import InMemoryTraceStore, Tracer


def test_trace_records_spans_and_tokens():
    store = InMemoryTraceStore()
    tracer = Tracer(store=store, run_id="r-obs-1")

    with tracer.span("outer") as outer:
        with tracer.span("inner", parent=outer) as inner:
            tracer.record_tokens(inner, "gpt-4o-mini", 100, 50, 0.0001)

    trace = tracer.finish()
    assert trace.run_id == "r-obs-1"
    assert [s.name for s in trace.spans] == ["outer", "inner"]
    assert trace.token_usage["input"] == 100
    assert trace.token_usage["output"] == 50
    assert trace.token_usage["total"] == 150
    stored = store.for_run("r-obs-1")
    assert stored and stored[0].trace_id == trace.trace_id


def test_span_error_status_and_event():
    store = InMemoryTraceStore()
    tracer = Tracer(store=store, run_id="r-obs-2")

    with pytest.raises(ValueError):
        with tracer.span("boom") as span:
            raise ValueError("kaput")

    trace = tracer.finish()
    span = trace.spans[0]
    assert span.status == "ERROR"
    assert span.events and span.events[0].name == "exception"
    assert span.events[0].attributes["type"] == "ValueError"


def test_suspended_status_survives_exception_path():
    store = InMemoryTraceStore()
    tracer = Tracer(store=store, run_id="r-obs-3")

    class Suspend(Exception):
        pass

    with pytest.raises(Suspend):
        with tracer.span("hitl-node") as span:
            span.status = "SUSPENDED"
            raise Suspend()

    trace = tracer.finish()
    assert trace.spans[0].status == "SUSPENDED"
