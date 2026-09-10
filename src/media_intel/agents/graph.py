"""Stateful graph engine (LangGraph-equivalent, dependency-free).

- Typed shared :class:`AgentState` with checkpointing (time-travel friendly).
- Nodes are async callables ``(state) -> dict`` returning *partial updates*.
- Edges: fixed, conditional, or terminal.
- :class:`Command` supports goto + state patch in one return.
- ``interrupt()`` marks a node as human-in-the-loop: the graph suspends,
  checkpoints, and resumes on ``resume()`` with an operator decision.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

from media_intel.core.observability import CURRENT_TRACER

@dataclass
class Command:
    """Node return value: patch state and/or jump to a specific node."""

    update: dict[str, Any] = field(default_factory=dict)
    goto: str | None = None


NodeFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any] | Command]]
Router = Callable[[dict[str, Any]], str]


@dataclass
class AgentState:
    """Shared state keys used across the platform's graphs."""

    run_id: str = ""
    objective: str = ""
    route: str = ""
    route_reasoning: str = ""
    facts: dict[str, Any] = field(default_factory=dict)
    context: list[dict[str, Any]] = field(default_factory=list)  # retrieved chunks
    messages: list[dict[str, str]] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    escalation: dict[str, Any] | None = None
    error: str | None = None
    budget_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "objective": self.objective,
            "route": self.route,
            "facts": self.facts,
            "context_count": len(self.context),
            "outputs": self.outputs,
            "quality": self.quality,
            "escalation": self.escalation,
            "error": self.error,
            "budget_summary": self.budget_summary,
        }


INTERRUPT_KEY = "__interrupt__"


def interrupt(payload: dict[str, Any]) -> dict[str, Any]:
    """Marker a node returns to request human input."""
    return {INTERRUPT_KEY: payload}


@dataclass
class NodeSpec:
    name: str
    fn: Callable[[dict[str, Any]], Awaitable[dict[str, Any] | Command]]
    human_in_the_loop: bool = False


class StateGraph:
    """Assemble nodes/edges, compile, run with checkpoints + interrupts."""

    def __init__(self, state_type: type = AgentState):
        self.state_type = state_type
        self._nodes: dict[str, NodeSpec] = {}
        self._edges: dict[str, str] = {}
        self._conditional: dict[str, Router] = {}
        self._entry: str | None = None
        self._checkpoint_dir: str = ".runs/checkpoints"
        self._trace_sink = None

    # ------------------------------------------------------------- assembly

    def add_node(self, name: str, fn: Callable[..., Any], *, human_in_the_loop: bool = False) -> "StateGraph":
        self._nodes[name] = NodeSpec(name=name, fn=fn, human_in_the_loop=human_in_the_loop)
        return self

    def add_edge(self, start: str, end: str) -> "StateGraph":
        self._edges[start] = end
        return self

    def add_conditional_edges(self, start: str, router: Router) -> "StateGraph":
        self._conditional[start] = router
        return self

    def set_entry(self, name: str) -> "StateGraph":
        self._entry = name
        return self

    def with_checkpointing(self, directory: str = ".runs/checkpoints") -> "StateGraph":
        self._checkpoint_dir = directory
        return self

    def with_trace_store(self, store: Any) -> "StateGraph":
        self._trace_sink = store
        return self

    def compile(self) -> "CompiledGraph":
        if not self._entry:
            raise ValueError("entry node not set")
        return CompiledGraph(
            nodes=self._nodes,
            edges=self._edges,
            conditional=self._conditional,
            entry=self._entry,
            checkpoint_dir=self._checkpoint_dir,
            trace_store=self._trace_sink,
        )


@dataclass
class _GraphConfig:
    nodes: dict[str, NodeSpec]
    edges: dict[str, str]
    conditional: dict[str, Router]
    entry: str
    checkpoint_dir: str
    trace_store: Any


class GraphInterrupted(Exception):
    """Raised when a HITL node suspends the graph (checkpoint is kept)."""

    def __init__(self, run_id: str, node: str, payload: dict[str, Any], state_snapshot: dict[str, Any]):
        self.run_id = run_id
        self.node = node
        self.payload = payload
        self.state_snapshot = state_snapshot
        super().__init__(f"interrupt at '{node}': {payload}")


class CompiledGraph:
    def __init__(self, nodes: dict[str, NodeSpec], edges: dict[str, str], conditional: dict[str, Router], entry: str, checkpoint_dir: str, trace_store: Any):
        self.nodes = nodes
        self.edges = edges
        self.conditional = conditional
        self.entry = entry
        self.checkpoint_dir = checkpoint_dir
        self.trace_store = trace_store
        from media_intel.core.observability import Tracer

        self._tracer_cls = Tracer

    # ------------------------------------------------------------------ run

    async def run(self, initial: dict[str, Any] | None = None, run_id: str | None = None, max_steps: int = 32) -> dict[str, Any]:
        state = dict(initial or {})
        state.setdefault("run_id", run_id or uuid.uuid4().hex[:12])
        tracer = self._tracer_cls(store=self.trace_store or _null_sink(), run_id=state["run_id"])
        current = self.entry
        steps = 0
        ctx_token = CURRENT_TRACER.set(tracer)

        try:
            with tracer.span("graph.run", {"entry": self.entry}) as outer:
              try:
                while steps < max_steps:
                    steps += 1
                    node = self.nodes.get(current)
                    if node is None:
                        raise ValueError(f"unknown node '{current}'")
                    with tracer.span(f"node.{node.name}", {"hitl": node.human_in_the_loop}) as span:
                        update = await node.fn(state)
                        interrupt_payload = update.get(INTERRUPT_KEY) if isinstance(update, dict) else None
                        if interrupt_payload is not None:
                            self._checkpoint(state, node.name, interrupt_payload)
                            span.status = "SUSPENDED"
                            raise GraphInterrupted(state["run_id"], node.name, interrupt_payload, dict(state))
                    if isinstance(update, Command):
                        state.update(update.update)
                        nxt = update.goto or self._next_node(current, state)
                    else:
                        state.update(update)
                        nxt = self._next_node(current, state)
                    if nxt is None:
                        break  # terminal node
                    current = nxt
                else:
                    state["error"] = f"max_steps={max_steps} reached"
              except GraphInterrupted:
                outer.status = "SUSPENDED"
                raise
        finally:
            tracer.finish()
            CURRENT_TRACER.reset(ctx_token)
        return state

    async def resume(self, run_id: str, decision: dict[str, Any], max_steps: int = 32) -> dict[str, Any]:
        """Continue a suspended HITL run from its checkpoint."""
        snap = self._load(run_id)
        if snap is None:
            raise ValueError(f"no checkpoint for run {run_id}")
        state, node_name, payload = snap
        state.setdefault("escalation", {})
        state["escalation"]["decision"] = decision
        state["pending_approval"] = {"node": node_name, "payload": payload}

        tracer = self._tracer_cls(store=self.trace_store or _null_sink(), run_id=run_id)
        current = self._next_after_interrupt(node_name, state)
        steps = 0
        ctx_token = CURRENT_TRACER.set(tracer)
        try:
            with tracer.span("graph.resume", {"from": node_name, "decision": decision}):
                while steps < max_steps:
                    steps += 1
                    node = self.nodes.get(current)
                    if node is None:
                        raise ValueError(f"unknown node '{current}'")
                    with tracer.span(f"node.{node.name}") as span:
                        update = await node.fn(state)
                        interrupt_payload = update.get(INTERRUPT_KEY) if isinstance(update, dict) else None
                        if interrupt_payload is not None:
                            self._checkpoint(state, node.name, interrupt_payload)
                            span.status = "SUSPENDED"
                            raise GraphInterrupted(run_id, node.name, interrupt_payload, dict(state))
                    if isinstance(update, Command):
                        state.update(update.update)
                        nxt = update.goto or self._next_node(current, state)
                    else:
                        state.update(update)
                        nxt = self._next_node(current, state)
                    if nxt is None:
                        break  # terminal node
                    current = nxt
        finally:
            tracer.finish()
            CURRENT_TRACER.reset(ctx_token)
        return state

    def _next_node(self, node_name: str, state: dict[str, Any]) -> str | None:
        """Resolve the successor of a node via its conditional router or fixed edge."""
        router = self.conditional.get(node_name)
        if router is not None:
            return router(state)
        return self.edges.get(node_name)

    def _next_after_interrupt(self, interrupted_node: str, state: dict[str, Any]) -> str:
        if interrupted_node in self.conditional:
            return self.conditional[interrupted_node](state)
        if interrupted_node in self.edges:
            return self.edges[interrupted_node]
        raise ValueError(f"no outgoing edge from interrupted node '{interrupted_node}'")

    # ---------------------------------------------------------- checkpoints

    def _checkpoint(self, state: dict[str, Any], node: str, payload: dict[str, Any]) -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # Runtime-only objects (e.g. the live RunBudget) are not serializable state.
        persisted = {k: v for k, v in state.items() if k != "budget"}
        record = {"state": persisted, "interrupted_at": node, "payload": payload, "ts": time.time()}
        with open(os.path.join(self.checkpoint_dir, f"{state['run_id']}.json"), "w", encoding="utf-8") as fh:
            json.dump(record, fh, default=str, indent=2)

    def _load(self, run_id: str) -> tuple[dict[str, Any], str, dict[str, Any]] | None:
        path = os.path.join(self.checkpoint_dir, f"{run_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as fh:
            record = json.load(fh)
        return record["state"], record["interrupted_at"], record["payload"]

    def pending_interruptions(self) -> list[dict[str, Any]]:
        out = []
        if not os.path.isdir(self.checkpoint_dir):
            return out
        for fname in os.listdir(self.checkpoint_dir):
            if fname.endswith(".json"):
                with open(os.path.join(self.checkpoint_dir, fname), encoding="utf-8") as fh:
                    rec = json.load(fh)
                out.append({"run_id": rec["state"].get("run_id"), "node": rec["interrupted_at"], "payload": rec["payload"]})
        return out


class _NullSink:
    def write_trace(self, trace: Any) -> None:
        pass


def _null_sink() -> _NullSink:
    return _NullSink()
