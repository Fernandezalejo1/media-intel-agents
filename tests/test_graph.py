"""Graph engine tests: flow, conditional edges, Command, interrupts, resume."""

from __future__ import annotations

import pytest

from media_intel.agents.graph import Command, GraphInterrupted, StateGraph, interrupt


@pytest.mark.asyncio
async def test_linear_flow_updates_state():
    async def a(state):
        return {"x": 1}

    async def b(state):
        return {"y": state["x"] + 1}

    g = StateGraph().add_node("a", a).add_node("b", b).set_entry("a").add_edge("a", "b").compile()
    state = await g.run({})
    assert state["x"] == 1 and state["y"] == 2


@pytest.mark.asyncio
async def test_conditional_edges_route_state():
    async def classify(state):
        return {"kind": "vip"}

    async def vip(state):
        return {"lane": "vip"}

    async def normal(state):
        return {"lane": "normal"}

    g = (
        StateGraph()
        .add_node("classify", classify)
        .add_node("vip", vip)
        .add_node("normal", normal)
        .set_entry("classify")
        .add_conditional_edges("classify", lambda s: "vip" if s.get("kind") == "vip" else "normal")
        .compile()
    )
    state = await g.run({})
    assert state["lane"] == "vip"


@pytest.mark.asyncio
async def test_command_goto_jumps():
    async def start(state):
        return Command(update={"n": 1}, goto="finish")

    async def finish(state):
        return {"done": True}

    g = StateGraph().add_node("start", start).add_node("finish", finish).set_entry("start").compile()
    state = await g.run({})
    assert state["done"] is True


@pytest.mark.asyncio
async def test_interrupt_suspends_and_resumes(tmp_path):
    async def work(state):
        return {"draft": "ready"}

    async def approve(state):
        return interrupt({"type": "approval_required", "item": state["draft"]})

    async def after(state):
        decision = state["escalation"]["decision"]["action"]
        return {"result": f"approved:{decision}"}

    g = (
        StateGraph()
        .add_node("work", work)
        .add_node("approve", approve, human_in_the_loop=True)
        .add_node("after", after)
        .set_entry("work")
        .add_edge("work", "approve")
        .add_edge("approve", "after")
        .with_checkpointing(str(tmp_path))
        .compile()
    )

    with pytest.raises(GraphInterrupted) as exc_info:
        await g.run({"run_id": "hitl-1"}, run_id="hitl-1")
    assert exc_info.value.payload["type"] == "approval_required"
    assert exc_info.value.payload["item"] == "ready"

    state = await g.resume("hitl-1", {"action": "approve"})
    assert state["result"] == "approved:approve"


@pytest.mark.asyncio
async def test_resume_without_checkpoint_fails(tmp_path):
    g = StateGraph().add_node("noop", lambda s: {}).set_entry("noop").with_checkpointing(str(tmp_path)).compile()
    with pytest.raises(ValueError, match="no checkpoint"):
        await g.resume("missing-run", {"action": "approve"})


@pytest.mark.asyncio
async def test_pending_interruptions_lists_checkpoint(tmp_path):
    async def approve(state):
        return interrupt({"q": "publish?"})

    g = (
        StateGraph()
        .add_node("approve", approve, human_in_the_loop=True)
        .set_entry("approve")
        .with_checkpointing(str(tmp_path))
        .compile()
    )
    with pytest.raises(GraphInterrupted):
        await g.run({"run_id": "p-1"}, run_id="p-1")
    pending = g.pending_interruptions()
    assert pending and pending[0]["run_id"] == "p-1" and pending[0]["payload"] == {"q": "publish?"}
