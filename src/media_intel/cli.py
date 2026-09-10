"""Command-line interface: seed mock data, run the E2E demo, serve the API."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from media_intel.bootstrap import build_platform
from media_intel.core.settings import Settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="media_intel", description="Media Intelligence Agent Platform")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("seed", help="populate local lakehouse + vector indexes with mock media data")
    sub.add_parser("demo", help="run the end-to-end editorial demo (offline, mock LLM)")
    serve = sub.add_parser("serve", help="start the FastAPI service")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    if args.command == "seed":
        from media_intel.data.seed import seed_all

        seed_all(build_platform(Settings.from_env()))
        print("seeded mock media data into .lakehouse/")
        return 0
    if args.command == "demo":
        return asyncio.run(_demo())
    if args.command == "serve":
        import uvicorn

        uvicorn.run("media_intel.api.main:create_default_app", factory=True, host="127.0.0.1", port=args.port)
        return 0
    return 1


async def _demo() -> int:
    from media_intel.agents.workflow import EditorialOrchestrator, WorkflowServices, build_editorial_graph

    platform = build_platform(Settings.from_env())
    from media_intel.data.seed import ensure_seeded

    ensure_seeded(platform)

    services = WorkflowServices(
        llm=platform.llm,
        model_large=platform.settings.model_large,
        model_small=platform.settings.model_small,
        registry=platform.registry,
        publish_queue=platform.publish_queue,
        trace_store=platform.trace_store,
        settings=platform.settings,
    )
    graph = build_editorial_graph(services)
    orch = EditorialOrchestrator(graph, services=services)

    print("=== editorial run: matchday objective ===")
    result = await orch.run("Cover the El Clasico match: goals, turning point, audience reaction")
    print(json.dumps(_summarize(result), indent=2, ensure_ascii=False, default=str))

    if result.get("pending_approval"):
        print("\n=== HITL: publication request is awaiting approval ===")
        decision = "approve"  # simulate operator approval
        resumed = await orch.decide(result["run_id"], decision, "demo operator")
        print(json.dumps(_summarize(resumed), indent=2, ensure_ascii=False, default=str))

    traces = platform.trace_store.for_run(result["run_id"]) if hasattr(platform.trace_store, "for_run") else []
    if traces:
        t = traces[0]
        print("\n=== trace summary (MLflow-compatible) ===")
        print(f"duration_ms={t.duration_ms} tokens={t.token_usage} cost_usd={round(t.cost_usd, 6)} spans={len(t.spans)}")
        for span in t.spans:
            print(f"  [{span.status:>9}] {span.name} ({span.duration_ms} ms)")
    return 0


def _summarize(state: dict) -> dict:
    outputs = state.get("outputs", {})
    return {
        "run_id": state.get("run_id"),
        "route": state.get("route"),
        "status": outputs.get("status"),
        "narrative": (outputs.get("narrative") or outputs.get("analysis") or "")[:220],
        "metadata": outputs.get("metadata"),
        "quality": state.get("quality"),
        "publication": outputs.get("publication"),
        "budget": state.get("budget_summary"),
        "pending_approval": bool(state.get("pending_approval")),
        "escalation": state.get("escalation"),
    }


if __name__ == "__main__":
    sys.exit(main())
