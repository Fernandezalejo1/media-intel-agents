# Architecture

## Layers

```
API / CLI            FastAPI service, CLI (seed | demo | serve)
Orchestration        EditorialOrchestrator → StateGraph engine
Agents               Supervisor, Narrative, Analyst, Metadata, Summarizer, QualityGate
Tools                ToolRegistry (retry, breakers, tracing) + media tools
Ports                LLMClient · EmbeddingProvider · VectorSearchBackend · LakehouseTable · TraceSink
Adapters (local)     MockLLM · MockEmbeddings · LocalVectorSearch · LocalDeltaTable · InMemoryTraceStore
Adapters (dbks)      OpenAICompatibleClient · DatabricksVectorSearch · DatabricksLakehouse · MlflowTraceSink · MosaicModelServingClient
```

## The editorial graph

```
route ──▶ matchday_lane ──▶ metadata_node ──▶ quality_gate ──▶ publish_request ──▶ finalize
   └────▶ analysis_lane ──────────────┘               │                 (HITL interrupt)
                                                      └── (failed) ──▶ finalize
```

- `route` — supervisor LLM call returns a `route_request` tool call (`matchday` | `analysis`).
- lanes — deterministic fact-gathering through tools (fixtures, events, audience, KB RAG),
  then specialist LLM calls (narrative or analysis+summary).
- `metadata_node` — strict-JSON metadata for CMS ingestion.
- `quality_gate` — JSON verdict; failure loops back to `route` once (bounded rewrite).
- `publish_request` — **HITL interrupt**: checkpoints state to `.runs/checkpoints/` and
  suspends; the operator approves/rejects via API or CLI; `resume()` continues from the
  checkpoint into `finalize`.
- `finalize` — applies the decision to the `PublishQueue` (published / rejected).

## Governance (cost & latency control)

`core/governance.py` keeps a per-run ledger: LLM call count, tokens in/out, elapsed
wall-clock. The budget is checked **before** each LLM call and recorded after; on breach
the run escalates (`escalation.level=operator`) instead of running away. Model tiering
(`model_small` for routing/metadata/summaries, `model_large` for narrative/analysis)
minimizes cost per call.

## Observability

Every run emits one trace: spans for `graph.run`, each `node.*`, each `agent.*`,
`llm.completion` (with token usage + cost) and `tool.*` calls (with arguments).
Traces are stored in-memory (queryable via `GET /v1/traces/{run_id}`), exportable as
JSONL, or forwarded to MLflow via `MlflowTraceSink`.

## Resilience

- LLM calls: retry w/ exponential backoff + jitter, circuit breaker, budget gate.
- Tool calls: per-tool retry policy and circuit breaker; errors returned to the model
  as tool results (self-correction) instead of crashing the run.
- Malformed JSON: `extract_json` fallbacks; quality-gate verdicts are validated.
- Unhandled errors: structured escalation envelope with budget summary attached.

## Data flow (event-driven option)

`runtime/events.py` provides a pub/sub bus (`match.*` wildcards) for wiring pipeline
events (e.g. `match.goal` → re-run insight) — the same shape as an Eventhub/Service Bus
integration on Databricks.
