# Media Intelligence Agent Platform

Plataforma multi-agente de grado producción para flujos de trabajo editoriales de
medios/deportes. Demuestra exactamente las responsabilidades del rol objetivo:
**sistemas agénticos (supervisor + sub-agentes), aplicaciones LLM, inteligencia
de medios + integración de datos, integración con el stack de Databricks y
producción (latencia, costo, observabilidad).**

Funciona 100% offline con un LLM mock determinista; se puede cambiar a cualquier
modelo compatible con OpenAI o un workspace de Databricks solo con variables de
entorno.

> **Multi-dominio:** el núcleo (grafo de agentes, supervisor, quality gate, HITL,
> presupuestos, tracing) es agnóstico al dominio. Cambiando las tools y los datos
> semilla se reutiliza para legal, finanzas, soporte, investigación, SRE, etc.

---

## Características

| Capacidad | Dónde |
|---|---|
| Arquitectura multi-agente (supervisor + sub-agentes) | `agents/workflow.py` — el supervisor enruta objetivos a lanes matchday/análisis |
| Orquestación con estado (LangGraph o equivalente) | `agents/graph.py` — `StateGraph` sin dependencias: estado tipado, edges condicionales, `Command(goto)`, interrupts, checkpoints |
| Agentes con herramientas (tool calling) | `agents/base.py` + `tools/registry.py` — specs tipo OpenAI, tool loop, retry, circuit breakers |
| Human-in-the-loop con escalación/aprobación | `interrupt()` del grafo + resume con checkpoint; el operador aprueba/rechaza vía API; `PublishQueue` |
| Servicios LLM: narrativa, metadata, resumen, análisis | `agents/specialists.py` — agentes Narrative / Metadata / Summarizer / Analyst |
| Minimizar llamadas LLM; control de costo y latencia | `core/governance.py` — topes de llamadas por run, topes de tokens, deadline verificados antes de cada llamada |
| RAG + búsqueda vectorial + KB | `data/retrieval.py` — embeddings + backend vectorial + `Retriever`; KB editorial e índices de transcripciones |
| Pipelines de datos estructurados + event-driven | `data/lakehouse.py` (tablas tipo Delta), `runtime/events.py` (event bus) |
| Databricks: Mosaic / Vector Search / Lakehouse / Lakebase / MLflow / DAB | capa de swap en `bootstrap.py`, `data/databricks.py` (adapters de Vector Search + Model Serving), `observability.py` (`MlflowTraceSink`), `deploy/databricks.yml` (DAB) |
| Tracing / observabilidad estilo MLflow | `core/observability.py` — spans, eventos, uso de tokens, costo por run; export JSONL |
| Manejo de errores, retries, fallbacks | `core/resilience.py` + breakers a nivel registry; pasada de reparación JSON; envelopes de escalación |
| APIs para apps downstream | Servicio FastAPI: runs, approvals, traces, búsqueda semántica |
| Requisitos estrictos de latencia | deadline del run en governance; timings por span; tiering de modelos small/large |

---

## Arquitectura

```
                         ┌────────────────────────────────────────────────┐
                         │                FastAPI service                  │
                         │  /v1/runs  /v1/approvals  /v1/traces  /v1/search│
                         └───────────────┬────────────────────────────────┘
                                         │
                     ┌───────────────────▼────────────────────┐
                     │        EditorialOrchestrator            │
                     │  (budgets, tracing, escalations, HITL)  │
                     └───────────────────┬────────────────────┘
                                         │
        ┌────────────────────────────────▼─────────────────────────────────┐
        │                     StateGraph (LangGraph-equivalent)            │
        │   route ─▶ matchday_lane / analysis_lane ─▶ metadata ─▶ quality  │
        │                                    └─▶ publish_request (INTERRUPT│
        │                                         waits for operator) ─▶ finalize │
        └──┬──────────────┬──────────────┬──────────────┬──────────────────┘
           │              │              │              │
      Supervisor    NarrativeAgent  AnalystAgent   MetadataAgent  QualityGate
      (router LLM)  (RAG + tools)   (RAG + tools)  (JSON output)  (JSON verdict)
           │              │              │
           └──────────────┴──────────────┴──▶ ToolRegistry (retry + circuit breaker)
                                     │
        ┌───────────────────────────┼─────────────────────────────┐
        │                           │                             │
   Lakehouse tables          Vector search / RAG            PublishQueue (HITL)
   (local JSONL ↔ Databricks SQL)  (local ↔ Databricks VS)   (checkpointed approvals)
```

### Flujo de un run (paso a paso)

1. **Entrada** — un editor envía un objetivo (p. ej. `"Cover the El Clasico: goles,
   punto de inflexión, reacción del público"`).
2. **Supervisor** — un LLM pequeño enruta el objetivo a la lane `matchday` o `analysis`.
3. **Recolección de hechos** — la lane junta datos vía las tools: fixtures, eventos
   del partido, métricas de audiencia, transcripciones y contexto de la KB editorial.
4. **Generación** — la lane llama a los especialistas (Narrative/Summary/Analyst).
5. **Metadata** — `MetadataAgent` genera título, keywords, tipo.
6. **Quality Gate** — `QualityGateAgent` evalúa el borrador. Si falla, reintenta una
   vez (loop acotado); si sigue mal, finaliza como `quality_rejected`.
7. **HITL** — `publish_request` levanta un `interrupt()`: el grafo se suspende, guarda
   checkpoint y espera al operador. `POST /v1/approvals/{run_id}` aprueba o rechaza.
   `resume()` continúa exactamente donde se detuvo.
8. **Finalize** — approve publica el contenido (PublishQueue), reject registra el
   rechazo, sin decisión = `quality_rejected`.

---

## Quickstart (offline, sin API keys)

```bash
# 1) Crear el entorno
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"     # Windows Git Bash
# ó: python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"  # Linux/macOS

# 2) Correr la demo E2E completa: route → tools → RAG → HITL → publish
.venv/Scripts/python -m media_intel.cli demo

# 3) Correr los 50 tests
.venv/Scripts/python -m pytest

# 4) Levantar la API
.venv/Scripts/python -m media_intel.cli serve --port 8000
```

La demo imprime la ruta elegida, la narrativa, el QA, la solicitud de publicación
que queda en espera y el trace final con tokens y costo.

---

## Quickstart con Docker (un comando)

```bash
docker compose up --build
# API:    http://localhost:8000
# Swagger: http://localhost:8000/docs
# Health:  http://localhost:8000/healthz
```

El compose arranca en modo `mock` (determinista, sin claves) y espera el
`healthcheck` de `/healthz` antes de reportar el servicio como sano.

---

## Pasar a un LLM real en 2 comandos (Ollama, sin claves de nube)

```bash
# 1) levanta la API + Ollama y baja un modelo
docker compose --profile llm up --build -d
docker compose exec ollama ollama pull qwen2.5:7b-instruct

# 2) apunta el provider OpenAI-compatible al Ollama del compose
docker compose up -d   -e LLM_PROVIDER=openai   -e OPENAI_BASE_URL=http://ollama:11434/v1   -e LLM_MODEL_LARGE=qwen2.5:7b-instruct   -e LLM_MODEL_SMALL=qwen2.5:7b-instruct
```

Con un modelo local chico el routing del supervisor y el veredicto del quality
gate van a ser de menor calidad que con GPT-4o o Claude — por eso la aprobación
sigue siendo humana. Lo importante es que **no cambia una línea de código**:
mismos agentes, mismas tools, mismos presupuestos, mismas trazas.

---

## API

```bash
curl -s localhost:8000/healthz

# iniciar un run (async)
curl -s -X POST localhost:8000/v1/runs -H 'content-type: application/json' \
     -d '{"objective": "Cubrir el Clasico: goles, punto de inflexión, audiencia"}'

# estado del run
curl -s localhost:8000/v1/runs/<run_id>

# aprobaciones pendientes (HITL)
curl -s localhost:8000/v1/approvals

# aprobar / rechazar un run suspendido
curl -s -X POST localhost:8000/v1/approvals/<run_id> -H 'content-type: application/json' \
     -d '{"action": "approve", "reason": "ok por el editor"}'

# traza del run (spans, tokens, costo)
curl -s localhost:8000/v1/traces/<run_id>

# búsqueda semántica
curl -s -X POST localhost:8000/v1/search -H 'content-type: application/json' \
     -d '{"query": "quién anotó el gol de la victoria", "index": "transcripts"}'
```

| Endpoint | Método | Descripción |
|---|---|---|
| `/healthz` | GET | Liveness |
| `/v1/runs` | POST | Inicia un run editorial (tarea async) |
| `/v1/runs/{id}` | GET | Estado / resultado del run |
| `/v1/approvals` | GET | Aprobaciones HITL pendientes |
| `/v1/approvals/{run_id}` | POST | Aprobar / rechazar un run suspendido |
| `/v1/traces/{run_id}` | GET | Spans + uso de tokens (estilo MLflow) |
| `/v1/search` | POST | Búsqueda semántica |

Si `API_KEY=...` está seteada, todos los endpoints (excepto `/healthz`) requieren el
header `X-API-Key`.

---

## Cambiar a modelos reales (compatibles con OpenAI)

```bash
cp .env.example .env
# LLM_PROVIDER=openai, OPENAI_API_KEY=..., OPENAI_BASE_URL=...
# (OpenAI / Azure / vLLM / Ollama / LM Studio / etc.)
```

El provider mock y el cliente OpenAI implementan el mismo protocolo, así que nada
más cambia: agentes, prompts, tools, presupuestos y trazas se comportan idéntico.

---

## Cambiar a Databricks

```bash
DATABRICKS_HOST=your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=dapi...
DATABRICKS_VECTOR_ENDPOINT=your-vs-endpoint
DATABRICKS_WAREHOUSE_ID=your-sql-warehouse
MLFLOW_ENABLED=true
MLFLOW_TRACKING_URI=databricks
```

`bootstrap.build_platform` cablea entonces:

- `DatabricksVectorSearch` en vez del índice coseno local;
- `DatabricksLakehouse` (Statement Execution API → Delta/Lakebase) en vez de tablas JSONL;
- `MlflowTraceSink` en vez del trace store en memoria;
- `MosaicModelServingClient` disponible para endpoints de Mosaic AI Model Serving.

Tasks batch/scheduled con el Databricks Asset Bundle incluido:
`deploy/databricks.yml` (`databricks bundle deploy`).

---

## Configuración de entorno

| Variable | Default | Descripción |
|---|---|---|
| `LLM_PROVIDER` | `mock` | `mock` (offline) o `openai` (cualquier endpoint compatible) |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Endpoint OpenAI-compatible |
| `OPENAI_API_KEY` | — | Secret del endpoint |
| `LLM_MODEL_LARGE` | `mock-large` | Modelo para agentes especialistas |
| `LLM_MODEL_SMALL` | `mock-small` | Modelo para supervisor/quality gate |
| `MOCK_LLM_LATENCY_MS` | `0` | Latencia simulada del mock |
| `EMBEDDING_PROVIDER` | `mock` | `mock` o `openai` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Modelo de embeddings |
| `AUTO_APPROVE` | `false` | Aprobar automáticamente la publicación |
| `MAX_LLM_CALLS_PER_RUN` | `24` | Tope de llamadas LLM por run |
| `MAX_TOTAL_TOKENS_PER_RUN` | `60000` | Tope de tokens por run |
| `RUN_DEADLINE_MS` | `20000` | Deadline wall-clock por run |
| `API_KEY` | — | Si se setea, requiere `X-API-Key` |
| `DATABRICKS_HOST` | — | Host del workspace Databricks |
| `DATABRICKS_TOKEN` | — | Token Databricks |
| `DATABRICKS_VECTOR_ENDPOINT` | — | Endpoint de Vector Search |
| `DATABRICKS_WAREHOUSE_ID` | — | SQL warehouse |
| `DATABRICKS_CATALOG` | `media_intel` | Catálogo Delta |
| `DATABRICKS_SCHEMA` | `default` | Schema Delta |
| `MLFLOW_ENABLED` | `false` | Enviar trazas a MLflow |
| `MLFLOW_TRACKING_URI` | — | URI de tracking MLflow |

---

## Estructura del proyecto

```
src/media_intel/
├── core/          settings, resilience (retry/breaker/fallback), governance, observability
├── llm/           protocolo del provider, provider mock, cliente compatible con OpenAI
├── data/          retrieval (RAG), tablas lakehouse, adaptadores Databricks, datos semilla
├── tools/         registry + herramientas de medios (fixtures, eventos, audiencia, transcripciones, KB, publish)
├── agents/        motor de grafo, agente base, especialistas, workflow editorial (HITL)
├── runtime/       event bus
├── api/           servicio FastAPI
├── bootstrap.py   composition root (selección de puertos por env)
└── cli.py         seed | demo | serve
tests/             50 tests: grafo, resiliencia, retrieval, lakehouse, agentes, workflow/HITL, API
deploy/            Databricks Asset Bundle de ejemplo
```

---

## Docker

```bash
docker build -t media-intel-agents .
docker run --rm -p 8000:8000 media-intel-agents
```

---

## Notas de diseño

- **Ports & adapters en todos lados**: cada dependencia externa (LLM, embeddings,
  búsqueda vectorial, tablas, tracing) es un protocolo pequeño; `bootstrap.py` es el
  único lugar que elige implementaciones. Los tests inyectan fakes directo.
- **Governance pre-call, no post-hoc**: el presupuesto del run se verifica antes de
  cada llamada LLM, así los loops descontrolados fallan rápido y escalan en vez de
  quemar costo.
- **HITL como primitiva del grafo**: `interrupt()` suspende con un checkpoint
  persistido; `resume()` continúa exactamente donde se detuvo el run — el mismo
  mecanismo que usarías con checkpoints de LangGraph en producción.
- **Tracing estructural, no opcional**: nodos del grafo, agentes, llamadas LLM y
  llamadas a tools emiten spans; el uso de tokens y el costo se acumulan por run.

---

## Tests

Cobertura de las piezas críticas:

- `test_graph.py` — flujos lineales, edges condicionales, `Command(goto)`,
  interrupt + resume, checkpoints.
- `test_resilience.py` — retry (éxito/exhaustado/no-retryable), circuit breaker
  (open/half-open), fallback chain, presupuesto (topes de llamadas y tokens).
- `test_retrieval.py` — embeddings mock, upsert/query vectorial, filtros por
  metadata, validación de dimensiones, RAG end-to-end.
- `test_lakehouse.py` — append + scan con `_ingested_at`, query por predicado.
- `test_agents_and_tools.py` — las 5 tools de medios, publish queue, registry
  (error + retry), enrutamiento del supervisor, quality gate.
- `test_workflow_hitl.py` — lane matchday → HITL, approve → publish, reject →
  block, lane de análisis, listing pendientes, tracing del run.
- `test_api.py` — healthz, ciclo de vida completo + approval, 404, búsqueda,
  trazas, rechazo de API key.
- `test_observability.py` — anidamiento de spans, records de tokens, estados de
  error y SUSPENDED.

```bash
.venv/Scripts/python -m pytest
```