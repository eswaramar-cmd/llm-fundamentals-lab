# AI Research & Knowledge Agent — Multi-User Production Architecture

A production-oriented, LangGraph-based AI agent with ChromaDB RAG, Ollama LLM,
tool calling, short/long-term memory, human-in-the-loop, streaming, retry/error
handling, Redis-backed shared state, distributed rate limiting, and horizontal
scaling via Docker Compose + Nginx load balancing.

---

## Architecture Diagram

### Current (Single-Instance) Architecture

```
                  USER
                    │
                    ▼
                Web Frontend
                    │
                    ▼
              FastAPI :8000  (single container)
                    │
                    ▼
                 LangGraph
                    │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
      Memory         RAG           Tools
        │             │             │
        ▼             ▼             ▼
     Chroma        Chroma       Calculator
     Memory        RAG          Web Search
     (local)       (local)      Save Memory
                      │
                      ▼
                   Ollama (host)
                      │
                      ▼
               llama3.2:3b
### Model Context Protocol (MCP) Architecture

```
                                USER / FRONTEND
                                      │
                                      ▼
                                FastAPI Server
                                      │
                                      ▼
                               LangGraph Agent
                             (Main Orchestrator)
                                      │
                                      ▼
                            LangGraph MCP Client
                         (Dynamic Tool Discovery)
                                      │
       ┌──────────────────────────────┼──────────────────────────────┐
       ▼                              ▼                              ▼
 Research MCP Server          Incident MCP Server             Gmail MCP Server
  • calculator                 • investigate_incident          • send_email
  • knowledge_base_search
  • web_search
```

---

## Model Context Protocol (MCP) in this Project

### 1. What is MCP in this Project?
The **Model Context Protocol (MCP)** standardizes how the LangGraph AI agent connects to and executes external tools, services, and diagnostic systems. Instead of hardcoding direct tool implementations into the orchestration engine, tools are exposed through modular FastMCP servers and dynamically discovered by an MCP Client.

### 2. MCP Client vs MCP Server
* **MCP Client (`LangGraphMCPClient`)**: Embedded inside the LangGraph agent layer. Connects to all registered MCP servers, queries their tool catalogs, translates MCP schemas into LangChain/LangGraph-compatible `BaseTool` instances, and forwards tool execution calls.
* **MCP Servers (`FastMCP`)**: Standalone, modular service endpoints that expose typed, validated tools conforming to the MCP specification. Each server isolates specific domain capabilities (Research, Production Diagnostics, Email Automation).

### 3. Registered MCP Servers
1. **`ResearchMCPServer`** (`backend/agent/mcp/servers/research_server.py`): Hosts research, mathematical, and knowledge retrieval tools.
2. **`IncidentMCPServer`** (`backend/agent/mcp/servers/incident_server.py`): Hosts production incident analysis and telemetry correlation tools.
3. **`GmailMCPServer`** (`backend/agent/mcp/servers/gmail_server.py`): Hosts safe preview and authenticated SMTP email delivery tools.

### 4. Available MCP Tools
| Tool Name | Server | Description | Input Arguments |
| :--- | :--- | :--- | :--- |
| **`calculator`** | Research Server | Safe arithmetic evaluation (`+`, `-`, `*`, `/`, `//`, `%`, `**`). | `expression: str` |
| **`knowledge_base_search`** | Research Server | Semantic search across indexed ChromaDB documents. | `question: str`, `k: int` |
| **`web_search`** | Research Server | Live web search via DuckDuckGo with fallback parsing. | `query: str`, `num_results: int` |
| **`investigate_incident`** | Incident Server | Dynamic incident diagnostic correlating real logs and latencies. | `service_name`, `endpoint`, `time_range`, `severity` |
| **`send_email`** | Gmail Server | Strict two-step email automation with preview mode & rate limiting. | `to_email`, `subject`, `body`, `confirmed`, `file_ids` |
| **`save_memory`** | Local Memory | Durable semantic key-value memory for user facts. | `user_id`, `key`, `value`, `category` |

### 5. Why MCP was Used
* **Modularity & Separation of Concerns**: Tools can be updated, scaled, or secured independently from the LangGraph agent graph.
* **Interoperability & Standards**: Any MCP-compliant client or tool can be attached without rewriting prompt logic or core graph nodes.
* **Safe Runtime Isolation**: Tool-specific dependencies (such as SMTP connections or DuckDuckGo parsers) are cleanly isolated behind standardized interfaces.

---


### Target (Multi-User, Horizontally Scalable)

```
                         MANY USERS
                             │
                             ▼
                    ┌──────────────────┐
                    │  LOAD BALANCER   │
                    │     NGINX        │  (port 80, SSE pass-through)
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ FastAPI  │   │ FastAPI  │   │ FastAPI  │
        │  API-1   │   │  API-2   │   │  API-3   │
        └────┬─────┘   └────┬─────┘   └────┬─────┘
             │              │              │
             └──────────────┼──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   LangGraph   │
                     │  Agent Engine │
                     └───────┬───────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
              ▼             ▼             ▼
        ┌──────────┐  ┌──────────┐   ┌───────────┐
        │  Redis   │  │ Vector DB  │   │   LLM     │
        │          │  │ (Chroma)   │   │  Layer    │
        │ Shared   │  │ Shared     │   │           │
        │ State    │  │ RAG        │   │ Ollama    │
        │          │  │            │   │ / API     │
        └──────────┘  └──────────┘   │           │
                                      └────┬──────┘
                                           │
                                           ▼
                                    Scalable Inference
                                      (current: laptop)

                            │
                            ▼
                     ┌─────────────┐
                     │  Workers    │  (Redis queue)
                     │ Background  │
                     │    Jobs     │
                     └──────┬──────┘
                            │
                            ▼
                     Shared Storage
                     (shared volumes / S3)
```

**Key differences:**
1. **Nginx load balancer** distributes requests across multiple FastAPI replicas
2. **Redis** provides shared checkpointing (replacing in-memory `MemorySaver`)
3. **Shared volumes** ensure all replicas access the same Chroma and documents
4. **Per-user isolation** via `user_id` + `thread_id` (session_id) keyed state
5. **Background workers** handle long-running jobs via Redis queue
6. **Rate limiting** is Redis-backed and shared across all replicas

---

## Files Modified

| File | Changes |
|---|---|
| `agent/config.py` | Added Redis, LLM abstraction, vector store provider, rate limiting, CORS, storage, metrics, logging settings |
| `agent/graph.py` | Replaced `MemorySaver` with `get_checkpoint_saver()` for shared Redis checkpointing |
| `agent/llm.py` | Added multi-provider LLM factory (Ollama, OpenAI, Anthropic, Groq) |
| `agent/tools/knowledge_base.py` | Added configurable vector store abstraction (local Chroma, Chroma server, Qdrant, Weaviate) |
| `agent/nodes/tools_node.py` | Added real timeout enforcement (ThreadPoolExecutor), retry with backoff, validation error detection |
| `agent/nodes/agent.py` | Configurable retry backoff, LLM timeout validation, metrics integration |
| `agent/nodes/error_handler.py` | Uses configured `max_retries` instead of hardcoded `3` |
| `agent/nodes/retrieval.py` | Added RAG latency metrics |
| `backend/main.py` | Added request ID middleware, Redis rate limiting, structured JSON logging, metrics endpoint, enhanced health/readiness, security hardening (CORS config, request size limit, sanitized errors), job status endpoint |
| `backend/requirements.txt` | Added `redis`, `langgraph-checkpoint-redis`, `locust`, `pypdf` |
| `backend/docker-compose.yml` | Added Redis service, Nginx load balancer, scaled API service |
| `frontend/script.js` | Updated to use load balancer port (80) |
| `.env` | Updated with all new environment variables |
| `.env.example` | Updated with all new environment variables |
| `README.md` | Complete rewrite for production architecture |

## Files Created

| File | Purpose |
|---|---|
| `agent/checkpoint.py` | Checkpoint saver abstraction (Redis or MemorySaver fallback) |
| `agent/middleware.py` | Request ID middleware for distributed tracing |
| `agent/rate_limiter.py` | Redis-backed distributed rate limiter |
| `agent/metrics.py` | Prometheus-format metrics collection |
| `agent/storage.py` | Document storage abstraction (local + S3) |
| `agent/jobs.py` | Redis-backed background job queue |
| `backend/nginx/nginx.conf` | Nginx load balancer configuration |
| `tests/test_checkpoint.py` | Tests for checkpoint abstraction |
| `tests/test_rate_limiter.py` | Tests for distributed rate limiting |
| `tests/test_metrics.py` | Tests for metrics collection |
| `tests/test_multiuser_isolation.py` | Tests for multi-user/session isolation |
| `load_tests/locustfile.py` | Locust load test suite |
| `load_tests/quick_test.py` | Quick load test (stdlib only, no Locust needed) |
| `load_tests/README.md` | Load test documentation |

## Dependencies Added

| Package | Version | Purpose |
|---|---|---|
| `redis` | >=5.2.1 | Redis client for checkpointing, rate limiting, job queue |
| `langgraph-checkpoint-redis` | 0.5.2 | Shared Redis checkpoint store for LangGraph |
| `locust` | >=2.20.0 | Load testing |
| `pypdf` | >=4.0.0 | PDF parsing for document ingestion |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API URL |
| `OLLAMA_MODEL` | `llama3.2:3b` | LLM model name |
| `LLM_PROVIDER` | `ollama` | `ollama`, `openai`, `anthropic`, `groq` |
| `LLM_BASE_URL` | `(default)` | Custom LLM API URL (for OpenAI-compatible endpoints) |
| `LLM_MODEL` | `llama3.2:3b` | LLM model for the selected provider |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model (must match ingestion model) |
| `EMBEDDING_PROVIDER` | `ollama` | `ollama` or `huggingface` |
| `VECTOR_STORE_PROVIDER` | `chroma_local` | `chroma_local`, `chroma_server`, `qdrant`, `weaviate` |
| `VECTOR_STORE_URL` | `""` | Remote vector store URL (production) |
| `CHROMA_RAG_PATH` | `./chroma_rag` | Local ChromaDB RAG path (dev) |
| `CHROMA_MEMORY_PATH` | `./chroma_memory` | Local ChromaDB memory path (dev) |
| `REDIS_URL` | `""` | Redis connection string (enables shared state) |
| `REDIS_TTL_SECONDS` | `86400` | Redis checkpoint TTL (24h) |
| `MAX_RETRIES` | `3` | Max LLM retry attempts |
| `TOOL_TIMEOUT_SECONDS` | `30` | Per-tool execution timeout |
| `MAX_TOOL_ITERATIONS` | `5` | Max agent tool-calling loop iterations |
| `RETRY_BACKOFF` | `2.0` | Exponential backoff base |
| `RATE_LIMIT_ENABLED` | `true` | Enable Redis-backed rate limiting |
| `RATE_LIMIT_REQUESTS` | `60` | Max requests per window per client |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate limit window size |
| `CORS_ORIGINS` | `localhost` | Comma-separated allowed origins |
| `MAX_REQUEST_SIZE_MB` | `10` | Max request body size |
| `METRICS_ENABLED` | `true` | Enable /metrics endpoint |
| `LOG_LEVEL` | `INFO` | Logging level |
| `DOC_STORAGE_PROVIDER` | `local` | `local` or `s3` |
| `DOC_STORAGE_BUCKET` | `""` | S3 bucket name (production) |

---

## Docker Compose Changes

The compose file at `backend/docker-compose.yml` now includes three services:

```yaml
services:
  redis:       # Redis 7 (shared state, rate limiting, job queue)
    image: redis:7-alpine
    ports: ["6379:6379"]

  api:         # FastAPI (scalable via --scale api=N)
    build:
      context: ..
      dockerfile: backend/Dockerfile
    ports: []  # No host port — accessed via Nginx

  nginx:       # Load balancer (SSE pass-through, retries)
    image: nginx:alpine
    ports: ["80:80"]
```

All API replicas share:
- **Redis** at `redis://redis:6379/0` for checkpoints, rate limiting, and jobs
- **Chroma volumes** via bind mounts (`./chroma_rag`, `./chroma_memory`)
- **Documents** via bind mount (`./documents`)

---

## Redis Configuration

Redis is the central shared-state layer:

1. **LangGraph Checkpoints** — `RedisSaver` replaces `MemorySaver`, storing
   conversation state per `thread_id` (session_id) in Redis. Any API replica
   can resume any session.

2. **Distributed Rate Limiting** — Sliding-window counter stored in Redis
   with SHA-256 hashed keys per `(user_id, client_ip, endpoint)`.

3. **Background Job Queue** — Redis list (`agent:jobs:queue`) for
   asynchronous document ingestion and long-running tasks.

4. **TTL** — Checkpoints expire after 24h by default (`REDIS_TTL_SECONDS`).

Redis runs as a standalone Docker container with:
```
redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
```

---

## Load Balancer Configuration (Nginx)

`backend/nginx/nginx.conf`:

- **SSE pass-through** — `proxy_buffering off` ensures streaming responses
  are forwarded immediately without buffering.
- **DNS re-resolution** — Uses Docker's embedded DNS (`resolver 127.0.0.11`)
  so Nginx picks up new API containers when scaling.
- **Forwarded headers** — `X-Real-IP`, `X-Forwarded-For`, `X-Request-ID`.
- **Graceful failover** — `proxy_next_upstream` retries on connection errors.
- **Timeouts** — 30s connect, 300s read/send (for long agent runs).

---

## Commands to Run Locally

### Development (single instance, no Docker)

```bash
# 1. Activate virtual environment
venv\Scripts\Activate.ps1  # Windows
# source venv/bin/activate  # Mac/Linux

# 2. Run unit tests
python -m pytest tests/ -v

# 3. Start the API locally
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 4. Ingest documents
python -m agent.ingest --clear
```

### Docker (production-like)

```bash
# 1. Start Ollama on the host
ollama serve
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# 2. Build and start all services
cd backend
docker compose up --build

# 3. API is available at http://localhost (Nginx port 80)
# 4. Metrics: http://localhost/metrics
# 5. Health: http://localhost/health
# 6. Readiness: http://localhost/health/ready
```

### Scaling to 3 API Replicas

```bash
# Stop existing stack
docker compose down

# Start with 3 API replicas
docker compose up --build --scale api=3

# Verify 3 containers are running
docker compose ps

# All 3 replicas share:
#   - Redis (checkpoints, rate limits, job queue)
#   - Chroma volumes (RAG + memory)
#   - Documents volume
#   - Ollama (via host.docker.internal)

# API requests are load-balanced across all 3 by Nginx
curl http://localhost/agent/run \
  -H "Content-Type: application/json" \
  -d '{"question": "What is 2+2?", "user_id": "user1"}'
```

### Background Workers

```bash
# Start a worker process (requires Redis)
docker compose run --rm api python -m agent.jobs

# Or with docker exec for an existing API container
docker compose exec api python -m agent.jobs --once
```

---

## Testing Commands

### Unit Tests

```bash
# All unit tests (no external services required)
python -m pytest tests/ -v

# Specific test files
python -m pytest tests/test_checkpoint.py tests/test_rate_limiter.py tests/test_metrics.py -v

# Multi-user isolation tests (require Ollama)
RUN_INTEGRATION_TESTS=1 python -m pytest tests/test_multiuser_isolation.py -v

# Existing tests
python -m pytest tests/ -v --ignore=tests/test_integration.py
```

### Load Testing

```bash
# Quick load test (stdlib only, no Locust needed)
python load_tests/quick_test.py --users 10 --requests 30
python load_tests/quick_test.py --users 25 --requests 60
python load_tests/quick_test.py --users 50 --requests 100

# All levels (10, 25, 50 users)
python load_tests/quick_test.py --all

# Locust (interactive web UI at http://localhost:8089)
locust -f load_tests/locustfile.py --host http://localhost

# Locust headless mode
locust -f load_tests/locustfile.py --headless -u 10 -r 2 --host http://localhost \
  --run-time 2m --csv load_tests/results_10
```

### Multi-User Isolation Test

```bash
# Verify User A and User B are isolated
RUN_INTEGRATION_TESTS=1 python -m pytest tests/test_multiuser_isolation.py -v

# Test HITL across replicas
python -m pytest tests/test_multiuser_isolation.py::TestHITLMultiReplica -v
```

---

## API Endpoints

| Method | Path | Description | Auth |
|---|---|---|---|
| `GET` | `/` | Service info + endpoint list | None |
| `GET` | `/health` | Liveness probe (Ollama, Chroma, Redis) | None |
| `GET` | `/health/ready` | Readiness probe (all dependencies) | None |
| `GET` | `/metrics` | Prometheus-format metrics | None |
| `POST` | `/chat` | Legacy streaming chat (backward compatible) | None |
| `POST` | `/agent/run` | Run agent to completion (JSON) | None |
| `POST` | `/agent/stream` | Stream agent events (SSE) | None |
| `POST` | `/agent/{session_id}/approve` | Resume after HITL interrupt | None |
| `POST` | `/memory` | Store a persistent user fact | None |
| `GET` | `/memory/{user_id}` | Retrieve user's persistent facts | None |
| `GET` | `/jobs/{job_id}` | Check background job status | None |

### Rate Limiting

- **Endpoint:** All API endpoints
- **Default:** 60 requests per 60 seconds per (user_id, IP, endpoint)
- **Headers:** `X-User-ID` for per-user rate limiting
- **Response on limit:** HTTP 429 with `Retry-After` header
- **Backend:** Redis sliding-window counter (shared across all replicas)
- **Fail-open:** When Redis is unavailable, all requests pass through

---

## Multi-User Isolation

### How It Works

1. **Session isolation:** Each user gets a unique `session_id` (thread_id).
   LangGraph stores conversation state per thread_id in Redis.

2. **Memory isolation:** Long-term memory uses ChromaDB with a `user_id`
   filter. User A's memories are never visible to User B.

3. **Request isolation:** The request ID middleware generates a UUID per
   request, included in all log lines for traceability.

4. **HITL isolation:** Interrupt/approval state is stored in the Redis
   checkpoint, keyed by thread_id. Any API replica can resume any session.

### Verification

```python
# User A and User B use different session_ids — no state leakage
# tested in tests/test_multiuser_isolation.py
```

### HITL Across Replicas

The HITL flow works across API replicas because:

1. `POST /agent/run` runs the graph until `interrupt()` — checkpoint is stored in Redis
2. API-1 can be handling the initial request
3. `POST /agent/{session_id}/approve` runs on API-2 — it loads the checkpoint from Redis and resumes
4. No sticky sessions required

```bash
# 1. Start agent run (may hit API-1 or API-2 or API-3 via Nginx)
curl -X POST http://localhost/agent/run \
  -H "Content-Type: application/json" \
  -d '{"question": "Delete the database", "require_approval": true}'

# Response: {"answer": "...", "approval_required": true, "session_id": "abc-123"}

# 2. Approve (may hit a different replica)
curl -X POST http://localhost/agent/abc-123/approve \
  -H "Content-Type: application/json" \
  -d '{"approved": true}'
```

---

## Current Bottlenecks

1. **Ollama (LLM inference)** — The single biggest bottleneck. `llama3.2:3b`
   runs on a single laptop. All API replicas share the same Ollama instance
   via `host.docker.internal:11434`. Scaling API replicas does NOT scale
   LLM throughput — they all queue at the same Ollama server.

2. **ChromaDB local persistence** — When using `VECTOR_STORE_PROVIDER=chroma_local`,
   all replicas share the same local Chroma volume. Concurrent writes
   (e.g., multiple users saving memory simultaneously) may conflict.
   For production, use `VECTOR_STORE_PROVIDER=chroma_server` with a
   dedicated Chroma server.

3. **Embedding model** — `nomic-embed-text` via Ollama is single-threaded.
   High concurrency for knowledge base search will saturate the embedding
   endpoint.

4. **Single-node Redis** — Redis runs on a single container. For
   production, use a Redis cluster or managed service.

---

## Why Ollama Is the LLM Bottleneck

**Scaling API replicas does NOT scale LLM inference.**

```
  API-1 ──┐
  API-2 ──┼──► all call the same Ollama
  API-3 ──┘      (host.docker.internal:11434)
       │
       ▼
  Ollama on laptop
  (single-threaded, CPU-bound)
```

Each API replica makes independent requests to the same Ollama instance.
Ollama processes one generation at a time per model. With 3 API replicas:

- **10 concurrent users:** Ollama queues requests — 5-10s per token
- **25 concurrent users:** Ollama queue depth grows — 15-30s per response
- **50 concurrent users:** 30s+ timeouts, 429s from rate limiting

**To truly scale LLM inference, replace Ollama with:**
- Managed API (OpenAI, Anthropic, Groq)
- vLLM on GPU instances
- GPU inference service

Set via environment variables:
```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
LLM_BASE_URL=https://api.openai.com/v1
```

---

## Production Migration Path

### Step 1: Infrastructure
```bash
# Deploy with Redis and Nginx
docker compose -f backend/docker-compose.yml up --build --scale api=3 -d
```

### Step 2: Vector Store (optional, for high write concurrency)
```bash
# Switch to Chroma server for concurrent writes
LLM_PROVIDER=chroma_server
VECTOR_STORE_URL=http://chroma:8000
```

### Step 3: LLM Scalability (the key bottleneck)
```bash
# Switch to a managed/scalable LLM provider
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
LLM_BASE_URL=https://api.openai.com/v1
# (set OPENAI_API_KEY as a secret)
```

### Step 4: Document Storage (for multi-region)
```bash
DOC_STORAGE_PROVIDER=s3
DOC_STORAGE_BUCKET=my-documents-bucket
# (AWS credentials via IAM roles, not env vars)
```

### Step 5: Ingress / TLS
- Add a real TLS termination proxy (Cloudflare, AWS ALB, etc.)
- Replace Nginx with your cloud load balancer
- Configure real certificates

---

## Structured Logging

All logs are JSON-formatted for multi-replica aggregation:

```json
{"timestamp": "2026-09-27T12:00:00Z", "level": "INFO", "logger": "uvicorn.error", "message": "Agent run: session=abc123 user=user1 question=What is 2+2? request_id=req-xyz"}
```

Each log entry includes:
- `request_id` — UUID per HTTP request
- `session_id` — LangGraph thread ID
- `user_id` — User identifier
- `endpoint` — API path
- `duration` — Request duration (ms)
- `status` — HTTP status code
- `error_type` — Exception type (on errors)

**Never logged:** API keys, passwords, tokens, private documents, private memory content.

---

## Metrics

`GET /metrics` returns Prometheus-format metrics:

| Metric | Labels | Description |
|---|---|---|
| `ai_requests_total` | `endpoint`, `status` | Request count |
| `ai_request_duration_seconds` | `endpoint`, `status`, `p50`, `p95`, `p99` | Latency histogram |
| `ai_llm_latency_seconds` | `model`, `p50`, `p95`, `p99` | LLM inference time |
| `ai_rag_latency_seconds` | `p50`, `p95`, `p99` | RAG retrieval time |
| `ai_tool_duration_seconds` | `tool`, `success`, `p50`, `p95`, `p99` | Tool execution time |
| `ai_tool_failures_total` | `tool` | Tool failure count |
| `ai_rate_limit_events_total` | (none) | Rate-limited requests |
| `ai_errors_total` | `error_type` | Error count |

---

## Security

| Concern | Mitigation |
|---|---|
| **CORS** | Configurable via `CORS_ORIGINS` (no wildcard with credentials) |
| **Request size** | `MAX_REQUEST_SIZE_MB` limit enforced in middleware (413 on exceed) |
| **Input validation** | Pydantic schema validation with `min_length`, `max_length` |
| **Path traversal** | `Storage._safe_path()` validates paths stay within base dir |
| **Error responses** | Generic messages — no stack traces or internal paths |
| **Calculator safety** | AST-based evaluation, no `eval()`, no code execution |
| **Rate limiting** | Redis-backed, distributed, fail-open |
| **Secrets** | All config via environment variables, never in source |
| **HITL approval** | Interrupt-based, state in Redis (not process-local) |

---

## How It All Works Together

### Request Flow (10 concurrent users → 3 API replicas)

```
User ──► Nginx:80 ──► [API-1 ──┐
              │    [API-2 ──┼──► Redis (checkpoint per session_id)
              │    [API-3 ──┘
              │       │
              │    LangGraph (runs on CPU in each container)
              │       │
              │    ┌──┼──┐
              │    ▼  ▼  ▼
              │   Redis  Chroma  Ollama (host)
              ▼
            SSE/streaming response back through Nginx
```

### HITL Flow (across replicas)

```
User ──► Nginx ──► API-1: POST /agent/run
                       │
                       │  interrupt() at check_approval node
                       ▼
                    Redis saves checkpoint (thread_id=session_abc)

User ──► Nginx ──► API-2: POST /agent/session_abc/approve
                       │
                       │  RedisSaver loads checkpoint from Redis
                       ▼
                    Graph resumes, completes
                       │
                       ▼
                    Redis saves final checkpoint
```

### Rate Limiting Flow

```
Request ──► API-1: RateLimitMiddleware
                  │
                  │  ZCOUNT(ZSET, now-60s, now)  →  count
                  │  ZADD(ZSET, now, request_id)
                  ▼
            Redis ZSET (shared, atomic)
            IF count > limit: return 429
```

### Background Job Flow

```
API ──► enqueue_job("ingest_document", args)
         │
         │  RPUSH redis:agent:jobs:queue, JSON{job}
         ▼
    Worker: BLPOP redis:agent:jobs:queue
         │
         │  Execute job handler
         ▼
    HSET redis:agent:jobs:status:job_id = "completed"
    GET /jobs/{job_id} → status check
```

---

## Known Limitations

1. **Ollama is the LLM bottleneck** — Scaling API replicas does not scale
   LLM inference (see above). Switch to a managed LLM for true scalability.

2. **Chroma local concurrency** — With `chroma_local`, concurrent writes
   from multiple replicas to the same local volume may conflict. Use
   `chroma_server` for production.

3. **Single Redis instance** — Redis is not clustered. For production,
   use Redis Cluster or a managed service (AWS ElastiCache, etc.).

4. **Single Nginx instance** — Nginx is not load-balanced itself. For
   production, use a cloud load balancer or multiple Nginx instances.

5. **No authentication** — API endpoints are open. Add OAuth2/JWT
   middleware for production.

6. **Document ingestion concurrency** — Running `python -m agent.ingest`
   from multiple containers simultaneously is not safe. Use the
   background job queue instead.

---

## Running & Testing MCP Architecture

### Starting the System
Start the complete stack (FastAPI, MCP servers, Redis, and Nginx) with a single command:
```bash
docker compose up
```

For local development:
```powershell
.\dev.ps1
```

### Running MCP Tests
Run the complete MCP test suite covering all tools and the LangGraph MCP client:
```powershell
.\venv\Scripts\python.exe -m pytest tests/test_mcp_*.py -v
```

### Checking Discovered Tools
Inspect all dynamically registered MCP tools via the REST endpoint:
```bash
curl http://localhost:8000/api/mcp/tools
```

