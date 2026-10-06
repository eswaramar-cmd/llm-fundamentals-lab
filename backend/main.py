"""AI Research & Knowledge Agent — FastAPI application.

Backward-compatible ``/chat`` endpoint plus new agent API:
- POST /chat
- POST /agent/run
- POST /agent/stream
- GET  /health
- GET  /health/ready
- GET  /metrics
- POST /memory
- GET  /memory/{user_id}
- POST /agent/{session_id}/approve

No LangSmith. No PostgreSQL. No paid APIs.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import re
import socket
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from langgraph.types import Command


from backend.agent.config import Settings
from backend.agent.middleware import RequestIDMiddleware
from backend.agent.rate_limiter import RedisRateLimiter, NoopRateLimiter, RateLimitExceeded
from backend.agent.metrics import get_metrics

from backend.agent.schemas import (
    AgentRunRequest,
    AgentRunResponse,
    AgentStreamRequest,
    ApprovalRequest,
    ChatRequest,
    HealthResponse,
    MemoryEntry,
    MemoryEntryRequest,
    IncidentInvestigationRequest,
    IncidentInvestigationResponse,
)

from backend.agent.state import AgentState
from backend.agent.uploads import (
    ALLOWED_EXTS,
    UploadError,
    resolve_attachments,
    max_file_bytes,
    max_files_per_request,
    max_total_attachment_bytes,
    max_total_upload_bytes,
    store_uploads_async,
)


logger = logging.getLogger("uvicorn.error")
settings = Settings()

# ------------------------------------------------------------------
# Rate limiter (initialised in lifespan)
# ------------------------------------------------------------------

_rate_limiter = NoopRateLimiter()


# ------------------------------------------------------------------
# Structured logging setup
# ------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    """JSON log formatter for structured, multi-replica logs."""

    def format(self, record: logging.LogRecord) -> str:
        base = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Attach contextual fields set by middleware
        for attr in ("request_id", "session_id", "user_id", "endpoint", "status", "duration", "error_type"):
            val = getattr(record, attr, None)
            if val is not None:
                base[attr] = val
        if record.exc_info:
            base["error_type"] = type(record.exc_info[1]).__name__
        return json.dumps(base)


def _configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))


# ------------------------------------------------------------------
# APPLICATION LIFESPAN
# ------------------------------------------------------------------

# STARTUP PREFLIGHT
# ---------------------------------------------------------------------------

def _resolve_bind_port() -> int:
    """Best-effort port this process will be asked to bind.

    The app is served by an externally launched uvicorn, so it is never told the
    port. ``--port`` on the command line is authoritative for the documented
    invocation; ``PORT`` covers script and container starts. Returns 8000 when
    neither is present, which is the port the frontend proxies to.
    """

    argv = sys.argv

    for index, token in enumerate(argv):
        value = None

        if token == "--port" and index + 1 < len(argv):
            value = argv[index + 1]
        elif token.startswith("--port="):
            value = token.split("=", 1)[1]

        if value is not None:
            try:
                return int(value)
            except ValueError:
                pass

    try:
        return int(os.environ.get("PORT", "8000"))
    except ValueError:
        return 8000


def _listening_pid(port: int) -> int | None:
    """PID listening on ``port``, or None when it cannot be determined.

    Deliberately best-effort. Every failure path returns None, which the caller
    treats as "cannot tell" and therefore does not block on: a preflight that
    cannot identify the owner must never be the reason a server refuses to
    start.
    """

    try:
        result = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        return None

    for line in result.stdout.splitlines():
        parts = line.split()

        # TCP  <local>  <foreign>  LISTENING  <pid>
        if len(parts) < 5 or parts[0].upper() != "TCP":
            continue

        if parts[3].upper() != "LISTENING":
            continue

        if parts[1].rsplit(":", 1)[-1] != str(port):
            continue

        try:
            return int(parts[4])
        except ValueError:
            return None

    return None


def _preflight_port_check() -> None:
    """Fail immediately when the port is already held by another process.

    uvicorn binds its socket only *after* this lifespan completes, so a conflict
    is otherwise reported only after every warm-up call has already run: the
    server appears to start, spends 20-40s loading Ollama and Gemini, then dies
    with WinError 10048. This turns that into an immediate, named failure.

    Under ``--reload`` the socket is already held by our own reloader parent,
    which is not a conflict and must not stop the child from starting.
    """

    port = _resolve_bind_port()

    probe = socket.socket()
    probe.settimeout(0.5)

    try:
        occupied = probe.connect_ex(("127.0.0.1", port)) == 0
    except OSError:
        occupied = False
    finally:
        probe.close()

    if not occupied:
        return

    owner = _listening_pid(port)

    if owner is None:
        # Port answers but the owner is unknown: proceed and let the real bind
        # decide, rather than guessing and refusing a legitimate start.
        logger.warning(
            "Port %d is already answering connections and its owner could not "
            "be identified; continuing.",
            port,
        )
        return

    if owner in {os.getpid(), os.getppid()}:
        return

    raise RuntimeError(
        f"Port {port} is already in use by process {owner}. "
        "Another copy of the backend is still running, so this one cannot "
        "start. Stop it first (taskkill /PID "
        f"{owner} /F), or run this server on a different port. "
        "The frontend proxies to /api on this port, so it will keep talking "
        "to the other process until this one is gone."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan — initialise heavy resources once."""

    # Before anything expensive. A port conflict must be reported in
    # milliseconds, not after the model warm-up below has run.
    _preflight_port_check()

    _configure_logging()

    logger.info("Starting AI Research & Knowledge Agent")

    logger.info("  LLM_PROVIDER: %s", settings.llm_provider)
    logger.info("  LLM_MODEL: %s", settings.llm_model)
    logger.info("  LLM_BASE_URL: %s", settings.llm_base_url or "(default)")

    logger.info("  EMBEDDING_MODEL: %s", settings.embedding_model)
    logger.info("  VECTOR_STORE_PROVIDER: %s", settings.vector_store_provider)

    logger.info("  CHROMA_RAG_PATH: %s", settings.chroma_rag_path)
    logger.info("  CHROMA_MEMORY_PATH: %s", settings.chroma_memory_path)

    logger.info("  REDIS_URL: %s", settings.redis_url or "(not set)")
    logger.info("  RATE_LIMIT_ENABLED: %s", settings.rate_limit_enabled)

    logger.info("  MAX_RETRIES: %d", settings.max_retries)
    logger.info("  TOOL_TIMEOUT_SECONDS: %d", settings.tool_timeout_seconds)

    # Initialise rate limiter
    global _rate_limiter
    if settings.rate_limit_enabled and settings.redis_url:
        try:
            _rate_limiter = RedisRateLimiter(
                redis_url=settings.redis_url,
                requests=settings.rate_limit_requests,
                window_seconds=settings.rate_limit_window_seconds,
            )
            logger.info("Rate limiter: Redis-backed")
        except Exception as exc:
            logger.warning("Rate limiter init failed, disabling: %s", exc)
            _rate_limiter = NoopRateLimiter()
    else:
        _rate_limiter = NoopRateLimiter()
        logger.info("Rate limiter: disabled (noop)")

    # Initialise checkpoint saver
    from backend.agent.checkpoint import get_checkpoint_saver

    saver = get_checkpoint_saver()
    saver_type = type(saver).__name__
    logger.info("Checkpoint saver: %s", saver_type)

    # Pre-warm graph (each replica has its own compiled graph object,
    # but all share the same Redis checkpoint store)
    from backend.agent.graph import build_graph

    build_graph()
    logger.info("Graph compiled and ready")

    _warm_models()

    yield

    logger.info("Shutting down agent")


def _warm_models() -> None:
    """Pull the LLM and embedding models into RAM.

    Runs on a background daemon thread (see the lifespan), so a request arriving
    during warm-up is served on a cold model rather than refused outright.

    Failures are logged and ignored: a cold model is slow, not broken.
    """

    import time as _time

    from backend.agent.config import get_settings

    settings = get_settings()

    started = _time.monotonic()

    try:
        from backend.agent.embeddings import get_embeddings

        get_embeddings().embed_query("warmup")
        logger.info("Embeddings warm")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Embeddings warm-up failed: %s", exc)

    lane_model = settings.parallel_llm_model or settings.llm_model

    if (settings.llm_provider or "ollama").lower() == "ollama":
        for label, model in (("lane", lane_model), ("agent", settings.llm_model)):
            try:
                # Both names are needed: get_llm_capped returns the reduced
                # context model used by parallel lanes, get_llm the full one.
                from backend.agent.llm import get_llm, get_llm_capped

                client = get_llm_capped() if label == "lane" else get_llm()
                client.invoke("hi")
                logger.info("Warmed %s model: %s", label, model)
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s model warm-up failed: %s", label, exc)

    logger.info(
        "Model warm-up finished in %.1fs",
        _time.monotonic() - started,
    )


# ------------------------------------------------------------------
# FASTAPI APP
# ------------------------------------------------------------------

app = FastAPI(
    title="AI Research & Knowledge Agent",
    description=(
        "A LangGraph-based AI agent with ChromaDB RAG, "
        "Ollama LLM, tool calling, short/long-term memory, "
        "and human-in-the-loop."
    ),
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(RequestIDMiddleware)


# ------------------------------------------------------------------
# CORS (config-driven, no wildcard with credentials)
# ------------------------------------------------------------------

_cors_origins = [
    o for o in settings.cors_origins.split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=r"^https:\/\/.*\.vercel\.app$|^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


# ------------------------------------------------------------------
# Request size limit
# ------------------------------------------------------------------

MAX_REQUEST_BYTES = settings.max_request_size_mb * 1024 * 1024

# Uploads carry their own, separately configurable caps (upload_max_file_mb and
# upload_max_total_mb), enforced inside the endpoint as each file streams to
# disk. Applying the global cap here as well silently capped every upload at
# max_request_size_mb no matter how far the upload limits were raised, and the
# rejection arrived as a bare 413 with no explanation of which limit applied.
# The headroom below is the true worst case plus multipart framing.
MAX_UPLOAD_BYTES = (
    max_total_upload_bytes() + max_file_bytes() + 8 * 1024 * 1024
)


class RequestSizeMiddleware(BaseHTTPMiddleware):
    """Reject requests whose body exceeds the configured size limit."""

    async def dispatch(self, request: Request, call_next):
        # Multipart framing is not predictable enough to pre-compute, so the
        # upload path is bounded by MAX_UPLOAD_BYTES and validated properly
        # inside the endpoint.
        limit = (
            MAX_UPLOAD_BYTES
            if request.url.path == "/upload"
            else MAX_REQUEST_BYTES
        )

        if request.headers.get("content-length"):
            try:
                size = int(request.headers["content-length"])
                if size > limit:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "detail": (
                                f"Request body too large. The limit for this "
                                f"endpoint is {limit // (1024 * 1024)} MB; raise "
                                f"it with upload_max_total_mb / upload_max_file_mb "
                                f"for uploads, or max_request_size_mb otherwise."
                            ),
                            "error_code": "payload_too_large",
                        },
                    )
            except ValueError:
                pass
        return await call_next(request)


app.add_middleware(RequestSizeMiddleware)


# ------------------------------------------------------------------
# Metrics middleware (request count + latency)
# ------------------------------------------------------------------

class MetricsMiddleware(BaseHTTPMiddleware):
    """Track request count and latency per endpoint."""

    async def dispatch(self, request: Request, call_next):
        start = time.monotonic()
        endpoint = request.url.path
        request_id = request.state.request_id if hasattr(request.state, "request_id") else "-"

        try:
            response = await call_next(request)
            duration = time.monotonic() - start
            metrics = get_metrics()
            metrics.record_request(endpoint, response.status_code, duration)
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception:
            duration = time.monotonic() - start
            get_metrics().record_request(endpoint, 500, duration)
            raise


app.add_middleware(MetricsMiddleware)


# ------------------------------------------------------------------
# Rate-limiting middleware (applied per-request, before routing)
# ------------------------------------------------------------------

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Distributed rate limiting via Redis sliding-window.

    Uses the client IP as the primary rate-limit key.  This avoids
    reading/consuming the request body in middleware and is the
    standard approach for API gateways.
    """

    async def dispatch(self, request: Request, call_next):
        if not settings.rate_limit_enabled:
            return await call_next(request)

        # Also honour an optional X-User-ID header for per-user limits
        user_id = request.headers.get("X-User-ID") or "anon"
        client_ip = request.client.host if request.client else "unknown"
        endpoint = request.url.path

        try:
            _rate_limiter.check(user_id, client_ip, endpoint)
        except RateLimitExceeded as exc:
            get_metrics().record_rate_limit()
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(exc.retry_after)},
                content={"detail": "Rate limit exceeded", "error_code": "rate_limited"},
            )

        return await call_next(request)


app.add_middleware(RateLimitMiddleware)


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------

def _thread_config(session_id: str) -> dict:
    """Build LangGraph configuration."""

    return {
        "configurable": {
            "thread_id": session_id
        }
    }


def _initial_state(request: AgentRunRequest) -> AgentState:
    """Build initial LangGraph state."""

    messages: list = []

    if request.question:
        messages.append(
            HumanMessage(
                content=request.question
            )
        )

    return {
        "question": request.question,
        "answer": "",
        "messages": messages,
        "retrieved_documents": [],
        "tool_results": [],
        "memory_context": "",
        "user_id": request.user_id,
        "session_id": request.session_id or str(uuid.uuid4()),
        "approval_required": request.require_approval,
        "approved": False,
        "error": None,
        "retry_count": 0,
        "research_results": [],
    }


def _serialize_message(msg: Any) -> dict:
    """Convert LangChain message into JSON."""

    if isinstance(msg, AIMessage):
        return {
            "role": "assistant",
            "content": msg.content or "",
        }

    if isinstance(msg, HumanMessage):
        return {
            "role": "user",
            "content": msg.content or "",
        }

    if isinstance(msg, ToolMessage):
        return {
            "role": "tool",
            "content": msg.content or "",
            "tool_call_id": msg.tool_call_id,
        }

    return {
        "role": "unknown",
        "content": str(msg),
    }


def _extract_sources(retrieved_docs: list) -> list[str]:
    """Extract unique source identifiers."""

    sources: set[str] = set()

    for doc in retrieved_docs:

        if isinstance(doc, dict):
            sources.add(
                doc.get("source", "unknown")
            )

        elif hasattr(doc, "source"):
            sources.add(doc.source)

    return sorted(sources)


def _chunk_text(
    text: str,
    chunk_size: int = 10,
):
    """Split text into small streaming chunks."""

    for i in range(
        0,
        len(text),
        chunk_size,
    ):
        yield text[i:i + chunk_size]


def _sse(payload: dict) -> str:
    """One SSE frame. Every byte on this endpoint comes through here.

    Named frames are sent alongside the data line purely for readability in a
    terminal; the data line alone is valid SSE and is what clients parse.
    """

    kind = payload.get("type", "message")

    return (
        f"event: {kind}\n"
        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    )


# Idle window before a keep-alive comment is written. Comfortably under the 15s
# idle timeout of common reverse proxies.
HEARTBEAT_SECONDS = 10.0

# A comment frame. Ignored by EventSource and by hand-rolled parsers, so it
# keeps the socket warm without inventing progress that did not happen.
_KEEPALIVE_FRAME = ": keep-alive\n\n"


def _extract_text(chunk) -> str:
    """Pull plain text out of a provider chunk without inventing any."""

    content = getattr(chunk, "content", None)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])

        return "".join(parts)

    return ""


# ------------------------------------------------------------------
# LIVE CHAT STREAM (SSE)
# ------------------------------------------------------------------

@app.post("/upload")
async def upload_files(
    request: Request,
    files: list[UploadFile] = File(...),
    user_id: str = "default_user",
):
    """Accept uploads for later attachment to an outgoing email.

    POST /upload?user_id=alice  (multipart/form-data, field name: files)

    Each file is validated against the shared allowlist in ``backend.agent.uploads``,
    stored under ``uploads/<user_id>/`` and given a ``file_id``. The id is what
    ``send_email(file_ids=[...])`` takes: the client never supplies a path, so a
    caller cannot direct the tool at a file outside its own upload directory.

    Returns the extracted text alongside the id, so the UI can preview what will
    be attached without the model having to guess at the contents.
    """

    req_id = getattr(request.state, "request_id", "-")

    payload = [(upload.filename or "", upload) for upload in files]

    try:
        stored = await store_uploads_async(payload, user_id)
    except UploadError as exc:
        logger.warning(
            "Upload rejected (request_id=%s user=%s): %s", req_id, user_id, exc
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info(
        "Upload accepted (request_id=%s user=%s files=%d bytes=%d)",
        req_id,
        user_id,
        len(stored),
        sum(record.size for record in stored),
    )

    return {
        "user_id": user_id,
        "count": len(stored),
        "total_bytes": sum(record.size for record in stored),
        "max_files_per_request": max_files_per_request(),
        "max_file_bytes": max_file_bytes(),
        "max_total_upload_bytes": max_total_upload_bytes(),
        "max_attachment_bytes": max_total_attachment_bytes(),
        "allowed_extensions": sorted(ALLOWED_EXTS),
        "files": [record.public() for record in stored],
    }


@app.get("/uploads/preview/{file_id}")
async def upload_preview(file_id: str, user_id: str = "default_user"):
    """Serve the generated thumbnail for an uploaded image.

    Takes ``user_id`` and resolves through the same per-user directory as
    everything else, so a preview cannot be used to read another user's upload
    by guessing an id.
    """

    try:
        resolved = resolve_attachments(user_id, [file_id])
    except UploadError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    record = resolved[0]

    if not record.thumbnail:
        raise HTTPException(
            status_code=404, detail="No thumbnail for this file."
        )

    path = record.path.parent / record.thumbnail

    if not path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail is missing.")

    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=300"},
    )


@app.post("/chat/stream")
async def chat_stream(
    request: Request,
    request_body: ChatRequest,
):
    """
    Stream a chat completion as Server-Sent Events.

    POST /chat/stream

    Every event is produced by real work. There is no simulated typing and no
    canned text: ``token`` frames carry verbatim provider output, ``tool_*``
    frames bracket real tool execution, and ``status`` frames name the node
    that is actually running.

    Event types: ``status``, ``tool_start``, ``tool_end``, ``token``,
    ``done``, ``error``.
    """

    message = request_body.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Question is required",
        )

    req_id = getattr(request.state, "request_id", "-")
    started_at = time.monotonic()

    # Honour the caller's thread_id instead of minting a fresh session per
    # request. A new id on every turn made each message its own thread, so the
    # checkpointer had nothing to resume, short-term memory never accumulated,
    # and a follow-up like "yes" arrived with an empty history and no way to
    # know what it was approving.
    session_id = (
        (request_body.thread_id or "").strip()
        or f"live_{uuid.uuid4().hex[:12]}"
    )
    user_id = (request_body.user_id or "default_user").strip() or "default_user"

    # A bare "yes" only means something next to the preview it approves.
    resolved = _resolve_pending_confirmation(user_id, session_id, message)
    if resolved:
        message = resolved

    logger.info(
        "Live chat stream: request_id=%s thread=%s user=%s question=%s",
        req_id,
        session_id,
        user_id,
        message[:80],
    )

    initial_state: AgentState = {
        "question": message,
        "answer": "",
        "messages": [HumanMessage(content=message)],
        "retrieved_documents": [],
        "tool_results": [],
        "memory_context": "",
        "user_id": user_id,
        "session_id": session_id,
        "approval_required": False,
        "approved": False,
        "error": None,
        "retry_count": 0,
        "research_results": [],
    }

    async def event_stream():

        try:

            from backend.agent.graph import get_agent_graph

            graph = get_agent_graph()

            config = _thread_config(session_id)

            token_count = 0
            char_count = 0
            first_token_ms: int | None = None
            answer_parts: list[str] = []
            final_state: dict | None = None

            status_texts = {
                "__start__": "Starting",
                "classify": "Classifying request",
                "retrieve_memory": "Checking memory",
                "retrieve_context": "Searching knowledge base",
                "agent": "Thinking",
                "execute_tools": "Running tools",
                "generate_answer": "Writing answer",
                "check_approval": "Checking approval",
                "handle_error": "Handling error",
            }

            queue: asyncio.Queue = asyncio.Queue()
            _STREAM_DONE = object()

            async def pump() -> None:
                """Move graph events onto a queue.

                The graph runs as a background task rather than being iterated
                inline. That is what makes the heartbeat safe: putting a timeout
                around the graph's own ``__anext__`` cancels the async generator
                when it expires, which tears down the running graph and ends the
                response early. Here the timeout applies only to ``queue.get()``,
                and cancelling that is harmless.
                """

                try:
                    async for evt in graph.astream_events(
                        initial_state,
                        config=config,
                        version="v2",
                    ):
                        await queue.put(evt)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    await queue.put(exc)
                finally:
                    await queue.put(_STREAM_DONE)

            producer = asyncio.create_task(pump())

            try:
                while True:
                    try:
                        event = await asyncio.wait_for(
                            queue.get(), timeout=HEARTBEAT_SECONDS
                        )
                    except asyncio.TimeoutError:
                        # A slow first token used to produce no bytes at all for
                        # its whole duration, which looks like a hang and
                        # invites proxy idle timeouts.
                        yield _KEEPALIVE_FRAME
                        continue

                    if event is _STREAM_DONE:
                        break

                    if isinstance(event, BaseException):
                        raise event

                    kind = event.get("event")
                    name = event.get("name", "")

                    # ----------------------------------------
                    # Node entered: real status
                    # ----------------------------------------

                    if kind == "on_chain_start" and name in status_texts:
                        yield _sse({
                            "type": "status",
                            "message": status_texts[name],
                            "node": name,
                        })

                    # ----------------------------------------
                    # Real model tokens, forwarded verbatim
                    # ----------------------------------------

                    elif kind == "on_chat_model_stream":
                        text = _extract_text(event.get("data", {}).get("chunk"))

                        if text:
                            if first_token_ms is None:
                                first_token_ms = round(
                                    (time.monotonic() - started_at) * 1000
                                )

                            token_count += 1
                            char_count += len(text)
                            answer_parts.append(text)

                            yield _sse({"type": "token", "text": text})

                    # ----------------------------------------
                    # Real tool calls
                    # ----------------------------------------

                    elif kind == "on_tool_start":
                        payload_in = event.get("data", {}).get("input")

                        yield _sse({
                            "type": "tool_start",
                            "tool": name,
                            "input": payload_in if isinstance(payload_in, dict) else {},
                        })

                    elif kind == "on_tool_end":
                        output = event.get("data", {}).get("output")

                        yield _sse({
                            "type": "tool_end",
                            "tool": name,
                            "result": _brief_result(output),
                            "time_ms": round(
                                (time.monotonic() - started_at) * 1000
                            ),
                        })

                    # ----------------------------------------
                    # Node failure and final state
                    #
                    # A node that raises is reported by LangGraph as an event
                    # rather than propagated, so without this branch a failed run
                    # looked identical to a successful one that said nothing.
                    # ----------------------------------------

                    elif kind == "on_chain_error":
                        raw_error = event.get("data", {}).get("error")
                        node_error = (
                            str(raw_error) if raw_error else name or "unknown node"
                        )
                        logger.warning("Graph node %s failed: %s", name, node_error)

                        yield _sse({
                            "type": "error",
                            "message": f"{name or 'agent'}: {node_error[:400]}",
                            "node": name,
                        })

                    elif kind == "on_chain_end":
                        output = event.get("data", {}).get("output")

                        if isinstance(output, dict) and "answer" in output:
                            final_state = output
            finally:
                # Only reached when the client disconnects or the response is
                # cut short; on the normal path the pump already finished.
                if not producer.done():
                    producer.cancel()

            elapsed_ms = round((time.monotonic() - started_at) * 1000)
            answer = "".join(answer_parts).strip()

            # With no streamed tokens the answer lives only in the graph's final
            # state. Reading it means a provider failure shows its real reason
            # instead of a generic "finished without producing any text".
            graph_error = None
            interrupted = False

            if isinstance(final_state, dict):
                graph_error = final_state.get("error")
                interrupted = bool(final_state.get("__interrupt__"))

                if not answer:
                    answer = str(final_state.get("answer") or "").strip()

            if interrupted:
                # The graph paused for human approval, so no answer exists yet.
                # Saying "done" here would be a lie the UI cannot recover from,
                # so report the pause and include the session id the approve
                # endpoint needs.
                yield _sse({
                    "type": "approval_required",
                    "session_id": session_id,
                    "reason": "This action needs your approval before it can finish.",
                })

                yield _sse({
                    "type": "done",
                    "latency_ms": round((time.monotonic() - started_at) * 1000),
                    "time_to_first_token_ms": first_token_ms,
                    "model": _active_model_name(),
                    "chunks": token_count,
                    "chars": char_count,
                    "chars_per_second": 0.0,
                    "error": None,
                    "approval_required": True,
                    "answer": answer or "Waiting for your approval to continue.",
                })

                return

            if not answer:
                answer = (
                    "The model provider did not return an answer. "
                    "This is usually a temporary outage or rate limit - "
                    "please try again."
                )

            model = _active_model_name()

            # Surfaced separately so the UI can render a definitive "Sent to X"
            # without parsing the model's prose. Only populated from a real
            # send_email tool result, so it can never claim a send that did not
            # happen.
            email_outcome = _extract_email_outcome(
                final_state if isinstance(final_state, dict) else None
            )

            if email_outcome and email_outcome.get("status") == "preview":
                _record_pending_preview(
                    user_id, session_id, email_outcome.get("detail", "")
                )

            # Provider token usage is not available on the streaming path, so
            # throughput is reported in characters. Counting streamed chunks as
            # "tokens" made a fast answer look like 0.2 tokens/second, because a
            # provider sends a whole clause per chunk.
            chars_per_second = (
                round(char_count / (elapsed_ms / 1000), 1) if elapsed_ms else 0
            )

            yield _sse({
                "type": "done",
                "latency_ms": elapsed_ms,
                "time_to_first_token_ms": first_token_ms,
                "model": model,
                "chunks": token_count,
                "chars": char_count,
                "chars_per_second": chars_per_second,
                "error": graph_error,
                "email": email_outcome,
                "answer": answer,
            })

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Live stream failed: %s",
                exc,
                exc_info=True,
            )

            yield _sse({
                "type": "error",
                "message": str(exc) or type(exc).__name__,
            })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _extract_email_outcome(final_state: dict | None) -> dict | None:
    """Pull a send_email result out of the final graph state.

    Returns ``{"status", "recipient", "detail"}`` or None. Derived only from a
    real tool result, so the UI can trust "sent" — a preview-only call reports
    ``preview`` instead, because that genuinely did not send anything.
    """

    if not isinstance(final_state, dict):
        return None

    for result in final_state.get("tool_results") or []:
        tool_name = getattr(result, "tool_name", None) or (
            result.get("tool_name") if isinstance(result, dict) else None
        )

        if tool_name != "send_email":
            continue

        text = (
            result.get("result")
            if isinstance(result, dict)
            else getattr(result, "result", "")
        ) or ""

        sent = getattr(result, "success", True)
        if isinstance(result, dict):
            sent = result.get("success", True)

        if text.startswith("Sent to "):
            recipient = text[len("Sent to ") :].split(" at ", 1)[0]
            status = "sent" if sent else "failed"
        elif text.startswith("PREVIEW ONLY"):
            recipient = ""
            status = "preview"
        else:
            recipient = ""
            status = "failed" if not sent else "sent"

        return {"status": status, "recipient": recipient, "detail": text[:400]}

    return None


# Approvals are tracked here rather than reconstructed from chat history.
#
# The send flow spans two requests ("send mail to X" then "yes"), and the
# approving turn arrives with no usable history: the request payload carries
# only the new message, Redis is usually down so the in-memory checkpointer is
# the only store, and a --reload dev server rebuilds the graph and loses even
# that. The model therefore saw a bare "yes" with nothing to approve and
# answered that it had no way to send mail. Holding the pending preview per
# thread makes the approval deterministic and independent of any of that.
_PENDING_EMAIL: "dict[str, dict]" = {}
_PENDING_LOCK = threading.Lock()
_PENDING_TTL_SECONDS = 15 * 60
_PENDING_MAX = 512

_PREVIEW_TO_RE = re.compile(r"^To:\s*(.+)$", re.MULTILINE)
_PREVIEW_SUBJECT_RE = re.compile(r"^Subject:\s*(.+)$", re.MULTILINE)
_PREVIEW_BODY_RE = re.compile(
    r"^Body:\s*(.+?)(?=\nAttachments \(\d+\):|\Z)",
    re.MULTILINE | re.DOTALL,
)

# "- fees.pdf (.pdf, 12 KB) file_id=<32 hex>" as emitted by the send tool.
_PREVIEW_ATTACHMENT_RE = re.compile(
    r"^-\s+(?P<name>.+?)\s+\((?P<ext>[^,]+),\s*[^)]*\)\s+file_id=(?P<id>[0-9a-f]{32})\s*$",
    re.MULTILINE,
)

_CONFIRM_RE = re.compile(
    r"^\s*(?:yes|yeah|yep|yup|ya|y|sure|ok|okay|k|confirm|confirmed|agreed)\b"
    r"|^\s*(?:please\s+)?(?:go\s+ahead|do\s+it|proceed"
    r"|send\s+(?:it|now|them|email|the\s+email)(?:\s+now)?)\b",
    re.IGNORECASE,
)


def _pending_key(user_id: str, session_id: str) -> str:
    return f"{user_id}::{session_id}"


def _record_pending_preview(
    user_id: str, session_id: str, detail: str
) -> None:
    """Remember a preview so a later bare "yes" can be resolved to a send."""

    to_match = _PREVIEW_TO_RE.search(detail or "")
    subject_match = _PREVIEW_SUBJECT_RE.search(detail or "")
    body_match = _PREVIEW_BODY_RE.search(detail or "")

    if not (to_match and subject_match and body_match):
        return

    record = {
        "to": to_match.group(1).strip(),
        "subject": subject_match.group(1).strip(),
        "body": body_match.group(1).strip(),
        "created": time.monotonic(),
        "file_ids": [
            m.group("id") for m in _PREVIEW_ATTACHMENT_RE.finditer(detail or "")
        ],
    }

    key = _pending_key(user_id, session_id)
    now = time.monotonic()

    with _PENDING_LOCK:
        expired = [
            k
            for k, v in _PENDING_EMAIL.items()
            if now - v.get("created", 0) > _PENDING_TTL_SECONDS
        ]
        for k in expired:
            _PENDING_EMAIL.pop(k, None)

        if len(_PENDING_EMAIL) >= _PENDING_MAX:
            oldest = min(
                _PENDING_EMAIL, key=lambda k: _PENDING_EMAIL[k]["created"]
            )
            _PENDING_EMAIL.pop(oldest, None)

        _PENDING_EMAIL[key] = record

    logger.info(
        "Pending email preview recorded for thread %s -> %s",
        session_id,
        record["to"],
    )


def _resolve_pending_confirmation(
    user_id: str, session_id: str, message: str
) -> str | None:
    """Turn a bare approval into an explicit, self-contained send request.

    Returns the rewritten question, or None when the turn is not an approval or
    nothing is pending. The rewrite names the recipient, subject and body so the
    classifier matches the email intent and the tool is bound, and states that
    approval has already been given so the model does not ask a second time.
    """

    if not message or len(message) > 400:
        return None

    # Matched as a prefix, not the whole turn: the UI's confirm button sends
    # "Yes, send it. to=... subject=... body=... confirmed=true". The stored
    # draft is authoritative and is preferred over re-reading those inline
    # fields, because it is what the user actually saw and approved.
    if not _CONFIRM_RE.match(message.strip()):
        return None

    key = _pending_key(user_id, session_id)
    now = time.monotonic()

    with _PENDING_LOCK:
        record = _PENDING_EMAIL.get(key)

        if record is None:
            return None

        if now - record.get("created", 0) > _PENDING_TTL_SECONDS:
            _PENDING_EMAIL.pop(key, None)
            return None

        # Consume it. A preview is good for exactly one approval, so a repeated
        # "yes" must not send the same message twice.
        _PENDING_EMAIL.pop(key, None)

        rewritten = (
            f'Send the email to {record["to"]} with subject "{record["subject"]}" '
            f'and body "{record["body"]}". The user has already seen this preview '
            f"and approved it, so call send_email exactly once with confirmed=true "
            f"and do not ask for confirmation again."
        )

        file_ids = record.get("file_ids") or []

        if file_ids:
            # Carried from the approved preview rather than re-derived, so the
            # message that goes out has exactly the attachments the user agreed to.
            rewritten += f" Attach these files by passing file_ids={file_ids!r}."

    logger.info(
        "Confirmation turn resolved to pending send for thread %s -> %s",
        session_id,
        record["to"],
    )

    return rewritten


def _active_model_name() -> str:
    """Name the model that actually served this request.

    Prefers a configured cloud provider, because that is what the user is
    waiting on; falls back to the local settings model.
    """

    try:
        from backend.agent.llm_fast import available_specs

        specs = available_specs()

        if specs:
            return f"{specs[0].name}/{specs[0].model}"
    except Exception:  # noqa: BLE001
        pass

    try:
        from backend.agent.config import get_settings

        settings = get_settings()
        return f"{settings.llm_provider}/{settings.llm_model}"
    except Exception:  # noqa: BLE001
        return "unknown"


def _brief_result(output) -> str:
    """Render a tool result compactly for the client."""

    if output is None:
        return ""

    if isinstance(output, str):
        return output[:800]

    content = getattr(output, "content", None)

    if isinstance(content, str):
        return content[:800]

    try:
        return json.dumps(output, default=str)[:800]
    except (TypeError, ValueError):
        return str(output)[:800]


# ------------------------------------------------------------------
# LLM LATENCY PROBE
# ------------------------------------------------------------------

@app.get("/health/llm")
async def health_llm():
    """
    Measure real round-trip and first-token latency for each configured LLM.

    Providers without an API key are reported as ``configured: false`` rather
    than skipped, so the endpoint always answers with the full picture.
    """

    from backend.agent.llm_fast import (
        get_fallback_spec,
        get_primary_spec,
        probe_latency,
        provider_order,
    )

    specs = [get_primary_spec(), get_fallback_spec()]
    by_name = {"groq": specs[0], "google": specs[1]}
    specs = [by_name[n] for n in provider_order()]

    results = await asyncio.gather(
        *(probe_latency(spec) for spec in specs)
    )

    return JSONResponse(
        content={
            "order": provider_order(),
            "active": next((r["model"] for r in results if r["ok"]), None),
            "providers": results,
            "any_usable": any(r["ok"] for r in results),
        }
    )


# ------------------------------------------------------------------
# ROOT
# ------------------------------------------------------------------

@app.get("/")
def root():

    # Built from the live route table rather than a hand-kept list, which had
    # drifted and was omitting /chat/stream and /health/llm.
    endpoints = sorted(
        f"{sorted(r.methods)[0]} {r.path}"
        for r in app.routes
        if getattr(r, "methods", None)
        and not r.path.startswith(("/openapi", "/docs", "/redoc"))
    )

    return {
        "message": "AI Research & Knowledge Agent",
        "version": "2.1.0",
        "endpoints": endpoints,
    }


# ------------------------------------------------------------------
# HEALTH (liveness)
# ------------------------------------------------------------------

def _dependency_status() -> dict[str, str]:
    """Probe Ollama, Chroma and Redis. Slow, and never on a hot path.

    Each check costs seconds: an HTTP round trip to Ollama, constructing the
    Chroma vector store, and a Redis ping. That is why the result is cached by
    ``health_check`` rather than recomputed per request.
    """

    ollama_status = "unknown"
    chroma_status = "unknown"
    redis_status = "unknown"

    try:
        import httpx

        with httpx.Client(timeout=5) as client:
            response = client.get(f"{settings.ollama_base_url}/api/tags")

        if response.status_code == 200:
            ollama_status = "ok"
        else:
            ollama_status = f"error ({response.status_code})"

    except Exception:
        ollama_status = "unreachable"

    try:
        from backend.agent.tools.knowledge_base import _get_vectorstore

        count = _get_vectorstore()._collection.count()
        chroma_status = f"ok ({count} docs)"

    except Exception:
        chroma_status = "unreachable"

    if settings.redis_url:
        try:
            import redis

            redis.from_url(
                settings.redis_url,
                socket_connect_timeout=1,
                socket_timeout=1,
            ).ping()
            redis_status = "ok"
        except Exception:
            redis_status = "unreachable"
    else:
        redis_status = "not-configured (local mode)"

    return {
        "ollama": ollama_status,
        "chroma": chroma_status,
        "redis": redis_status,
    }


# Dependency probes cost seconds, so /health caches them briefly. Without this a
# liveness probe took 7.2s per call, which defeats its purpose and stalls
# anything polling for readiness — including the frontend on load.
_HEALTH_CACHE: dict[str, Any] = {"at": 0.0, "statuses": None}
_HEALTH_CACHE_TTL_SECONDS = 15.0
_HEALTH_CACHE_LOCK = threading.Lock()


def _cached_dependency_status() -> dict[str, str]:
    """Return the dependency probes, recomputed at most every TTL."""

    now = time.monotonic()
    cached = _HEALTH_CACHE["statuses"]

    if cached is not None and (now - _HEALTH_CACHE["at"]) < _HEALTH_CACHE_TTL_SECONDS:
        return cached

    # The lock keeps a burst of concurrent probes from each paying the cost;
    # the loser re-checks the timestamp and reuses the winner's result.
    with _HEALTH_CACHE_LOCK:
        cached = _HEALTH_CACHE["statuses"]
        now = time.monotonic()

        if (
            cached is not None
            and (now - _HEALTH_CACHE["at"]) < _HEALTH_CACHE_TTL_SECONDS
        ):
            return cached

        statuses = _dependency_status()
        _HEALTH_CACHE["statuses"] = statuses
        _HEALTH_CACHE["at"] = now
        return statuses


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health_check():
    """Liveness probe — is the process alive?

    Dependency detail is included but cached: probing Ollama, Chroma and Redis
    costs ~7s, and a liveness check that takes 7s cannot be used to decide
    whether the process is up. ``/health/ready`` reports dependencies live.
    """

    statuses = _cached_dependency_status()

    return HealthResponse(
        status="ok",
        ollama=statuses["ollama"],
        chroma=statuses["chroma"],
        version="2.1.0",
        redis=statuses["redis"],
    )


# ------------------------------------------------------------------
# READINESS (are dependencies available?)
# ------------------------------------------------------------------

@app.get(
    "/health/ready",
    response_model=HealthResponse,
)
def readiness_check():
    """Readiness probe — are all dependencies ready?"""

    statuses = {}

    # Ollama
    try:
        import httpx

        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{settings.ollama_base_url}/api/tags")
            statuses["ollama"] = "ok" if resp.status_code == 200 else f"error ({resp.status_code})"
    except Exception:
        statuses["ollama"] = "unreachable"

    # Chroma
    try:
        from backend.agent.tools.knowledge_base import _get_vectorstore

        vs = _get_vectorstore()
        count = vs._collection.count()
        statuses["chroma"] = f"ok ({count} docs)"
    except Exception:
        statuses["chroma"] = "unreachable"

    # Redis
    if settings.redis_url:
        try:
            import redis

            r = redis.from_url(settings.redis_url, socket_connect_timeout=3)
            r.ping()
            statuses["redis"] = "ok"
        except Exception:
            statuses["redis"] = "unreachable"
    else:
        statuses["redis"] = "not-configured"

    # Checkpoint saver
    try:
        from backend.agent.checkpoint import get_checkpoint_saver

        saver = get_checkpoint_saver()
        statuses["checkpoint"] = type(saver).__name__
    except Exception as exc:
        statuses["checkpoint"] = f"error: {type(exc).__name__}"

    all_ok = all(
        v in ("ok", "not-configured")
        for v in statuses.values()
    )

    return JSONResponse(
        content={
            "status": "ready" if all_ok else "degraded",
            "ollama": statuses.get("ollama", "unknown"),
            "chroma": statuses.get("chroma", "unknown"),
            "redis": statuses.get("redis", "unknown"),
            "checkpoint": statuses.get("checkpoint", "unknown"),
            "version": "2.1.0",
        }
    )


# ------------------------------------------------------------------
# METRICS
# ------------------------------------------------------------------

@app.get("/metrics")
def metrics_endpoint():
    """Prometheus-format metrics endpoint."""

    if not settings.metrics_enabled:
        return JSONResponse(
            status_code=404,
            content={"detail": "Metrics disabled"},
        )

    prometheus_text = get_metrics().to_prometheus()
    from fastapi.responses import PlainTextResponse

    return PlainTextResponse(content=prometheus_text, media_type="text/plain; version=0.0.4")


# ------------------------------------------------------------------
# LEGACY CHAT
# ------------------------------------------------------------------

@app.post("/chat")
async def chat(
    request: Request,
    request_body: ChatRequest,
):
    """
    Backward-compatible chat endpoint.

    POST /chat

    {
        "question": "hello"
    }
    """

    question = request_body.message.strip()

    if not question:

        return JSONResponse(
            status_code=400,
            content={
                "answer": "Please enter a question."
            },
        )

    req_id = getattr(request.state, "request_id", "-")
    logger.info(
        "Legacy /chat: %s request_id=%s",
        question[:80],
        req_id,
    )

    from backend.agent.graph import get_agent_graph

    session_id = (
        f"chat_{uuid.uuid4().hex[:12]}"
    )

    initial_state: AgentState = {

        "question": question,

        "answer": "",

        "messages": [
            HumanMessage(
                content=question
            )
        ],

        "retrieved_documents": [],

        "tool_results": [],

        "memory_context": "",

        "user_id": "default_user",

        "session_id": session_id,

        "approval_required": False,

        "approved": False,

        "error": None,

        "retry_count": 0,

        "research_results": [],
    }

    async def generate():

        graph = get_agent_graph()

        config = _thread_config(
            session_id
        )

        final_answer = ""

        try:

            async for event in graph.astream(
                initial_state,
                config=config,
            ):

                for node_id, data in event.items():

                    # ----------------------------------------
                    # Generate answer node
                    # ----------------------------------------

                    if node_id == "generate_answer":

                        answer = data.get(
                            "answer",
                            "",
                        )

                        if (
                            answer
                            and answer != final_answer
                        ):

                            new_text = answer[
                                len(final_answer):
                            ]

                            for chunk in _chunk_text(
                                new_text
                            ):

                                yield chunk

                            final_answer = answer

                    # ----------------------------------------
                    # Agent node
                    # ----------------------------------------

                    elif node_id == "agent":

                        messages = data.get(
                            "messages"
                        )

                        if (
                            isinstance(messages, list)
                            and messages
                        ):

                            msg = messages[0]

                            if (
                                isinstance(
                                    msg,
                                    AIMessage,
                                )
                                and not msg.tool_calls
                            ):

                                content = (
                                    msg.content or ""
                                )

                                if (
                                    content
                                    and content
                                    != final_answer
                                ):

                                    new_text = content[
                                        len(final_answer):
                                    ]

                                    for chunk in _chunk_text(
                                        new_text
                                    ):

                                        yield chunk

                                    final_answer = content

        except Exception as exc:

            logger.error(
                "Chat stream failed: %s",
                exc,
                exc_info=True,
            )

            if not final_answer:

                yield (
                    "I encountered an error. "
                    "Please try again."
                )

        if not final_answer:

            yield (
                "I could not find an answer "
                "to your question."
            )

    return StreamingResponse(
        generate(),
        media_type="text/plain",
    )


# ------------------------------------------------------------------
# AGENT RUN
# ------------------------------------------------------------------

@app.post(
    "/agent/run",
    response_model=AgentRunResponse,
)
async def agent_run(
    request: Request,
    request_body: AgentRunRequest,
):

    question = request_body.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question is required",
        )

    session_id = (
        request_body.session_id
        or str(uuid.uuid4())
    )

    user_id = request_body.user_id or "default_user"
    req_id = getattr(request.state, "request_id", "-")

    logger.info(
        "Agent run: session=%s user=%s question=%s request_id=%s",
        session_id,
        user_id,
        question[:80],
        req_id,
    )

    from backend.agent.graph import get_agent_graph

    graph = get_agent_graph()

    config = _thread_config(
        session_id
    )

    initial_state = _initial_state(
        request_body
    )

    try:

        result: dict[str, Any] = (
            await graph.ainvoke(
                initial_state,
                config=config,
            )
        )

    except Exception as exc:

        logger.error(
            "Agent run failed: %s",
            exc,
            exc_info=True,
        )

        return AgentRunResponse(
            answer=(
                "I encountered an error "
                "while processing your request."
            ),
            session_id=session_id,
            approval_required=False,
            tool_results=None,
            sources=None,
        )

    # --------------------------------------------------------
    # HITL interrupt
    # --------------------------------------------------------

    if result.get("__interrupt__"):

        return AgentRunResponse(

            answer=result.get(
                "answer",
                "",
            ),

            session_id=session_id,

            approval_required=True,

            tool_results=[
                (
                    tr.model_dump()
                    if hasattr(
                        tr,
                        "model_dump",
                    )
                    else tr
                )

                for tr in result.get(
                    "tool_results",
                    [],
                )
            ] or None,

            sources=_extract_sources(
                result.get(
                    "retrieved_documents",
                    [],
                )
            ),
        )

    # --------------------------------------------------------
    # Normal response
    # --------------------------------------------------------

    return AgentRunResponse(

        answer=result.get(
            "answer",
            "",
        ),

        session_id=session_id,

        approval_required=False,

        tool_results=[
            (
                tr.model_dump()
                if hasattr(
                    tr,
                    "model_dump",
                )
                else tr
            )

            for tr in result.get(
                "tool_results",
                [],
            )
        ] or None,

        sources=_extract_sources(
            result.get(
                "retrieved_documents",
                [],
            )
        ),
    )


# ------------------------------------------------------------------
# AGENT STREAM
# ------------------------------------------------------------------

@app.post("/agent/stream")
async def agent_stream(
    request: Request,
    request_body: AgentStreamRequest,
):
    """
    Stream agent execution using Server-Sent Events.

    Events:
    - message
    - tool_call
    - tool_result
    - done
    - approval_required
    - error
    """

    question = request_body.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question is required",
        )

    session_id = (
        request_body.session_id
        or str(uuid.uuid4())
    )

    user_id = request_body.user_id or "default_user"
    req_id = getattr(request.state, "request_id", "-")

    logger.info(
        "Agent stream: session=%s user=%s question=%s request_id=%s",
        session_id,
        user_id,
        question[:80],
        req_id,
    )

    from backend.agent.graph import get_agent_graph

    graph = get_agent_graph()

    config = _thread_config(
        session_id
    )

    initial_state = _initial_state(
        AgentRunRequest(
            **request_body.model_dump()
        )
    )

    async def event_stream():

        try:

            async for event in graph.astream(
                initial_state,
                config=config,
            ):

                for node_id, data in event.items():

                    # ----------------------------------------
                    # HITL interrupt
                    # ----------------------------------------

                    if node_id == "__interrupt__":

                        yield (
                            "event: approval_required\n"
                            f"data: {json.dumps({'type': 'approval_required', 'data': data})}\n\n"
                        )

                        return

                    # ----------------------------------------
                    # Agent
                    # ----------------------------------------

                    if (
                        node_id == "agent"
                        and isinstance(data, dict)
                    ):

                        messages = data.get(
                            "messages"
                        )

                        if messages:

                            msg = messages[-1]

                            if isinstance(
                                msg,
                                AIMessage,
                            ):

                                # Tool calls

                                if msg.tool_calls:

                                    yield (
                                        "event: tool_call\n"
                                        f"data: {json.dumps({'tool_calls': msg.tool_calls})}\n\n"
                                    )

                                # Normal message

                                elif msg.content:

                                    yield (
                                        "event: message\n"
                                        f"data: {json.dumps({'content': msg.content})}\n\n"
                                    )

                    # ----------------------------------------
                    # Tool results
                    # ----------------------------------------

                    if (
                        node_id == "execute_tools"
                        and isinstance(data, dict)
                    ):

                        results = data.get(
                            "tool_results",
                            [],
                        )

                        for tr in results:

                            payload = (
                                tr.model_dump()
                                if hasattr(
                                    tr,
                                    "model_dump",
                                )
                                else tr
                            )

                            yield (
                                "event: tool_result\n"
                                f"data: {json.dumps({'result': payload})}\n\n"
                            )

                    # ----------------------------------------
                    # Final answer
                    # ----------------------------------------

                    if (
                        node_id == "generate_answer"
                        and isinstance(data, dict)
                    ):

                        answer = data.get(
                            "answer",
                            "",
                        )

                        if answer:

                            yield (
                                "event: done\n"
                                f"data: {json.dumps({'answer': answer})}\n\n"
                            )

                    # ----------------------------------------
                    # Approval check
                    # ----------------------------------------

                    if (
                        node_id == "check_approval"
                        and isinstance(data, dict)
                    ):

                        if data.get(
                            "approval_required"
                        ):

                            yield (
                                "event: approval_required\n"
                                f"data: {json.dumps({'answer': data.get('answer', '')})}\n\n"
                            )

                            return

        except Exception as exc:

            logger.error(
                "Agent stream failed: %s",
                exc,
                exc_info=True,
            )

            yield (
                "event: error\n"
                f"data: {json.dumps({'error': 'Agent stream failed', 'detail': 'An internal error occurred'})}\n\n"
            )

        yield (
            "event: end\n"
            f"data: {json.dumps({})}\n\n"
        )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )


# ------------------------------------------------------------------
# PARALLEL TASK RUNNER
# ------------------------------------------------------------------

@app.post("/task/run")
async def task_run(
    request: Request,
    request_body: AgentStreamRequest,
):
    """
    Split one request into subtasks and run them concurrently.

    POST /task/run

    Events (Server-Sent Events):

    - plan            : the subtasks that will run
    - subtask_start   : a lane began
    - subtask_phase   : an internal stage changed (load index, embed query,
                        select context, build prompt, generate)
    - subtask_delta   : streamed tokens from a lane (may interleave across lanes)
    - subtask_done    : a lane finished (answer included)
    - subtask_failed  : a lane failed; other lanes continue
    - merged          : the combined answer
    - done            : final answer
    - error           : the whole request failed
    - end             : stream closed

    Splitting and merging are rule-based and cost no LLM calls, so the first
    frame is emitted immediately. Phase updates and lane tokens are relayed as
    they happen, so the UI can show internal progress and text long before the
    lanes finish.
    """

    question = request_body.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question is required",
        )

    req_id = getattr(request.state, "request_id", "-")
    session_id = request_body.session_id or f"task_{uuid.uuid4().hex[:12]}"

    logger.info(
        "Parallel task run: session=%s request_id=%s question=%s",
        session_id,
        req_id,
        question[:80],
    )

    from backend.agent.parallel import (
        merge,
        run_subtask,
        split_task,
    )

    subtasks = split_task(question)

    logger.info(
        "Split %d char request into %d subtask(s): request_id=%s",
        len(question),
        len(subtasks),
        req_id,
    )

    async def event_stream():

        def frame(name: str, payload: dict) -> str:
            return (
                f"event: {name}\n"
                f"data: {json.dumps(payload)}\n\n"
            )

        try:

            yield frame(
                "plan",
                {
                    "subtasks": [s.to_dict() for s in subtasks],
                    "session_id": session_id,
                },
            )

            # Emit a start frame per lane, then resolve them all together.
            for subtask in subtasks:
                yield frame(
                    "subtask_start",
                    {"id": subtask.id, "text": subtask.text},
                )

            queue: asyncio.Queue = asyncio.Queue()

            def relay_delta(lane_id: int, text: str) -> None:
                queue.put_nowait(("delta", {"id": lane_id, "text": text}))

            def relay_phase(
                lane_id: int,
                stage: str,
                status: str,
                detail: str,
            ) -> None:
                queue.put_nowait(
                    (
                        "phase",
                        {
                            "id": lane_id,
                            "stage": stage,
                            "status": status,
                            "detail": detail,
                        },
                    )
                )

            async def run_and_report(subtask):

                try:
                    finished = await run_subtask(
                        subtask,
                        relay_delta,
                        relay_phase,
                    )
                except Exception as exc:  # noqa: BLE001
                    finished.status = "failed"
                    finished.error = f"{type(exc).__name__}: {exc}"

                queue.put_nowait(("final", finished))

            runners = [
                asyncio.create_task(run_and_report(s))
                for s in subtasks
            ]

            # Interleave phase updates and streamed tokens from every lane
            # until all are final.
            remaining = len(runners)

            while remaining:
                kind, payload = await queue.get()

                if kind == "delta":
                    yield frame(
                        "subtask_delta",
                        {"id": payload["id"], "text": payload["text"]},
                    )
                    continue

                if kind == "phase":
                    yield frame("subtask_phase", payload)
                    continue

                finished = payload

                name = (
                    "subtask_done"
                    if finished.status == "done"
                    else "subtask_failed"
                )

                yield frame(
                    name,
                    {
                        "id": finished.id,
                        "text": finished.text,
                        "answer": finished.answer,
                        "error": finished.error,
                        "sources": finished.sources,
                        "duration_ms": finished.duration_ms,
                    },
                )

                remaining -= 1

            await asyncio.gather(*runners)

            final_answer = merge(subtasks)

            # The merge is deterministic, so there is no generation to stream
            # token by token. Emitting raw text here would also corrupt the SSE
            # framing -- every byte must sit inside a proper event/data pair.
            yield frame("merged", {"answer": final_answer})

            yield frame(
                "done",
                {
                    "answer": final_answer,
                    "subtasks": [s.to_dict() for s in subtasks],
                },
            )

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Parallel task run failed: %s",
                exc,
                exc_info=True,
            )
            yield frame(
                "error",
                {
                    "error": "Task run failed",
                    "detail": str(exc) or type(exc).__name__,
                },
            )

        yield frame("end", {})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )


# ------------------------------------------------------------------
# SPLIT PREVIEW
# ------------------------------------------------------------------

@app.post("/task/split")
async def task_split(
    request: Request,
    request_body: AgentStreamRequest,
):
    """
    Show how a request would be split, without running anything.

    POST /task/split

    Instant and free: the splitter is pure rules, so the UI can show the plan
    the moment the user hits send.
    """

    from backend.agent.parallel import split_task

    subtasks = split_task(request_body.question.strip())

    return JSONResponse(
        content={
            "count": len(subtasks),
            "subtasks": [s.to_dict() for s in subtasks],
        },
    )


# ------------------------------------------------------------------
# HUMAN APPROVAL
# ------------------------------------------------------------------

@app.post(
    "/agent/{session_id}/approve"
)
async def approve_session(
    request: Request,
    session_id: str,
    request_body: ApprovalRequest,
):
    """Resume a paused agent after human approval."""

    req_id = getattr(request.state, "request_id", "-")

    logger.info(
        "Approving session %s approved=%s request_id=%s",
        session_id,
        request_body.approved,
        req_id,
    )

    from backend.agent.graph import get_agent_graph

    graph = get_agent_graph()

    config = _thread_config(
        session_id
    )

    try:

        resume_value = {
            "approved": request_body.approved
        }

        if request_body.feedback:

            resume_value["feedback"] = (
                request_body.feedback
            )

        result: dict[str, Any] = await graph.ainvoke(
            Command(
                resume=resume_value
            ),
            config=config,
        )

        return JSONResponse(
            content={
                "answer": result.get(
                    "answer",
                    "",
                ),
                "session_id": session_id,
                "approved": result.get(
                    "approved",
                    request_body.approved,
                ),
                "approval_required": result.get(
                    "approval_required",
                    False,
                ),
                "sources": _extract_sources(
                    result.get(
                        "retrieved_documents",
                        [],
                    )
                ),
            }
        )

    except Exception as exc:

        logger.error(
            "Resume failed: %s",
            exc,
            exc_info=True,
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to resume agent",
                "error_code": "resume_failed",
            },
        )


# ------------------------------------------------------------------
# JOB STATUS (background workers)
# ------------------------------------------------------------------

@app.get("/jobs/{job_id}")
async def get_job(
    request: Request,
    job_id: str,
):
    """Check the status of a background job."""

    from backend.agent.jobs import get_job_status

    try:
        info = get_job_status(job_id)
    except Exception:
        return JSONResponse(
            status_code=503,
            content={"detail": "Job queue unavailable", "error_code": "queue_unavailable"},
        )

    return info


# ------------------------------------------------------------------
# MEMORY
# ------------------------------------------------------------------

@app.post(
    "/memory",
    response_model=MemoryEntry,
)
async def store_memory(
    request: Request,
    request_body: MemoryEntryRequest,
):

    from backend.agent.memory.long_term import (
        get_long_term_memory
    )

    mem = get_long_term_memory()

    try:

        mem.save(
            request_body.user_id,
            request_body.key,
            request_body.value,
            request_body.category or "preference",
        )

    except Exception as exc:

        logger.error("Memory save failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to save memory",
        )

    return MemoryEntry(

        key=request_body.key,

        value=request_body.value,

        category=(
            request_body.category
            or "preference"
        ),

        created_at=(
            datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()
        ),
    )


@app.get(
    "/memory/{user_id}"
)
async def get_user_memory(
    request: Request,
    user_id: str,
):

    from backend.agent.memory.long_term import (
        get_long_term_memory
    )

    mem = get_long_term_memory()

    try:

        facts = mem.retrieve_all(
            user_id
        )

    except Exception as exc:

        logger.error("Memory retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve memory",
        )

    return {
        "user_id": user_id,
        "memories": facts,
    }


# ------------------------------------------------------------------
# PRODUCTION INCIDENT INVESTIGATION (MCP TOOL / FLOW)
# ------------------------------------------------------------------

@app.post(
    "/incidents/investigate",
    response_model=IncidentInvestigationResponse,
)
async def investigate_incident_endpoint(
    request: Request,
    payload: IncidentInvestigationRequest,
):
    """Investigate a production incident dynamically based on runtime user input.

    Correlates evidence from existing application logs and metrics, computes confidence,
    and returns ranked root causes and non-destructive recommendations.
    """
    from backend.agent.mcp.engine import IncidentInvestigator
    from backend.agent.mcp.sources import IncidentQueryParams

    req_id = getattr(request.state, "request_id", None) or payload.request_id

    params = IncidentQueryParams(
        service_name=payload.service_name,
        error_message=payload.error_message,
        time_range=payload.time_range,
        severity=payload.severity,
        request_id=req_id,
        endpoint=payload.endpoint,
        details=payload.details,
        user_id=payload.user_id,
    )

    logger.info(
        "Incident investigation requested (service=%s, error=%s, time_range=%s, severity=%s, request_id=%s)",
        payload.service_name,
        payload.error_message,
        payload.time_range,
        payload.severity,
        req_id,
    )

    investigator = IncidentInvestigator()
    result = investigator.investigate(params)
    return result


@app.get("/api/mcp/tools")
async def list_mcp_tools():
    """List all tools discovered across registered MCP servers."""
    from backend.agent.mcp.client import get_mcp_client

    client = get_mcp_client()
    tools = client.discover_tools()

    return {
        "mcp_servers": list(client.servers.keys()),
        "tools_count": len(tools),
        "tools": [
            {
                "name": name,
                "description": getattr(tool, "description", ""),
                "server": client._tool_to_server.get(name, "local"),
            }
            for name, tool in tools.items()
        ],
    }



# ------------------------------------------------------------------
# ERROR HANDLERS
# ------------------------------------------------------------------

@app.exception_handler(
    HTTPException
)
async def http_exception_handler(
    request: Request,
    exc: HTTPException,
):

    return JSONResponse(

        status_code=exc.status_code,

        content={
            "detail": exc.detail,
            "error_code": "http_error",
        },

        headers=exc.headers,
    )


@app.exception_handler(
    Exception
)
async def general_exception_handler(
    request: Request,
    exc: Exception,
):

    logger.error(
        "Unhandled error: %s",
        exc,
        exc_info=True,
    )

    return JSONResponse(

        status_code=500,

        content={
            "detail": "Internal server error",
            "error_code": "internal_error",
        },
    )
