"""Background job processing via Redis.

A lightweight, Redis-backed job queue — no Celery required.
Workers are separate processes that pull jobs from a Redis list.

Usage:
    # Enqueue a job from the API:
    from backend.agent.jobs import enqueue_job

    job_id = enqueue_job("ingest_document", {"file_path": "/documents/foo.pdf"})

    # Run a worker (separate process, inside the container):
    python -m backend.agent.jobs
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from backend.agent.config import get_settings

logger = logging.getLogger(__name__)

# Redis list key holding JSON-encoded jobs
_QUEUE_KEY = "agent:jobs:queue"
# Redis hash mapping job_id -> status ("pending", "running", "completed", "failed")
_STATUS_KEY = "agent:jobs:status"
# Redis hash mapping job_id -> result JSON
_RESULT_KEY = "agent:jobs:results"


def _redis_client():
    settings = get_settings()
    if not settings.redis_url:
        raise RuntimeError("REDIS_URL is required for background jobs")

    import redis

    return redis.from_url(settings.redis_url, socket_connect_timeout=5, socket_timeout=5)


# ------------------------------------------------------------------
# Job registry — maps job type string to callable
# ------------------------------------------------------------------

_JOBS: dict[str, Callable] = {}


def register_job(name: str):
    """Decorator to register a background job handler."""

    def decorator(func: Callable):
        _JOBS[name] = func
        return func

    return decorator


@register_job("ingest_document")
def _job_ingest_document(args: dict[str, Any]) -> str:
    """Background job: ingest a single document into the vector store."""
    from backend.agent.ingest import load_documents, chunk_documents, _get_vectorstore

    file_path = args["file_path"]
    if os.path.isdir(file_path):
        docs = load_documents(file_path)
    else:
        from langchain_core.documents import Document

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(file_path)
            docs = [
                Document(
                    page_content=page.extract_text() or "",
                    metadata={"source": file_path, "page": i},
                )
                for i, page in enumerate(reader.pages)
            ]
        else:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                docs = [Document(page_content=f.read(), metadata={"source": file_path})]

    chunked = chunk_documents(docs)
    vs = _get_vectorstore()
    vs.add_documents(chunked)
    return f"Ingested {len(chunked)} chunks from {file_path}"


# ------------------------------------------------------------------
# Enqueue / dequeue
# ------------------------------------------------------------------


def enqueue_job(job_type: str, args: dict[str, Any]) -> str:
    """Enqueue a job for background processing. Returns the job ID."""
    if job_type not in _JOBS:
        raise ValueError(f"Unknown job type: {job_type}")

    job_id = uuid.uuid4().hex

    payload = {
        "job_id": job_id,
        "job_type": job_type,
        "args": args,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    client = _redis_client()
    client.hset(_STATUS_KEY, job_id, "pending")
    client.rpush(_QUEUE_KEY, json.dumps(payload))

    logger.info("Enqueued job %s (%s)", job_id, job_type)
    return job_id


def get_job_status(job_id: str) -> dict[str, Any]:
    """Return the status and result of a job."""
    client = _redis_client()
    status = client.hget(_STATUS_KEY, job_id) or "unknown"
    result = client.hget(_RESULT_KEY, job_id)

    info: dict[str, Any] = {"job_id": job_id, "status": status}
    if result:
        try:
            info["result"] = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            info["result"] = result
    return info


def run_worker(once: bool = False) -> None:
    """Run the background worker loop.

    Pulls jobs from the Redis queue and executes them.
    If ``once`` is True, processes a single job and exits.
    """
    if not get_settings().redis_url:
        logger.error("Cannot start worker: REDIS_URL not set")
        return

    client = _redis_client()
    logger.info("Background worker started (once=%s)", once)

    while True:
        job_json = client.blpop(_QUEUE_KEY, timeout=5)
        if job_json is None:
            if once:
                logger.info("No jobs in queue, exiting")
                break
            continue

        payload = json.loads(job_json[1])
        job_id = payload["job_id"]
        job_type = payload["job_type"]
        args = payload.get("args", {})

        logger.info("Processing job %s (%s)", job_id, job_type)
        client.hset(_STATUS_KEY, job_id, "running")

        handler = _JOBS.get(job_type)
        if handler is None:
            client.hset(_STATUS_KEY, job_id, "failed")
            client.hset(_RESULT_KEY, job_id, json.dumps({"error": f"Unknown job type: {job_type}"}))
            continue

        try:
            result = handler(args)
            client.hset(_STATUS_KEY, job_id, "completed")
            client.hset(_RESULT_KEY, job_id, json.dumps({"result": result}))
            logger.info("Job %s completed", job_id)
        except Exception as exc:
            logger.error("Job %s failed: %s", job_id, exc, exc_info=True)
            client.hset(_STATUS_KEY, job_id, "failed")
            client.hset(_RESULT_KEY, job_id, json.dumps({"error": type(exc).__name__}))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    once = os.environ.get("WORKER_ONCE", "0") == "1"
    run_worker(once=once)
