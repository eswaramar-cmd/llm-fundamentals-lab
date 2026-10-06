"""Configuration management using Pydantic Settings.

All configuration comes from environment variables with sensible defaults.
Never hardcode secrets — everything is environment-driven.

Usage:
    from backend.agent.config import get_settings
    settings = get_settings()
    llm = ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# backend/agent/config.py -> backend/ is the package's parent, the repository
# root is one above that. Data stores and documents live at the repository
# root, so resolving them from the wrong level silently creates a fresh empty
# index and loses the existing documents.
BACKEND_DIR = PROJECT_ROOT / "backend"

# Absolute paths, because the app is started both from the repository root
# (uvicorn backend.main:app) and from backend/ (docker). A relative ".env"
# resolves against the working directory and is missed in one of the two.
ENV_FILES = (BACKEND_DIR / ".env", PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Ollama / LLM ----
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_temperature: float = 0.1
    ollama_num_predict: int = 512
    # How long Ollama keeps a model resident in RAM. "-1" pins it forever, which
    # starves the page cache on low-RAM machines and slows every later request.
    ollama_keep_alive: str = "5m"
    # Context window. Smaller = less KV cache RAM and less prompt to prefill.
    ollama_num_ctx: int = 2048

    # ---- LLM abstraction ----
    # Development: LLM_PROVIDER=ollama, LLM_BASE_URL=http://host.docker.internal:11434
    # Production: LLM_PROVIDER=openai, LLM_BASE_URL=https://api.openai.com/v1, LLM_MODEL=gpt-4o
    llm_provider: str = "ollama"  # "ollama" | "openai" | "anthropic" | "groq" | "vertexai"
    llm_base_url: str = ""
    llm_model: str = "llama3.2:3b"

    # ---- Embeddings ----
    # Ollama embedding model (nomic-embed-text) — fast and local.
    embedding_model: str = "nomic-embed-text"
    embedding_provider: str = "ollama"  # "ollama" or "huggingface"

    # ---- Vector store abstraction ----
    # Development: local ChromaDB on shared Docker volumes
    # Production: shared Chroma server (chroma://host:port) or alternative
    vector_store_provider: str = "chroma_local"  # "chroma_local" | "chroma_server" | "weaviate" | "qdrant"
    vector_store_url: str = ""
    chroma_rag_path: str = str(PROJECT_ROOT / "chroma_rag")
    chroma_rag_collection: str = "documents"

    chroma_memory_path: str = str(PROJECT_ROOT / "chroma_memory")
    chroma_memory_collection: str = "long_term_memory"

    # ---- Redis / shared state ----
    # Shared checkpoint store for multi-replica session/state persistence.
    # When REDIS_URL is set, RedisSaver is used; otherwise MemorySaver
    # (for single-instance local development).
    redis_url: str = ""
    redis_ttl_seconds: int = 86400  # 24h default TTL for checkpoints

    # ---- Agent behaviour ----
    max_retries: int = 3
    tool_timeout_seconds: int = 30
    max_tool_iterations: int = 5
    retry_backoff: float = 2.0  # exponential backoff base

    # When false, the router never falls back to an LLM classification call —
    # unmatched questions default to "direct". Saves one full LLM round trip.
    classify_llm_fallback: bool = True

    # Characters of each retrieved chunk handed to the LLM.
    rag_chunk_char_limit: int = 1200

    # ---- Parallel task execution (agent/parallel.py) ----
    # Optional model override for lane answers. Lets the lanes run a small, fast
    # model while the single-agent path keeps a larger one that can actually
    # emit valid tool calls.
    parallel_llm_model: str = ""
    # Upper bound on subtasks produced by one request.
    parallel_max_subtasks: int = 4
    # Chunks retrieved per subtask.
    parallel_retrieve_k: int = 2
    # Tokens allowed per subtask answer. Kept tight: with several lanes running
    # the total is what the user waits for, not any single lane.
    parallel_num_predict: int = 120

    # Keywords that trigger human-in-the-loop approval.
    # If the user query or a tool result contains any of these,
    # the agent pauses for human review.
    approval_keywords: list[str] = [
        "delete",
        "drop",
        "destroy",
        "remove",
        "rm -rf",
        "format",
        "shutdown",
        "restart",
        "kill",
    ]

    # ---- Documents ----
    documents_path: str = str(PROJECT_ROOT / "documents")
    doc_storage_provider: str = "local"  # "local" | "s3"
    doc_storage_bucket: str = ""
    doc_storage_prefix: str = "documents"

    # ---- Rate limiting (Redis-backed, distributed) ----
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 60  # max requests per window per user
    rate_limit_window_seconds: int = 60

    # ---- Gmail (real SMTP send) ----
    # Blank credentials disable the tool entirely: it refuses rather than
    # pretending, so a half-configured deployment cannot silently no-op.
    gmail_user: str = ""
    gmail_app_password: str = ""
    gmail_from_name: str = "AI Agent"
    gmail_smtp_host: str = "smtp.gmail.com"
    gmail_smtp_port: int = 587
    # Per-socket timeout. A hung SMTP handshake must not hold the request open.
    gmail_timeout_seconds: int = 30
    gmail_max_retries: int = 2
    # Sending is a side effect on a third party, so it is capped far more
    # tightly than chat traffic.
    gmail_max_per_hour: int = 10

    # Raw attachment bytes allowed in one message.
    #
    # This is not an arbitrary policy number. Gmail rejects any message over
    # 25 MB, and base64 inflates binary content by a third on the wire, so
    # ~18 MB of raw attachment is the most that reliably fits. Exceeding this is
    # reported clearly here rather than discovered as an opaque bounce later.
    gmail_max_attachment_mb: int = 18

    # ---- Uploads ----
    #
    # Storage-side limits, deliberately far more generous than the email ceiling
    # above: a large scanned PDF is worth keeping on disk even when it cannot be
    # mailed in one piece. Raise these freely; they exist to stop a runaway
    # upload filling the disk, not to ration document sizes.
    upload_max_file_mb: int = 200
    upload_max_files: int = 50
    upload_max_total_mb: int = 1000
    # Refuse new uploads when free space drops below this, so the app fails with
    # an explanation instead of a disk-full error part-way through a write.
    upload_min_free_mb: int = 512

    # ---- CORS ----
    cors_origins: str = "http://localhost:3000,http://localhost:8000,http://localhost:80"

    # ---- Request size limit ----
    max_request_size_mb: int = 10

    # ---- Metrics ----
    metrics_enabled: bool = True

    # ---- Logging ----
    log_level: str = "INFO"

    # ---- Docker / host detection ----
    @property
    def is_docker(self) -> bool:
        """Detect whether running inside a Docker container."""
        return os.path.exists("/.dockerenv")

    @field_validator("chroma_rag_path", "chroma_memory_path", "documents_path")
    @classmethod
    def _anchor_relative_paths(cls, value: str) -> str:
        """Resolve a relative store path against PROJECT_ROOT.

        .env conventionally holds values like ./chroma_rag. Left relative,
        those silently point at whatever the process working directory
        happens to be, so the app works when launched from the repository
        root and silently starts an empty store when launched from backend/.
        Absolute paths and Docker mount points are left untouched.
        """
        path = Path(value)

        if path.is_absolute():
            return str(path)

        return str((PROJECT_ROOT / path).resolve())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton."""
    return Settings()
