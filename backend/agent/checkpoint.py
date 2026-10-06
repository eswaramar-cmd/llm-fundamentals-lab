"""Checkpoint / state abstraction for LangGraph.

Provides a shared, production-compatible checkpoint store backed by Redis
when ``REDIS_URL`` is set. Falls back to in-memory ``MemorySaver`` for
single-instance local development when Redis is unavailable.

This module is the single place where the checkpointer is selected, so
``graph.py`` and ``main.py`` never import ``MemorySaver`` or ``RedisSaver``
directly.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from langgraph.checkpoint.base import BaseCheckpointSaver

from backend.agent.config import get_settings

logger = logging.getLogger(__name__)

# Module-level optional imports (lazy-safe for environments without redis)
try:
    import redis as _redis_lib
except ImportError:
    _redis_lib = None

try:
    from langgraph.checkpoint.redis import RedisSaver as _RedisSaver
except ImportError:
    _RedisSaver = None


def _redis_available() -> bool:
    """Check whether Redis is configured and reachable."""
    settings = get_settings()
    if not settings.redis_url:
        return False
    if _redis_lib is None:
        logger.warning("redis-py not installed; cannot use Redis checkpointing")
        return False
    try:
        # One second, not the redis-py default. This probe runs during startup
        # and its failure is non-fatal — we fall back to InMemorySaver — so the
        # only thing a long timeout buys is a slower start. On Windows a connect
        # is attempted per address family, so a 2s timeout cost 4.1s of startup
        # for a Redis that was never going to answer.
        r = _redis_lib.from_url(
            settings.redis_url,
            socket_connect_timeout=1,
            socket_timeout=1,
            retry_on_timeout=False,
        )
        r.ping()
        return True
    except Exception as exc:
        logger.warning("Redis ping failed: %s", exc)
        return False


@lru_cache(maxsize=1)
def get_checkpoint_saver() -> BaseCheckpointSaver:
    """Return a shared checkpoint saver.

    Priority:
        1. RedisSaver  — when REDIS_URL is set and reachable (production,
           multi-replica).
        2. MemorySaver — fallback for local development without Redis.
    """
    settings = get_settings()

    if settings.redis_url and _redis_available() and _RedisSaver is not None:
        try:
            ttl: Optional[dict] = None
            if settings.redis_ttl_seconds > 0:
                ttl = {"default_ttl": settings.redis_ttl_seconds}

            saver = _RedisSaver(
                redis_url=settings.redis_url,
                ttl=ttl,
            )
            logger.info("Using RedisSaver for shared checkpointing (redis=%s)", settings.redis_url)
            return saver
        except Exception as exc:
            logger.warning("Failed to initialise RedisSaver, falling back to MemorySaver: %s", exc)

    # Fallback: in-memory (single-instance only)
    from langgraph.checkpoint.memory import MemorySaver

    logger.info("Using MemorySaver (in-memory, single-instance only)")
    return MemorySaver()
