"""Distributed, Redis-backed rate limiter.

Uses a sliding-window counter stored in Redis so every API replica
shares the same counter for a given user/session.

When Redis is unavailable or rate limiting is disabled, all requests
pass through (fail-open).
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Raised when a client exceeds the configured rate limit."""

    def __init__(self, retry_after: int) -> None:
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded, retry after {retry_after}s")


def _client_key(user_id: str, client_ip: str, endpoint: str) -> str:
    """Build a Redis key that uniquely identifies a rate-limit bucket."""
    raw = f"{user_id}:{client_ip}:{endpoint}"
    h = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"ratelimit:{h}"


class RedisRateLimiter:
    """Sliding-window rate limiter backed by Redis."""

    def __init__(self, redis_url: str, requests: int, window_seconds: int) -> None:
        self._redis_url = redis_url
        self._requests = requests
        self._window_seconds = window_seconds
        self._client: Optional[object] = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        import redis

        self._client = redis.from_url(self._redis_url, socket_connect_timeout=2, socket_timeout=2)
        return self._client

    def check(self, user_id: str, client_ip: str, endpoint: str) -> None:
        """Raise ``RateLimitExceeded`` if the limit has been exceeded."""

        # Fail open when Redis is unavailable
        try:
            client = self._get_client()
        except Exception as exc:
            logger.warning("Redis unavailable for rate limiting, failing open: %s", exc)
            return

        key = _client_key(user_id, client_ip, endpoint)
        now = time.time()
        window_start = now - self._window_seconds

        try:
            pipe = client.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zcard(key)
            pipe.zadd(key, {str(now): now})
            pipe.expire(key, self._window_seconds)
            results = pipe.execute()

            count = results[1]
            if count > self._requests:
                # Remove the request we just added (over-count)
                client.zremrangebyscore(key, now, now)
                retry_after = self._window_seconds
                raise RateLimitExceeded(retry_after)

        except RateLimitExceeded:
            raise
        except Exception as exc:
            logger.warning("Rate-limit check failed, failing open: %s", exc)


class NoopRateLimiter:
    """No-op rate limiter used when rate limiting is disabled."""

    def check(self, user_id: str, client_ip: str, endpoint: str) -> None:
        return None
