"""Tests for the Redis-backed distributed rate limiter."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from backend.agent.rate_limiter import RedisRateLimiter, NoopRateLimiter, RateLimitExceeded


class TestNoopRateLimiter:
    """The no-op limiter should never block."""

    def test_noop_always_passes(self):
        limiter = NoopRateLimiter()
        # Should not raise
        limiter.check("user1", "1.2.3.4", "/agent/run")


class TestRedisRateLimiter:
    """Tests for the Redis rate limiter logic."""

    def test_allows_under_limit(self):
        """Requests under the limit should not raise."""
        limiter = RedisRateLimiter(
            redis_url="redis://localhost:6379/0",
            requests=5,
            window_seconds=60,
        )

        mock_client = MagicMock()
        mock_client.pipeline.return_value = MagicMock()
        # zremrangebyscore -> pipeline, zcard returns 1 (under limit)
        mock_pipe = MagicMock()
        mock_pipe.execute.return_value = [None, 1, 1, True]
        mock_client.pipeline.return_value = mock_pipe

        limiter._client = mock_client
        limiter.check("user1", "1.2.3.4", "/agent/run")

    def test_blocks_over_limit(self):
        """Requests over the limit should raise RateLimitExceeded."""
        limiter = RedisRateLimiter(
            redis_url="redis://localhost:6379/0",
            requests=5,
            window_seconds=60,
        )

        mock_client = MagicMock()
        mock_pipe = MagicMock()
        # zcard returns 6 (over limit of 5)
        mock_pipe.execute.return_value = [None, 6, 1, True]
        mock_client.pipeline.return_value = mock_pipe

        limiter._client = mock_client
        with pytest.raises(RateLimitExceeded) as exc_info:
            limiter.check("user1", "1.2.3.4", "/agent/run")

        assert exc_info.value.retry_after > 0

    def test_fails_open_on_redis_error(self):
        """When Redis is unavailable, should fail open (not block)."""
        limiter = RedisRateLimiter(
            redis_url="redis://localhost:6379/0",
            requests=5,
            window_seconds=60,
        )

        mock_client = MagicMock()
        mock_client.pipeline.side_effect = Exception("Redis down")
        limiter._client = mock_client

        # Should not raise
        limiter.check("user1", "1.2.3.4", "/agent/run")

    def test_different_users_have_separate_limits(self):
        """User A's requests should not count toward User B's limit."""
        limiter = RedisRateLimiter(
            redis_url="redis://localhost:6379/0",
            requests=5,
            window_seconds=60,
        )

        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_pipe.execute.return_value = [None, 3, 1, True]
        mock_client.pipeline.return_value = mock_pipe
        limiter._client = mock_client

        # Both should pass independently
        from backend.agent.rate_limiter import _client_key

        key_a = _client_key("userA", "1.2.3.4", "/agent/run")
        key_b = _client_key("userB", "1.2.3.4", "/agent/run")
        assert key_a != key_b
