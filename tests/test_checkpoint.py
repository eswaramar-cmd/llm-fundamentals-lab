"""Tests for the checkpoint abstraction (Redis + fallback)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


class TestCheckpointFactory:
    """Tests for the checkpoint saver factory."""

    def test_falls_back_to_memory_saver_without_redis(self):
        """Without REDIS_URL, should use MemorySaver."""
        os.environ.pop("REDIS_URL", None)

        from backend.agent.config import get_settings

        get_settings.cache_clear()
        from backend.agent.checkpoint import get_checkpoint_saver

        get_checkpoint_saver.cache_clear()

        saver = get_checkpoint_saver()
        assert saver is not None
        assert "Memory" in type(saver).__name__

    def test_uses_redis_saver_when_redis_available(self):
        """When REDIS_URL is set and reachable, should use RedisSaver."""
        os.environ["REDIS_URL"] = "redis://localhost:6379/0"

        from backend.agent.config import get_settings

        get_settings.cache_clear()
        from backend.agent.checkpoint import get_checkpoint_saver

        get_checkpoint_saver.cache_clear()

        # Mock both the redis library and the RedisSaver class
        mock_redis = MagicMock()
        mock_redis.from_url.return_value.ping.return_value = True
        mock_saver_instance = MagicMock(name="RedisSaver")

        with patch("backend.agent.checkpoint._redis_lib", mock_redis), \
             patch("backend.agent.checkpoint._RedisSaver", return_value=mock_saver_instance):
            saver = get_checkpoint_saver()
            assert saver is mock_saver_instance

        os.environ.pop("REDIS_URL", None)
        get_settings.cache_clear()
        get_checkpoint_saver.cache_clear()

    def test_falls_back_when_redis_unreachable(self):
        """When Redis is unreachable, should fall back to MemorySaver."""
        os.environ["REDIS_URL"] = "redis://localhost:6379/0"

        from backend.agent.config import get_settings

        get_settings.cache_clear()
        from backend.agent.checkpoint import get_checkpoint_saver

        get_checkpoint_saver.cache_clear()

        # Mock redis to raise on ping
        mock_redis = MagicMock()
        mock_redis.from_url.return_value.ping.side_effect = Exception("Connection refused")

        with patch("backend.agent.checkpoint._redis_lib", mock_redis):
            saver = get_checkpoint_saver()
            assert "Memory" in type(saver).__name__

        os.environ.pop("REDIS_URL", None)
        get_settings.cache_clear()
        get_checkpoint_saver.cache_clear()
