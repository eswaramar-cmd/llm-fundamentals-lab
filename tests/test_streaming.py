"""Tests for streaming behavior.

These tests require a running Ollama instance.
Set RUN_INTEGRATION_TESTS=1 to enable.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

INTEGRATION = os.environ.get("RUN_INTEGRATION_TESTS", "0") == "1"
integration = pytest.mark.skipif(not INTEGRATION, reason="Set RUN_INTEGRATION_TESTS=1 to enable")


@pytest.fixture
def client():
    from backend.main import app

    return TestClient(app, raise_server_exceptions=False)


class TestStreaming:
    @integration
    def test_chat_streaming_returns_text(self, client):
        """Legacy /chat should return a streaming text response."""
        with client.stream("POST", "/chat", json={"question": "Hello"}) as resp:
            assert resp.status_code == 200
            content_type = resp.headers.get("content-type", "")
            assert "text/plain" in content_type or "text/event-stream" in content_type

    @integration
    def test_agent_stream_returns_sse(self, client):
        """Agent stream should return Server-Sent Events."""
        with client.stream("POST", "/agent/stream", json={"question": "Hello"}) as resp:
            assert resp.status_code == 200
            content_type = resp.headers.get("content-type", "")
            assert "text/event-stream" in content_type or "text/plain" in content_type
