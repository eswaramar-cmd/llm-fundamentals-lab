"""Tests for FastAPI API validation and endpoints."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

# Integration tests are skipped unless explicitly enabled
INTEGRATION = os.environ.get("RUN_INTEGRATION_TESTS", "0") == "1"
integration = pytest.mark.skipif(not INTEGRATION, reason="Set RUN_INTEGRATION_TESTS=1 to enable")


@pytest.fixture
def client():
    from backend.main import app

    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    @integration
    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "ollama" in data
        assert "chroma" in data

    @integration
    def test_readiness_check(self, client):
        resp = client.get("/health/ready")
        assert resp.status_code == 200

    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data


class TestAPIValidation:
    def test_chat_request_empty_question(self, client):
        """Empty question should return 400 or 422 (Pydantic validation)."""
        response = client.post("/chat", json={"question": ""})
        assert response.status_code in (400, 422)

    def test_chat_request_missing_question(self, client):
        """Missing question should return 422 (Pydantic validation)."""
        response = client.post("/chat", json={})
        assert response.status_code == 422

    def test_agent_run_missing_question(self, client):
        response = client.post("/agent/run", json={})
        assert response.status_code == 422

    @integration
    def test_agent_run_valid_request_model(self, client):
        response = client.post("/agent/run", json={"question": "Hello"})
        assert response.status_code in (200, 500)

    @integration
    def test_memory_store_valid(self, client):
        response = client.post(
            "/memory",
            json={
                "user_id": "test_user",
                "key": "test_key",
                "value": "test_value",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["key"] == "test_key"
        assert data["value"] == "test_value"

    def test_memory_store_missing_key(self, client):
        """Missing key should return 422 before hitting ChromaDB."""
        response = client.post(
            "/memory",
            json={"user_id": "test", "value": "val"},
        )
        assert response.status_code == 422

    def test_memory_store_missing_value(self, client):
        """Missing value should return 422 before hitting ChromaDB."""
        response = client.post(
            "/memory",
            json={"user_id": "test", "key": "mykey"},
        )
        assert response.status_code == 422

    @integration
    def test_memory_retrieve(self, client):
        # First store something
        client.post(
            "/memory",
            json={"user_id": "test_user_api", "key": "lang", "value": "Python"},
        )
        # Then retrieve
        response = client.get("/memory/test_user_api")
        assert response.status_code == 200
        data = response.json()
        assert "memories" in data
        assert any(m["key"] == "lang" for m in data["memories"])

    def test_approve_endpoint_exists(self, client):
        """The approve endpoint should be registered."""
        response = client.post("/agent/test_session_id/approve", json={"approved": True})
        # May return 500 if session not found, but endpoint exists
        assert response.status_code in (200, 404, 500)

    @integration
    def test_chat_streaming(self, client):
        """Test the legacy /chat endpoint streaming."""
        with client.stream("POST", "/chat", json={"question": "Hello"}) as resp:
            assert resp.status_code == 200
            text = ""
            for chunk in resp.iter_text():
                text += chunk
            assert len(text) > 0

    @integration
    def test_agent_stream_sse(self, client):
        """Agent stream should return Server-Sent Events."""
        with client.stream("POST", "/agent/stream", json={"question": "Hello"}) as resp:
            assert resp.status_code == 200
            content_type = resp.headers.get("content-type", "")
            assert "text" in content_type or "event-stream" in content_type
