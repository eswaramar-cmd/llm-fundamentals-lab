"""Tests for config and Pydantic validation."""

from __future__ import annotations

import pytest

from backend.agent.config import Settings, get_settings


class TestConfig:
    def test_default_settings(self, monkeypatch):
        # Settings() reads the local .env, so assert on the class defaults
        # instead -- otherwise this fails on any machine whose .env pins a
        # different model.
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        s = Settings(_env_file=None)
        assert s.ollama_model == Settings.model_fields["ollama_model"].default
        assert "11434" in s.ollama_base_url
        assert s.max_retries == 3
        assert s.tool_timeout_seconds == 30

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b")
        monkeypatch.setenv("MAX_RETRIES", "5")
        s = Settings()
        assert s.ollama_model == "qwen2.5:7b"
        assert s.max_retries == 5

    def test_get_settings_cached(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2  # lru_cache returns same instance


class TestSchemaValidation:
    """Test Pydantic request/response model validation."""

    def test_chat_request_valid(self):
        from backend.agent.schemas import ChatRequest

        req = ChatRequest(question="Hello, how are you?")
        assert req.question == "Hello, how are you?"

    def test_chat_request_too_long(self):
        from backend.agent.schemas import ChatRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ChatRequest(question="x" * 5000)

    def test_agent_run_request_defaults(self):
        from backend.agent.schemas import AgentRunRequest

        req = AgentRunRequest(question="What is 2+2?")
        assert req.user_id == "default_user"
        assert req.session_id is None
        assert req.require_approval is False

    def test_approval_request_defaults(self):
        from backend.agent.schemas import ApprovalRequest

        req = ApprovalRequest()
        assert req.approved is True
        assert req.feedback is None
