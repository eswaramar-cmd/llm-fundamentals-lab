"""Shared pytest fixtures."""

from __future__ import annotations

import os
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import pytest


@pytest.fixture
def tmp_chroma(tmp_path):
    """Provide a temporary Chroma directory."""
    return str(tmp_path / "chroma_test")


@pytest.fixture
def mock_llm_config(monkeypatch):
    """Patch LLM so tests don't hit Ollama unless explicitly requested."""
    from langchain_ollama import ChatOllama

    class FakeChatOllama(ChatOllama):
        """A ChatOllama subclass that can be overridden per-test."""

        def __init__(self, **kwargs):
            # Don't actually connect; just store kwargs
            self._mock_responses = kwargs.pop("_mock_responses", [])
            self._mock_index = 0
            for k, v in kwargs.items():
                setattr(self, k, v)

        def invoke(self, *args, **kwargs):
            # Simulate tool-calling
            from langchain_core.messages import AIMessage

            if self._mock_responses:
                return self._mock_responses[self._mock_index % len(self._mock_responses)]
            self._mock_index += 1
            return AIMessage(content="Mock response")

        def stream(self, *args, **kwargs):
            from langchain_core.messages import AIMessage

            yield AIMessage(content="Mock stream")

    monkeypatch.setattr("backend.agent.llm.ChatOllama", FakeChatOllama)
