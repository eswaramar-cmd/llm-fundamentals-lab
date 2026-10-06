"""Tests for error handling and retry logic."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from backend.agent.nodes.error_handler import handle_error
from backend.agent.nodes.agent import agent_node
from backend.agent.state import AgentState


class TestErrorHandling:
    def test_handle_error_increments_retry(self):
        state: AgentState = {"error": "timeout", "retry_count": 0}
        result = handle_error(state)
        assert result["retry_count"] == 1
        assert result["answer"] == "Retrying..."

    def test_handle_error_max_retries(self):
        state: AgentState = {"error": "timeout", "retry_count": 3}
        result = handle_error(state)
        assert result["retry_count"] == 3
        assert "error" in result["answer"].lower()

    def test_handle_error_preserves_original_error(self):
        state: AgentState = {"error": "Ollama unavailable", "retry_count": 1}
        result = handle_error(state)
        assert result["error"] == "Ollama unavailable"


class TestAgentNodeRetry:
    def test_agent_node_exhausts_retries(self):
        """Agent node should return error after max retries."""
        state: AgentState = {
            "question": "What is the meaning of life?",
            "messages": [HumanMessage(content="What is the meaning of life?")],
            "retry_count": 5,  # already past max
        }

        # Mock LLM to always raise. The node streams now (astream, not invoke)
        # and picks its model through models_for_intent, so both are patched.
        with patch("backend.agent.nodes.agent.models_for_intent") as mock_get:
            mock_llm = MagicMock()
            mock_llm.astream.side_effect = ConnectionError("Ollama down")
            mock_get.return_value = [("mock", mock_llm)]

            result = asyncio.run(agent_node(state))

        assert result["error"] is not None
        assert "error" in result["answer"].lower() or "unavailable" in result["answer"].lower()
