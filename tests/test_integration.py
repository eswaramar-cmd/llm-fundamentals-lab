"""Integration tests that require Ollama and ChromaDB.

Run with: RUN_INTEGRATION_TESTS=1 python -m pytest tests/test_integration.py -v
"""

from __future__ import annotations

import os

import pytest

INTEGRATION = os.environ.get("RUN_INTEGRATION_TESTS", "0") == "1"
pytestmark = pytest.mark.skipif(not INTEGRATION, reason="Set RUN_INTEGRATION_TESTS=1")


@pytest.fixture
def ollama_available():
    import httpx
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def test_ollama_reachable(ollama_available):
    assert ollama_available, "Ollama must be running for integration tests"


@pytest.mark.asyncio
async def test_graph_runs_simple_conversation():
    """Run the full graph with a simple greeting."""
    from langchain_core.messages import HumanMessage
    from backend.agent.graph import get_agent_graph
    from backend.agent.state import AgentState

    graph = get_agent_graph()
    session_id = "test_conv_001"

    state: AgentState = {
        "question": "Hello, what is your name?",
        "answer": "",
        "messages": [HumanMessage(content="Hello, what is your name?")],
        "retrieved_documents": [],
        "tool_results": [],
        "memory_context": "",
        "user_id": "test_user",
        "session_id": session_id,
        "approval_required": False,
        "approved": False,
        "error": None,
        "retry_count": 0,
        "research_results": [],
    }

    config = {"configurable": {"thread_id": session_id}}
    result = await graph.ainvoke(state, config=config)

    assert result.get("answer") is not None
    assert len(result["answer"]) > 0
    assert result.get("error") is None


@pytest.mark.asyncio
async def test_graph_calculator_tool():
    """Test that the agent uses the calculator tool."""
    from langchain_core.messages import HumanMessage
    from backend.agent.graph import get_agent_graph
    from backend.agent.state import AgentState

    graph = get_agent_graph()
    session_id = "test_calc_001"

    state: AgentState = {
        "question": "What is 25 multiplied by 40?",
        "answer": "",
        "messages": [HumanMessage(content="What is 25 multiplied by 40?")],
        "retrieved_documents": [],
        "tool_results": [],
        "memory_context": "",
        "user_id": "test_user",
        "session_id": session_id,
        "approval_required": False,
        "approved": False,
        "error": None,
        "retry_count": 0,
        "research_results": [],
    }

    config = {"configurable": {"thread_id": session_id}}
    result = await graph.ainvoke(state, config=config)

    assert result.get("answer") is not None
    assert "1000" in result["answer"]
