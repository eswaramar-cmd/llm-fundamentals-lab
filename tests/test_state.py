"""Tests for the AgentState TypedDict."""

from __future__ import annotations

from backend.agent.state import AgentState, RetrievedDocument, ToolResult, ResearchResult


def test_agent_state_has_all_required_fields():
    """AgentState must declare all fields listed in the spec."""
    expected_fields = {
        "question",
        "answer",
        "messages",
        "retrieved_documents",
        "tool_results",
        "memory_context",
        "user_id",
        "session_id",
        "approval_required",
        "approved",
        "error",
        "retry_count",
        "research_results",
    }
    assert set(AgentState.__annotations__.keys()) == expected_fields


def test_agent_state_defaults():
    """AgentState should allow total=False (partial initialization)."""
    state: AgentState = {"question": "hi"}
    assert state["question"] == "hi"


def test_retrieved_document_model():
    doc = RetrievedDocument(content="test content", source="test.pdf", score=0.95)
    assert doc.content == "test content"
    assert doc.source == "test.pdf"
    assert doc.score == 0.95


def test_tool_result_model():
    result = ToolResult(tool_name="calculator", result="1250", success=True)
    assert result.tool_name == "calculator"
    assert result.result == "1250"
    assert result.success is True
    assert result.error is None


def test_tool_result_failure():
    result = ToolResult(tool_name="web_search", result="", success=False, error="timeout")
    assert result.success is False
    assert result.error == "timeout"


def test_research_result_model():
    research = ResearchResult(
        query="latest AI news",
        findings="AI is advancing rapidly",
        sources=["https://example.com"],
    )
    assert research.query == "latest AI news"
    assert len(research.sources) == 1
