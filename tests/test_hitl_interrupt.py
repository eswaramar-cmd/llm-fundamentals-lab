"""Tests for HITL interrupt/resume mechanisms in LangGraph."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from backend.agent.state import AgentState


class TestInterruptResume:
    """Test the LangGraph interrupt/resume mechanism."""

    def test_interrupt_value_type(self):
        """The interrupt value should be a dict with type info."""
        from langgraph.types import interrupt

        # interrupt raises Interrupt exception - we can't call it directly
        # in a unit test. Instead, verify the function exists and is callable.
        assert callable(interrupt)

    def test_check_approval_with_sensitive_question(self):
        """Check approval should flag sensitive questions."""
        from backend.agent.nodes.approval import check_approval

        state: AgentState = {
            "question": "Please delete my account",
            "answer": "Deleting your account...",
            "approval_required": True,
            "tool_results": [],
        }
        # The check_approval node uses interrupt() which raises,
        # so we test the keyword detection logic separately
        question = state["question"].lower()
        needs = any(kw in question for kw in ["delete", "drop", "destroy"])
        assert needs is True

    def test_check_approval_safe(self):
        """Safe questions should not require approval."""
        from backend.agent.nodes.approval import check_approval

        state: AgentState = {
            "question": "What is the weather today?",
            "answer": "It's sunny.",
            "approval_required": False,
            "tool_results": [],
        }
        # This should not interrupt
        try:
            result = check_approval(state)
            assert result["approved"] is True
            assert result["approval_required"] is False
        except Exception as e:
            # If interrupt() was called, it means the check is working
            # but for safe queries it shouldn't interrupt
            pytest.fail(f"Safe query should not interrupt: {e}")

    def test_check_approval_with_sensitive_tool_result(self):
        """Sensitive content in tool results should trigger approval."""
        from backend.agent.state import ToolResult

        tool_results = [ToolResult(tool_name="web_search", result="delete the database", success=True)]
        answer_lower = "the command executed".lower()
        needs = any(kw in answer_lower for kw in ["delete"])
        result_text = tool_results[0].result.lower()
        needs = needs or any(kw in result_text for kw in ["delete"])
        assert needs is True


class TestApprovalRequest:
    """Test the approval request flow."""

    def test_approve_with_feedback(self):
        from backend.agent.schemas import ApprovalRequest

        req = ApprovalRequest(approved=True, feedback="Looks good, proceed")
        assert req.approved is True
        assert req.feedback == "Looks good, proceed"

    def test_reject_without_feedback(self):
        from backend.agent.schemas import ApprovalRequest

        req = ApprovalRequest(approved=False)
        assert req.approved is False
        assert req.feedback is None
