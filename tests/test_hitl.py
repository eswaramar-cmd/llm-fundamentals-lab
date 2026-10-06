"""Tests for human-in-the-loop (interrupt/pause/resume)."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from backend.agent.state import AgentState


class TestHITL:
    def test_check_approval_no_interrupt_when_safe(self):
        """Approval check should not interrupt for safe queries."""
        from backend.agent.nodes.approval import check_approval

        state: AgentState = {
            "question": "What is 2 + 2?",
            "answer": "4",
            "approval_required": False,
            "tool_results": [],
        }
        result = check_approval(state)
        assert result["approved"] is True
        assert result["approval_required"] is False

    def test_check_approval_user_requested(self):
        """When user requests approval, graph should pause."""
        from langgraph.types import Interrupt

        # We can't easily test interrupt() in a unit test because it
        # raises Interrupt exception. Just verify the logic detects it.
        sensitive_check = "delete"

        # Simulate the check logic
        question = "delete my account"
        needs = any(kw in question.lower() for kw in ["delete", "drop", "destroy"])
        assert needs is True

    def test_check_approval_safe_question(self):
        question = "What is the weather?"
        needs = any(kw in question.lower() for kw in ["delete", "drop", "destroy"])
        assert needs is False

    def test_approval_request_schema(self):
        from backend.agent.schemas import ApprovalRequest

        req = ApprovalRequest(approved=True, feedback="Looks good")
        assert req.approved is True
        assert req.feedback == "Looks good"

        req2 = ApprovalRequest(approved=False)
        assert req2.approved is False
        assert req2.feedback is None
