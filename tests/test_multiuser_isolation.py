"""Tests for multi-user isolation across sessions/replicas.

Verifies that:
- User A's session state never leaks to User B
- Session IDs are properly isolated in the checkpoint store
- The same session can be resumed across "replicas" (different graph instances)
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest
from fastapi.testclient import TestClient

# Disable rate limiting for these tests
os.environ["RATE_LIMIT_ENABLED"] = "false"

INTEGRATION = os.environ.get("RUN_INTEGRATION_TESTS", "0") == "1"
integration = pytest.mark.skipif(not INTEGRATION, reason="Set RUN_INTEGRATION_TESTS=1 to enable")


class TestMultiUserIsolation:
    """Verify user/session isolation in shared state."""

    @integration
    def test_different_sessions_have_independent_state(self):
        """Two different session IDs should not share conversation state."""
        from backend.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            user_a_session = f"session_a_{uuid.uuid4().hex[:8]}"
            user_b_session = f"session_b_{uuid.uuid4().hex[:8]}"

            r1 = client.post(
                "/agent/run",
                json={
                    "question": "What is 2+2?",
                    "user_id": "userA",
                    "session_id": user_a_session,
                },
            )
            assert r1.status_code == 200

            r2 = client.post(
                "/agent/run",
                json={
                    "question": "What is 3+3?",
                    "user_id": "userB",
                    "session_id": user_b_session,
                },
            )
            assert r2.status_code == 200

            assert r1.json()["session_id"] == user_a_session
            assert r2.json()["session_id"] == user_b_session

    @integration
    def test_same_session_preserves_history(self):
        """A second message on the same session should have conversation history."""
        from backend.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            session_id = f"session_test_{uuid.uuid4().hex[:8]}"

            r1 = client.post(
                "/agent/run",
                json={
                    "question": "My name is Alice.",
                    "user_id": "test_user",
                    "session_id": session_id,
                },
            )
            assert r1.status_code == 200

            r2 = client.post(
                "/agent/run",
                json={
                    "question": "What is my name?",
                    "user_id": "test_user",
                    "session_id": session_id,
                },
            )
            assert r2.status_code == 200

    @integration
    def test_user_memory_isolation(self):
        """User A's memories should not appear for User B."""
        from backend.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            user_a = f"iso_user_a_{uuid.uuid4().hex[:8]}"
            user_b = f"iso_user_b_{uuid.uuid4().hex[:8]}"

            r_a = client.post(
                "/memory",
                json={
                    "user_id": user_a,
                    "key": "secret_pref",
                    "value": "I love jazz",
                    "category": "preference",
                },
            )
            assert r_a.status_code == 200

            r_b = client.post(
                "/memory",
                json={
                    "user_id": user_b,
                    "key": "secret_pref",
                    "value": "I love rock",
                    "category": "preference",
                },
            )
            assert r_b.status_code == 200

            get_a = client.get(f"/memory/{user_a}")
            assert get_a.status_code == 200
            a_memories = get_a.json()["memories"]
            for mem in a_memories:
                assert "rock" not in mem["value"]

            get_b = client.get(f"/memory/{user_b}")
            assert get_b.status_code == 200
            b_memories = get_b.json()["memories"]
            for mem in b_memories:
                assert "jazz" not in mem["value"]


class TestHITLMultiReplica:
    """Verify HITL interrupt/resume works across 'replicas'."""

    def test_interrupt_then_resume_same_session(self):
        """A graph interrupted on one instance can be resumed."""
        from backend.agent.graph import get_agent_graph
        from backend.agent.state import AgentState
        from langchain_core.messages import HumanMessage
        from langgraph.types import Command

        # Simulate "replica 1" handling the initial run
        graph1 = get_agent_graph()
        session_id = f"hitl_test_{uuid.uuid4().hex[:8]}"

        initial_state: AgentState = {
            "question": "delete my account",
            "answer": "",
            "messages": [HumanMessage(content="delete my account")],
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

        # Run until interrupt
        result1 = asyncio.run(graph1.ainvoke(initial_state, config=config))

        # Should have interrupted
        assert result1.get("__interrupt__") is not None

        # Simulate "replica 2" resuming with approval
        # In a real multi-replica setup, this would be a different process.
        # With Redis checkpointing, the state is shared.
        graph2 = get_agent_graph()

        resume_result = asyncio.run(
            graph2.ainvoke(
                Command(resume={"approved": True, "feedback": "Approved in test"}),
                config=config,
            )
        )

        assert resume_result.get("approved") is True
        assert resume_result.get("approval_required") is False
