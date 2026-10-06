"""Tests for short-term memory."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from backend.agent.memory.short_term import ShortTermMemory


class TestShortTermMemory:
    def test_summarize_empty(self):
        assert ShortTermMemory.summarize_messages([]) == ""

    def test_summarize_user_and_assistant(self):
        messages = [
            HumanMessage(content="My name is Amar"),
            AIMessage(content="Nice to meet you, Amar."),
        ]
        summary = ShortTermMemory.summarize_messages(messages)
        assert "My name is Amar" in summary
        assert "Nice to meet you" in summary

    def test_summarize_with_tool_calls(self):
        messages = [
            HumanMessage(content="What is 25 * 50?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculator",
                        "args": {"expression": "25 * 50"},
                        "id": "call_1",
                    }
                ],
            ),
            ToolMessage(content="1250", tool_call_id="call_1"),
            AIMessage(content="25 * 50 = 1250"),
        ]
        summary = ShortTermMemory.summarize_messages(messages)
        assert "25 * 50" in summary
        assert "1250" in summary

    def test_summarize_only_messages(self):
        """Only HumanMessage and AIMessage should appear in summary."""
        messages = [
            SystemMessage(content="system prompt"),
            HumanMessage(content="hello"),
        ]
        summary = ShortTermMemory.summarize_messages(messages)
        assert "hello" in summary
