"""Tests for tool calling (LLM → tools → LLM loop)."""

from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from backend.agent.nodes.tools_node import execute_tools


class TestToolCalling:
    """Test the agent's tool calling loop."""

    def test_tool_call_and_result_added(self):
        """Verify that tool results are properly formatted."""
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calculator",
                            "args": {"expression": "25 * 50"},
                            "id": "call_calc_1",
                        }
                    ],
                )
            ],
        }

        result = execute_tools(state)

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], ToolMessage)
        assert "1250" in result["messages"][0].content
        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0].tool_name == "calculator"

    def test_tool_execution_unknown_tool(self):
        from backend.agent.nodes.tools_node import execute_tools

        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "nonexistent_tool",
                            "args": {},
                            "id": "call_unknown_1",
                        }
                    ],
                )
            ],
        }

        result = execute_tools(state)

        assert result["messages"][0].content == "[Error: Tool 'nonexistent_tool' is not available.]"
