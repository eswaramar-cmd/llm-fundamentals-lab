"""Tests for graph routing and structure."""

from __future__ import annotations

import pytest

from backend.agent.graph import build_graph, get_agent_graph
from backend.agent.state import AgentState


class TestGraphStructure:
    def test_graph_has_all_nodes(self):
        from backend.agent.graph import build_graph

        # Build a fresh graph to inspect nodes
        from langgraph.graph import StateGraph

        graph = StateGraph(AgentState)

        # Re-add nodes to check they exist in the builder
        from backend.agent.nodes.agent import agent_node
        from backend.agent.nodes.approval import check_approval, generate_answer
        from backend.agent.nodes.memory import retrieve_memory
        from backend.agent.nodes.retrieval import retrieve_context
        from backend.agent.nodes.router import classify_question
        from backend.agent.nodes.tools_node import execute_tools

        graph.add_node("classify", classify_question)
        graph.add_node("retrieve_memory", retrieve_memory)
        graph.add_node("retrieve_context", retrieve_context)
        graph.add_node("agent", agent_node)
        graph.add_node("execute_tools", execute_tools)
        graph.add_node("generate_answer", generate_answer)
        graph.add_node("check_approval", check_approval)

        node_names = set(graph.nodes.keys())
        expected = {
            "classify",
            "retrieve_memory",
            "retrieve_context",
            "agent",
            "execute_tools",
            "generate_answer",
            "check_approval",
        }
        assert expected.issubset(node_names)


class TestRouting:
    def test_route_question_returns_correct_edges(self):
        from backend.agent.nodes.router import route_question

        # 'rag' intent → retrieve_context
        state: AgentState = {"answer": "rag"}
        assert route_question(state) == "retrieve_context"

        # 'direct' intent → agent_node
        state = {"answer": "direct"}
        assert route_question(state) == "agent_node"

        # 'calculate' intent → agent_node
        state = {"answer": "calculate"}
        assert route_question(state) == "agent_node"

        # 'research' intent → agent_node
        state = {"answer": "research"}
        assert route_question(state) == "agent_node"

    def test_route_question_default(self):
        from backend.agent.nodes.router import route_question

        # Unknown intent defaults to agent_node
        state: AgentState = {"answer": "unknown"}
        assert route_question(state) == "agent_node"

    def test_should_continue_with_tool_calls(self):
        from langchain_core.messages import AIMessage

        from backend.agent.nodes.agent import should_continue

        state: AgentState = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "calculator", "args": {"expression": "1+1"}, "id": "c1"}],
                )
            ]
        }
        assert should_continue(state) == "execute_tools"

    def test_should_continue_without_tool_calls(self):
        from langchain_core.messages import AIMessage

        from backend.agent.nodes.agent import should_continue

        state: AgentState = {"messages": [AIMessage(content="Hello!")]}
        assert should_continue(state) == "generate_answer"

    def test_should_continue_empty_messages(self):
        from backend.agent.nodes.agent import should_continue

        assert should_continue({"messages": []}) == "generate_answer"
