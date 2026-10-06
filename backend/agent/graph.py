"""LangGraph state graph assembly.

Flow:

    START
      |
      v
    classify_question
      |
      v
    retrieve_memory
      |
      v
    route_question
      |
      +--------------------+
      |                    |
      v                    v
   sensitive          retrieve_context
      |                    |
      v                    v
 check_approval           agent
      |                    |
      |                    v
      |              should_continue
      |                /          \
      |               /            \
      |              v              v
      |       execute_tools    generate_answer
      |              |              |
      |              v              v
      |            agent       check_approval
      |                             |
      +-----------------------------+
                                    |
                                   END
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from backend.agent.checkpoint import get_checkpoint_saver
from backend.agent.nodes.agent import agent_node, should_continue
from backend.agent.nodes.approval import check_approval, generate_answer
from backend.agent.nodes.error_handler import handle_error
from backend.agent.nodes.memory import retrieve_memory
from backend.agent.nodes.retrieval import retrieve_context
from backend.agent.nodes.router import classify_question, route_question
from backend.agent.nodes.tools_node import execute_tools
from backend.agent.state import AgentState

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """Build and compile the AI Research & Knowledge Agent graph."""

    graph = StateGraph(AgentState)

    # ---------------------------------------------------------
    # Nodes
    # ---------------------------------------------------------

    graph.add_node("classify", classify_question)
    graph.add_node("retrieve_memory", retrieve_memory)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("agent", agent_node)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("check_approval", check_approval)
    graph.add_node("handle_error", handle_error)

    # ---------------------------------------------------------
    # Initial flow
    # ---------------------------------------------------------

    graph.add_edge(START, "classify")

    graph.add_edge(
        "classify",
        "retrieve_memory",
    )

    # ---------------------------------------------------------
    # Route after classification
    # ---------------------------------------------------------

    graph.add_conditional_edges(
    "retrieve_memory",
    route_question,
    {
        "retrieve_context": "retrieve_context",
        "agent_node": "agent",
        "check_approval": "check_approval",
    },
)
    # ---------------------------------------------------------
    # RAG → Agent
    # ---------------------------------------------------------

    graph.add_edge(
        "retrieve_context",
        "agent",
    )

    # ---------------------------------------------------------
    # Agent → Tools OR Answer
    # ---------------------------------------------------------

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "execute_tools": "execute_tools",
            "generate_answer": "generate_answer",
        },
    )

    # ---------------------------------------------------------
    # Tool → Agent loop
    # ---------------------------------------------------------

    graph.add_edge(
        "execute_tools",
        "agent",
    )

    # ---------------------------------------------------------
    # Final answer → HITL
    # ---------------------------------------------------------

    graph.add_edge(
        "generate_answer",
        "check_approval",
    )

    # ---------------------------------------------------------
    # HITL → END
    # ---------------------------------------------------------

    graph.add_edge(
        "check_approval",
        END,
    )

    # ---------------------------------------------------------
    # Compile with shared checkpointing (Redis or MemorySaver)
    # ---------------------------------------------------------

    compiled = graph.compile(
        checkpointer=get_checkpoint_saver(),
        name="ResearchAgent",
    )

    logger.info(
        "LangGraph compiled with %d nodes",
        len(graph.nodes),
    )

    return compiled


_agent_graph: StateGraph | None = None


def get_agent_graph() -> StateGraph:
    """Return the compiled LangGraph singleton."""

    global _agent_graph

    if _agent_graph is None:
        _agent_graph = build_graph()

    return _agent_graph