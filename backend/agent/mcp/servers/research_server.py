"""Research MCP Server — exposes calculator, knowledge base search, and web search tools."""

from __future__ import annotations

import logging
from fastmcp import FastMCP

from backend.agent.tools.calculator import calculate
from backend.agent.tools.knowledge_base import search_knowledge_base
from backend.agent.tools.web_search import perform_web_search

logger = logging.getLogger(__name__)

# Initialize FastMCP Research Server
research_mcp = FastMCP("ResearchMCPServer")


@research_mcp.tool(
    name="calculator",
    description=(
        "Evaluate a safe arithmetic expression and return the numeric result. "
        "Supports addition, subtraction, multiplication, division, modulo, "
        "power, and parentheses. Input must be a string like '25 * 50 + 100'. "
        "Do not use this for non-arithmetic tasks."
    ),
)
def calculator(expression: str) -> str:
    """Evaluate a safe arithmetic expression.

    Args:
        expression: Arithmetic expression to evaluate, e.g. '25 * 16'.
    """
    try:
        result = calculate(expression)
        return str(result)
    except Exception as exc:
        logger.warning("Calculator tool error: %s", exc)
        return f"Error: {exc}"


@research_mcp.tool(
    name="knowledge_base_search",
    description=(
        "Search the personal knowledge base for relevant documents. "
        "Use this when the user asks questions about their uploaded PDFs, documents, "
        "or any content stored in the knowledge base. "
        "Input: a 'question' string and optional 'k' (1-20) for number of results."
    ),
)
def knowledge_base_search(question: str, k: int = 2) -> str:
    """Search documents in the knowledge base.

    Args:
        question: Question or search query to look up in the vector store.
        k: Number of relevant chunks to retrieve (default 2, min 1, max 20).
    """
    try:
        return search_knowledge_base(question=question, k=k)
    except Exception as exc:
        logger.error("Knowledge base search tool error: %s", exc)
        return f"Error searching knowledge base: {exc}"


@research_mcp.tool(
    name="web_search",
    description=(
        "Search the web for current information about a topic. "
        "Use this when the user asks about recent events, current "
        "facts, or anything not in the local knowledge base. "
        "Input: 'query' (search string) and optional 'num_results' (1-10)."
    ),
)
def web_search(query: str, num_results: int = 5) -> str:
    """Search the web for recent information.

    Args:
        query: Search keywords or query string.
        num_results: Number of search results to retrieve (default 5, max 10).
    """
    try:
        return perform_web_search(query=query, num_results=num_results)
    except Exception as exc:
        logger.error("Web search tool error: %s", exc)
        return f"Error performing web search: {exc}"


if __name__ == "__main__":
    research_mcp.run()

