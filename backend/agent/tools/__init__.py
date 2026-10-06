"""Tool registry — discover and list all available agent tools via MCP Client."""

from __future__ import annotations

from typing import Any
from langchain_core.tools import BaseTool

from backend.agent.memory.long_term import save_memory
from backend.agent.tools.calculator import calculator
from backend.agent.tools.gmail_send import send_email
from backend.agent.tools.investigate_incident import investigate_incident
from backend.agent.tools.knowledge_base import knowledge_base_search
from backend.agent.tools.web_search import web_search

# Direct tools fallback mapping
_DIRECT_TOOLS: dict[str, BaseTool] = {
    "calculator": calculator,
    "knowledge_base_search": knowledge_base_search,
    "web_search": web_search,
    "save_memory": save_memory,
    "send_email": send_email,
    "investigate_incident": investigate_incident,
}


def get_all_tools() -> dict[str, BaseTool]:
    """Discover and return all registered tools across MCP servers."""
    try:
        from backend.agent.mcp.client import get_mcp_client
        return get_mcp_client().discover_tools()
    except Exception:
        return _DIRECT_TOOLS


class _LazyToolsDict(dict):
    """Dictionary that delegates to MCP Client while preserving dictionary interface."""

    def __getitem__(self, key: str) -> BaseTool:
        try:
            from backend.agent.mcp.client import get_mcp_client
            tools = get_mcp_client().discover_tools()
            if key in tools:
                return tools[key]
        except Exception:
            pass
        return _DIRECT_TOOLS[key]

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: object) -> bool:
        return key in get_all_tools()

    def __iter__(self):
        return iter(get_all_tools())

    def __len__(self):
        return len(get_all_tools())

    def items(self):
        return get_all_tools().items()

    def keys(self):
        return get_all_tools().keys()

    def values(self):
        return get_all_tools().values()


# Registry of all tools by name — dynamic MCP discovery with zero circular imports
ALL_TOOLS: dict[str, BaseTool] = _LazyToolsDict(_DIRECT_TOOLS)

# Default tool set used by the agent (excluding send_email which requires explicit email intent)
DEFAULT_TOOLS: list[BaseTool] = [
    knowledge_base_search,
    calculator,
    web_search,
    save_memory,
    investigate_incident,
]

__all__ = [
    "ALL_TOOLS",
    "DEFAULT_TOOLS",
    "get_all_tools",
    "calculator",
    "knowledge_base_search",
    "web_search",
    "save_memory",
    "send_email",
    "investigate_incident",
]
