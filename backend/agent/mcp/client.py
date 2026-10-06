"""MCP Client Adapter — connects LangGraph Agent to MCP Servers.

Discovers tools across Research, Incident, and Gmail MCP servers and wraps
them into LangChain-compatible BaseTool instances for LangGraph.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import inspect
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from langchain_core.tools import BaseTool, StructuredTool

from backend.agent.mcp.servers.gmail_server import gmail_mcp
from backend.agent.mcp.servers.incident_server import incident_mcp
from backend.agent.mcp.servers.research_server import research_mcp
from backend.agent.memory.long_term import save_memory

logger = logging.getLogger(__name__)


def _extract_text(result: Any) -> str:
    """Safely extract plain text output from MCP tool execution result."""
    if hasattr(result, "content") and result.content:
        first = result.content[0]
        return first.text if hasattr(first, "text") else str(first)
    if isinstance(result, list) and result:
        first = result[0]
        return first.text if hasattr(first, "text") else str(first)
    return str(result)


def _run_async_in_thread(coro: Any) -> Any:
    """Safely execute an async coroutine from synchronous context without loop conflicts."""
    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        return asyncio.run(coro)
    except Exception as exc:
        logger.error("Error executing async coroutine in thread: %s", exc)
        raise


class LangGraphMCPClient:
    """MCP Client that manages connections to FastMCP servers and exposes tools to LangGraph."""

    def __init__(self) -> None:
        self.servers: Dict[str, Any] = {
            "research": research_mcp,
            "incident": incident_mcp,
            "gmail": gmail_mcp,
        }
        self._tool_to_server: Dict[str, str] = {}
        self._langchain_tools: Dict[str, BaseTool] = {}
        self._discovered: bool = False

    def discover_tools(self) -> Dict[str, BaseTool]:
        """Discover tools from all MCP servers and wrap them as LangChain tools."""
        if self._discovered and self._langchain_tools:
            return self._langchain_tools

        for server_name, server in self.servers.items():
            try:
                mcp_tools = _run_async_in_thread(server.list_tools())
                for mcp_tool in mcp_tools:
                    tool_name = mcp_tool.name
                    self._tool_to_server[tool_name] = server_name
                    lc_tool_instance = self._create_langchain_wrapper(server_name, server, tool_name, mcp_tool)
                    self._langchain_tools[tool_name] = lc_tool_instance
            except Exception as exc:
                logger.error("Failed discovering tools from MCP server %s: %s", server_name, exc)

        # Include local persistent memory tool
        self._langchain_tools["save_memory"] = save_memory
        self._discovered = True
        logger.info("LangGraph MCP Client discovered %d tools across %d servers", len(self._langchain_tools), len(self.servers))
        return self._langchain_tools

    def _create_langchain_wrapper(self, server_name: str, server: Any, tool_name: str, mcp_tool: Any) -> BaseTool:
        """Create a callable LangChain BaseTool from an MCP tool."""
        description = getattr(mcp_tool, "description", f"MCP tool {tool_name} from {server_name} server")
        fn = getattr(mcp_tool, "fn", None)

        if fn is not None:
            @functools.wraps(fn)
            def _wrapped(*args: Any, **kwargs: Any) -> str:
                try:
                    res = fn(*args, **kwargs)
                    if inspect.iscoroutine(res):
                        res = _run_async_in_thread(res)
                    return _extract_text(res)
                except Exception as exc:
                    logger.error("Error executing MCP tool '%s' on '%s': %s", tool_name, server_name, exc)
                    return f"Error executing {tool_name}: {exc}"

            return StructuredTool.from_function(
                func=_wrapped,
                name=tool_name,
                description=description,
            )

        def _fallback_call(**kwargs: Any) -> str:
            try:
                call_coro = server.call_tool(tool_name, kwargs)
                res = _run_async_in_thread(call_coro)
                return _extract_text(res)
            except Exception as exc:
                logger.error("Error executing MCP fallback for '%s': %s", tool_name, exc)
                return f"Error executing {tool_name}: {exc}"

        return StructuredTool.from_function(
            func=_fallback_call,
            name=tool_name,
            description=description,
        )

    def call_tool_sync(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call a tool through the MCP client by name."""
        tools = self.discover_tools()
        if tool_name not in tools:
            raise KeyError(f"Tool '{tool_name}' not found in MCP registry")
        tool = tools[tool_name]
        return tool.invoke(arguments)

    def get_tools_for_intent(self, intent: str) -> List[BaseTool]:
        """Return subset of tools appropriate for the given classified intent."""
        tools = self.discover_tools()
        
        intent_mapping = {
            "calculate": ["calculator"],
            "rag": ["knowledge_base_search"],
            "research": ["web_search"],
            "incident": ["investigate_incident"],
            "email": ["send_email"],
            "direct": [],
            "sensitive": [],
        }

        names = intent_mapping.get(intent)
        if names is None:
            return [t for k, t in tools.items() if k != "send_email"]

        return [tools[n] for n in names if n in tools]


# Singleton MCP Client instance for application runtime
_mcp_client: Optional[LangGraphMCPClient] = None


def get_mcp_client() -> LangGraphMCPClient:
    """Get or initialize the global LangGraph MCP Client."""
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = LangGraphMCPClient()
        _mcp_client.discover_tools()
    return _mcp_client
