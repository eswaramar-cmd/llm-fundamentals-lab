"""Graph nodes: tool execution with timeout and retry support."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextvars import copy_context

from langchain_core.messages import ToolMessage

from backend.agent.config import get_settings
from backend.agent.metrics import get_metrics
from backend.agent.state import AgentState, ToolResult

logger = logging.getLogger(__name__)

# Map of tool name -> tool object for execution
_TOOL_MAP: dict | None = None


def _get_tool_map() -> dict:
    """Lazily build the tool name -> tool object mapping."""
    global _TOOL_MAP
    if _TOOL_MAP is None:
        from backend.agent.tools import ALL_TOOLS

        _TOOL_MAP = dict(ALL_TOOLS)
    return _TOOL_MAP


def _invoke_with_timeout(tool, args: dict, timeout: int) -> str:
    """Invoke a tool with a hard timeout.

    Uses a thread pool so it works on all platforms (including Windows).
    Returns the tool's content string or raises the original exception.

    The context is copied explicitly because LangChain delivers run callbacks
    through ``contextvars``. Without it the tool run starts in an empty
    context, so ``astream_events`` never sees ``on_tool_start``/``on_tool_end``
    and the client shows a blank timeline.
    """

    def _call():
        return tool.invoke(args)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(copy_context().run, _call)
        result = future.result(timeout=timeout)

    if hasattr(result, "content"):
        return result.content
    return str(result)


def _is_transient(exc: Exception) -> bool:
    """Return True if the error is transient and worth retrying."""
    name = type(exc).__name__
    # Network / connection errors are transient
    transient_names = {
        "ConnectionError",
        "TimeoutError",
        "httpx.ConnectError",
        "httpx.ReadTimeout",
        "httpx.WriteTimeout",
        "urllib3.exceptions.MaxRetryError",
        "requests.exceptions.ConnectionError",
        "requests.exceptions.Timeout",
        "requests.exceptions.ReadTimeout",
    }
    if name in transient_names:
        return True
    # Check by message for common patterns
    msg = str(exc).lower()
    if "timeout" in msg or "connection" in msg or "network" in msg:
        return True
    return False


def execute_tools(state: AgentState) -> AgentState:
    """Node: execute tool calls requested by the LLM.

    Reads tool_calls from the last message in ``state['messages']``,
    executes each tool with timeout and retry, and appends ``ToolMessage``
    results.
    """
    messages = state.get("messages", [])
    if not messages:
        return {"error": "No messages to extract tool calls from"}

    last_msg = messages[-1]
    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return {"error": "No tool calls to execute"}

    settings = get_settings()
    tool_map = _get_tool_map()
    metrics = get_metrics()

    tool_results: list[ToolResult] = []
    new_messages: list = []

    # Some providers repeat a tool call verbatim several times inside one
    # assistant message. Executing those duplicates is not just wasted work:
    # for a side-effecting tool it means the same email delivered N times, and
    # it fills the UI timeline with identical rows. Collapse byte-identical
    # (name, args) pairs and keep the first occurrence, whose id is the one the
    # model will match the ToolMessage against.
    seen_calls: set[tuple[str, str]] = set()
    deduped: list = []
    dropped: list = []

    for tool_call in last_msg.tool_calls:
        try:
            fingerprint = (
                tool_call.get("name", ""),
                json.dumps(
                    tool_call.get("args", {}),
                    sort_keys=True,
                    default=str,
                ),
            )
        except (TypeError, ValueError):
            fingerprint = None

        if fingerprint is not None:
            if fingerprint in seen_calls:
                dropped.append(tool_call)
                continue
            seen_calls.add(fingerprint)

        deduped.append(tool_call)

    if dropped:
        logger.info(
            "Collapsed %d duplicate tool call(s) out of %d",
            len(dropped),
            len(last_msg.tool_calls),
        )
        # The AIMessage still advertises every tool_call id it emitted, and
        # providers reject the history if any id goes unanswered ("function
        # call turn comes immediately after a user turn"). Answer the dropped
        # ids explicitly instead of leaving the conversation malformed.
        for tool_call in dropped:
            new_messages.append(
                ToolMessage(
                    content=(
                        "[Duplicate tool call collapsed: identical to an "
                        "earlier call in this turn, executed once.]"
                    ),
                    tool_call_id=tool_call.get("id", ""),
                )
            )

    for tool_call in deduped:
        name = tool_call.get("name", "")
        args = tool_call.get("args", {})
        call_id = tool_call.get("id", "")

        tool = tool_map.get(name)

        if tool is None:
            logger.warning("Unknown tool requested: %s", name)
            result_text = f"[Error: Tool '{name}' is not available.]"
            tool_results.append(
                ToolResult(
                    tool_name=name,
                    result=result_text,
                    success=False,
                    error="tool_not_found",
                )
            )
            new_messages.append(ToolMessage(content=result_text, tool_call_id=call_id))
            continue

        # Clamp args to allowed input size
        if isinstance(args, dict):
            for k, v in args.items():
                if isinstance(v, str) and len(v) > 8192:
                    args[k] = v[:8192]

        # Retry loop with exponential backoff
        last_error: Exception | None = None
        max_attempts = settings.max_retries + 1
        result_text = ""
        success = False

        for attempt in range(1, max_attempts + 1):
            start = time.monotonic()
            try:
                result_text = _invoke_with_timeout(
                    tool, args, settings.tool_timeout_seconds
                )
                success = True
                metrics.record_tool(name, time.monotonic() - start, True)
                break

            except FutureTimeout:
                elapsed = time.monotonic() - start
                metrics.record_tool(name, elapsed, False)
                metrics.record_error("tool_timeout")
                last_error = TimeoutError(f"Tool '{name}' timed out after {settings.tool_timeout_seconds}s")
                logger.warning(
                    "Tool %s timed out (attempt %d/%d)",
                    name, attempt, max_attempts,
                )
                # Timeouts may warrant retry
                if attempt < max_attempts:
                    delay = settings.retry_backoff ** (attempt - 1)
                    time.sleep(delay)
                else:
                    result_text = f"[Tool '{name}' timed out after {settings.tool_timeout_seconds}s.]"
                    tool_results.append(
                        ToolResult(
                            tool_name=name,
                            result=result_text,
                            success=False,
                            error="timeout",
                        )
                    )
                    new_messages.append(ToolMessage(content=result_text, tool_call_id=call_id))
                    break

            except Exception as exc:  # noqa: BLE001
                elapsed = time.monotonic() - start
                metrics.record_tool(name, elapsed, False)
                last_error = exc

                # Don't retry validation errors
                error_name = type(exc).__name__
                if "ValidationError" in error_name or "ValidationError" in str(
                    exc.__class__.__mro__
                ):
                    logger.warning("Tool %s validation error (not retrying): %s", name, exc)
                    result_text = f"[Tool '{name}' received invalid arguments.]"
                    success = False
                    break

                transient = _is_transient(exc)
                logger.warning(
                    "Tool %s failed (attempt %d/%d, transient=%s): %s",
                    name, attempt, max_attempts, transient, exc,
                )

                if transient:
                    if attempt < max_attempts:
                        delay = settings.retry_backoff ** (attempt - 1)
                        time.sleep(delay)
                        continue
                    result_text = f"[Tool '{name}' failed after {max_attempts} attempts: {error_name}]"
                    success = False
                    break

                # Non-transient: the same input will fail the same way, so
                # retrying only multiplies the side effects and floods the SSE
                # timeline with one tool_start per attempt. Fail on the first.
                result_text = f"[Tool '{name}' failed: {exc}]"
                success = False
                break

        if success:
            logger.info("Tool %s returned result (%d chars)", name, len(str(result_text)))
            tool_results.append(
                ToolResult(
                    tool_name=name,
                    result=str(result_text),
                    success=True,
                    error=None,
                )
            )
            new_messages.append(ToolMessage(content=str(result_text), tool_call_id=call_id))
        else:
            if last_error and not isinstance(last_error, TimeoutError):
                error_name = type(last_error).__name__
                if not result_text:
                    result_text = f"[Tool '{name}' failed: {error_name}]"
            tool_results.append(
                ToolResult(
                    tool_name=name,
                    result=result_text,
                    success=False,
                    error=type(last_error).__name__ if last_error else "unknown",
                )
            )
            new_messages.append(ToolMessage(content=result_text, tool_call_id=call_id))

    return {
        "messages": new_messages,
        "tool_results": tool_results,
    }
