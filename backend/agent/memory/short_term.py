"""Short-term (session) memory.

LangGraph's ``MemorySaver`` checkpoint store already keeps the full
conversation state (messages) in memory keyed by ``thread_id``.
This module provides a thin wrapper so the rest of the code does not
couple directly to LangGraph internals.

Short-term memory = conversation history for one session/thread.
It is ephemeral — if the process restarts, only what was persisted
to disk (long-term memory) survives.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """Wrapper around LangGraph's ``MemorySaver`` checkpoint store.

    This class does not store messages itself — the LangGraph graph
    state already carries ``messages``.  Its responsibilities are:

    * Provide the ``MemorySaver`` instance to the graph compiler.
    * Offer helpers to extract a readable conversation summary.
    """

    def __init__(self) -> None:
        self._saver = MemorySaver()

    @property
    def saver(self) -> MemorySaver:
        """Return the underlying LangGraph MemorySaver."""
        return self._saver

    @staticmethod
    def summarize_messages(messages: list) -> str:
        """Return a human-readable summary of the conversation so far.

        Used to populate ``state['memory_context']`` before the
        generation node runs.
        """
        if not messages:
            return ""

        parts: list[str] = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                parts.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                content = msg.content or ""
                if msg.tool_calls:
                    calls = "; ".join(
                        f"{c.get('name', '?')}({c.get('args', {})})"
                        for c in msg.tool_calls
                    )
                    parts.append(f"Assistant: [tool calls: {calls}]")
                else:
                    parts.append(f"Assistant: {content}")

        return "\n".join(parts)


# Singleton instance
_short_term = ShortTermMemory()


def get_short_term_memory() -> ShortTermMemory:
    """Return the singleton ShortTermMemory instance."""
    return _short_term
