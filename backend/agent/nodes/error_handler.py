"""Graph nodes: error handling and retry management."""

from __future__ import annotations

import logging

from backend.agent.config import get_settings
from backend.agent.state import AgentState

logger = logging.getLogger(__name__)


def handle_error(state: AgentState) -> AgentState:
    """Node: centralised error handler.

    Called when the graph encounters an error. Increments retry_count
    and returns a clean error message to the user.
    """
    error = state.get("error", "")
    retry_count = state.get("retry_count", 0)
    max_retries = get_settings().max_retries

    if retry_count < max_retries:
        retry_count += 1
        logger.warning("Retrying after error (attempt %d): %s", retry_count, error)
        return {
            "error": error,
            "retry_count": retry_count,
            "answer": "Retrying...",
        }

    logger.error("Max retries exceeded: %s", error)
    return {
        "error": error,
        "retry_count": retry_count,
        "answer": "I encountered an error and could not complete the task. Please try again.",
    }
