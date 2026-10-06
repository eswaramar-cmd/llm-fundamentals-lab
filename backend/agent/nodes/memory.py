"""Graph nodes: memory retrieval."""

from __future__ import annotations

import logging

from backend.agent.memory.long_term import get_long_term_memory
from backend.agent.memory.short_term import ShortTermMemory
from backend.agent.state import AgentState

logger = logging.getLogger(__name__)


def retrieve_memory(state: AgentState) -> AgentState:
    """Node: load short-term conversation summary + long-term user facts.

    Results are stored in ``state['memory_context']`` as a single
    string the LLM can read.
    """
    user_id = state.get("user_id", "default_user")
    messages = state.get("messages", [])

    # Deliberately left empty. Stored facts are few per user, so relevance
    # ranking is not worth an embedding call on a CPU-only box.
    query = ""

    parts: list[str] = []

    # --- Short-term: conversation history summary ---
    # Skipped when the only message is the current question. On a fresh turn
    # there is no history to summarise, and summarising it produced
    # "User: <the question>", which was then injected as a second copy of the
    # question — pure prompt-token cost on every request.
    if len(messages) >= 2:
        summary = ShortTermMemory.summarize_messages(messages[:-1])
        if summary:
            parts.append("=== Conversation History ===")
            parts.append(summary)

    # --- Long-term: persistent user facts ---
    try:
        mem = get_long_term_memory()

        # No query means "everything for this user", which is a metadata filter.
        # similarity_search would embed the query first, and embeddings are the
        # slowest thing on a CPU-only box, so that path is reserved for the
        # cases that actually need relevance ranking.
        facts = mem.retrieve_all(user_id) if not query else mem.retrieve(
            user_id, query=query, k=5
        )

        if facts:
            parts.append("")
            parts.append("=== What you know about this user ===")
            for f in facts:
                parts.append(f"{f['key']} = {f['value']}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Long-term memory retrieval failed: %s", exc)

    memory_context = "\n".join(parts) if parts else ""

    logger.info("Memory context length: %d chars", len(memory_context))

    return {"memory_context": memory_context}
