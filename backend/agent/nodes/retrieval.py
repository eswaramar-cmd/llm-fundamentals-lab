"""Graph nodes: context retrieval (pre-fetch RAG documents)."""

from __future__ import annotations

import logging
import time

from backend.agent.metrics import get_metrics
from backend.agent.tools.knowledge_base import _get_vectorstore
from backend.agent.state import AgentState, RetrievedDocument

logger = logging.getLogger(__name__)


def retrieve_context(state: AgentState) -> AgentState:
    """Node: search the knowledge base for documents relevant to the question.

    Results are stored in ``state['retrieved_documents']`` and also
    formatted into a string that gets appended to the conversation.
    """
    question = state.get("question", "")
    if not question:
        return {"retrieved_documents": []}

    metrics = get_metrics()
    start = time.monotonic()

    try:
        vs = _get_vectorstore()
        docs = vs.similarity_search(question, k=4)
        metrics.record_rag(time.monotonic() - start)
    except Exception as exc:  # noqa: BLE001
        metrics.record_rag(time.monotonic() - start)
        logger.error("RAG retrieval failed: %s", exc)
        return {
            "retrieved_documents": [],
            "error": f"RAG retrieval failed: {type(exc).__name__}",
        }

    retrieved = []
    for doc in docs:
        retrieved.append(
            RetrievedDocument(
                content=doc.page_content,
                source=doc.metadata.get("source", "unknown") if doc.metadata else "unknown",
                score=None,
            )
        )

    logger.info("Retrieved %d documents for question", len(retrieved))

    return {"retrieved_documents": retrieved}
