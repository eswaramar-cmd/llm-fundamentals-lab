"""Graph nodes: answer generation and HITL approval."""

from __future__ import annotations

import logging
import re

from langgraph.types import interrupt

from backend.agent.state import AgentState

logger = logging.getLogger(__name__)


# Checked against the *request*, never the generated answer. An earlier version
# scanned the answer too, and plain substring matching made "format" inside
# "information" trip the interrupt, which silently emptied the response.
_DESTRUCTIVE_PATTERN = re.compile(
    r"\b(?:delete|destroy|erase|wipe|shutdown)\b"
    r"|rm\s+-rf"
    r"|drop\s+(?:table|database)"
    r"|format\s+(?:disk|drive)",
    re.IGNORECASE,
)

# "How do I delete a file in Python?" asks about deletion; it does not perform
# it. Informational questions are never treated as approval-worthy.
_INFORMATIONAL_PATTERN = re.compile(
    r"^\s*(?:how|what|why|when|where|who|which|explain|describe|define|"
    r"is|are|does|do|can|could|should|would|will)\b",
    re.IGNORECASE,
)

# Tool activity that actually performed something destructive.
_TOOL_DESTRUCTIVE_PATTERN = re.compile(
    "|".join(
        (
            r"rm\s+-rf",
            r"drop\s+(?:table|database)",
            r"deleted?\s+\d+",
            r"destroyed",
            r"wiped",
            r"shutdown",
        )
    ),
    re.IGNORECASE,
)


def _needs_approval(question: str, tool_results: list) -> bool:
    """True when a destructive action is actually being requested or performed."""

    text = question or ""

    if not _INFORMATIONAL_PATTERN.match(text) and _DESTRUCTIVE_PATTERN.search(text):
        return True

    combined = ""

    for result in tool_results:
        combined += f" {getattr(result, 'result', '')}"

    return bool(_TOOL_DESTRUCTIVE_PATTERN.search(combined))


def generate_answer(state: AgentState) -> AgentState:
    """Node: finalize the answer."""

    messages = state.get("messages", [])
    answer = state.get("answer", "")

    if not answer and messages:
        last_msg = messages[-1]
        if hasattr(last_msg, "content"):
            answer = last_msg.content or ""

    if not answer:
        answer = "I'm not sure how to answer that. Could you rephrase?"

    logger.info("Generated answer (%d chars)", len(answer))

    return {
        "answer": answer,
        "error": None,
    }


def check_approval(state: AgentState) -> AgentState:
    """Node: pause for human approval when a sensitive action is requested."""

    question = state.get("question", "")
    answer = state.get("answer", "")
    tool_results = state.get("tool_results", [])
    user_requested = state.get("approval_required", False)

    needs_approval = user_requested or _needs_approval(question, tool_results)

    if needs_approval:
        logger.info("Approval required - pausing graph")

        approval = interrupt(
            {
                "type": "approval_required",
                "question": question,
                "reason": "Human approval required before proceeding.",
                "current_answer": answer,
            }
        )

        # ---------------------------------
        # APPROVED
        # ---------------------------------
        if isinstance(approval, dict) and approval.get("approved") is True:
            logger.info("Human approval received")

            # Only the keys this node owns are returned. Spreading the whole
            # state back would re-add every message, because `messages` uses an
            # `add` reducer, and the duplicated history would be resent to the
            # model on the next turn.
            return {
                "approved": True,
                "approval_required": False,
                "answer": (
                    "Approval received. However, no file-deletion "
                    "tool is configured, so no files were deleted."
                ),
                "error": None,
            }

        # ---------------------------------
        # REJECTED
        # ---------------------------------
        logger.info("Human approval rejected")

        return {
            "approved": False,
            "approval_required": False,
            "answer": "Operation cancelled by the user.",
            "error": None,
        }

    # ---------------------------------
    # NO APPROVAL REQUIRED
    # ---------------------------------
    return {
        "approval_required": False,
        "approved": True,
    }