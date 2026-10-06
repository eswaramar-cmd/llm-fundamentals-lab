"""Graph nodes: router (question classification)."""

from __future__ import annotations

import logging
import re

from backend.agent.config import get_settings
from backend.agent.llm import get_llm
from backend.agent.state import AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Email intent
# ---------------------------------------------------------

# Both an explicit send verb and a recipient are required. Matching a bare "send"
# would capture "send me the retrieval pipeline", which is a document question.
_EMAIL_PATTERNS = [
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    r"\b(?:send|email|e-mail|mail|messag\w*)\b[^.?]*@\w+\.\w+",
    r"\b(?:send|email|e-mail)\b[^.?]*\b(?:to|@)\b",
    r"\b(?:draft|compose|write)\b[^.?]*\b(?:email|e-mail|mail|message)\b",
    r"\b(?:send\s+(?:the\s+)?(?:email|mail|message))\b",
]

# The send flow is two turns: the first asks for a preview, the second approves
# it. The approving turn is almost always a bare "yes", which carries no email
# vocabulary at all. Classified on its own wording, that reply became `direct`,
# which binds no tools, so the approval could never actually trigger the send
# and the conversation dead-ended with the model claiming it had no way to mail
# anyone. These patterns recognise the approval itself.
_CONFIRM_PATTERNS = [
    # The whole turn is nothing but an assent.
    r"^\s*(?:yes|yeah|yep|yup|ya|y|sure|ok|okay|k|confirm|confirmed|agreed)\b[\s.!?]*$",
    # Or an explicit instruction to go ahead with the send.
    r"^\s*(?:please\s+)?(?:go\s+ahead|do\s+it|proceed|send\s+(?:it|now|them|email|the\s+email)(?:\s+now)?)\b[\s.!?]*$",
]

# Marker the send tool puts in the preview ToolMessage it returns.
_EMAIL_PREVIEW_MARKER = "PREVIEW ONLY"


def _looks_like_confirmation(question: str) -> bool:
    """Whether this turn is approving the previous one.

    Deliberately narrow. Anything loose enough to catch "yes, and also explain
    retrieval" would drag unrelated questions back into the email path.
    """

    text = (question or "").strip().lower()

    if not text or len(text) > 60:
        return False

    return any(re.search(pattern, text) for pattern in _CONFIRM_PATTERNS)


def _has_pending_email_preview(state: AgentState) -> bool:
    """Whether the thread is sitting on an unapproved email preview."""

    for message in state.get("messages") or []:
        content = getattr(message, "content", None)

        if isinstance(content, str) and _EMAIL_PREVIEW_MARKER in content:
            return True

        if isinstance(content, list):
            for block in content:
                text = block.get("text", "") if isinstance(block, dict) else ""
                if _EMAIL_PREVIEW_MARKER in text:
                    return True

    return False


# ---------------------------------------------------------
# Sensitive / dangerous action patterns
# ---------------------------------------------------------

_SENSITIVE_PATTERNS = [
    r"\bdelete\b",
    r"\bdestroy\b",
    r"\bdrop\b",
    r"\bremove\b",
    r"\brm\s+-rf\b",
    r"\bformat\b",
    r"\bshutdown\b",
    r"\brestart\b",
    r"\bkill\b",
]


# ---------------------------------------------------------
# RAG patterns
# ---------------------------------------------------------

_RAG_PATTERNS = [
    r"my (pdf|document|file|files|notes)",
    r"(in|from|in my) (the |my )?(documents|knowledge base|pdf|notes|files)",
    r"(what|explain|tell me) (is |are )?(written|mentioned|described) (in|in my)",
    r"(read|search) (my |the )?(pdf|document|file)",
    r"apex care|knowledge base",
]


# ---------------------------------------------------------
# Calculator patterns
# ---------------------------------------------------------

_CALCULATE_PATTERNS = [
    r"\d+\s*[\*\+\-\/x×]\s*\d+",
    r"what is\s+\d+",
    r"(calculate|compute|math)",
    r"(multiply|divide|add|subtract|minus|plus|times)",
    r"\d+\s*%\s*\d+",
    r"square root|cube root|power|exponent",
]


# ---------------------------------------------------------
# Research patterns
# ---------------------------------------------------------

_RESEARCH_PATTERNS = [
    r"latest",
    r"recent",
    r"current (state|status|news)",
    r"what (is|are) .*(today|now)",
    r"trends?",
    r"breaking news",
    r"202(?:\d|5|6)",
    r"this year",
]

# ---------------------------------------------------------
# Production Incident / Reliability patterns
# ---------------------------------------------------------

_INCIDENT_PATTERNS = [
    r"\b(?:investigate|incident|postmortem|triage|outage|downtime|root\s*cause)\b",
    r"\b(?:why\s+is\s+my\s+service\s+(?:slow|down|failing|erroring|crashing|degraded))\b",
    r"\b(?:why\s+are\s+requests?\s+(?:slow|failing|dropping|timing\s*out|rejected|failing))\b",
    r"\b(?:what\s+caused\s+(?:the\s+)?(?:latency|error|spike|incident|failure|timeout|outage))\b",
    r"\b(?:service\s+(?:down|degraded|slow|unhealthy|erroring))\b",
    r"\b(?:latency\s+(?:spike|increase|jump|high|issue|alert))\b",
    r"\b(?:error\s+rate\s+(?:spike|high|jump|increase))\b",
]


def _keyword_classify(question: str) -> str | None:
    """Try to classify using keywords."""

    q = question.lower()

    # Email is checked before sensitive on purpose. A request like "email bob
    # saying delete the old report" contains a sensitive keyword, and routing it
    # to the approval interrupt would mean the send tool is never bound at all.
    # Sending is already gated by an explicit confirmation inside the tool, so
    # classifying it as email loses nothing.
    for pattern in _EMAIL_PATTERNS:
        if re.search(pattern, q):
            return "email"

    # Sensitive actions MUST be checked first among the remaining intents.
    for pattern in _SENSITIVE_PATTERNS:
        if re.search(pattern, q):
            return "sensitive"

    # Calculator
    for pattern in _CALCULATE_PATTERNS:
        if re.search(pattern, q):
            return "calculate"

    # RAG
    for pattern in _RAG_PATTERNS:
        if re.search(pattern, q):
            return "rag"

    # Web research
    for pattern in _RESEARCH_PATTERNS:
        if re.search(pattern, q):
            return "research"

    # Incident investigation
    for pattern in _INCIDENT_PATTERNS:
        if re.search(pattern, q):
            return "incident"

    # Greetings
    greetings = [
        "hello",
        "hi ",
        "hey",
        "how are you",
        "what's up",
        "good morning",
        "good evening",
        "good afternoon",
        "thank",
        "thanks",
        "ok",
        "okay",
        "cool",
    ]

    for greeting in greetings:
        if q.startswith(greeting):
            return "direct"

    # Very short questions
    if len(q) < 25 and not any(
        keyword in q
        for keyword in ["what", "why", "how", "is", "are", "can"]
    ):
        return "direct"

    return None


def classify_question(state: AgentState) -> AgentState:
    """Classify the user's question."""

    question = state.get("question", "")

    if not question:
        return {
            "error": "Empty question provided"
        }

    # Fast keyword classification
    intent = _keyword_classify(question)

    # An approval for a preview on screen has no email vocabulary of its own.
    # Carry the email intent over from the pending preview, otherwise the very
    # reply that authorises the send is the reply that loses the tool.
    if (
        intent is None
        and _looks_like_confirmation(question)
        and _has_pending_email_preview(state)
    ):
        logger.info("Confirmation turn inherits pending email intent")
        intent = "email"

    # LLM fallback
    if intent is None:
        if not get_settings().classify_llm_fallback:
            logger.info(
                "No keyword match and LLM fallback disabled — defaulting to direct"
            )
            intent = "direct"
        else:
            llm = get_llm()

            try:
                response = llm.invoke(
                    f"Classify this question. Reply with only ONE word:\n"
                    f"'direct', 'rag', 'calculate', 'research', 'email', or 'incident'.\n"
                    f"Use 'email' only when the user wants a message actually sent "
                    f"to an address. Use 'incident' when investigating failures, errors, "
                    f"or latency. Use 'direct' for ordinary conversation.\n"
                    f"Question: {question}"
                )

                intent = response.content.strip().lower()

            except Exception as exc:
                logger.warning(
                    "Classification failed, defaulting to direct: %s",
                    exc,
                )

                intent = "direct"

    # Valid intents
    #
    # `email` and `incident` must be listed here.
    valid_intents = {
        "direct",
        "rag",
        "calculate",
        "research",
        "sensitive",
        "email",
        "incident",
    }

    if intent not in valid_intents:
        if intent is not None:
            logger.warning(
                "Classifier returned unknown intent %r — defaulting to direct",
                intent,
            )
        intent = "direct"

    logger.info(
        "Question classified as: %s",
        intent,
    )

    return {
        "answer": intent
    }


def route_question(state: AgentState) -> str:
    """Route the question based on classification."""

    intent = state.get("answer", "direct")

    # RAG
    if intent == "rag":
        return "retrieve_context"

    # Sensitive action
    if intent == "sensitive":
        return "check_approval"

    # Everything else
    return "agent_node"