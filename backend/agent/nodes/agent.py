"""Graph nodes: agent (LLM with tools — core decision engine)."""

from __future__ import annotations

import asyncio
import logging
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.agent.config import get_settings
from backend.agent.llm import models_for_intent
from backend.agent.metrics import get_metrics
from backend.agent.state import AgentState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = SystemMessage(
    "You are an AI Research & Knowledge Agent.\n\n"
    "Tools:\n"
    "  - knowledge_base_search: search the user's uploaded documents\n"
    "  - calculator: evaluate arithmetic\n"
    "  - web_search: search the web for current information\n"
    "  - save_memory: remember a durable fact about the user\n"
    "  - send_email: send real email through Gmail\n"
    "  - investigate_incident: investigate production incidents, service latency, failures, and errors using real logs and metrics\n\n"
    "Rules:\n"
    "1. Use knowledge_base_search for questions about the user's documents.\n"
    "2. Use calculator for arithmetic.\n"
    "3. Use web_search for current or recent information.\n"
    "4. Use investigate_incident when asked to investigate production incidents, slow services, failing requests, latency spikes, or errors.\n"
    "5. Answer directly when no tool is needed.\n"
    "6. Be concise and accurate. Never invent document contents or hallucinate telemetry.\n\n"
    "Email (strict two-step, never skip):\n"
    "6. To send email you need all three: recipient address, subject, body.\n"
    "   If any is missing, ask ONLY for the missing one. Do not guess an address.\n"
    "7. First call send_email once with confirmed=false. Its arguments are\n"
    "   to_email, subject and body - use exactly those names.\n"
    "8. The reply will begin 'PREVIEW ONLY'. Show the user the To / Subject /\n"
    "   Body from that reply and ask them to confirm. Do NOT call send_email\n"
    "   again until the user replies; calling it repeatedly just wastes time.\n"
    "9. Only after the user explicitly says yes, call send_email again with\n"
    "   confirmed=true. Never set confirmed=true on the first call.\n"
    "10. Never invent a recipient address, and never send on your own initiative.\n"
    "11. After sending, report the real result returned by the tool.\n\n"
    "OUTPUT FORMAT - always follow this, never reply as plain unstructured text:\n"
    "- Write every response in Markdown.\n"
    "- Use **bold** for key terms, `inline code` for values, file names, or commands.\n"
    "- Use ## headings to separate major sections when the answer has multiple parts.\n"
    "- Use bullet lists ( - item ) for enumerations of 3 or more items.\n"
    "- Use numbered lists ( 1. step ) for sequential steps or ranked items.\n"
    "- Use triple-backtick fenced code blocks with a language tag for ALL code, commands, and JSON.\n"
    "- Use Markdown tables for structured comparisons or multi-column data.\n"
    "- Use > blockquotes for direct quotes from a source document.\n"
    "- For short answers (1-2 sentences) no headings needed - keep it tight.\n\n"
    "ACCURACY RULES - critical, never violate these:\n"
    "- Only state facts you retrieved from a tool result or that the user told you.\n"
    "- If a tool returned no result, say so explicitly; do NOT fill gaps with guesses.\n"
    "- When quoting a document, cite it inline: According to [document name].\n"
    "- When using web_search results, cite the source URL after the relevant fact.\n"
    "- If you are unsure, say I do not know rather than speculating.\n"
    "- Never fabricate statistics, names, dates, prices, or URLs."
)

# Only the most recent turns are replayed to the model. Older context costs
# prompt tokens on every call without changing the answer.
MAX_HISTORY_MESSAGES = 6

# Cap on tool calls issued for a single answer. Without a cap the model can loop
# on retrieval, and each call is a full round trip.
MAX_TOOL_CALLS = 2


def _build_messages(state: AgentState, messages: list) -> list:
    """Build the complete message list for the LLM call.

    Only the last ``MAX_HISTORY_MESSAGES`` conversation turns are sent. Older
    turns are almost never relevant to the current question and every extra
    token is paid for twice: once on the way in, and again on the way out.
    """

    result: list = [_SYSTEM_PROMPT]

    memory_context = state.get("memory_context", "")
    if memory_context:
        result.append(
            SystemMessage(f"Context from memory:\n{memory_context}")
        )

    retrieved = state.get("retrieved_documents", [])
    if retrieved:
        context_text = "\n\n".join(
            f"[Doc {i + 1}] {d.content}"
            for i, d in enumerate(retrieved)
        )

        result.append(
            SystemMessage(
                f"Relevant documents from the knowledge base:\n\n{context_text}"
            )
        )

    result.extend(messages[-MAX_HISTORY_MESSAGES:])

    return result

def _response_text(response) -> str:
    """Flatten a model response to plain text.

    Content is a string on Ollama and Groq but a list of blocks on Gemini, so
    this normalises both shapes.
    """
    content = getattr(response, "content", None)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )

    return ""


def _is_validation_error(exc: Exception) -> bool:
    """Check whether an exception is a validation/schema error (not retryable)."""

    name = type(exc).__name__
    if "Validation" in name or "Schema" in name or "BadRequest" in name:
        return True
    return False


def _chunk_text(chunk) -> str:
    """Pull plain text out of a streamed chunk.

    Gemini yields content as a list of blocks and Ollama/Groq as a string, so
    both shapes are normalised here.
    """

    content = getattr(chunk, "content", None)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )

    return ""


def _stream_with_timeout(model, messages: list, seconds: float, partial: list[str]):
    """Deprecated shim kept so imports do not break; the node is now async."""

    raise RuntimeError("use _astream_with_timeout")


async def _astream_with_timeout(model, messages: list, seconds: float, partial: list[str]):
    """Stream the model to completion under a hard wall-clock limit.

    Returns ``(message, streamed_chars)``. ``partial`` is filled with the text
    produced so far and stays useful even when the call raises, so the caller
    can tell a failure that happened before any output from one that happened
    mid-answer.

    Streaming rather than ``invoke`` is what makes the UI feel live. ``invoke``
    blocks until the provider has produced the entire answer, so every token
    event arrives in one burst at the end and the user watches a spinner for
    the whole generation.

    This is an async node on purpose. Draining the stream on a worker thread
    with a copied context loses the LangChain callback handler, and
    ``on_chat_model_stream`` events then never reach ``astream_events`` — the
    endpoint silently falls back to buffering the whole answer. Awaiting
    ``astream`` keeps the callback config on the event loop, where it belongs.
    """

    async def _call():
        accumulated = None
        streamed_chars = 0

        async for chunk in model.astream(messages):
            text = _chunk_text(chunk)

            if text:
                partial.append(text)

            streamed_chars += len(text)
            accumulated = chunk if accumulated is None else accumulated + chunk

        if accumulated is None:
            raise RuntimeError("Model stream produced no chunks")

        message = AIMessage(
            content=accumulated.content,
            tool_calls=list(getattr(accumulated, "tool_calls", []) or []),
        )

        return message, streamed_chars

    try:
        return await asyncio.wait_for(_call(), timeout=seconds)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"LLM call exceeded {seconds:.0f}s") from exc


async def agent_node(state: AgentState) -> AgentState:
    """Node: stream the LLM with tools, rotating providers and retrying."""

    question = state.get("question", "")
    messages = list(state.get("messages", []))
    settings = get_settings()
    metrics = get_metrics()

    if not messages and question:
        messages.append(HumanMessage(content=question))

    # classify_question stores the intent in state["answer"] and route_question
    # reads it from there, so it is still the intent when this node runs. It is
    # what selects the tool slice below, and this node overwrites "answer" with
    # the real response further down.
    intent = state.get("answer", "")

    llm_with_tools = models_for_intent(intent)
    full_messages = _build_messages(state, messages)

    logger.info(
        "Agent node: intent=%s providers=%s",
        intent or "unclassified",
        [name for name, _ in llm_with_tools],
    )

    last_error: Exception | None = None
    last_streamed_chars = 0
    partial_answer = ""
    attempt_partial: list[str] = []

    for attempt in range(1, settings.max_retries + 1):
        start = time.monotonic()

        # Rotate provider on each retry. A cloud endpoint returning 503 is a
        # property of that endpoint, so retrying it unchanged just burns the
        # backoff; the next candidate is usually a working one.
        provider_name, model = llm_with_tools[
            (attempt - 1) % len(llm_with_tools)
        ]

        try:
            logger.info(
                "LLM attempt %d/%d via %s",
                attempt,
                settings.max_retries,
                provider_name,
            )

            attempt_partial = []

            response, streamed_chars = await _astream_with_timeout(
                model,
                full_messages,
                settings.tool_timeout_seconds,
                attempt_partial,
            )
            metrics.record_llm(settings.llm_model, time.monotonic() - start)
            last_streamed_chars = streamed_chars

            if isinstance(response, AIMessage) and response.tool_calls:
                tool_calls = response.tool_calls[:MAX_TOOL_CALLS]

                if len(tool_calls) < len(response.tool_calls):
                    logger.info(
                        "Trimmed %d tool call(s) down to %d",
                        len(response.tool_calls),
                        MAX_TOOL_CALLS,
                    )

                logger.info(
                    "LLM requested %d tool call(s)",
                    len(tool_calls),
                )

                trimmed = response.model_copy(
                    update={"tool_calls": tool_calls}
                )

                return {
                    "messages": [trimmed],
                    "answer": "",
                    "retry_count": 0,
                    "error": None,
                }

            # Gemini returns content as a list of blocks, not a string, so
            # reading .content directly would put a list in the answer.
            answer = _response_text(response)

            logger.info(
                "LLM generated final answer (%d chars)",
                len(answer),
            )

            return {
                "messages": [response],
                "answer": answer,
                "retry_count": 0,
                "error": None,
            }

        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - start
            metrics.record_llm(settings.llm_model, elapsed)
            last_error = exc

            partial_answer = "".join(attempt_partial)
            last_streamed_chars = len(partial_answer)

            # Validation errors are not transient - do not retry
            if _is_validation_error(exc):
                logger.warning(
                    "LLM validation error (not retrying): %s", exc
                )
                break

            # Once tokens have reached the client, retrying would append a
            # second competing copy of the answer to the same bubble. Report the
            # partial result instead.
            if last_streamed_chars > 0:
                logger.warning(
                    "LLM failed after streaming %d chars; not retrying: %s",
                    last_streamed_chars,
                    exc,
                )
                break

            logger.warning(
                "LLM attempt %d/%d via %s failed: %s",
                attempt,
                settings.max_retries,
                provider_name,
                exc,
            )

            if attempt < settings.max_retries:
                delay = settings.retry_backoff ** (attempt - 1)
                logger.info("Retrying in %s second(s)...", delay)
                await asyncio.sleep(delay)

    logger.error(
        "LLM failed after %d attempts: %s",
        settings.max_retries,
        last_error,
    )

    reason = str(last_error) if last_error else "unknown error"

    if last_streamed_chars > 0:
        # Keep what the user already has on screen and explain the gap.
        return {
            "error": f"Generation interrupted: {reason}",
            "answer": partial_answer,
            "retry_count": settings.max_retries,
        }

    return {
        "error": f"LLM failed after {settings.max_retries} attempts: {reason}",
        "answer": (
            "The model provider is unavailable right now, so I could not "
            f"generate an answer. ({reason[:160]})"
        ),
        "retry_count": settings.max_retries,
    }


def should_continue(state: AgentState) -> str:
    """Conditional edge: decide whether to execute tools or finish."""

    messages = state.get("messages", [])

    if not messages:
        return "generate_answer"

    last_msg = messages[-1]

    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "execute_tools"

    return "generate_answer"
