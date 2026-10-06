"""Parallel task execution.

Splits one user request into independent subtasks, runs them concurrently, and
merges the answers deterministically.

Design goals, in priority order:

1. **No LLM call for planning or merging.** Both are done with rules. An LLM
   planner would add a full round trip before any real work starts, which on a
   CPU-only box costs more than the parallelism ever saves.
2. **One LLM call per subtask.** The LangGraph path costs two or three
   sequential calls (decide tool -> execute tool -> answer). Each subtask here
   does retrieve-then-answer in a single call.
3. **Fail soft.** One failing subtask must never sink the whole request; its
   error is reported inline and the other lanes still deliver.

The concurrency here overlaps retrieval, embedding and (when the provider can
actually serve concurrent requests) inference. On a GPU or a cloud provider that
is a real wall-clock win. On a single CPU core Ollama serialises the
generation itself, so the win there comes from doing fewer, shorter calls --
not from the parallelism.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from backend.agent.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Subtask
# ---------------------------------------------------------


@dataclass
class Subtask:
    """One independently executable slice of the user's request."""

    id: int
    text: str
    status: str = "pending"  # pending | running | done | failed | skipped
    answer: str = ""
    error: str = ""
    sources: int = 0
    duration_ms: int = 0
    started_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "status": self.status,
            "answer": self.answer,
            "error": self.error,
            "sources": self.sources,
            "duration_ms": self.duration_ms,
        }


# ---------------------------------------------------------
# Splitting (pure rules, no LLM)
# ---------------------------------------------------------

_QUESTION_SPLIT = re.compile(r"(?<=[?.!])\s+(?=[A-Z0-9])")

_LIST_ITEM = re.compile(
    r"^\s*(?:\d+[.)]|[-*•]|step\s+\d+)\s+",
    re.IGNORECASE,
)

_THEN_SPLIT = re.compile(
    r"\s*(?:,\s*)?\b(?:and\s+then|then\s+also|after\s+that|followed\s+by)\b\s*",
    re.IGNORECASE,
)

_SEMICOLON = re.compile(r"\s*;\s+")

_MIN_PART_LEN = 6
_MAX_PART_LEN = 300

_INLINE_MARKER = re.compile(r"\s+(?:\d+[.)]|[-*•])\s+(?=[A-Z0-9])")

_CONNECTIVES = re.compile(
    r"^(?:and|also|additionally|furthermore|moreover)\b\s*",
    re.IGNORECASE,
)


def _clean(part: str) -> str:
    part = _CONNECTIVES.sub("", part.strip())
    part = _LIST_ITEM.sub("", part).strip()

    if not part:
        return ""

    # Keep a trailing question mark so the answering prompt stays natural.
    if part[-1] not in ".?!":
        part += "?"

    return part


def split_task(text: str) -> list[Subtask]:
    """Split a request into subtasks using structure, never an LLM.

    Recognises three shapes, most specific first:

    * explicit lists -- ``1. do x 2. do y`` or ``- do x``
    * multiple sentences that each ask something
    * ``and then`` / ``;`` chains

    Anything that cannot be split cleanly is returned as a single subtask, so
    this is always safe to call.
    """

    settings = get_settings()
    limit = settings.parallel_max_subtasks

    raw = text.strip()

    if not raw:
        return []

    candidates: list[str] = []

    lines = [ln for ln in raw.splitlines() if ln.strip()]

    # 1. Explicit multi-line list.
    if len(lines) > 1 and sum(bool(_LIST_ITEM.match(ln)) for ln in lines) >= 2:
        candidates = [_clean(ln) for ln in lines]

    # 1b. Inline numbered/bulleted list on a single line.
    if not candidates:
        markers = re.findall(r"(?:\d+[.)]|[-*•])\s+(?=[A-Z0-9])", raw)
        if len(markers) >= 2:
            candidates = [p.strip() for p in _INLINE_MARKER.split(raw) if p.strip()]

    # 2. Semicolon or "and then" chain.
    if not candidates:
        parts = [p for p in _SEMICOLON.split(raw) if p.strip()]
        if len(parts) > 1:
            candidates = [p.strip() for p in parts]
        else:
            parts = [p for p in _THEN_SPLIT.split(raw) if p.strip()]
            if len(parts) > 1:
                candidates = [p.strip() for p in parts]

    # 3. Several independent questions in one message.
    if not candidates:
        parts = [p for p in _QUESTION_SPLIT.split(raw) if p.strip()]
        if len(parts) > 1:
            candidates = [p.strip() for p in parts]

    candidates = [c for c in candidates if c]

    if not candidates:
        return [Subtask(id=1, text=raw[:_MAX_PART_LEN])]

    # Carry each part's offset through the pipeline. Cleaning rewrites the
    # text, so re-finding it in ``raw`` later would fail and fall back to an
    # arbitrary order.
    located = [(max(raw.find(c), 0), c) for c in candidates]

    # Keep the largest N parts -- never silently drop the tail.
    if len(located) > limit:
        keep = sorted(range(len(located)), key=lambda i: len(located[i][1]), reverse=True)[:limit]
        located = [located[i] for i in keep]

    located.sort(key=lambda item: item[0])

    located = _merge_short(located)

    parts = [_clean(text) for _, text in located]
    parts = [p for p in parts if p]

    if not parts:
        return [Subtask(id=1, text=raw[:_MAX_PART_LEN])]

    return [Subtask(id=i + 1, text=p) for i, p in enumerate(parts)]


def _merge_short(located: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Fold too-short fragments into the preceding part, keeping offsets."""

    merged: list[list] = []

    for pos, part in located:
        part = part.strip()

        if not part:
            continue

        if merged and len(part) < _MIN_PART_LEN:
            merged[-1][1] = f"{merged[-1][1]} {part}".strip()
            continue

        merged.append([pos, part])

    # A leading fragment with nothing to fold into still has to survive.
    if len(merged) > 1 and len(merged[0][1]) < _MIN_PART_LEN:
        merged[1][1] = f"{merged[0][1]} {merged[1][1]}".strip()
        merged = merged[1:]

    return [(pos, text) for pos, text in merged]


# ---------------------------------------------------------
# Execution
# ---------------------------------------------------------


_ANSWER_SYSTEM = (
    "You are a fast, precise assistant. Answer the user's question using the "
    "provided context when it is relevant. Be direct and concise: at most three "
    "sentences. If the context does not answer it, say so plainly instead of "
    "guessing."
)


def _retrieve(question: str, k: int, phase=None) -> tuple[str, int]:
    """Blocking retrieval. Returns formatted context and hit count.

    ``phase`` is called before and after each internal stage so the caller can
    surface progress. Retrieval is several seconds of opaque work -- embedding,
    then the vector search -- and without these callbacks the UI has nothing to
    show during it.
    """

    from backend.agent.tools.knowledge_base import _get_vectorstore

    settings = get_settings()

    def report(stage: str, status: str, detail: str = "") -> None:
        if phase is not None:
            phase(stage, status, detail)

    try:
        report("load index", "running", "opening the vector store")

        store = _get_vectorstore()

        report("load index", "done", "vector store ready")
        report("embed query", "running", settings.embedding_model)

        docs = store.similarity_search(question, k=k)

        report("embed query", "done", f"{len(docs)} candidate(s)")
        report("select context", "running", f"taking top {k}")

    except Exception as exc:  # noqa: BLE001
        logger.warning("Parallel retrieval failed: %s", exc)
        report("embed query", "failed", str(exc)[:120])
        return "", 0

    if not docs:
        report("select context", "done", "no matches")
        return "", 0

    limit = settings.rag_chunk_char_limit

    parts = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown") if doc.metadata else "unknown"
        body = doc.page_content.strip()

        if limit > 0 and len(body) > limit:
            body = body[:limit] + " …[truncated]"

        parts.append(f"[{i}] (source={source}) {body}")

    joined = "\n\n".join(parts)

    report(
        "select context",
        "done",
        f"{len(docs)} chunk(s), {len(joined)} chars",
    )

    return joined, len(docs)


# Ordered so the UI can show the full plan before the first phase completes.
_PHASE_PLAN = ("load index", "embed query", "select context", "build prompt", "generate")


def _answer_sync(subtask: "Subtask", on_delta, on_phase=None) -> tuple[str, int]:
    """Blocking retrieve + streamed answer for one lane.

    ``on_delta`` is invoked for every chunk so the caller can relay tokens while
    generation is still running, instead of waiting for the full answer.
    ``on_phase`` reports internal stage transitions.
    """

    settings = get_settings()

    def report(stage: str, status: str, detail: str = "") -> None:
        if on_phase is not None:
            on_phase(stage, status, detail)

    context, hits = _retrieve(subtask.text, settings.parallel_retrieve_k, phase=on_phase)

    report("build prompt", "running", "assembling messages")

    messages = [SystemMessage(content=_ANSWER_SYSTEM)]

    if context:
        messages.append(
            SystemMessage(content=f"Context:\n\n{context}")
        )

    messages.append(HumanMessage(content=subtask.text))

    chars = sum(len(getattr(m, "content", "") or "") for m in messages)

    report(
        "build prompt",
        "done",
        f"{len(messages)} messages, ~{chars // 4} tokens",
    )

    from backend.agent.llm import get_llm_capped

    model = settings.parallel_llm_model or settings.llm_model

    report("generate", "running", model)

    llm = get_llm_capped()

    parts: list[str] = []

    for chunk in llm.stream(messages):
        text = getattr(chunk, "content", "") or ""

        if not text:
            continue

        parts.append(text)

        try:
            on_delta(text)
        except Exception:  # noqa: BLE001
            pass

    answer = "".join(parts).strip()

    report("generate", "done", f"{len(parts)} chunk(s)")

    return answer, hits


async def run_subtask(
    subtask: "Subtask",
    on_delta=None,
    on_phase=None,
) -> "Subtask":
    """Execute one lane off the event loop so SSE frames keep flowing."""

    subtask.status = "running"
    subtask.started_at = time.monotonic()

    loop = asyncio.get_running_loop()

    def relay(text: str) -> None:
        if on_delta is not None:
            loop.call_soon_threadsafe(on_delta, subtask.id, text)

    def phase(stage: str, status: str, detail: str) -> None:
        if on_phase is not None:
            loop.call_soon_threadsafe(on_phase, subtask.id, stage, status, detail)

    try:
        answer, hits = await asyncio.to_thread(_answer_sync, subtask, relay, phase)

        if not answer:
            subtask.status = "skipped"
            subtask.error = "empty answer"
        else:
            subtask.status = "done"
            subtask.answer = answer
            subtask.sources = hits

    except Exception as exc:  # noqa: BLE001
        logger.exception("Lane %d failed", subtask.id)
        subtask.status = "failed"
        subtask.error = f"{type(exc).__name__}: {exc}"

    subtask.duration_ms = int((time.monotonic() - subtask.started_at) * 1000)

    return subtask


async def run_all(subtasks: list["Subtask"]) -> list["Subtask"]:
    """Run every lane concurrently, discarding streamed deltas."""

    if not subtasks:
        return []

    return list(await asyncio.gather(*(run_subtask(s) for s in subtasks)))


# ---------------------------------------------------------
# Merging (deterministic, no LLM call)
# ---------------------------------------------------------


def merge(subtasks: list[Subtask]) -> str:
    """Combine subtask answers into one response.

    Deliberately not an LLM call: a synthesis pass is the most expensive part
    of the chain and buys nothing when every lane already answered in prose.
    """

    done = [s for s in subtasks if s.status == "done" and s.answer]
    failed = [s for s in subtasks if s.status in {"failed", "skipped"}]

    if not done:
        return ""

    tail = ""

    if failed:
        reasons = "; ".join(f"part {s.id} ({s.status})" for s in failed)
        tail = (
            f"\n\n---\n\n_Incomplete: {reasons}. "
            "Re-ask those parts individually for full detail._"
        )

    # Only unwrap when the request really was fully answered. A single
    # surviving lane alongside failures still has to disclose the gap.
    if len(done) == 1 and not failed:
        return done[0].answer

    blocks = [f"**{s.id}. {s.text}**\n\n{s.answer}" for s in done]

    return "\n\n".join(blocks) + tail
