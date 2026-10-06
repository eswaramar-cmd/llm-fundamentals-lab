"""Long-term memory — persistent user facts stored in ChromaDB.

Unlike short-term memory (which holds conversation history for one
session), long-term memory stores **useful, persistent facts** about
a user across sessions.

Examples:
    User: "My name is Amar."       → stored as preferred_name = Amar
    User: "I prefer Python."       → stored as preferred_language = Python
    User: "My budget is $500."     → stored as budget = 500

Storage backend: ChromaDB (``chroma_memory`` directory), with
HuggingFace embeddings for vector search.  The abstraction is clean
so the backend can be swapped later (e.g. to PostgreSQL or Redis).
"""

from __future__ import annotations

import datetime as _dt
import logging

from langchain_chroma import Chroma
from langchain_core.tools import tool as lc_tool
from pydantic import BaseModel, Field

from backend.agent.config import get_settings
from backend.agent.embeddings import get_embeddings

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Singleton Chroma client for long-term memory
# ------------------------------------------------------------------

_collection: Chroma | None = None


def _get_embeddings():
    """Return cached embeddings from the factory."""
    return get_embeddings()


def _get_collection() -> Chroma:
    """Return (and lazily create) the long-term memory Chroma collection."""
    global _collection
    if _collection is None:
        settings = get_settings()
        _collection = Chroma(
            collection_name=settings.chroma_memory_collection,
            embedding_function=_get_embeddings(),
            persist_directory=settings.chroma_memory_path,
        )
    return _collection


# ------------------------------------------------------------------
# Extract facts from conversation text using a lightweight LLM call.
# ------------------------------------------------------------------

def _extract_facts(text: str, llm) -> list[dict]:
    """Ask the LLM to extract persistent facts from the given text.

    Returns a list of ``{"key": ..., "value": ..., "category": ...}``.
    Falls back to an empty list if the LLM call fails.
    """
    if not text.strip():
        return []

    prompt = (
        "Extract any persistent facts, preferences, or user-provided "
        "information from the following conversation text. "
        "Return ONLY a JSON list of objects with 'key', 'value', 'category' fields. "
        "If nothing is worth remembering, return an empty list.\n\n"
        f"Text:\n{text}\n\n"
        "JSON:"
    )

    try:
        raw = llm.invoke(prompt)
        content = raw.content if hasattr(raw, "content") else str(raw)
        # Parse the JSON response
        import json
        facts = json.loads(content)
        if isinstance(facts, list):
            return [f for f in facts if isinstance(f, dict) and f.get("key") and f.get("value")]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Fact extraction failed: %s", exc)

    return []


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

class LongTermMemory:
    """Persistent user-memory backed by ChromaDB."""

    def __init__(self) -> None:
        self._collection = _get_collection()

    def save(self, user_id: str, key: str, value: str, category: str = "preference") -> str:
        """Store a fact for a user. Returns the memory ID."""
        memory_id = f"user:{user_id}:{key}"
        timestamp = _dt.datetime.now(_dt.timezone.utc).isoformat()

        self._collection.add_texts(
            texts=[f"{key}: {value}"],
            ids=[memory_id],
            metadatas=[{
                "user_id": user_id,
                "key": key,
                "value": value,
                "category": category,
                "created_at": timestamp,
            }],
        )
        logger.info("Saved memory: user=%s key=%s", user_id, key)
        return memory_id

    def retrieve(self, user_id: str, query: str = "", k: int = 5) -> list[dict]:
        """Retrieve relevant facts for a user."""
        try:
            results = self._collection.similarity_search(
                query or "user preferences",
                k=k,
                filter={"user_id": user_id},
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Long-term memory retrieval failed: %s", exc)
            return []

        facts = []
        for doc in results:
            meta = doc.metadata
            facts.append({
                "key": meta.get("key", ""),
                "value": meta.get("value", ""),
                "category": meta.get("category", "preference"),
                "created_at": meta.get("created_at", ""),
            })
        return facts

    def retrieve_all(self, user_id: str) -> list[dict]:
        """Retrieve ALL facts for a user (not vector-based)."""
        try:
            data = self._collection._collection.get(
                where={"user_id": user_id},
                include=["metadatas", "documents"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Long-term memory list failed: %s", exc)
            return []

        facts = []
        for meta in data.get("metadatas", []):
            facts.append({
                "key": meta.get("key", ""),
                "value": meta.get("value", ""),
                "category": meta.get("category", "preference"),
                "created_at": meta.get("created_at", ""),
            })
        return facts

    def save_conversation_facts(self, user_id: str, text: str, llm) -> int:
        """Extract and store facts from conversation text. Returns count saved."""
        facts = _extract_facts(text, llm)
        count = 0
        for fact in facts:
            try:
                self.save(user_id, fact["key"], str(fact["value"]), fact.get("category", "preference"))
                count += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to save fact %s: %s", fact, exc)
        return count


# Lazy singleton — created on first access, not at import time.
_long_term: LongTermMemory | None = None


def get_long_term_memory() -> LongTermMemory:
    """Return the singleton LongTermMemory instance (lazily created)."""
    global _long_term
    if _long_term is None:
        _long_term = LongTermMemory()
    return _long_term


# ------------------------------------------------------------------
# Tool: save_memory
# ------------------------------------------------------------------

class SaveMemoryInput(BaseModel):
    """Input for the save_memory tool."""
    user_id: str = Field(..., description="User identifier")
    key: str = Field(..., description="Fact key (e.g. 'preferred_language')", min_length=1, max_length=128)
    value: str = Field(..., description="Fact value", min_length=1, max_length=4096)
    category: str = Field(default="preference", description="Category of fact")


def save_memory_tool():
    """Create the save_memory LangChain tool."""

    def _save(user_id: str, key: str, value: str, category: str = "preference") -> str:
        try:
            mem = get_long_term_memory()
            mem_id = mem.save(user_id, key, value, category)
            return f"Saved memory '{key}' = '{value}' (id: {mem_id})"
        except Exception as exc:  # noqa: BLE001
            return f"[save_memory error: {type(exc).__name__}: {exc}]"

    return lc_tool(
        "save_memory",
        description=(
            "Store a persistent fact or preference for the user. "
            "Use this when the user explicitly tells you something "
            "worth remembering (name, preference, settings, etc.). "
            "Input: user_id, key, value, and optional category."
        ),
        args_schema=SaveMemoryInput,
    )(_save)


# Module-level tool instance
save_memory = save_memory_tool()
