"""Knowledge-base search tool with configurable vector store.

Supports multiple backends via ``VECTOR_STORE_PROVIDER``:

    Development (default):
        VECTOR_STORE_PROVIDER=chroma_local
        CHROMA_RAG_PATH=/chroma_rag

    Production (shared):
        VECTOR_STORE_PROVIDER=chroma_server
        VECTOR_STORE_URL=http://chroma:8000

All replicas must query the *same* production vector store to avoid
per-container data silos.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_chroma import Chroma
from langchain_core.tools import tool as lc_tool
from pydantic import BaseModel, Field

from backend.agent.config import get_settings
from backend.agent.embeddings import get_embeddings

logger = logging.getLogger(__name__)


class KnowledgeBaseInput(BaseModel):
    question: str = Field(
        ...,
        description="The user's question to search for in the knowledge base.",
        min_length=1,
        max_length=4096,
    )
    k: int = Field(default=2, description="Number of relevant documents to retrieve.", ge=1, le=20)


# ------------------------------------------------------------------
# Vector-store factory
# ------------------------------------------------------------------

_vectorstore: Chroma | None = None


def _get_chroma_local(settings) -> Chroma:
    """Local ChromaDB with persistent storage (development default)."""
    return Chroma(
        collection_name=settings.chroma_rag_collection,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_rag_path,
    )


def _get_chroma_server(settings) -> Chroma:
    """Remote ChromaDB server (production)."""
    from chromadb import HttpClient

    client = HttpClient(
        host=settings.vector_store_url.split("//")[-1].split("/")[0].split(":")[0],
        port=int(
            settings.vector_store_url.split(":")[-1].split("/")[0]
            if ":" in settings.vector_store_url
            else "8000"
        ),
    )
    return Chroma(
        client=client,
        collection_name=settings.chroma_rag_collection,
        embedding_function=get_embeddings(),
    )


def _get_qdrant(settings) -> Any:
    """Qdrant vector store (production alternative).

    Return type is Any because langchain_qdrant is an optional dependency and
    is imported lazily below; annotating it directly would make the annotation
    unresolvable on installs that do not include it.
    """
    from langchain_qdrant import Qdrant

    return Qdrant.from_existing_collection(
        embedding=get_embeddings(),
        collection_name=settings.chroma_rag_collection,
        url=settings.vector_store_url,
    )


def _get_weaviate(settings) -> Any:
    """Weaviate vector store (production alternative).

    See _get_qdrant for why the return type is Any.
    """
    from langchain_weaviate import WeaviateVectorStore

    return WeaviateVectorStore.from_existing_index(
        embedding=get_embeddings(),
        index_name=settings.chroma_rag_collection,
        text_key="text",
        url=settings.vector_store_url,
    )


_VECTOR_STORE_FACTORIES = {
    "chroma_local": _get_chroma_local,
    "chroma_server": _get_chroma_server,
    "qdrant": _get_qdrant,
    "weaviate": _get_weaviate,
}


def _get_vectorstore() -> Chroma | "Qdrant | WeaviateVectorStore":  # noqa: F821
    """Return a cached vector store based on configuration.

    Development: local Chroma on a shared Docker volume.
    Production: remote Chroma server / Qdrant / Weaviate.
    """
    global _vectorstore
    if _vectorstore is None:
        settings = get_settings()
        provider = settings.vector_store_provider
        factory = _VECTOR_STORE_FACTORIES.get(provider)
        if factory is None:
            logger.warning("Unknown vector store provider '%s', falling back to chroma_local", provider)
            factory = _VECTOR_STORE_FACTORIES["chroma_local"]
        _vectorstore = factory(settings)
        logger.info("Vector store provider: %s", provider)
    return _vectorstore


# ------------------------------------------------------------------
# Tool
# ------------------------------------------------------------------

def search_knowledge_base(question: str, k: int = 2) -> str:
    """Search the vector database for relevant documents."""
    settings = get_settings()
    try:
        vs = _get_vectorstore()
        docs = vs.similarity_search(question, k=k)
    except Exception as exc:  # noqa: BLE001
        logger.error("Vector store search failed: %s", exc)
        return f"[knowledge_base_search error: {type(exc).__name__}]"

    if not docs:
        return "[knowledge_base_search: No relevant documents found in the knowledge base.]"

    parts = []
    limit = settings.rag_chunk_char_limit
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown") if doc.metadata else "unknown"
        chunk = doc.metadata.get("chunk", "?") if doc.metadata else "?"
        body = doc.page_content.strip()
        if limit > 0 and len(body) > limit:
            body = body[:limit] + " …[truncated]"
        parts.append(
            f"--- Document {i} (source={source}, chunk={chunk}) ---\n"
            f"{body}"
        )

    return "\n\n".join(parts)


def knowledge_base_search_tool():
    return lc_tool(
        "knowledge_base_search",
        description=(
            "Search the personal knowledge base for relevant documents. "
            "Use this when the user asks questions about their uploaded PDFs, documents, "
            "or any content stored in the knowledge base. "
            "Input: a 'question' string and optional 'k' (1-20) for number of results."
        ),
        args_schema=KnowledgeBaseInput,
    )(search_knowledge_base)


knowledge_base_search = knowledge_base_search_tool()

