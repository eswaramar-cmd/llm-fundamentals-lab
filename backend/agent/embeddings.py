"""Embedding model factory — supports Ollama and HuggingFace backends.

Created because HuggingFace embeddings can be slow to load in some
environments.  Ollama-based embeddings are the default and are fast
since Ollama is already running.

Usage:
    from backend.agent.embeddings import get_embeddings
    emb = get_embeddings()
    vs = Chroma(embedding_function=emb, ...)
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.embeddings import Embeddings

from backend.agent.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Return a cached embedding model based on configuration.

    Default: OllamaEmbeddings with nomic-embed-text (768-dim).
    Alternative: HuggingFaceEmbeddings with a sentence-transformers model.
    """
    settings = get_settings()

    if settings.embedding_provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info("Using HuggingFace embeddings: %s", settings.embedding_model)
        return HuggingFaceEmbeddings(model_name=settings.embedding_model)

    # Default: Ollama
    from langchain_ollama import OllamaEmbeddings

    logger.info("Using Ollama embeddings: %s", settings.embedding_model)
    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )
