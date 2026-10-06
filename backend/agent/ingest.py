"""Document ingestion script — populate the ChromaDB knowledge base.

Usage:
    python -m backend.agent.ingest
    python -m backend.agent.ingest --file path/to/document.txt
    python -m backend.agent.ingest --clear  # wipe and re-create the collection
"""

from __future__ import annotations

import argparse
import logging
import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.agent.config import get_settings
from backend.agent.embeddings import get_embeddings
from langchain_chroma import Chroma

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_text_file(filepath: str) -> list[Document]:
    """Load a text file into Document objects."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    return [Document(page_content=content, metadata={"source": filepath})]


def load_pdf_file(filepath: str) -> list[Document]:
    """Load a PDF file using pypdf."""
    from pypdf import PdfReader

    reader = PdfReader(filepath)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            docs.append(Document(
                page_content=text,
                metadata={"source": filepath, "page": i},
            ))
    return docs


def load_documents(source_dir: str) -> list[Document]:
    """Load all supported documents from a source directory."""
    if not os.path.isdir(source_dir):
        logger.warning("Documents directory not found: %s", source_dir)
        return []

    docs: list[Document] = []
    for filename in sorted(os.listdir(source_dir)):
        filepath = os.path.join(source_dir, filename)
        if not os.path.isfile(filepath):
            continue
        ext = os.path.splitext(filename)[1].lower()
        if ext not in (".pdf", ".txt", ".md"):
            continue
        if ext == ".pdf":
            try:
                docs.extend(load_pdf_file(filepath))
                logger.info("Ingested PDF: %s", filename)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to ingest %s: %s", filename, exc)
        else:
            docs.extend(load_text_file(filepath))
            logger.info("Ingested text: %s", filename)

    return docs


def chunk_documents(docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


def create_or_replace_collection(clear: bool = True) -> Chroma:
    settings = get_settings()
    emb = get_embeddings()

    if clear:
        # Delete existing collection via client (Windows-safe)
        import chromadb
        try:
            client = chromadb.PersistentClient(path=settings.chroma_rag_path)
            client.delete_collection(settings.chroma_rag_collection)
            logger.info("Deleted old collection: %s", settings.chroma_rag_collection)
        except Exception:
            logger.info("No existing collection to delete")

    vs = Chroma(
        collection_name=settings.chroma_rag_collection,
        embedding_function=emb,
        persist_directory=settings.chroma_rag_path,
    )
    return vs


def _create_sample_documents() -> list[Document]:
    """Create sample documents for the knowledge base."""
    return [
        Document(
            page_content=(
                "Apex Care is a technology company focused on AI research and "
                "knowledge management. The company provides tools for researchers "
                "and developers to build intelligent agents. Apex Care was founded "
                "in 2024 and is headquartered in Hyderabad, India."
            ),
            metadata={"source": "sample_overview.txt", "chunk": 0},
        ),
        Document(
            page_content=(
                "The llama3.2:3b model is a 3-billion parameter language model "
                "optimized for efficiency while maintaining quality on a wide "
                "range of tasks including question answering, reasoning, and "
                "instruction following. It runs locally via Ollama."
            ),
            metadata={"source": "sample_models.txt", "chunk": 0},
        ),
        Document(
            page_content=(
                "ChromaDB is a vector database designed for AI applications. It "
                "stores embeddings and supports similarity search. ChromaDB can "
                "be used as an in-memory store or with persistent storage on disk."
            ),
            metadata={"source": "sample_tech.txt", "chunk": 0},
        ),
    ]


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    parser.add_argument("--file", type=str, default=None, help="Path to a single file")
    parser.add_argument("--clear", action="store_true", default=True, help="Clear before ingest (default: True)")
    args = parser.parse_args()

    settings = get_settings()

    if args.file:
        docs = load_documents(os.path.dirname(args.file) or ".")
    else:
        docs = load_documents(settings.documents_path)

    if not docs:
        logger.info("No documents found in documents/. Creating sample documents.")
        docs = _create_sample_documents()

    chunked = chunk_documents(docs)
    logger.info("Total chunks to ingest: %d", len(chunked))

    vs = create_or_replace_collection(clear=args.clear)
    vs.add_documents(chunked)
    count = vs._collection.count()
    logger.info("Ingestion complete. Collection '%s' now has %d documents.", settings.chroma_rag_collection, count)


if __name__ == "__main__":
    main()
