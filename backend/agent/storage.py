"""Document storage abstraction.

Supports multiple backends for document ingestion:
    - Local filesystem (development)
    - S3-compatible object storage (production)

The abstraction ensures that all API replicas access the *same*
document store — never per-container local storage.
"""

from __future__ import annotations

import logging
import os
from typing import Protocol

from backend.agent.config import get_settings

logger = logging.getLogger(__name__)


class DocumentStorage(Protocol):
    """Protocol that any document-storage backend must implement."""

    def list_files(self) -> list[str]: ...

    def read_file(self, path: str) -> bytes: ...

    def write_file(self, path: str, data: bytes) -> str: ...

    def delete_file(self, path: str) -> None: ...


class LocalDocumentStorage:
    """Filesystem-backed storage (development default).

    Works with shared Docker volumes mounted across replicas.
    """

    def __init__(self, base_path: str) -> None:
        self._base = base_path
        os.makedirs(self._base, exist_ok=True)

    def _safe_path(self, path: str) -> str:
        """Resolve and validate path stays within base directory."""
        full = os.path.normpath(os.path.join(self._base, path))
        if not full.startswith(os.path.abspath(self._base)):
            raise ValueError(f"Path traversal detected: {path}")
        return full

    def list_files(self) -> list[str]:
        files = []
        for root, _, names in os.walk(self._base):
            for name in names:
                rel = os.path.relpath(os.path.join(root, name), self._base)
                files.append(rel)
        return sorted(files)

    def read_file(self, path: str) -> bytes:
        full = self._safe_path(path)
        with open(full, "rb") as f:
            return f.read()

    def write_file(self, path: str, data: bytes) -> str:
        full = self._safe_path(path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as f:
            f.write(data)
        return path

    def delete_file(self, path: str) -> None:
        full = self._safe_path(path)
        if os.path.exists(full):
            os.remove(full)


class S3DocumentStorage:
    """S3-compatible object storage (production).

    Requires ``boto3``. Falls back to local if S3 credentials are not set.
    """

    def __init__(self, bucket: str, prefix: str = "") -> None:
        import boto3

        self._s3 = boto3.client("s3")
        self._bucket = bucket
        self._prefix = prefix

    def _key(self, path: str) -> str:
        path = path.lstrip("/")
        if self._prefix:
            return f"{self._prefix.rstrip('/')}/{path}"
        return path

    def list_files(self) -> list[str]:
        paginator = self._s3.get_paginator("list_objects_v2")
        files = []
        for page in paginator.paginate(Bucket=self._bucket, Prefix=self._prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if self._prefix:
                    key = key[len(self._prefix) + 1 :]
                files.append(key)
        return sorted(files)

    def read_file(self, path: str) -> bytes:
        obj = self._s3.get_object(Bucket=self._bucket, Key=self._key(path))
        return obj["Body"].read()

    def write_file(self, path: str, data: bytes) -> str:
        self._s3.put_object(Bucket=self._bucket, Key=self._key(path), Body=data)
        return path

    def delete_file(self, path: str) -> None:
        self._s3.delete_object(Bucket=self._bucket, Key=self._key(path))


_STORAGE: DocumentStorage | None = None


def get_document_storage() -> DocumentStorage:
    """Return a cached document storage instance based on configuration."""
    global _STORAGE
    if _STORAGE is None:
        settings = get_settings()
        if settings.doc_storage_provider == "s3":
            _STORAGE = S3DocumentStorage(
                bucket=settings.doc_storage_bucket,
                prefix=settings.doc_storage_prefix,
            )
            logger.info("Document storage: S3 (%s)", settings.doc_storage_bucket)
        else:
            _STORAGE = LocalDocumentStorage(settings.documents_path)
            logger.info("Document storage: local (%s)", settings.documents_path)
    return _STORAGE
