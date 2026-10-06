"""Middleware for structured request logging and request-ID tracking.

Adds a UUID4 ``X-Request-ID`` header to every request (if not already
present) and attaches it to the request state so downstream code can
include it in log lines.
"""

from __future__ import annotations

import logging
import uuid

from starlette.types import ASGIApp, Receive, Scope, Send


def _get_logger() -> logging.Logger:
    return logging.getLogger("uvicorn.access")


class RequestIDMiddleware:
    """Inject / propagate a request ID on every request."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        existing_id = headers.get(b"x-request-id") or headers.get(b"x-correlation-id")
        request_id = existing_id.decode() if existing_id else uuid.uuid4().hex

        scope["state"] = scope.get("state") or {}
        scope["state"]["request_id"] = request_id

        # Inject into response headers via a response wrapper
        original_send = send

        async def _send(message):
            if message["type"] == "http.response.start":
                raw_headers = message.get("headers", [])
                raw_headers.append((b"x-request-id", request_id.encode()))
                message["headers"] = raw_headers
            await original_send(message)

        await self.app(scope, receive, _send)
