"""Gmail MCP Server — exposes the send_email tool for email automation and previews."""

from __future__ import annotations

import logging
from typing import Optional
from fastmcp import FastMCP

from backend.agent.tools.gmail_send import _send_email

logger = logging.getLogger(__name__)

# Initialize FastMCP Gmail Server
gmail_mcp = FastMCP("GmailMCPServer")


@gmail_mcp.tool(
    name="send_email",
    description=(
        "Send real email through Gmail SMTP. Always follow the two-step confirmation flow: "
        "first call with confirmed=false to get a preview, show the preview to the user, "
        "and only call with confirmed=true after the user explicitly confirms."
    ),
)
def send_email(
    to_email: str,
    subject: str,
    body: str,
    confirmed: bool = False,
    user_id: str = "default_user",
    file_ids: Optional[list[str]] = None,
) -> str:
    """Send an email or generate a preview.

    Args:
        to_email: Recipient email address (e.g. 'user@example.com').
        subject: Subject line of the email.
        body: Plain text email body.
        confirmed: Must be True to send, or False to generate a safe preview (default False).
        user_id: User identifier for hourly rate limiting.
        file_ids: Optional list of uploaded file IDs to attach.
    """
    try:
        return _send_email(
            to_email=to_email,
            subject=subject,
            body=body,
            confirmed=confirmed,
            user_id=user_id,
            file_ids=file_ids or [],
        )
    except Exception as exc:
        logger.error("Gmail MCP tool error: %s", exc)
        return f"Error: {exc}"


if __name__ == "__main__":
    gmail_mcp.run()
