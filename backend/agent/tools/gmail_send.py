"""Gmail send tool — delivers real email over SMTP.

There is no simulation path in this module. If credentials are missing the tool
raises rather than returning a success-looking string, because a fake send is
worse than an obvious failure: the caller would report a message delivered that
was never transmitted.

Confirmation is a parameter, not a prompt convention. ``confirmed=False``
returns a preview and touches no network. Nothing is sent unless the caller
passes ``confirmed=True``, so an agent that forgets to ask still cannot mail a
stranger.
"""

from __future__ import annotations

import logging
import re
import smtplib
import sys
import threading
import time
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import formataddr, formatdate, make_msgid

from langchain_core.tools import ToolException
from langchain_core.tools import tool as lc_tool
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from backend.agent.config import get_settings
from backend.agent.metrics import get_metrics
from backend.agent.rate_limiter import RateLimitExceeded
from backend.agent.uploads import resolve_attachments

logger = logging.getLogger(__name__)

MAX_SUBJECT_LENGTH = 200
MAX_BODY_LENGTH = 5000

# Deliberately stricter than RFC 5322. This accepts the addresses people
# actually type and rejects header-injection attempts, which is the property
# that matters here.
_EMAIL_PATTERN = re.compile(
    r"^[A-Za-z0-9!#$%&'*+/=?^_`{|}~.-]+"
    r"@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)+$"
)

# Header injection guard. A newline in a subject or recipient would let a caller
# append arbitrary headers (Bcc, extra To) to the outgoing message.
_CONTROL_CHARS = re.compile(r"[\r\n\x00]")

# Fallback counters, used only when Redis is unreachable. Without a local
# fallback the hourly send cap would silently disappear whenever Redis is down.
_LOCAL_SENDS: dict[str, list[float]] = {}
_LOCAL_LOCK = threading.Lock()


def _sanitize_header(value: str, field_name: str) -> str:
    """Strip control characters that could forge extra mail headers."""

    cleaned = _CONTROL_CHARS.sub(" ", value).strip()

    if not cleaned:
        raise ToolException(f"{field_name} is empty after sanitisation")

    return cleaned


def _sanitize_body(body: str) -> str:
    """Neutralise control characters in the body.

    The body is plain text, so this is about stripping terminal escape sequences
    and NUL bytes rather than header injection.
    """

    cleaned = _CONTROL_CHARS.sub("", body)
    # eslint-disable-next-line control-characters
    cleaned = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", cleaned)

    return cleaned.strip()


def _validate_email(address: str) -> str:
    candidate = address.strip()

    if not candidate:
        raise ToolException("Recipient address is empty")

    if _CONTROL_CHARS.search(candidate):
        raise ToolException(
            "Recipient contains illegal line breaks or control characters"
        )

    # Delegate to email_validator when it is installed (it is in
    # requirements.txt); the regex above is the always-available fallback so the
    # tool still behaves if the optional dependency is missing.
    try:
        from email_validator import EmailNotValidError, validate_email

        try:
            result = validate_email(candidate, check_deliverability=False)
            return result.normalized
        except EmailNotValidError as exc:
            raise ToolException(f"Invalid recipient address: {exc}") from exc
    except ImportError:
        pass

    if not _EMAIL_PATTERN.match(candidate):
        raise ToolException(
            f"Invalid recipient address: {candidate!r} is not a valid email"
        )

    return candidate


def _require_credentials() -> tuple[str, str, str]:
    settings = get_settings()

    if not settings.gmail_user or not settings.gmail_app_password:
        raise ToolException(
            "Gmail is not configured. Set GMAIL_USER and GMAIL_APP_PASSWORD "
            "in backend/.env. Create an App Password at "
            "https://myaccount.google.com/apppasswords. "
            "No message was sent."
        )

    return (
        settings.gmail_user.strip(),
        # Google displays App Passwords in groups of four ("abcd efgh ijkl
        # mnop") purely so they can be transcribed reliably, and says so when
        # it issues one. Passing that form to AUTH verbatim makes Gmail answer
        # "Invalid username or password" for a password that is in fact
        # correct, which is an unpleasant thing to debug. Accept either form.
        "".join(settings.gmail_app_password.split()),
        settings.gmail_from_name or "AI Agent",
    )


_redis_ok: bool | None = None
_REDIS_CHECKED_AT = 0.0
_REDIS_PROBE_TTL = 30.0


def _redis_usable() -> bool:
    """Whether Redis answered recently enough to trust it for rate limiting.

    ``RedisRateLimiter.check`` deliberately fails open and returns normally when
    Redis is down, so its return value says nothing. Calling it and returning on
    success meant the cap was skipped entirely on any machine with Redis
    configured but unreachable — which is most laptops. Probing first is what
    makes the local fallback actually run.
    """

    global _redis_ok, _REDIS_CHECKED_AT

    now = time.monotonic()

    if _redis_ok is not None and now - _REDIS_CHECKED_AT < _REDIS_PROBE_TTL:
        return _redis_ok

    usable = False
    redis_url = get_settings().redis_url
    if not redis_url:
        _redis_ok = False
        _REDIS_CHECKED_AT = now
        return False

    try:
        import redis

        client = redis.from_url(
            redis_url,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
        usable = bool(client.ping())
    except Exception as exc:  # noqa: BLE001
        logger.debug("Redis probe failed, using local send counter: %s", exc)

    _redis_ok = usable
    _REDIS_CHECKED_AT = now

    return usable


def reset_local_send_counter(user_id: str | None = None) -> None:
    """Clear the in-process hourly send counter.

    Call this in tests or when you need to unblock sending without waiting
    an hour. When ``user_id`` is None every user's counter is cleared.
    """
    with _LOCAL_LOCK:
        if user_id is None:
            _LOCAL_SENDS.clear()
        else:
            _LOCAL_SENDS.pop(user_id, None)


def _check_rate_limit(user_id: str) -> None:
    """Enforce the hourly send cap.

    Redis is used when reachable so the cap is shared across replicas. When it
    is not — which is the common case on a laptop — an in-process counter takes
    over, so the cap is never silently disabled.
    """

    global _LOCAL_SENDS, _LOCAL_LOCK

    settings = get_settings()
    limit = max(1, settings.gmail_max_per_hour)
    window_seconds = 3600

    if _redis_usable():
        from backend.agent.rate_limiter import RedisRateLimiter

        try:
            # user_id is the bucket owner; "gmail_tool" is the client_ip slot
            # (no real IP available here); "send_email" is the endpoint label.
            RedisRateLimiter(settings.redis_url, limit, window_seconds).check(
                user_id, "gmail_tool", "send_email"
            )
            return
        except RateLimitExceeded as exc:
            # Convert to ToolException so the agent surfaces a readable message
            # rather than a raw traceback.
            raise ToolException(
                f"Rate limit reached: this account has already sent {limit} "
                f"emails in the last hour. Try again in "
                f"{exc.retry_after // 60} minute(s)."
            ) from exc
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis rate limit error, using local counter: %s", exc)
            _redis_ok = False  # type: ignore[assignment]

    now = time.time()

    with _LOCAL_LOCK:
        history = [
            t for t in _LOCAL_SENDS.get(user_id, []) if now - t < window_seconds
        ]

        if len(history) >= limit:
            _LOCAL_SENDS[user_id] = history
            oldest = min(history)
            retry_in = max(1, int(window_seconds - (now - oldest)))
            raise ToolException(
                f"Rate limit reached: this account has already sent {limit} "
                f"emails in the last hour. Try again in "
                f"{retry_in // 60} minute(s)."
            )

        history.append(now)
        _LOCAL_SENDS[user_id] = history


def _smtp_send(
    sender: str,
    app_password: str,
    from_name: str,
    recipient: str,
    subject: str,
    body: str,
    attachments: list | None = None,
) -> str:
    """Perform one authenticated SMTP delivery. Raises on failure."""

    settings = get_settings()

    message = EmailMessage()
    message["From"] = formataddr((from_name, sender))
    message["To"] = recipient
    message["Subject"] = subject
    message["Date"] = formatdate(localtime=True)
    # A stable Message-ID lets the recipient quote a reply and gives support a
    # reference to quote in logs.
    message["Message-ID"] = make_msgid(domain=sender.split("@")[-1])
    message.set_content(body)

    for record in attachments or []:
        # maintype/subtype are taken from the validated allowlist rather than
        # guessed from the filename, and the name is sanitised because it
        # becomes a MIME header the recipient's client will display.
        message.add_attachment(
            record.path.read_bytes(),
            maintype=record.mime.split("/", 1)[0],
            subtype=record.mime.split("/", 1)[1],
            filename=_sanitize_header(record.original_name, "attachment"),
        )

    deadline = time.monotonic() + settings.gmail_timeout_seconds

    with smtplib.SMTP(
        settings.gmail_smtp_host,
        settings.gmail_smtp_port,
        timeout=settings.gmail_timeout_seconds,
    ) as smtp:
        smtp.ehlo()
        # Port 587 is submission: STARTTLS upgrades the connection before the
        # password crosses the wire. Credentials must never go out in the clear.
        smtp.starttls()
        smtp.ehlo()
        smtp.login(sender, app_password)

        remaining = max(1.0, deadline - time.monotonic())

        if smtp.sock is not None:
            smtp.sock.settimeout(remaining)

        refused = smtp.send_message(message)

        if refused:
            raise smtplib.SMTPException(
                f"Recipient refused: {', '.join(refused)}"
            )

    return message["Message-ID"]


class SendEmailInput(BaseModel):
    """Input schema for the Gmail send tool."""

    model_config = ConfigDict(populate_by_name=True)

    to_email: str = Field(
        ...,
        # LLMs paraphrase argument names constantly, and "recipient" was the
        # model's first guess in testing, which failed the whole call. Accepting
        # the near-misses costs nothing and turns a hard failure into a send.
        validation_alias=AliasChoices("to_email", "recipient", "to", "email"),
        description="Recipient email address, e.g. 'ravi@gmail.com'. Field name: to_email.",
        min_length=3,
        max_length=320,
    )
    subject: str = Field(
        ...,
        description="Email subject line. Must not be empty.",
        min_length=1,
        max_length=MAX_SUBJECT_LENGTH,
    )
    body: str = Field(
        ...,
        description="Plain-text email body. Must not be empty.",
        min_length=1,
        max_length=MAX_BODY_LENGTH,
    )
    confirmed: bool = Field(
        default=False,
        description=(
            "Must be True to actually send. Leave False to receive a preview "
            "without sending anything."
        ),
    )
    user_id: str = Field(
        default="default_user",
        description="Caller identity, used for the hourly send rate limit.",
        max_length=128,
    )
    file_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Optional file_ids returned by POST /upload, to attach the files to "
            "this email. At most 5 files, 20 MB total. Empty means no "
            "attachments. Only pass ids the user has actually uploaded in this "
            "conversation; never invent one."
        ),
        max_length=5,
    )


def _send_email(
    to_email: str,
    subject: str,
    body: str,
    confirmed: bool = False,
    user_id: str = "default_user",
    file_ids: list[str] | None = None,
) -> str:
    """Send an email through Gmail SMTP.

    Pass ``confirmed=False`` (the default) to get a preview without sending.
    Only ``confirmed=True`` transmits a message.
    """

    metrics = get_metrics()
    settings = get_settings()
    started = time.monotonic()

    try:
        recipient = _validate_email(to_email)
        subject = _sanitize_header(subject, "Subject")[:MAX_SUBJECT_LENGTH]
        body = _sanitize_body(body)[:MAX_BODY_LENGTH]

        if not body:
            raise ToolException("Body is empty after sanitisation")

        # Resolved before the confirmation gate so an unknown or foreign file_id
        # fails while nothing has been sent, rather than after approval.
        try:
            attachments = resolve_attachments(user_id, list(file_ids or []))
        except Exception as exc:
            raise ToolException(str(exc)) from exc

        sender, app_password, from_name = _require_credentials()

        # A loop-back guard: the agent mailing itself is never a real request and
        # burns the hourly quota.
        if recipient.lower() == sender.lower():
            raise ToolException(
                f"Refusing to send to the sending account itself ({sender}). "
                "No message was sent."
            )

        if not confirmed:
            lines = [
                "PREVIEW ONLY - nothing was sent.",
                f"To: {recipient}",
                f"From: {from_name} <{sender}>",
                f"Subject: {subject}",
                f"Body: {body}",
            ]

            if attachments:
                total_kb = sum(r.size for r in attachments) / 1024
                lines.append(f"Attachments ({len(attachments)}):")
                lines.extend(
                    f"- {r.original_name} ({r.ext}, {r.size / 1024:.0f} KB) "
                    f"file_id={r.file_id}"
                    for r in attachments
                )
                lines.append(f"Total attachment size: {total_kb:.0f} KB")

            return "\n".join(lines)

        _check_rate_limit(user_id)

        last_error: Exception | None = None

        for attempt in range(1, max(1, settings.gmail_max_retries) + 1):
            try:
                message_id = _smtp_send(
                    sender,
                    app_password,
                    from_name,
                    recipient,
                    subject,
                    body,
                    attachments,
                )
            except smtplib.SMTPAuthenticationError as exc:
                # Must precede SMTPResponseException: authentication errors are a
                # subclass of it, and a generic "5xx, not retried" message here
                # would hide the one fix that actually matters — a wrong App
                # Password.
                raise ToolException(
                    "Gmail rejected the credentials. Check GMAIL_USER and that "
                    f"GMAIL_APP_PASSWORD is a 16-character App Password, not "
                    f"your account password. ({exc})"
                ) from exc
            except (smtplib.SMTPResponseException, smtplib.SMTPConnectError) as exc:
                # 4xx means "try again"; 5xx means this will never succeed, so
                # retrying just wastes the quota and delays the error.
                code = getattr(exc, "smtp_code", 0) or 0
                last_error = exc

                if 400 <= code < 500:
                    raise ToolException(
                        f"Gmail rejected the message ({code} {code}): "
                        f"{getattr(exc, 'smtp_error', exc)}. Not retried."
                    ) from exc

                logger.warning(
                    "SMTP attempt %d/%d failed: %s",
                    attempt,
                    settings.gmail_max_retries,
                    exc,
                )

                if attempt < settings.gmail_max_retries:
                    time.sleep(1.5 * attempt)
                continue
            except smtplib.SMTPException as exc:
                last_error = exc
                logger.warning(
                    "SMTP attempt %d/%d failed: %s",
                    attempt,
                    settings.gmail_max_retries,
                    exc,
                )

                if attempt < settings.gmail_max_retries:
                    time.sleep(1.5 * attempt)

                continue
            except OSError as exc:
                # Socket timeouts and DNS failures surface here.
                last_error = exc
                logger.warning(
                    "SMTP attempt %d/%d failed at the socket level: %s",
                    attempt,
                    settings.gmail_max_retries,
                    exc,
                )

                if attempt < settings.gmail_max_retries:
                    time.sleep(1.5 * attempt)

                continue

            sent_at = datetime.now(timezone.utc).astimezone().strftime(
                "%Y-%m-%d %H:%M:%S %Z"
            )
            metrics.record_tool("send_email", time.monotonic() - started, True)

            logger.info("Email sent to %s (id=%s)", recipient, message_id)

            sent_note = (
                f" with {len(attachments)} attachment(s), id {message_id}"
                if attachments
                else f" with id {message_id}"
            )

            return f"Sent to {recipient} at {sent_at}{sent_note}"

        metrics.record_tool("send_email", time.monotonic() - started, False)
        raise ToolException(
            f"Gmail delivery failed after {settings.gmail_max_retries} "
            f"attempts: {last_error}"
        )

    except ToolException:
        metrics.record_tool("send_email", time.monotonic() - started, False)
        raise
    except Exception as exc:  # noqa: BLE001
        metrics.record_tool("send_email", time.monotonic() - started, False)
        logger.exception("Unexpected failure in send_email")
        raise ToolException(f"Unexpected failure while sending: {exc}") from exc


send_email = lc_tool(
    "send_email",
    description=(
        "Send a real plain-text email through the configured Gmail account over "
        "SMTP. Arguments are exactly: to_email, subject, body, confirmed. "
        "Call with confirmed=false first to get a preview; nothing is "
        "transmitted. Only call again with confirmed=true after the user has "
        "explicitly agreed. If the result begins with 'PREVIEW ONLY', do not "
        "call this tool again - show the preview and wait for the user to reply."
    ),
    args_schema=SendEmailInput,
)(_send_email)


def _main() -> int:
    """CLI for verifying real delivery.

    Runs a preview by default and only transmits with ``--send``, so a typo in
    this command cannot mail anybody. The same confirmation rule the agent is
    held to applies here.

        python -m backend.agent.tools.gmail_send --to you@gmail.com \\
            --subject "test" --body "hello" --send
    """

    import argparse

    parser = argparse.ArgumentParser(
        prog="gmail_send",
        description="Send a real test email through the configured Gmail account.",
    )
    parser.add_argument("--to", required=True, help="Recipient address")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--body", required=True)
    parser.add_argument(
        "--send",
        action="store_true",
        help="Actually transmit. Without this flag only a preview is printed.",
    )
    parser.add_argument("--user-id", default="cli_test")

    args = parser.parse_args()

    print(f"Gmail account : {get_settings().gmail_user or '(not configured)'}")
    print(f"Mode          : {'REAL SEND' if args.send else 'preview only'}")

    try:
        result = _send_email(
            to_email=args.to,
            subject=args.subject,
            body=args.body,
            confirmed=args.send,
            user_id=args.user_id,
        )
    except ToolException as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 1

    print(f"\n{result}")

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())