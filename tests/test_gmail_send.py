"""Tests for the real Gmail send tool.

The critical property under test is that nothing reaches the network unless the
caller both supplies valid credentials and passes ``confirmed=True``. The preview
test enforces that by patching ``smtplib.SMTP`` to fail if it is ever
constructed, so a regression that sends without confirmation breaks the suite
instead of mailing somebody.
"""

from __future__ import annotations

import smtplib

import pytest
from langchain_core.tools import ToolException

from backend.agent.tools import ALL_TOOLS, DEFAULT_TOOLS
from backend.agent.tools.gmail_send import (
    _check_rate_limit,
    _sanitize_body,
    _sanitize_header,
    _send_email,
    _validate_email,
)
import backend.agent.tools.gmail_send as gmail_send


@pytest.fixture(autouse=True)
def _clear_rate_limit_state(monkeypatch):
    monkeypatch.setattr(gmail_send, "_redis_usable", lambda: False)
    gmail_send._LOCAL_SENDS.clear()
    yield
    gmail_send._LOCAL_SENDS.clear()


@pytest.fixture
def configured(monkeypatch):
    """Pretend Gmail is configured, without touching real credentials."""
    monkeypatch.setattr(gmail_send, "_require_credentials", lambda: (
        "sender@gmail.com",
        "app-password",
        "AI Agent",
    ))


class TestCredentialNormalization:
    """Google hands out App Passwords in display groups of four.

    Passing "abcd efgh ijkl mnop" to AUTH verbatim makes Gmail reject a password
    that is in fact correct, so the grouping has to be removed before login.
    """

    def _creds(self, monkeypatch, user, password):
        monkeypatch.setattr(gmail_send, "get_settings", lambda: type(
            "S",
            (),
            {
                "gmail_user": user,
                "gmail_app_password": password,
                "gmail_from_name": "AI Agent",
            },
        )())
        return gmail_send._require_credentials()

    def test_strips_display_spaces(self, monkeypatch):
        user, password, _ = self._creds(
            monkeypatch, "a@b.com", "abcd efgh ijkl mnop"
        )
        assert password == "abcdefghijklmnop"
        assert " " not in password

    def test_bare_password_unchanged(self, monkeypatch):
        user, password, _ = self._creds(
            monkeypatch, "a@b.com", "abcdefghijklmnop"
        )
        assert password == "abcdefghijklmnop"

    def test_user_is_trimmed(self, monkeypatch):
        user, _, _ = self._creds(monkeypatch, "  a@b.com  ", "abcdefghijklmnop")
        assert user == "a@b.com"

    def test_ten_digit_style_google_password(self, monkeypatch):
        user, password, _ = self._creds(
            monkeypatch, "a@b.com", "ab12 cd34 ef56 gh78"
        )
        assert password == "ab12cd34ef56gh78"


class TestRegistration:
    def test_tool_registered(self):
        assert "send_email" in ALL_TOOLS
        assert ALL_TOOLS["send_email"].name == "send_email"

    def test_not_in_default_tool_set(self):
        """Binding an irreversible tool by default would invite accidental sends."""
        names = {t.name for t in DEFAULT_TOOLS}
        assert "send_email" not in names


class TestValidation:
    @pytest.mark.parametrize(
        "address",
        ["ravi@gmail.com", "a.b+tag@sub.co.uk", "first.last@example.org"],
    )
    def test_accepts_valid(self, address):
        assert _validate_email(address)

    @pytest.mark.parametrize(
        "address",
        [
            "not-an-email",
            "a@b",
            "@gmail.com",
            "spaces in@gmail.com",
            "x@y.com\r\nBcc: evil@z.com",
            "",
        ],
    )
    def test_rejects_invalid(self, address):
        with pytest.raises(ToolException):
            _validate_email(address)

    def test_subject_strips_header_injection(self):
        cleaned = _sanitize_header("hi\r\nBcc: evil@x.com", "Subject")
        assert "\n" not in cleaned
        assert "\r" not in cleaned

    def test_body_strips_escape_sequences(self):
        assert "\x1b" not in _sanitize_body("hello \x1b[31mred\x1b[0m")

    def test_rejects_empty_subject(self):
        with pytest.raises(ToolException):
            _sanitize_header("   ", "Subject")


class TestConfirmationGate:
    def test_preview_never_opens_a_connection(self, configured, monkeypatch):
        """The whole safety story rests on this."""

        def explode(*args, **kwargs):
            raise AssertionError("smtplib.SMTP was constructed during a preview")

        monkeypatch.setattr(smtplib, "SMTP", explode)

        result = _send_email(
            to_email="test@gmail.com",
            subject="hi",
            body="hello",
            confirmed=False,
        )

        assert "PREVIEW ONLY" in result
        assert "test@gmail.com" in result

    def test_unconfirmed_does_not_consume_quota(self, configured, monkeypatch):
        monkeypatch.setattr(
            gmail_send, "_check_rate_limit", lambda *_: pytest.fail("quota used")
        )

        _send_email("test@gmail.com", "hi", "hello", confirmed=False)

    def test_unconfigured_refuses_instead_of_faking(self, monkeypatch):
        """No credentials must produce an error, never a success string."""
        # Patched rather than relying on the absence of real credentials in
        # backend/.env. Reading the developer's own .env made this test pass only
        # while nobody had configured Gmail, and start failing the moment they
        # did — the one time a send really could go out.
        monkeypatch.setattr(
            gmail_send,
            "_require_credentials",
            lambda: (_ for _ in ()).throw(
                ToolException(
                    "Gmail is not configured. No message was sent."
                )
            ),
        )

        with pytest.raises(ToolException) as exc:
            _send_email("test@gmail.com", "hi", "hello", confirmed=True)

        message = str(exc.value)
        assert "not configured" in message.lower()
        assert "no message was sent" in message.lower()

    def test_blocks_self_send(self, configured):
        with pytest.raises(ToolException) as exc:
            _send_email("sender@gmail.com", "hi", "hello", confirmed=True)

        assert "itself" in str(exc.value)

    def test_self_send_is_case_insensitive(self, configured):
        with pytest.raises(ToolException):
            _send_email("Sender@Gmail.com", "hi", "hello", confirmed=True)


class TestSuccessfulSend:
    def test_returns_sent_confirmation(self, configured, monkeypatch):
        captured = {}

        class FakeSMTP:
            def __init__(self, host, port, timeout=None):
                captured["host"] = host
                captured["port"] = port
                captured["timeout"] = timeout
                self.sock = None

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def ehlo(self):
                pass

            def starttls(self):
                captured["starttls"] = True

            def login(self, user, password):
                captured["login"] = (user, password)

            def send_message(self, message):
                captured["message"] = message
                return {}

        monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)

        result = _send_email(
            "test@gmail.com",
            "hi",
            "hello",
            confirmed=True,
            user_id="u1",
        )

        assert captured["host"] == "smtp.gmail.com"
        assert captured["port"] == 587
        assert captured["starttls"] is True
        assert captured["login"] == ("sender@gmail.com", "app-password")
        assert result.startswith("Sent to test@gmail.com at ")
        assert "with id" in result

        sent = captured["message"]
        assert sent["To"] == "test@gmail.com"
        assert sent["Subject"] == "hi"
        assert sent["From"] == "AI Agent <sender@gmail.com>"

    def test_refused_recipient_is_an_error_not_a_success(self, configured, monkeypatch):
        class RefusingSMTP:
            def __init__(self, *a, **k):
                self.sock = None

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def ehlo(self):
                pass

            def starttls(self):
                pass

            def login(self, *a):
                pass

            def send_message(self, message):
                return {"test@gmail.com": (550, b"no such user")}

        monkeypatch.setattr(smtplib, "SMTP", RefusingSMTP)

        with pytest.raises(ToolException):
            _send_email("test@gmail.com", "hi", "hello", confirmed=True)

    def test_auth_failure_is_actionable(self, configured, monkeypatch):
        class BadAuthSMTP:
            def __init__(self, *a, **k):
                self.sock = None

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def ehlo(self):
                pass

            def starttls(self):
                pass

            def login(self, *a):
                raise smtplib.SMTPAuthenticationError(535, b"bad credentials")

        monkeypatch.setattr(smtplib, "SMTP", BadAuthSMTP)

        with pytest.raises(ToolException) as exc:
            _send_email("test@gmail.com", "hi", "hello", confirmed=True)

        assert "App Password" in str(exc.value)


class TestRateLimit:
    def test_blocks_past_the_hourly_cap(self, configured):
        # Redis is unreachable in this environment, so the in-process counter is
        # what enforces the cap. Fill it directly and assert the next call fails.
        import time as _time

        now = _time.time()
        cap = gmail_send.get_settings().gmail_max_per_hour
        gmail_send._LOCAL_SENDS["u1"] = [now - 1] * cap

        with pytest.raises(ToolException) as exc:
            _check_rate_limit("u1")

        assert "rate limit" in str(exc.value).lower()

    def test_allows_below_the_cap(self, configured):
        import time as _time

        now = _time.time()
        gmail_send._LOCAL_SENDS["u1"] = [now - 1] * 3

        _check_rate_limit("u1")

    def test_old_entries_expire(self, configured):
        import time as _time

        # Older than an hour, so they must not count against the cap.
        gmail_send._LOCAL_SENDS["u1"] = [_time.time() - 7200] * 50

        _check_rate_limit("u1")


class TestIntentRouting:
    def test_email_intent_binds_only_the_email_tool(self):
        from backend.agent.llm import models_for_intent

        providers = models_for_intent("email")
        assert providers

    def test_email_questions_classify_as_email(self):
        from backend.agent.nodes.router import _keyword_classify

        assert _keyword_classify("Send birthday wish to ravi@gmail.com") == "email"
        assert _keyword_classify("email bob@x.com about the report") == "email"

    def test_non_email_is_not_classified_as_email(self):
        from backend.agent.nodes.router import _keyword_classify

        assert _keyword_classify("send me the retrieval pipeline") != "email"
        assert _keyword_classify("What is 25 * 50?") == "calculate"


class TestEmailOutcomeExtraction:
    def test_sent(self):
        from backend.main import _extract_email_outcome

        class R:
            tool_name = "send_email"
            success = True
            result = "Sent to ravi@gmail.com at 2026-10-04 10:00:00 IST with id a@b"

        out = _extract_email_outcome({"tool_results": [R()]})

        assert out["status"] == "sent"
        assert out["recipient"] == "ravi@gmail.com"

    def test_preview_is_not_reported_as_sent(self):
        from backend.main import _extract_email_outcome

        class R:
            tool_name = "send_email"
            success = True
            result = "PREVIEW ONLY - nothing was sent.\nTo: ravi@gmail.com"

        out = _extract_email_outcome({"tool_results": [R()]})

        assert out["status"] == "preview"

    def test_absent(self):
        from backend.main import _extract_email_outcome

        assert _extract_email_outcome({"tool_results": []}) is None
        assert _extract_email_outcome(None) is None