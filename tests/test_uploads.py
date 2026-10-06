"""Tests for upload storage, validation and attachment resolution.

The properties that matter here are the hostile ones: a rejected type stays
rejected, a file over the cap is never fully written, and — most importantly —
an attachment id cannot be used to read a file belonging to somebody else. The
happy paths are covered too, because a guard that rejects everything also
passes every rejection test.
"""

from __future__ import annotations

import io

import pytest

from backend.agent import uploads as U


@pytest.fixture(autouse=True)
def _clean():
    U.clear_user_uploads("alice")
    U.clear_user_uploads("mallory")
    yield
    U.clear_user_uploads("alice")
    U.clear_user_uploads("mallory")


@pytest.fixture
def small_limits(monkeypatch):
    """Shrink the caps so limit tests allocate kilobytes, not megabytes.

    The limits are settings now and their defaults are deliberately generous
    (200 MB per file), so testing them at their real values would write
    hundreds of megabytes to disk on every run.
    """

    monkeypatch.setattr(U, "max_file_bytes", lambda: 64 * 1024)
    monkeypatch.setattr(U, "max_files_per_request", lambda: 3)
    monkeypatch.setattr(U, "max_total_upload_bytes", lambda: 256 * 1024)
    monkeypatch.setattr(U, "max_total_attachment_bytes", lambda: 96 * 1024)
    monkeypatch.setattr(U, "_check_free_space", lambda: None)

    return {"file": 64 * 1024, "files": 3, "total": 256 * 1024, "email": 96 * 1024}


def blob(data: bytes, name: str = "f.txt") -> tuple[str, io.BytesIO]:
    return name, io.BytesIO(data)


class TestAllowlist:
    @pytest.mark.parametrize(
        "ext",
        [
            ".pdf", ".doc", ".docx", ".txt", ".csv",
            ".xls", ".xlsx", ".pptx",
            ".jpg", ".jpeg", ".png", ".webp",
        ],
    )
    def test_every_documented_type_is_accepted(self, ext):
        record = U.store_upload(f"sample{ext}", io.BytesIO(b"data"), "alice")
        assert record.ext == ext
        assert record.mime == U.MIME_BY_EXT[ext]

    @pytest.mark.parametrize("ext", [".exe", ".bat", ".sh", ".msi", ".js"])
    def test_blocked_types_are_always_refused(self, ext):
        with pytest.raises(U.UploadError):
            U.store_upload(f"payload{ext}", io.BytesIO(b"MZ"), "alice")

    def test_blocked_type_is_named_as_blocked(self):
        with pytest.raises(U.UploadError, match="never accepted"):
            U.store_upload("run.exe", io.BytesIO(b"MZ"), "alice")

    def test_unknown_type_refused(self):
        with pytest.raises(U.UploadError, match="not an allowed"):
            U.store_upload("thing.xyz", io.BytesIO(b"x"), "alice")

    def test_missing_extension_refused(self):
        with pytest.raises(U.UploadError, match="no file extension"):
            U.store_upload("README", io.BytesIO(b"x"), "alice")

    def test_extension_match_is_case_insensitive(self):
        record = U.store_upload("REPORT.PDF", io.BytesIO(b"x"), "alice")
        assert record.ext == ".pdf"

    def test_double_extension_cannot_smuggle(self):
        # "invoice.pdf.exe" must be judged on its real final extension.
        with pytest.raises(U.UploadError):
            U.store_upload("invoice.pdf.exe", io.BytesIO(b"MZ"), "alice")


class TestLimits:
    def test_more_than_five_files_refused(self, small_limits):
        batch = [(f"f{i}.txt", io.BytesIO(b"x")) for i in range(small_limits["files"] + 1)]

        with pytest.raises(U.UploadError, match="At most 3 files"):
            U.store_uploads(batch, "alice")

    def test_exactly_five_files_allowed(self, small_limits):
        batch = [(f"f{i}.txt", io.BytesIO(b"x")) for i in range(small_limits["files"])]
        assert len(U.store_uploads(batch, "alice")) == small_limits["files"]

    def test_oversized_file_is_cut_off(self, small_limits):
        payload = io.BytesIO(b"a" * (small_limits["file"] + 4096))

        with pytest.raises(U.UploadError, match="per-file limit"):
            U.store_upload("big.txt", payload, "alice")

    def test_rejected_file_leaves_nothing_on_disk(self, small_limits):
        directory = U.user_upload_dir("alice")

        with pytest.raises(U.UploadError):
            U.store_upload(
                "big.txt", io.BytesIO(b"a" * (small_limits["file"] + 1)), "alice"
            )

        assert list(directory.glob("*")) == []

    def test_failed_batch_rolls_back_earlier_files(self):
        batch = [
            ("ok1.txt", io.BytesIO(b"a")),
            ("ok2.txt", io.BytesIO(b"b")),
            ("evil.exe", io.BytesIO(b"MZ")),
        ]

        with pytest.raises(U.UploadError):
            U.store_uploads(batch, "alice")

        # The two that succeeded before the rejection must not survive, or they
        # would remain attachable after the caller was told the upload failed.
        assert [p for p in U.user_upload_dir("alice").glob("*") if p.suffix != ".json"] == []

    def test_empty_file_refused(self):
        with pytest.raises(U.UploadError, match="empty"):
            U.store_upload("nothing.txt", io.BytesIO(b""), "alice")

    def test_limits_come_from_settings(self):
        # Guards against the caps drifting back into module constants, which
        # would silently ignore configuration.
        assert U.max_file_bytes() >= 1024
        assert U.max_files_per_request() >= 1
        assert U.max_total_upload_bytes() >= U.max_file_bytes()
        assert U.max_total_attachment_bytes() >= 1


class TestUserIsolation:
    @pytest.mark.parametrize(
        "bad",
        ["../evil", "..", "a/b", "a\\b", "", ".", "x" * 100],
    )
    def test_traversing_user_id_refused(self, bad):
        with pytest.raises(U.UploadError):
            U.user_upload_dir(bad)

    def test_other_users_file_is_not_resolvable(self):
        record = U.store_upload("secret.txt", io.BytesIO(b"top secret"), "mallory")

        with pytest.raises(U.UploadError, match="not found"):
            U.resolve_attachments("alice", [record.file_id])

    def test_path_shaped_id_refused(self):
        for bad in ["../../etc/passwd", "..%2f..%2fetc", "/etc/passwd", "abc"]:
            with pytest.raises(U.UploadError):
                U.resolve_attachments("alice", [bad])

    def test_non_hex_id_refused(self):
        with pytest.raises(U.UploadError, match="Invalid attachment id"):
            U.resolve_attachments("alice", ["A" * 32])

    def test_own_file_resolves(self):
        record = U.store_upload("mine.txt", io.BytesIO(b"hello"), "alice")
        resolved = U.resolve_attachments("alice", [record.file_id])

        assert len(resolved) == 1
        assert resolved[0].original_name == "mine.txt"

    def test_email_size_ceiling_is_enforced(self, small_limits):
        # Gmail's own 25 MB limit is the reason this cap exists, so exceeding it
        # must fail here with an explanation rather than bouncing later.
        ids = [
            U.store_upload(f"f{i}.txt", io.BytesIO(b"x" * 60_000), "alice").file_id
            for i in range(2)
        ]

        with pytest.raises(U.UploadError, match="fits in one email"):
            U.resolve_attachments("alice", ids)


class TestExtraction:
    def test_txt_extracted(self):
        record = U.store_upload("note.txt", io.BytesIO(b"plain body"), "alice")
        assert "plain body" in record.text

    def test_docx_extracted(self):
        docx = pytest.importorskip("docx")
        document = docx.Document()
        document.add_paragraph("semester marks")

        buffer = io.BytesIO()
        document.save(buffer)

        record = U.store_upload("marks.docx", io.BytesIO(buffer.getvalue()), "alice")
        assert "semester marks" in record.text

    def test_csv_rows_shown(self):
        pd = pytest.importorskip("pandas")
        buffer = io.BytesIO()
        pd.DataFrame({"name": ["a", "b"], "score": [1, 2]}).to_csv(buffer, index=False)

        record = U.store_upload("data.csv", io.BytesIO(buffer.getvalue()), "alice")
        assert "name" in record.text and "score" in record.text

    def test_csv_truncates_at_row_limit(self):
        pd = pytest.importorskip("pandas")
        buffer = io.BytesIO()
        pd.DataFrame({"n": range(500)}).to_csv(buffer, index=False)

        record = U.store_upload("big.csv", io.BytesIO(buffer.getvalue()), "alice")
        assert f"first {U.EXTRACT_ROW_LIMIT} of 500" in record.text

    def test_image_gets_thumbnail_and_no_invented_caption(self):
        Image = pytest.importorskip("PIL.Image")
        buffer = io.BytesIO()
        Image.new("RGB", (80, 60), (10, 120, 200)).save(buffer, format="PNG")

        record = U.store_upload("photo.png", io.BytesIO(buffer.getvalue()), "alice")

        assert record.thumbnail is not None
        # No vision model runs here, so the text must not claim to describe it.
        assert "ready to send" in record.text.lower()
        assert record.category == "image"

    def test_corrupt_pdf_still_attachable(self):
        # An unparseable file must not become unsendable.
        record = U.store_upload(
            "broken.pdf", io.BytesIO(b"not really a pdf"), "alice"
        )
        assert "could not be extracted" in record.text.lower()

    def test_resolved_record_knows_its_thumbnail(self):
        # Regression: resolve_attachments left thumbnail unset, so the preview
        # endpoint 404'd on every image even though the thumbnail existed.
        Image = pytest.importorskip("PIL.Image")
        buffer = io.BytesIO()
        Image.new("RGB", (50, 50), (5, 5, 5)).save(buffer, format="JPEG")

        record = U.store_upload("pic.jpg", io.BytesIO(buffer.getvalue()), "alice")
        resolved = U.resolve_attachments("alice", [record.file_id])

        assert resolved[0].thumbnail == f"thumbnails/{record.file_id}.jpg"
        assert (resolved[0].path.parent / resolved[0].thumbnail).exists()

    def test_non_image_has_no_thumbnail(self):
        record = U.store_upload("a.txt", io.BytesIO(b"x"), "alice")
        resolved = U.resolve_attachments("alice", [record.file_id])
        assert resolved[0].thumbnail is None

    def test_legacy_doc_is_attached_without_fake_text(self):
        record = U.store_upload("old.doc", io.BytesIO(b"\xd0\xcf\x11\xe0"), "alice")
        assert record.ext == ".doc"
        assert "not supported" in record.text.lower()


class TestAsyncEntryPoint:
    """The HTTP path spools Starlette's async UploadFile before storing."""

    @pytest.mark.asyncio
    async def test_async_source_is_stored(self):
        pytest.importorskip("asyncio")

        class AsyncSource:
            def __init__(self, data: bytes):
                self._data = data
                self._pos = 0

            async def read(self, size: int = -1) -> bytes:
                if size is None or size < 0:
                    size = len(self._data) - self._pos
                chunk = self._data[self._pos : self._pos + size]
                self._pos += len(chunk)
                return chunk

        stored = await U.store_uploads_async(
            [("async.txt", AsyncSource(b"async body"))], "alice"
        )

        assert len(stored) == 1
        assert stored[0].path.read_bytes() == b"async body"

    @pytest.mark.asyncio
    async def test_async_oversize_is_refused(self, small_limits):
        class AsyncSource:
            def __init__(self, size: int):
                self._left = size

            async def read(self, size: int = -1) -> bytes:
                take = min(size if size and size > 0 else self._left, self._left)
                self._left -= take
                return b"a" * take

        with pytest.raises(U.UploadError, match="per-file limit"):
            await U.store_uploads_async(
                [("big.txt", AsyncSource(small_limits["file"] + 10_000))], "alice"
            )

    @pytest.mark.asyncio
    async def test_async_batch_cap_enforced(self, small_limits):
        class AsyncSource:
            async def read(self, size: int = -1) -> bytes:
                return b""

        items = [
            (f"f{i}.txt", AsyncSource()) for i in range(small_limits["files"] + 1)
        ]

        with pytest.raises(U.UploadError, match="At most 3 files"):
            await U.store_uploads_async(items, "alice")


class TestSendIntegration:
    def test_preview_lists_attachments_with_ids(self):
        from backend.agent.tools import gmail_send as G

        G._require_credentials = lambda: ("me@gmail.com", "pw", "AI Agent")

        record = U.store_upload("fees.pdf", io.BytesIO(b"%PDF-1.4"), "alice")

        out = G._send_email(
            "them@university.ac.in", "Fees", "here",
            confirmed=False, user_id="alice", file_ids=[record.file_id],
        )

        assert "Attachments (1)" in out
        assert "fees.pdf" in out
        assert record.file_id in out

    def test_unknown_file_id_refused_before_sending(self):
        from backend.agent.tools import gmail_send as G

        sent = []
        G._require_credentials = lambda: ("me@gmail.com", "pw", "AI Agent")
        G._smtp_send = lambda *a, **k: sent.append(a) or "<id>"

        from langchain_core.tools import ToolException

        with pytest.raises(ToolException):
            G._send_email(
                "them@university.ac.in", "S", "b",
                confirmed=True, user_id="alice", file_ids=["f" * 32],
            )

        assert sent == [], "SMTP ran despite an unresolvable attachment"

    def test_attachments_carry_correct_mime(self):
        from backend.agent.tools import gmail_send as G
        from email.message import EmailMessage

        captured = {}

        def fake(sender, pw, name, rcpt, subj, body, attachments=None):
            message = EmailMessage()
            message.set_content(body)
            for item in attachments or []:
                message.add_attachment(
                    item.path.read_bytes(),
                    maintype=item.mime.split("/")[0],
                    subtype=item.mime.split("/")[1],
                    filename=item.original_name,
                )
            captured["message"] = message
            return "<id>"

        G._require_credentials = lambda: ("me@gmail.com", "pw", "AI Agent")
        G._smtp_send = fake

        pdf = U.store_upload("a.pdf", io.BytesIO(b"%PDF-1.4"), "alice")
        csv = U.store_upload("b.csv", io.BytesIO(b"x,y\n1,2"), "alice")

        G._send_email(
            "them@university.ac.in", "Files", "body",
            confirmed=True, user_id="alice", file_ids=[pdf.file_id, csv.file_id],
        )

        types = {
            part.get_filename(): part.get_content_type()
            for part in captured["message"].iter_attachments()
        }

        assert types["a.pdf"] == "application/pdf"
        assert types["b.csv"] == "text/csv"