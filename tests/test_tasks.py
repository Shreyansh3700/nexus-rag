import os
import unittest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("JWT_SECRET", "test-jwt-secret")

from arq import Retry

from src.config.config import ARQ_MAX_TRIES
from src.documents.errors import PermanentIngestionError
from src.workers import tasks

DOC_ROW = {
    "user_id": "user-1",
    "session_id": "session-1",
    "filename": "report.pdf",
    "object_name": "doc-1.pdf",
}


class FakeDb:
    """Minimal asyncpg pool stand-in that records status updates."""

    def __init__(self, row=DOC_ROW, update_result="UPDATE 1"):
        self.conn = MagicMock()
        self.conn.fetchrow = AsyncMock(return_value=row)
        self.conn.execute = AsyncMock(return_value=update_result)

    @asynccontextmanager
    async def _acquire(self):
        yield self.conn

    def acquire(self):
        return self._acquire()

    def statuses(self):
        """(status, reason, chunk_count) for every UPDATE made by _set_status."""
        return [call.args[2:5] for call in self.conn.execute.await_args_list]


def make_ctx(db, job_try=1):
    minio = MagicMock()
    minio.download_file.return_value = b"%PDF bytes"
    return {"db": db, "milvus": MagicMock(), "minio": minio, "job_try": job_try}


class ProcessDocumentTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_marks_ready_with_chunk_count(self):
        db = FakeDb()
        with patch.object(tasks, "ingest_document", AsyncMock(return_value=7)):
            await tasks.process_document(make_ctx(db), "doc-1")

        self.assertEqual(db.statuses()[-1], ("ready", None, 7))

    async def test_permanent_error_fails_without_retry(self):
        db = FakeDb()
        error = PermanentIngestionError("No extractable text found")
        with patch.object(tasks, "ingest_document", AsyncMock(side_effect=error)):
            await tasks.process_document(make_ctx(db), "doc-1")

        status, reason, _ = db.statuses()[-1]
        self.assertEqual(status, "failed")
        self.assertIn("No extractable text", reason)

    async def test_transient_error_requeues_and_raises_retry(self):
        db = FakeDb()
        with patch.object(tasks, "ingest_document", AsyncMock(side_effect=TimeoutError("openai"))):
            with self.assertRaises(Retry):
                await tasks.process_document(make_ctx(db, job_try=1), "doc-1")

        status, reason, _ = db.statuses()[-1]
        self.assertEqual(status, "queued")
        self.assertIn("Retrying after attempt 1", reason)
        self.assertNotIn("failed", [s[0] for s in db.statuses()])

    async def test_transient_error_on_last_attempt_marks_failed(self):
        db = FakeDb()
        with patch.object(tasks, "ingest_document", AsyncMock(side_effect=TimeoutError("openai"))):
            await tasks.process_document(make_ctx(db, job_try=ARQ_MAX_TRIES), "doc-1")

        status, reason, _ = db.statuses()[-1]
        self.assertEqual(status, "failed")
        self.assertIn(f"Failed after {ARQ_MAX_TRIES} attempts", reason)

    async def test_missing_document_is_skipped(self):
        db = FakeDb(row=None)
        ingest = AsyncMock()
        with patch.object(tasks, "ingest_document", ingest):
            await tasks.process_document(make_ctx(db), "doc-1")

        ingest.assert_not_awaited()

    async def test_vectors_removed_when_document_deleted_mid_processing(self):
        db = FakeDb(update_result="UPDATE 0")
        ctx = make_ctx(db)
        with patch.object(tasks, "ingest_document", AsyncMock(return_value=3)):
            await tasks.process_document(ctx, "doc-1")

        ctx["milvus"].delete.assert_called_once()
        self.assertIn('"doc-1"', ctx["milvus"].delete.call_args.kwargs["filter"])

    async def test_status_update_error_does_not_delete_vectors(self):
        db = FakeDb()
        ctx = make_ctx(db)
        db.conn.execute.side_effect = [
            "UPDATE 1",  # processing
            ConnectionError("db down"),  # ready
        ]
        with patch.object(tasks, "ingest_document", AsyncMock(return_value=3)):
            await tasks.process_document(ctx, "doc-1")

        ctx["milvus"].delete.assert_not_called()


if __name__ == "__main__":
    unittest.main()
