"""HTTP-level tests: FastAPI routes through TestClient.

The app's lifespan does not run (TestClient is not used as a context
manager), so no real Postgres, Redis, Milvus or MinIO is needed. The tests
put fakes on app.state and patch the MinIO client factory. Rate limiting is
switched off, because it needs Redis.
"""
import os
import unittest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("JWT_SECRET", "test-jwt-secret")

import asyncpg
from fastapi.testclient import TestClient

import app as app_module
from src.auth import get_current_user, hash_password
from src.config.settings import settings
from src.rate_limit import limiter
from src.routes import chat as chat_routes
from src.routes import documents as documents_routes
from src.schema.models import CurrentUser

USER = CurrentUser(id="user-1", email="owner@example.com")


class FakeConn:
    def __init__(self):
        self.fetchrow = AsyncMock(return_value=None)
        self.fetch = AsyncMock(return_value=[])
        self.execute = AsyncMock(return_value="OK")

    def executed_sql(self):
        return [" ".join(call.args[0].split()) for call in self.execute.await_args_list]


class FakePool:
    def __init__(self, conn):
        self.conn = conn

    @asynccontextmanager
    async def _acquire(self):
        yield self.conn

    def acquire(self):
        return self._acquire()


class RouteTestCase(unittest.TestCase):
    authenticated = True

    def setUp(self):
        self.app = app_module.app
        self.conn = FakeConn()
        self.redis = MagicMock()
        self.redis.enqueue_job = AsyncMock(return_value=MagicMock(job_id="job-1"))
        self.milvus = MagicMock()
        self.minio = MagicMock()
        self.app.state.db = FakePool(self.conn)
        self.app.state.redis = self.redis
        self.app.state.milvus = self.milvus
        self.app.state.chain = MagicMock()
        self.app.state.title_chain = MagicMock()

        limiter_enabled = limiter.enabled
        limiter.enabled = False
        self.addCleanup(setattr, limiter, "enabled", limiter_enabled)

        minio_patch = patch.object(documents_routes, "get_minio_client", return_value=self.minio)
        minio_patch.start()
        self.addCleanup(minio_patch.stop)

        if self.authenticated:
            self.app.dependency_overrides[get_current_user] = lambda: USER
        self.addCleanup(self.app.dependency_overrides.clear)

        self.client = TestClient(self.app)

    def assert_no_email_in_logs(self, logs, email):
        for line in logs.output:
            self.assertNotIn(email, line.lower())


# ---------------------------------------------------------------------------
# /auth
# ---------------------------------------------------------------------------
class AuthRouteTests(RouteTestCase):
    authenticated = False
    EMAIL = "alice@example.com"

    def test_signup_returns_token_and_logs_no_email(self):
        with self.assertLogs(level="DEBUG") as logs:
            response = self.client.post(
                "/auth/signup", json={"email": "Alice@Example.com", "password": "long-password"}
            )

        self.assertEqual(response.status_code, 201)
        body = response.json()
        self.assertTrue(body["access_token"])
        self.assertEqual(body["user"]["email"], self.EMAIL)
        self.assert_no_email_in_logs(logs, self.EMAIL)

    def test_signup_with_registered_email_returns_409(self):
        self.conn.execute.side_effect = asyncpg.exceptions.UniqueViolationError("duplicate")

        with self.assertLogs(level="DEBUG") as logs:
            response = self.client.post(
                "/auth/signup", json={"email": self.EMAIL, "password": "long-password"}
            )

        self.assertEqual(response.status_code, 409)
        self.assert_no_email_in_logs(logs, self.EMAIL)

    def test_login_with_wrong_password_returns_401_and_logs_no_email(self):
        self.conn.fetchrow.return_value = {
            "id": "user-1",
            "email": self.EMAIL,
            "password_hash": hash_password("the-right-password"),
        }

        with self.assertLogs(level="DEBUG") as logs:
            response = self.client.post(
                "/auth/login", json={"email": self.EMAIL, "password": "the-wrong-password"}
            )

        self.assertEqual(response.status_code, 401)
        self.assert_no_email_in_logs(logs, self.EMAIL)

    def test_login_with_right_password_returns_token(self):
        self.conn.fetchrow.return_value = {
            "id": "user-1",
            "email": self.EMAIL,
            "password_hash": hash_password("the-right-password"),
        }

        response = self.client.post(
            "/auth/login", json={"email": self.EMAIL, "password": "the-right-password"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["user"]["id"], "user-1")

    def test_protected_route_without_token_returns_401(self):
        response = self.client.get("/documents", params={"session_id": "s1"})

        self.assertEqual(response.status_code, 401)


# ---------------------------------------------------------------------------
# /documents/upload
# ---------------------------------------------------------------------------
class UploadRouteTests(RouteTestCase):
    def upload(self, *files):
        return self.client.post(
            "/documents/upload",
            files=[("files", file) for file in files],
            data={"session_id": "session-1"},
        )

    def test_new_file_is_stored_registered_and_enqueued(self):
        response = self.upload(("notes.txt", b"hello world", "text/plain"))

        self.assertEqual(response.status_code, 202)
        [entry] = response.json()["uploaded"]
        self.assertEqual(entry["status"], "queued")
        self.assertFalse(entry["duplicate"])
        self.minio.upload_file.assert_called_once()
        self.assertTrue(
            any(sql.startswith("INSERT INTO documents") for sql in self.conn.executed_sql())
        )
        self.redis.enqueue_job.assert_awaited_once_with("process_document", entry["document_id"])

    def test_unsupported_extension_returns_422_and_stores_nothing(self):
        response = self.upload(("script.exe", b"MZ", "application/octet-stream"))

        self.assertEqual(response.status_code, 422)
        self.minio.upload_file.assert_not_called()
        self.redis.enqueue_job.assert_not_awaited()

    def test_file_over_size_limit_returns_413(self):
        with patch.object(settings, "max_upload_bytes", 10):
            response = self.upload(("big.txt", b"x" * 11, "text/plain"))

        self.assertEqual(response.status_code, 413)
        self.minio.upload_file.assert_not_called()

    def test_bad_file_in_batch_rejects_whole_batch(self):
        response = self.upload(
            ("good.txt", b"fine", "text/plain"),
            ("empty.txt", b"", "text/plain"),
        )

        self.assertEqual(response.status_code, 422)
        self.minio.upload_file.assert_not_called()

    def test_file_already_in_session_is_returned_as_duplicate(self):
        self.conn.fetchrow.return_value = {
            "id": "doc-old",
            "filename": "notes.txt",
            "status": "ready",
        }

        response = self.upload(("notes.txt", b"hello world", "text/plain"))

        self.assertEqual(response.status_code, 202)
        [entry] = response.json()["uploaded"]
        self.assertEqual(entry["document_id"], "doc-old")
        self.assertTrue(entry["duplicate"])
        self.minio.upload_file.assert_not_called()
        self.redis.enqueue_job.assert_not_awaited()

    def test_same_file_twice_in_one_request_is_processed_once(self):
        response = self.upload(
            ("a.txt", b"same bytes", "text/plain"),
            ("b.txt", b"same bytes", "text/plain"),
        )

        first, second = response.json()["uploaded"]
        self.assertFalse(first["duplicate"])
        self.assertTrue(second["duplicate"])
        self.assertEqual(first["document_id"], second["document_id"])
        self.redis.enqueue_job.assert_awaited_once()

    def test_enqueue_failure_marks_document_failed(self):
        self.redis.enqueue_job.side_effect = ConnectionError("redis down")

        response = self.upload(("notes.txt", b"hello world", "text/plain"))

        self.assertEqual(response.status_code, 500)
        self.assertTrue(
            any("SET status = 'failed'" in sql for sql in self.conn.executed_sql())
        )


# ---------------------------------------------------------------------------
# /documents/{id} and /documents/{id}/status
# ---------------------------------------------------------------------------
class DocumentRouteTests(RouteTestCase):
    def test_status_of_unknown_or_foreign_document_returns_404(self):
        response = self.client.get("/documents/doc-1/status")

        self.assertEqual(response.status_code, 404)
        # The query is scoped to the current user.
        self.assertEqual(self.conn.fetchrow.await_args.args[1:], ("doc-1", USER.id))

    def test_delete_unknown_document_returns_404(self):
        response = self.client.delete("/documents/doc-1")

        self.assertEqual(response.status_code, 404)
        self.milvus.delete.assert_not_called()

    def test_delete_keeps_row_when_milvus_delete_fails(self):
        self.conn.fetchrow.return_value = {"id": "doc-1", "object_name": "doc-1.pdf"}
        self.milvus.delete.side_effect = ConnectionError("milvus down")

        response = self.client.delete("/documents/doc-1")

        self.assertEqual(response.status_code, 502)
        self.assertFalse(
            any(sql.startswith("DELETE FROM documents") for sql in self.conn.executed_sql())
        )
        self.minio.delete_file.assert_not_called()

    def test_delete_runs_milvus_then_postgres_then_minio(self):
        self.conn.fetchrow.return_value = {"id": "doc-1", "object_name": "doc-1.pdf"}
        order = []
        self.milvus.delete.side_effect = lambda **_: order.append("milvus")

        async def execute(sql, *args):
            order.append("postgres")
            return "DELETE 1"

        self.conn.execute.side_effect = execute
        self.minio.delete_file.side_effect = lambda *_: order.append("minio")

        response = self.client.delete("/documents/doc-1")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(order, ["milvus", "postgres", "minio"])

    def test_minio_delete_failure_does_not_fail_request(self):
        self.conn.fetchrow.return_value = {"id": "doc-1", "object_name": "doc-1.pdf"}
        self.minio.delete_file.side_effect = RuntimeError("minio down")

        response = self.client.delete("/documents/doc-1")

        self.assertEqual(response.status_code, 200)


# ---------------------------------------------------------------------------
# /chat
# ---------------------------------------------------------------------------
class ChatRouteTests(RouteTestCase):
    def test_foreign_session_returns_404(self):
        foreign = {"exists": False, "foreign": True, "title": None, "history": []}
        with patch.object(
            chat_routes, "get_session_context_from_db", AsyncMock(return_value=foreign)
        ), patch.object(chat_routes, "get_answer", AsyncMock()) as get_answer:
            response = self.client.post(
                "/chat", json={"session_id": "s1", "user_query": "hello"}
            )

        self.assertEqual(response.status_code, 404)
        get_answer.assert_not_awaited()

    def test_model_failure_returns_502(self):
        context = {"exists": True, "foreign": False, "title": "t", "history": []}
        with patch.object(
            chat_routes, "get_session_context_from_db", AsyncMock(return_value=context)
        ), patch.object(
            chat_routes, "get_answer", AsyncMock(side_effect=TimeoutError("llm"))
        ):
            response = self.client.post(
                "/chat", json={"session_id": "s1", "user_query": "hello"}
            )

        self.assertEqual(response.status_code, 502)


if __name__ == "__main__":
    unittest.main()
