"""Integration tests against a real PostgreSQL 16 (testcontainers).

The unit tests mock asyncpg, so they cannot catch SQL errors, a broken
schema bootstrap, or a wrong ON CONFLICT / CASCADE. These tests run the real
SQL. They need Docker and `pip install -r requirements-dev.lock`, and skip
when either is missing.

Run only these:
    python -m unittest discover -s tests/integration -t . -v
"""
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("JWT_SECRET", "test-jwt-secret")

import asyncpg

from src.database.schema import ensure_postgres_schema
from src.documents import embedder
from src.documents.chunker import DocumentChunk
from src.routes.documents import _find_duplicate

try:
    from testcontainers.community.postgres import PostgresContainer
except ImportError:  # requirements-dev.txt not installed
    PostgresContainer = None


def _docker_available() -> bool:
    try:
        import docker

        docker.from_env().ping()
        return True
    except Exception:
        return False


@unittest.skipUnless(
    PostgresContainer is not None and _docker_available(),
    "needs Docker and testcontainers (pip install -r requirements-dev.lock)",
)
class PostgresIntegrationTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.container = PostgresContainer("postgres:16-alpine")
        cls.container.start()

    @classmethod
    def tearDownClass(cls):
        cls.container.stop()

    async def asyncSetUp(self):
        c = self.container
        self.pool = await asyncpg.create_pool(
            host=c.get_container_host_ip(),
            port=int(c.get_exposed_port(5432)),
            user=c.username,
            password=c.password,
            database=c.dbname,
            min_size=1,
            max_size=2,
        )
        async with self.pool.acquire() as conn:
            await ensure_postgres_schema(conn)
            await conn.execute(
                "TRUNCATE users, sessions, messages, documents, document_chunks CASCADE"
            )

    async def asyncTearDown(self):
        await self.pool.close()

    async def insert_document(self, document_id, status="ready", content_hash="h1"):
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO documents (id, user_id, session_id, filename, status, content_hash)
                VALUES ($1, 'user-1', 'session-1', 'report.pdf', $2, $3)
                """,
                document_id,
                status,
                content_hash,
            )

    async def chunk_rows(self, document_id):
        async with self.pool.acquire() as conn:
            return await conn.fetch(
                "SELECT id, chunk_text FROM document_chunks WHERE document_id = $1 ORDER BY chunk_index",
                document_id,
            )

    async def store(self, document_id, texts):
        chunks = [DocumentChunk(text=t, chunk_index=i, page=1) for i, t in enumerate(texts)]
        vectors = [[0.0] * 4 for _ in texts]
        with patch.object(
            embedder.embedding_service, "aembed_documents", AsyncMock(return_value=vectors)
        ):
            return await embedder.embed_and_store(
                chunks, document_id, "user-1", "session-1", "report.pdf",
                db=self.pool, milvus_client=MagicMock(),
            )

    async def test_schema_bootstrap_runs_twice_without_error(self):
        async with self.pool.acquire() as conn:
            await ensure_postgres_schema(conn)
            columns = {
                row["column_name"]
                for row in await conn.fetch(
                    "SELECT column_name FROM information_schema.columns WHERE table_name = 'documents'"
                )
            }

        self.assertTrue({"content_hash", "failure_reason", "object_name"} <= columns)

    async def test_rerun_of_ingestion_replaces_chunks(self):
        await self.insert_document("doc-1")

        await self.store("doc-1", ["one", "two", "three"])
        await self.store("doc-1", ["one v2", "two v2"])  # retry with fewer chunks

        rows = await self.chunk_rows("doc-1")
        self.assertEqual([r["chunk_text"] for r in rows], ["one v2", "two v2"])
        self.assertEqual([r["id"] for r in rows], ["doc-1:0", "doc-1:1"])

    async def test_duplicate_lookup_ignores_failed_documents(self):
        await self.insert_document("doc-failed", status="failed", content_hash="same")
        self.assertIsNone(await _find_duplicate(self.pool, "user-1", "session-1", "same"))

        await self.insert_document("doc-ready", status="ready", content_hash="same")
        row = await _find_duplicate(self.pool, "user-1", "session-1", "same")
        self.assertEqual(row["id"], "doc-ready")

        self.assertIsNone(await _find_duplicate(self.pool, "user-2", "session-1", "same"))

    async def test_deleting_document_removes_its_chunks(self):
        await self.insert_document("doc-1")
        await self.store("doc-1", ["one", "two"])

        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM documents WHERE id = 'doc-1'")

        self.assertEqual(await self.chunk_rows("doc-1"), [])


if __name__ == "__main__":
    unittest.main()
