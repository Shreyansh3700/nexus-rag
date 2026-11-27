"""PostgreSQL schema bootstrap.

ensure_postgres_schema() creates every table and index the app needs. Each
statement uses IF NOT EXISTS, so it is safe to run on every API startup and
in integration tests. It only adds missing tables, columns and indexes; it
does not change existing ones (that needs a migration tool such as Alembic).
"""
import asyncpg


async def ensure_postgres_schema(connection: asyncpg.Connection) -> None:
    # --- existing tables ---
    await connection.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT,
            title TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS messages (
            id BIGSERIAL PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
            role TEXT NOT NULL CHECK (role IN ('Human', 'AI')),
            content TEXT NOT NULL,
            sources JSONB NOT NULL DEFAULT '[]'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS messages_session_id_id_idx
            ON messages (session_id, id);
        CREATE INDEX IF NOT EXISTS sessions_user_id_idx
            ON sessions (user_id);
        """
    )
    await connection.execute(
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS user_id TEXT"
    )
    await connection.execute(
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS sources JSONB NOT NULL DEFAULT '[]'::jsonb"
    )

    # --- RAG: document metadata table ---
    await connection.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            chunk_count INT NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'processing'
        );
        CREATE INDEX IF NOT EXISTS documents_user_session_idx
            ON documents (user_id, session_id);
        """
    )
    # Add new columns for queue-based architecture
    await connection.execute(
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS object_name TEXT"
    )
    await connection.execute(
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS failure_reason TEXT"
    )
    # SHA-256 of the file bytes, used to skip duplicate uploads.
    await connection.execute(
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_hash TEXT"
    )
    await connection.execute(
        """
        CREATE INDEX IF NOT EXISTS documents_user_session_hash_idx
            ON documents (user_id, session_id, content_hash)
        """
    )

    # --- RAG: chunk persistence table (Milvus fallback) ---
    await connection.execute(
        """
        CREATE TABLE IF NOT EXISTS document_chunks (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            user_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            chunk_index INT NOT NULL,
            chunk_text TEXT NOT NULL,
            page INT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS document_chunks_session_id_idx
            ON document_chunks (session_id);
        CREATE INDEX IF NOT EXISTS document_chunks_document_id_idx
            ON document_chunks (document_id);
        """
    )
    await connection.execute(
        "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS page INT"
    )
