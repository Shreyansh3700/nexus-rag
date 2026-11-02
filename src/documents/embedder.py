"""
Embedding service — embeds text chunks with OpenAI and writes them to Milvus
and PostgreSQL.

Writes are idempotent so a retried job never duplicates chunks:
  1. Embed all chunks (batched by the embedding client).
  2. In one Postgres transaction: delete this document's old chunk rows, then
     insert the new ones.
  3. In Milvus: delete this document's old vectors, then upsert the new ones.

Chunk IDs are deterministic (``{document_id}:{chunk_index}``), so the same
chunk always maps to the same primary key in both stores.

Milvus calls are synchronous, so they run in a worker thread to keep the
event loop free.
"""
import asyncio

from pymilvus import MilvusClient

from src.config.config import MILVUS_COLLECTION_NAME
from src.documents.chunker import DocumentChunk
from src.documents.filters import document_filter
from src.logger import get_logger
from src.services.embedding_service import embedding_service

logger = get_logger(__name__)

# Milvus VARCHAR field character limit (must match the schema in config.py)
_TEXT_MAX_LEN = 4096


def _truncate(text: str, max_len: int = _TEXT_MAX_LEN) -> str:
    """Truncate text to fit within the Milvus VARCHAR field limit."""
    return text[:max_len] if len(text) > max_len else text


def chunk_id(document_id: str, chunk_index: int) -> str:
    """Deterministic primary key for a chunk (fits Milvus VARCHAR(64))."""
    return f"{document_id}:{chunk_index}"


async def embed_and_store(
    chunks: list[DocumentChunk],
    document_id: str,
    user_id: str,
    session_id: str,
    filename: str,
    db,
    milvus_client: MilvusClient,
) -> int:
    """Embed *chunks* and write them to both Milvus and Postgres.

    Args:
        chunks:       Page-aware chunks produced by the chunker.
        document_id:  UUID of the parent document row in Postgres.
        user_id:      ID of the owning user (for scoping / filtering).
        session_id:   ID of the chat session (for scoping / filtering).
        filename:     Original filename (stored as source metadata in Milvus).
        db:           asyncpg connection pool.
        milvus_client: Connected MilvusClient instance.

    Returns:
        Number of chunks successfully stored.

    Raises:
        RuntimeError: If the embedding API call or either write fails.
    """
    if not chunks:
        logger.warning(
            "embed_and_store called with empty chunks list: document_id=%s", document_id
        )
        return 0

    logger.info(
        "Embedding %s chunks: document_id=%s user_id=%s session_id=%s",
        len(chunks),
        document_id,
        user_id,
        session_id,
    )

    # ------------------------------------------------------------------
    # 1. Embed all chunks in one batched call
    # ------------------------------------------------------------------
    try:
        vectors = await embedding_service.aembed_documents([chunk.text for chunk in chunks])
    except Exception as exc:
        logger.exception(
            "OpenAI embedding failed: document_id=%s chunks=%s", document_id, len(chunks)
        )
        raise RuntimeError(f"Embedding failed: {exc}") from exc

    # ------------------------------------------------------------------
    # 2. Bulk-insert into Postgres document_chunks
    # ------------------------------------------------------------------
    chunk_ids = [chunk_id(document_id, chunk.chunk_index) for chunk in chunks]

    pg_rows = [
        (
            chunk_ids[i],
            document_id,
            user_id,
            session_id,
            chunk.chunk_index,
            chunk.text,
            chunk.page,
        )
        for i, chunk in enumerate(chunks)
    ]

    try:
        async with db.acquire() as conn:
            async with conn.transaction():
                # Remove rows from any earlier, partial attempt first.
                await conn.execute(
                    "DELETE FROM document_chunks WHERE document_id = $1",
                    document_id,
                )
                await conn.executemany(
                    """
                    INSERT INTO document_chunks
                        (id, document_id, user_id, session_id, chunk_index, chunk_text, page)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (id) DO UPDATE SET
                        chunk_text = EXCLUDED.chunk_text,
                        page = EXCLUDED.page
                    """,
                    pg_rows,
                )
        logger.debug(
            "Postgres chunk rows replaced: document_id=%s count=%s",
            document_id,
            len(pg_rows),
        )
    except Exception as exc:
        logger.exception(
            "Postgres chunk insert failed: document_id=%s", document_id
        )
        raise RuntimeError(f"Postgres chunk insert failed: {exc}") from exc

    # ------------------------------------------------------------------
    # 3. Upsert vectors into Milvus
    # ------------------------------------------------------------------
    milvus_rows = [
        {
            "chunk_id": chunk_ids[i],
            "document_id": document_id,
            "user_id": user_id,
            "session_id": session_id,
            "chunk_index": chunk.chunk_index,
            "page": chunk.page,
            "filename": _truncate(filename, 512),
            "text": _truncate(chunk.text),
            "dense_vector": vectors[i],
        }
        for i, chunk in enumerate(chunks)
    ]

    try:
        # Remove vectors from any earlier, partial attempt, then upsert.
        await asyncio.to_thread(
            milvus_client.delete,
            collection_name=MILVUS_COLLECTION_NAME,
            filter=document_filter(document_id),
        )
        await asyncio.to_thread(
            milvus_client.upsert,
            collection_name=MILVUS_COLLECTION_NAME,
            data=milvus_rows,
        )
        logger.info(
            "Milvus upsert complete: document_id=%s chunks=%s",
            document_id,
            len(milvus_rows),
        )
    except Exception as exc:
        logger.exception(
            "Milvus upsert failed: document_id=%s — job will retry",
            document_id,
        )
        raise RuntimeError(f"Milvus upsert failed: {exc}") from exc

    return len(chunks)
