"""ARQ task function for the document ingestion pipeline.

`process_document` owns the document status lifecycle:

    queued → processing → ready
                        → failed                    (permanent error, or last attempt)
                        → queued (+ retry reason)   (transient error, retry scheduled)

Permanent errors (PermanentIngestionError) fail at once. Any other error is
transient: the task schedules a retry with exponential backoff via arq.Retry
until ARQ_MAX_TRIES attempts are used, then marks the document failed.
Writes in the pipeline are idempotent, so retries never duplicate chunks.
"""
import asyncio
import time

from arq import Retry

from src.config.config import ARQ_MAX_TRIES, MILVUS_COLLECTION_NAME, MINIO_BUCKET
from src.documents.errors import PermanentIngestionError
from src.documents.filters import document_filter
from src.documents.ingestor import ingest_document
from src.logger import bind_user_id, get_logger
from src.storage.minio_client import get_minio_client

logger = get_logger(__name__)

_REASON_MAX_LEN = 500


def retry_delay_seconds(job_try: int) -> int:
    """Exponential backoff: 2s after attempt 1, 4s after attempt 2, ..."""
    return 2 ** job_try


async def process_document(ctx: dict, document_id: str) -> None:
    """Download a queued document from MinIO and run the ingestion pipeline.

    Args:
        ctx: ARQ worker context. Needs `db` (asyncpg pool) and `milvus`
            (MilvusClient); ARQ adds `job_try` (1-based attempt number).
        document_id: UUID of the document to process.

    Raises:
        arq.Retry: On a transient error when attempts remain.
    """
    start_time = time.perf_counter()
    db = ctx["db"]
    milvus_client = ctx["milvus"]
    minio_client = ctx.get("minio") or get_minio_client()
    job_try = int(ctx.get("job_try", 1))

    logger.info(
        "ARQ task started: document_id=%s attempt=%s/%s",
        document_id,
        job_try,
        ARQ_MAX_TRIES,
    )

    async with db.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT user_id, session_id, filename, object_name
            FROM documents
            WHERE id = $1
            """,
            document_id,
        )

    if row is None:
        # Deleted while queued — nothing to do, and nothing to retry.
        logger.warning("Document no longer exists, skipping: document_id=%s", document_id)
        return

    user_id = row["user_id"]
    if not row["object_name"]:
        await _set_status(db, document_id, "failed", reason="Missing MinIO object reference")
        return

    with bind_user_id(user_id):
        await _set_status(db, document_id, "processing", reason=None)

        try:
            content = await asyncio.to_thread(
                minio_client.download_file, MINIO_BUCKET, row["object_name"]
            )
            count = await ingest_document(
                document_id=document_id,
                user_id=user_id,
                session_id=row["session_id"],
                filename=row["filename"],
                content=content,
                db=db,
                milvus_client=milvus_client,
            )
        except PermanentIngestionError as exc:
            logger.warning(
                "Permanent ingestion failure, not retrying: document_id=%s reason=%s",
                document_id,
                exc,
            )
            await _set_status(db, document_id, "failed", reason=str(exc))
            return
        except Exception as exc:
            if job_try < ARQ_MAX_TRIES:
                delay = retry_delay_seconds(job_try)
                logger.warning(
                    "Transient ingestion failure, retrying in %ss: document_id=%s attempt=%s/%s",
                    delay,
                    document_id,
                    job_try,
                    ARQ_MAX_TRIES,
                    exc_info=True,
                )
                await _set_status(
                    db,
                    document_id,
                    "queued",
                    reason=f"Retrying after attempt {job_try}/{ARQ_MAX_TRIES}: {exc}",
                )
                raise Retry(defer=delay) from exc

            logger.exception(
                "Ingestion failed on final attempt: document_id=%s attempts=%s",
                document_id,
                job_try,
            )
            await _set_status(
                db,
                document_id,
                "failed",
                reason=f"Failed after {job_try} attempts: {exc}",
            )
            return

        still_exists = await _set_status(
            db, document_id, "ready", reason=None, chunk_count=count
        )
        if still_exists is False:
            # The user deleted the document while it was being processed.
            # Remove the vectors this run just wrote so they cannot leak into
            # retrieval for the session. Postgres chunks cascade-deleted already.
            logger.warning(
                "Document deleted during processing, removing its vectors: document_id=%s",
                document_id,
            )
            await asyncio.to_thread(
                milvus_client.delete,
                collection_name=MILVUS_COLLECTION_NAME,
                filter=document_filter(document_id),
            )
            return

        logger.info(
            "Document processing complete: document_id=%s chunks=%s elapsed=%.2fs",
            document_id,
            count,
            time.perf_counter() - start_time,
        )


async def _set_status(
    db,
    document_id: str,
    status: str,
    reason: str | None,
    chunk_count: int | None = None,
) -> bool | None:
    """Update status, failure_reason, and (optionally) chunk_count.

    Returns:
        True if the row was updated, False if the row no longer exists,
        None if the update itself failed (state unknown).
    """
    if reason is not None:
        reason = reason[:_REASON_MAX_LEN]
    try:
        async with db.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE documents
                SET status = $2,
                    failure_reason = $3,
                    chunk_count = COALESCE($4, chunk_count)
                WHERE id = $1
                """,
                document_id,
                status,
                reason,
                chunk_count,
            )
    except Exception:
        logger.exception(
            "Failed to update document status: document_id=%s status=%s",
            document_id,
            status,
        )
        return None

    logger.debug(
        "Document status updated: document_id=%s status=%s reason=%s",
        document_id,
        status,
        (reason or "")[:100],
    )
    # asyncpg returns a command tag such as "UPDATE 1".
    return str(result).rsplit(" ", 1)[-1] != "0"
