"""
Document management routes.

POST   /documents/upload               — Upload files to MinIO and enqueue for processing
GET    /documents                      — List documents for a session
GET    /documents/{document_id}/status — Poll document processing status
DELETE /documents/{document_id}        — Delete a document from Milvus + Postgres + MinIO

MinIO and Milvus clients are synchronous, so their calls run in a worker
thread (asyncio.to_thread) to keep the event loop free.
"""
import asyncio
import hashlib
import os
import uuid
from dataclasses import dataclass
from typing import List

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile, status

from src.auth import get_current_user
from src.config.config import MAX_UPLOAD_BYTES, MILVUS_COLLECTION_NAME, MINIO_BUCKET
from src.documents.filters import document_filter
from src.logger import get_logger
from src.rate_limit import get_user_or_ip, limiter
from src.routes.dependencies import get_db, get_milvus, get_redis
from src.storage.minio_client import get_minio_client

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}
_READ_CHUNK_BYTES = 1024 * 1024


@dataclass
class _PreparedUpload:
    filename: str
    ext: str
    content: bytes
    content_hash: str


def _check_extension(filename: str) -> str:
    ext = os.path.splitext(filename.lower())[1]
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"File '{filename}' has unsupported extension '{ext}'. "
                f"Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
            ),
        )
    return ext


async def _read_limited(upload: UploadFile, filename: str, limit: int) -> bytes:
    """Read the upload in chunks and stop as soon as it exceeds *limit* bytes."""
    buffer = bytearray()
    while chunk := await upload.read(_READ_CHUNK_BYTES):
        buffer.extend(chunk)
        if len(buffer) > limit:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"File '{filename}' is larger than the "
                    f"{limit // (1024 * 1024)} MB limit."
                ),
            )
    return bytes(buffer)


async def _prepare_uploads(files: List[UploadFile]) -> list[_PreparedUpload]:
    """Validate every file before any is stored, so a bad file in a batch
    does not leave the earlier files half-registered."""
    prepared = []
    for upload in files:
        filename = upload.filename or "untitled"
        ext = _check_extension(filename)
        content = await _read_limited(upload, filename, MAX_UPLOAD_BYTES)
        if not content:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"File '{filename}' is empty.",
            )
        prepared.append(
            _PreparedUpload(
                filename=filename,
                ext=ext,
                content=content,
                content_hash=hashlib.sha256(content).hexdigest(),
            )
        )
    return prepared


async def _find_duplicate(db, user_id: str, session_id: str, content_hash: str):
    """Return an existing, non-failed document with the same bytes in this session."""
    async with db.acquire() as conn:
        return await conn.fetchrow(
            """
            SELECT id, filename, status
            FROM documents
            WHERE user_id = $1 AND session_id = $2 AND content_hash = $3
              AND status <> 'failed'
            ORDER BY uploaded_at DESC
            LIMIT 1
            """,
            user_id,
            session_id,
            content_hash,
        )


# ---------------------------------------------------------------------------
# POST /documents/upload
# ---------------------------------------------------------------------------
@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
@limiter.limit("10/minute", key_func=get_user_or_ip)
async def upload_documents(
    request: Request,
    files: List[UploadFile],
    session_id: str = Form(..., min_length=1, max_length=128),
    db=Depends(get_db),
    redis_pool=Depends(get_redis),
    current_user=Depends(get_current_user),
):
    """Upload files to MinIO and enqueue them for async processing.

    1. Validates extension, size, and non-emptiness of every file.
    2. Skips files whose bytes already exist (not failed) in this session.
    3. Uploads raw files to MinIO.
    4. Inserts metadata rows with status='queued'.
    5. Enqueues jobs in Redis for ARQ workers.

    Poll GET /documents/{document_id}/status to watch progress.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one file must be provided.",
        )

    prepared = await _prepare_uploads(files)
    minio_client = get_minio_client()
    uploaded = []
    seen_in_request: dict[str, dict] = {}

    for item in prepared:
        # ------------------------------------------------------------------ #
        # 0. Duplicate check (same bytes, same user + session)
        # ------------------------------------------------------------------ #
        duplicate = seen_in_request.get(item.content_hash)
        if duplicate is None:
            row = await _find_duplicate(db, current_user.id, session_id, item.content_hash)
            if row is not None:
                duplicate = {
                    "document_id": row["id"],
                    "filename": row["filename"],
                    "status": row["status"],
                }
        if duplicate is not None:
            logger.info(
                "Skipping duplicate upload: filename=%s existing_document_id=%s",
                item.filename,
                duplicate["document_id"],
            )
            uploaded.append({**duplicate, "duplicate": True})
            continue

        document_id = str(uuid.uuid4())
        object_name = f"{document_id}{item.ext}"

        # ------------------------------------------------------------------ #
        # 1. Upload file to MinIO
        # ------------------------------------------------------------------ #
        try:
            await asyncio.to_thread(
                minio_client.upload_file,
                bucket=MINIO_BUCKET,
                object_name=object_name,
                data=item.content,
            )
        except Exception as exc:
            logger.exception(
                "Failed to upload file to MinIO: filename=%s user_id=%s",
                item.filename,
                current_user.id,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to upload file '{item.filename}' to storage.",
            ) from exc

        # ------------------------------------------------------------------ #
        # 2. Insert metadata row with status='queued'
        # ------------------------------------------------------------------ #
        try:
            async with db.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO documents
                        (id, user_id, session_id, filename, object_name, content_hash, status)
                    VALUES ($1, $2, $3, $4, $5, $6, 'queued')
                    """,
                    document_id,
                    current_user.id,
                    session_id,
                    item.filename,
                    object_name,
                    item.content_hash,
                )
        except Exception as exc:
            logger.exception(
                "Failed to insert document row: filename=%s user_id=%s",
                item.filename,
                current_user.id,
            )
            try:
                await asyncio.to_thread(minio_client.delete_file, MINIO_BUCKET, object_name)
            except Exception:
                logger.warning(
                    "Failed to clean up MinIO object after metadata insert failure: %s",
                    object_name,
                )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to register document.",
            ) from exc

        # ------------------------------------------------------------------ #
        # 3. Enqueue job in Redis
        # ------------------------------------------------------------------ #
        try:
            job = await redis_pool.enqueue_job("process_document", document_id)
            logger.info(
                "Enqueued document for processing: document_id=%s job_id=%s filename=%s user_id=%s",
                document_id,
                job.job_id if job else "unknown",
                item.filename,
                current_user.id,
            )
        except Exception as exc:
            logger.exception(
                "Failed to enqueue document job: document_id=%s filename=%s",
                document_id,
                item.filename,
            )
            try:
                async with db.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE documents
                        SET status = 'failed', failure_reason = 'Failed to enqueue job'
                        WHERE id = $1
                        """,
                        document_id,
                    )
            except Exception:
                logger.exception("Failed to update document status after enqueue failure")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to queue document '{item.filename}' for processing.",
            ) from exc

        entry = {"document_id": document_id, "filename": item.filename, "status": "queued"}
        seen_in_request[item.content_hash] = entry
        uploaded.append({**entry, "duplicate": False})

    return {"uploaded": uploaded}


# ---------------------------------------------------------------------------
# GET /documents/{document_id}/status
# ---------------------------------------------------------------------------
@router.get("/{document_id}/status")
async def get_document_status(
    document_id: str,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Poll document processing status.

    Returns:
        {
            "document_id": str,
            "status": "queued" | "processing" | "ready" | "failed",
            "failure_reason": str | null
        }
    """
    try:
        async with db.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT status, failure_reason
                FROM documents
                WHERE id = $1 AND user_id = $2
                """,
                document_id,
                current_user.id,
            )
    except Exception as exc:
        logger.exception(
            "Failed to fetch document status: document_id=%s user_id=%s",
            document_id,
            current_user.id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document status.",
        ) from exc

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    return {
        "document_id": document_id,
        "status": row["status"],
        "failure_reason": row["failure_reason"],
    }


# ---------------------------------------------------------------------------
# GET /documents
# ---------------------------------------------------------------------------
@router.get("")
async def list_documents(
    session_id: str,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """List all documents uploaded by the current user for a given session."""
    try:
        async with db.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, filename, status, chunk_count, uploaded_at, failure_reason
                FROM documents
                WHERE user_id = $1 AND session_id = $2
                ORDER BY uploaded_at DESC
                """,
                current_user.id,
                session_id,
            )
    except Exception as exc:
        logger.exception(
            "Failed to list documents: user_id=%s session_id=%s",
            current_user.id,
            session_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documents.",
        ) from exc

    return [
        {
            "document_id": row["id"],
            "filename": row["filename"],
            "status": row["status"],
            "chunk_count": row["chunk_count"],
            "failure_reason": row["failure_reason"],
            "uploaded_at": row["uploaded_at"].isoformat() if row["uploaded_at"] else None,
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# DELETE /documents/{document_id}
# ---------------------------------------------------------------------------
@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
async def delete_document(
    document_id: str,
    db=Depends(get_db),
    milvus=Depends(get_milvus),
    current_user=Depends(get_current_user),
):
    """Delete a document from Milvus, then Postgres (cascades to chunks), then MinIO.

    Milvus goes first: if its delete fails, the request fails and the Postgres
    row stays, so the user can retry. The reverse order could leave vectors
    that still answer questions for a document the user thinks is gone.
    """
    async with db.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, object_name FROM documents
            WHERE id = $1 AND user_id = $2
            """,
            document_id,
            current_user.id,
        )
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )
    object_name = row["object_name"]

    # 1. Milvus vectors — must succeed.
    try:
        await asyncio.to_thread(
            milvus.delete,
            collection_name=MILVUS_COLLECTION_NAME,
            filter=document_filter(document_id),
        )
    except Exception as exc:
        logger.exception(
            "Milvus delete failed; keeping document so the delete can be retried: document_id=%s",
            document_id,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to delete document vectors. Please try again.",
        ) from exc

    # 2. Postgres — ON DELETE CASCADE removes document_chunks rows too.
    try:
        async with db.acquire() as conn:
            await conn.execute("DELETE FROM documents WHERE id = $1", document_id)
    except Exception as exc:
        logger.exception(
            "Failed to delete document from Postgres: document_id=%s", document_id
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document.",
        ) from exc

    # 3. MinIO original file — best-effort; an orphan file is not user-visible.
    if object_name:
        try:
            await asyncio.to_thread(
                get_minio_client().delete_file, MINIO_BUCKET, object_name
            )
        except Exception:
            logger.warning(
                "MinIO delete failed (non-fatal): document_id=%s object=%s — file may linger in storage",
                document_id,
                object_name,
            )

    logger.info(
        "Document deleted: document_id=%s user_id=%s", document_id, current_user.id
    )
    return {"deleted": document_id}
