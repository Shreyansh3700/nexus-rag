"""
Ingestion pipeline — parse → chunk → embed → store for one uploaded document.

Called by the ARQ task in src/workers/tasks.py, which owns the document's
status lifecycle. This module only does the work and raises on failure:

  - PermanentIngestionError: the file can never succeed (unsupported type,
    corrupt file, no extractable text). The worker does not retry.
  - Any other exception: treated as transient (OpenAI, Postgres, or Milvus
    errors). The worker retries it.

Writes are idempotent (see embedder.py), so a retry after a partial failure
does not duplicate chunks.
"""
from src.documents.chunker import chunk_units
from src.documents.embedder import embed_and_store
from src.documents.errors import PermanentIngestionError
from src.documents.parser import parse_file
from src.logger import get_logger

logger = get_logger(__name__)


async def ingest_document(
    document_id: str,
    user_id: str,
    session_id: str,
    filename: str,
    content: bytes,
    db,
    milvus_client,
) -> int:
    """Parse, chunk, embed, and store a document.

    Returns:
        Number of chunks stored.

    Raises:
        PermanentIngestionError: The document cannot be ingested; do not retry.
        Exception: Any other error is transient; the caller may retry.
    """
    logger.info(
        "Ingestion started: document_id=%s filename=%s user_id=%s session_id=%s",
        document_id,
        filename,
        user_id,
        session_id,
    )

    # 1. Parse raw bytes → page-aware source units. Parsing is deterministic,
    #    so any parse error (unsupported type, corrupt PDF/DOCX) is permanent.
    try:
        units = parse_file(filename, content)
    except ValueError as exc:
        raise PermanentIngestionError(str(exc)) from exc
    except Exception as exc:
        raise PermanentIngestionError(f"Could not parse file: {exc}") from exc

    if not units:
        raise PermanentIngestionError(
            "No extractable text found (the file may be empty or a scanned image)."
        )

    # 2. Chunk units independently so PDF chunks never cross pages.
    chunks = chunk_units(units)
    if not chunks:
        raise PermanentIngestionError("Chunker produced zero chunks.")

    # 3. Embed chunks and write to Postgres + Milvus (idempotent).
    count = await embed_and_store(
        chunks=chunks,
        document_id=document_id,
        user_id=user_id,
        session_id=session_id,
        filename=filename,
        db=db,
        milvus_client=milvus_client,
    )

    logger.info(
        "Ingestion complete: document_id=%s filename=%s chunks=%s",
        document_id,
        filename,
        count,
    )
    return count
