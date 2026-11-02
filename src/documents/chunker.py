"""
Document chunker — splits page-aware text into overlapping chunks for embedding.

Uses LangChain's RecursiveCharacterTextSplitter with sensible defaults:
  - chunk_size=1000 chars
  - chunk_overlap=150 chars

These defaults keep each chunk well within the token budget for
text-embedding-3-small while maintaining enough overlap to preserve context
across chunk boundaries.
"""
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.logger import get_logger

logger = get_logger(__name__)

_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 150

# Module-level splitter instance — stateless, safe to reuse across calls.
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=_CHUNK_SIZE,
    chunk_overlap=_CHUNK_OVERLAP,
    length_function=len,
    # Prefer splitting on paragraph → sentence → word boundaries
    separators=["\n\n", "\n", ". ", " ", ""],
)


@dataclass(frozen=True)
class DocumentChunk:
    """A chunk and the source metadata required to cite it."""

    text: str
    chunk_index: int
    page: int | None = None


def chunk_units(units) -> list[DocumentChunk]:
    """Split parsed units without ever crossing a PDF page boundary."""
    chunks: list[DocumentChunk] = []
    for unit in units:
        if not unit.text or not unit.text.strip():
            continue
        for text in _splitter.split_text(unit.text):
            if text.strip():
                chunks.append(
                    DocumentChunk(
                        text=text.strip(),
                        chunk_index=len(chunks),
                        page=unit.page,
                    )
                )

    logger.debug(
        "Chunked units: units=%s chunks=%s avg_chunk_chars=%s",
        len(units),
        len(chunks),
        round(sum(len(chunk.text) for chunk in chunks) / len(chunks)) if chunks else 0,
    )
    return chunks
