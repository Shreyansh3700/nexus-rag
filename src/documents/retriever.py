"""Hybrid document retrieval using Milvus dense vectors, BM25, and RRF.

MilvusClient is synchronous, so every Milvus call runs in a worker thread
(asyncio.to_thread) to keep the FastAPI event loop free.
"""
import asyncio
import re
from dataclasses import dataclass

from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker

from src.config.config import (
    MILVUS_COLLECTION_NAME,
    MILVUS_HYBRID_CANDIDATE_K,
    MILVUS_RRF_K,
    MILVUS_TOP_K,
    RERANKER_CANDIDATE_K,
    RERANKER_ENABLED,
    RERANKER_TOP_K,
    RETRIEVAL_MAX_COSINE_DISTANCE,
)
from src.documents.filters import milvus_filter_value
from src.documents.reranker import rerank_chunks
from src.logger import get_logger
from src.services.embedding_service import embedding_service

logger = get_logger(__name__)


@dataclass(frozen=True)
class RetrievalResult:
    """LLM context plus the exact source metadata used to build it."""

    context: str
    sources: list[dict]


_CITATION_PATTERN = re.compile(r"\\?\[\^?(\d+(?:\s*,\s*\d+)*)\\?]")


def _dedupe_by_page(sources: list[dict]) -> list[dict]:
    """Collapse multiple chunk citations that point at the same
    (filename, page) into a single entry, keeping the first (lowest
    citation number) occurrence. Several retrieved chunks commonly come
    from the same page, which would otherwise list that page once per chunk."""
    seen = set()
    deduped = []
    for source in sources:
        key = (source["filename"], source["page"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(source)
    return deduped


def cited_sources(answer: str, sources: list[dict]) -> list[dict]:
    """Return sources explicitly cited as ``[n]`` in the answer.

    Tolerates the formatting variants models commonly produce alongside
    markdown output: comma-separated citations (``[1, 2]``), adjacent
    brackets (``[1][2]``), markdown-escaped brackets (``\\[1\\]``), and
    footnote-style markers (``[^1]``). If the model retrieved context but
    tagged nothing explicitly, falls back to all retrieved sources rather
    than losing citation/page data entirely. Either way, sources that share
    the same (filename, page) are collapsed to one entry.
    """
    cited_numbers = {
        int(number)
        for group in _CITATION_PATTERN.findall(answer)
        for number in re.split(r"\s*,\s*", group)
    }
    explicitly_cited = [source for source in sources if source["citation"] in cited_numbers]
    return _dedupe_by_page(explicitly_cited or sources)


def _format_context(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    lines = []
    for index, chunk in enumerate(chunks, start=1):
        page = (
            f"PDF page: {chunk['page']}"
            if chunk["page"] is not None
            else "PDF page: unavailable"
        )
        lines.append(
            f"[{index}] Source: {chunk['filename']}\n{page}\n\n{chunk['text'].strip()}"
        )
    return (
        "Relevant document context (use this to answer the user's question):\n\n"
        + "\n\n".join(lines)
    )


def _scope_filter(user_id: str, session_id: str) -> str:
    return (
        f'user_id == "{milvus_filter_value(user_id)}" '
        f'and session_id == "{milvus_filter_value(session_id)}"'
    )


def _has_chunks(milvus_client: MilvusClient, filter_expr: str) -> bool:
    rows = milvus_client.query(
        collection_name=MILVUS_COLLECTION_NAME,
        filter=filter_expr,
        output_fields=["chunk_id"],
        limit=1,
    )
    return bool(rows)


def _hybrid_search(
    milvus_client: MilvusClient,
    query: str,
    query_vector: list[float],
    filter_expr: str,
    top_k: int,
) -> list[dict]:
    candidate_k = max(top_k, MILVUS_HYBRID_CANDIDATE_K)
    # radius/range_filter is Milvus's server-side range search: for COSINE
    # (larger score = more similar), only candidates with
    # radius < similarity <= range_filter survive the ANN search on this
    # leg, before RRF fusion ever sees them.
    min_similarity = 1.0 - RETRIEVAL_MAX_COSINE_DISTANCE
    dense_request = AnnSearchRequest(
        data=[query_vector],
        anns_field="dense_vector",
        param={
            "metric_type": "COSINE",
            "params": {"nprobe": 16, "radius": min_similarity, "range_filter": 1.0},
        },
        limit=candidate_k,
        expr=filter_expr,
    )
    sparse_request = AnnSearchRequest(
        data=[query],
        anns_field="sparse_vector",
        param={"metric_type": "BM25"},
        limit=candidate_k,
        expr=filter_expr,
    )
    results = milvus_client.hybrid_search(
        collection_name=MILVUS_COLLECTION_NAME,
        reqs=[dense_request, sparse_request],
        ranker=RRFRanker(k=MILVUS_RRF_K),
        limit=top_k,
        output_fields=["text", "filename", "document_id", "chunk_index", "page"],
    )
    hits = results[0] if results else []
    return [
        {
            "text": entity["text"],
            "filename": entity.get("filename", "Unknown document"),
            "page": entity.get("page"),
            "chunk": entity.get("chunk_index", 0),
        }
        for hit in hits
        if (entity := hit.get("entity", {})).get("text")
    ]


async def retrieve_context(
    query: str,
    session_id: str,
    user_id: str,
    milvus_client: MilvusClient,
    top_k: int = MILVUS_TOP_K,
) -> RetrievalResult:
    """Retrieve hybrid-ranked context from the active session's documents only."""
    if not query.strip():
        return RetrievalResult(context="", sources=[])

    filter_expr = _scope_filter(user_id, session_id)
    try:
        if not await asyncio.to_thread(_has_chunks, milvus_client, filter_expr):
            logger.info(
                "SOURCES-TRACE: no indexed chunks for this session — sources will be empty: "
                "user_id=%s session_id=%s",
                user_id,
                session_id,
            )
            return RetrievalResult(context="", sources=[])

        logger.debug(
            "Running hybrid retrieval: user_id=%s session_id=%s candidates=%s "
            "rerank_enabled=%s max_cosine_distance=%s",
            user_id,
            session_id,
            RERANKER_CANDIDATE_K
            if RERANKER_ENABLED
            else max(top_k, MILVUS_HYBRID_CANDIDATE_K),
            RERANKER_ENABLED,
            RETRIEVAL_MAX_COSINE_DISTANCE,
        )
        query_vector = await embedding_service.aembed_query(query)
        retrieval_k = RERANKER_CANDIDATE_K if RERANKER_ENABLED else top_k
        chunks = await asyncio.to_thread(
            _hybrid_search,
            milvus_client=milvus_client,
            query=query,
            query_vector=query_vector,
            filter_expr=filter_expr,
            top_k=retrieval_k,
        )
        if RERANKER_ENABLED:
            try:
                chunks = await rerank_chunks(
                    query=query,
                    chunks=chunks,
                    top_k=min(RERANKER_TOP_K, top_k),
                )
            except Exception:
                logger.warning(
                    "Reranking failed; using hybrid-ranked candidates: user_id=%s session_id=%s",
                    user_id,
                    session_id,
                    exc_info=True,
                )
                chunks = chunks[:top_k]
        logger.info(
            "Hybrid retrieval: user_id=%s session_id=%s chunks_returned=%s reranked=%s",
            user_id,
            session_id,
            len(chunks),
            RERANKER_ENABLED,
        )
        sources = [
            {
                "filename": chunk["filename"],
                "page": chunk["page"],
                "chunk": chunk["chunk"],
                "citation": index,
            }
            for index, chunk in enumerate(chunks, start=1)
        ]
        logger.info(
            "SOURCES-TRACE: retrieval built sources: user_id=%s session_id=%s sources_count=%s sources=%s",
            user_id,
            session_id,
            len(sources),
            sources,
        )
        return RetrievalResult(context=_format_context(chunks), sources=sources)
    except Exception:
        logger.warning(
            "SOURCES-TRACE: hybrid retrieval raised an exception — falling back to "
            "empty context/sources: user_id=%s session_id=%s",
            user_id,
            session_id,
            exc_info=True,
        )
        return RetrievalResult(context="", sources=[])
