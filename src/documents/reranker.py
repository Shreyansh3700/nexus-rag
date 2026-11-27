"""Cross-encoder reranking for hybrid-retrieval candidates.

Two modes, selected by RERANKER_URL:

* Remote (RERANKER_URL set; the Docker setup): send the query and the
  candidate texts to the reranker service (src/reranker_service.py) and get
  one score per text. The API process never imports torch or loads the
  model, so API workers stay small and a slow rerank cannot block them.
* Local (RERANKER_URL unset; local development only): load the model in
  this process. Needs ``pip install -r requirements-reranker.txt``.

rerank_chunks raises on any failure (timeout, HTTP error, bad response). The
retriever catches that and keeps the hybrid (RRF) order, so a reranker
outage makes answers slightly less precise but never fails a chat request.
"""
import asyncio
from functools import lru_cache

import httpx

from src.config.settings import settings
from src.logger import get_logger

logger = get_logger(__name__)


class CrossEncoderScorer:
    """Scores (query, text) pairs with a sentence-transformers CrossEncoder.

    Shared by local mode and the reranker service.
    """

    def __init__(self, model_name: str, max_length: int):
        # Heavy import (torch): only processes that score pairs pay for it.
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder reranker: model=%s", model_name)
        self.model_name = model_name
        self._model = CrossEncoder(model_name, max_length=max_length)

    def score(self, query: str, texts: list[str]) -> list[float]:
        pairs = [(query, text) for text in texts]
        return [float(score) for score in self._model.predict(pairs)]


@lru_cache(maxsize=1)
def get_local_scorer() -> CrossEncoderScorer:
    """Load the local model once, on the first request that needs it."""
    return CrossEncoderScorer(settings.reranker_model, settings.reranker_max_length)


_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    """One shared connection pool to the reranker service per process."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            base_url=settings.reranker_url or "",
            timeout=settings.reranker_timeout_seconds,
        )
    return _http_client


async def close_reranker_client() -> None:
    """Close the shared HTTP client (called on API shutdown)."""
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


async def _score_remote(query: str, texts: list[str]) -> list[float]:
    response = await _get_http_client().post(
        "/rerank", json={"query": query, "texts": texts}
    )
    response.raise_for_status()
    scores = response.json().get("scores")
    if not isinstance(scores, list) or len(scores) != len(texts):
        raise RuntimeError(
            f"Reranker service returned {len(scores) if isinstance(scores, list) else 'no'} "
            f"scores for {len(texts)} texts"
        )
    return [float(score) for score in scores]


async def _score_local(query: str, texts: list[str]) -> list[float]:
    # CPU-bound inference: run it in a thread to keep the event loop free.
    return await asyncio.to_thread(lambda: get_local_scorer().score(query, texts))


async def rerank_chunks(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Return the *top_k* chunks ordered by cross-encoder relevance.

    Each returned chunk gets a ``rerank_score`` key.

    Raises:
        Exception: The scoring call failed. The caller keeps the old order.
    """
    if not chunks:
        return []

    texts = [chunk["text"] for chunk in chunks]
    if settings.reranker_url:
        scores = await _score_remote(query, texts)
    else:
        scores = await _score_local(query, texts)

    ranked_chunks = [
        {**chunk, "rerank_score": score}
        for chunk, score in zip(chunks, scores)
    ]
    ranked_chunks.sort(key=lambda chunk: chunk["rerank_score"], reverse=True)
    return ranked_chunks[:top_k]
