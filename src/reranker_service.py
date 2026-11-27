"""Standalone cross-encoder reranking service.

Run:
    uvicorn src.reranker_service:app --host 0.0.0.0 --port 8080

Endpoints:
    POST /rerank   {"query": str, "texts": [str, ...]}  ->  {"scores": [float, ...]}
    GET  /health   {"status": "ok", "model": str}

Why a separate process: the model and torch use several hundred MB of RAM,
and inference is CPU-heavy. Inside the API, every uvicorn worker loaded its
own copy, and a slow rerank competed with chat requests for the CPU. Here
the model loads once, and the service can scale (or move to a GPU) on its own.

The model loads at startup, so /health only answers when the service can
score. Inference runs in a worker thread. A semaphore
(RERANKER_MAX_CONCURRENCY) caps parallel inference, so extra requests wait
instead of slowing every request down. The API client timeout
(RERANKER_TIMEOUT_SECONDS) decides when the API stops waiting and keeps the
hybrid order.

This module imports only FastAPI, settings and the scorer. It must not import
the API's config (LangChain, Milvus, Postgres).
"""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from src.config.settings import settings
from src.documents.reranker import CrossEncoderScorer
from src.logger import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

# Upper bounds keep one request from holding the model for a long time.
MAX_TEXTS = 200
MAX_TEXT_CHARS = 8192


class RerankRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    texts: list[str] = Field(min_length=1, max_length=MAX_TEXTS)


class RerankResponse(BaseModel):
    scores: list[float]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tests set app.state.scorer to a fake before startup.
    if getattr(app.state, "scorer", None) is None:
        app.state.scorer = await asyncio.to_thread(
            CrossEncoderScorer, settings.reranker_model, settings.reranker_max_length
        )
    app.state.semaphore = asyncio.Semaphore(settings.reranker_max_concurrency)
    logger.info(
        "Reranker service ready: model=%s max_concurrency=%s",
        settings.reranker_model,
        settings.reranker_max_concurrency,
    )
    yield


app = FastAPI(title="nexus-rag reranker", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health(request: Request) -> dict:
    return {"status": "ok", "model": settings.reranker_model}


@app.post("/rerank", response_model=RerankResponse)
async def rerank(body: RerankRequest, request: Request) -> RerankResponse:
    # Texts longer than the model's window are truncated by the tokenizer
    # anyway; cutting them here bounds the request cost.
    texts = [text[:MAX_TEXT_CHARS] for text in body.texts]
    async with request.app.state.semaphore:
        scores = await asyncio.to_thread(request.app.state.scorer.score, body.query, texts)
    return RerankResponse(scores=scores)
