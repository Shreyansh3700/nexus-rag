"""Reranker client (remote + local mode), retriever fallback, and the service."""
import json
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("JWT_SECRET", "test-jwt-secret")

import httpx
from fastapi.testclient import TestClient

from src import reranker_service
from src.config.settings import settings
from src.documents import reranker, retriever

CHUNKS = [
    {"text": "alpha", "filename": "a.pdf", "page": 1, "chunk": 0},
    {"text": "beta", "filename": "b.pdf", "page": 2, "chunk": 1},
    {"text": "gamma", "filename": "c.pdf", "page": 3, "chunk": 2},
]


class FakeScorer:
    """Scores a text by its length, so the expected order is easy to read."""

    def score(self, query, texts):
        return [float(len(text)) for text in texts]


def mock_service(handler):
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://reranker"
    )
    return patch.object(reranker, "_get_http_client", return_value=client)


class RemoteRerankTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        url_patch = patch.object(settings, "reranker_url", "http://reranker")
        url_patch.start()
        self.addCleanup(url_patch.stop)

    async def test_sorts_by_service_scores_and_cuts_to_top_k(self):
        requests = []

        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"scores": [0.1, 0.9, 0.5]})

        with mock_service(handler):
            result = await reranker.rerank_chunks("q", CHUNKS, top_k=2)

        self.assertEqual([c["text"] for c in result], ["beta", "gamma"])
        self.assertEqual(result[0]["rerank_score"], 0.9)
        self.assertEqual(requests, [{"query": "q", "texts": ["alpha", "beta", "gamma"]}])

    async def test_wrong_number_of_scores_raises(self):
        with mock_service(lambda request: httpx.Response(200, json={"scores": [1.0]})):
            with self.assertRaises(RuntimeError):
                await reranker.rerank_chunks("q", CHUNKS, top_k=2)

    async def test_http_error_raises(self):
        with mock_service(lambda request: httpx.Response(503)):
            with self.assertRaises(httpx.HTTPStatusError):
                await reranker.rerank_chunks("q", CHUNKS, top_k=2)

    async def test_no_chunks_makes_no_call(self):
        handler = MagicMock()
        with mock_service(handler):
            self.assertEqual(await reranker.rerank_chunks("q", [], top_k=2), [])
        handler.assert_not_called()


class LocalRerankTests(unittest.IsolatedAsyncioTestCase):
    async def test_unset_url_uses_in_process_scorer(self):
        with patch.object(settings, "reranker_url", None), patch.object(
            reranker, "get_local_scorer", return_value=FakeScorer()
        ):
            result = await reranker.rerank_chunks("q", CHUNKS, top_k=1)

        self.assertEqual([c["text"] for c in result], ["alpha"])  # ties keep input order


class RetrieverFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_reranker_failure_keeps_hybrid_order(self):
        with patch.object(settings, "reranker_enabled", True), patch.object(
            retriever, "_has_chunks", return_value=True
        ), patch.object(
            retriever.embedding_service, "aembed_query", AsyncMock(return_value=[0.1])
        ), patch.object(
            retriever, "_hybrid_search", return_value=list(CHUNKS)
        ), patch.object(
            retriever, "rerank_chunks", AsyncMock(side_effect=httpx.ConnectTimeout("down"))
        ):
            result = await retriever.retrieve_context(
                "q", session_id="s1", user_id="u1", milvus_client=MagicMock(), top_k=2
            )

        self.assertEqual([s["filename"] for s in result.sources], ["a.pdf", "b.pdf"])
        self.assertIn("alpha", result.context)


class RerankerServiceTests(unittest.TestCase):
    def setUp(self):
        reranker_service.app.state.scorer = FakeScorer()
        self.addCleanup(setattr, reranker_service.app.state, "scorer", None)

    def test_rerank_returns_one_score_per_text(self):
        with TestClient(reranker_service.app) as client:
            response = client.post("/rerank", json={"query": "q", "texts": ["a", "bbb"]})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"scores": [1.0, 3.0]})

    def test_empty_texts_are_rejected(self):
        with TestClient(reranker_service.app) as client:
            response = client.post("/rerank", json={"query": "q", "texts": []})

        self.assertEqual(response.status_code, 422)

    def test_too_many_texts_are_rejected(self):
        texts = ["t"] * (reranker_service.MAX_TEXTS + 1)
        with TestClient(reranker_service.app) as client:
            response = client.post("/rerank", json={"query": "q", "texts": texts})

        self.assertEqual(response.status_code, 422)

    def test_health(self):
        with TestClient(reranker_service.app) as client:
            response = client.get("/health")

        self.assertEqual(response.json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
