"""Embedding service with OpenAI-only model fallback.

Configure the fallback chain with EMBEDDING_FALLBACK_CHAIN, a comma-separated
"provider:model" list tried in order, e.g.:

    EMBEDDING_FALLBACK_CHAIN=openai:text-embedding-3-small,openai:text-embedding-3-large

Only the first entry is required (defaults to the previous single-model
behavior).

IMPORTANT: every configured model must produce vectors of the same
dimensionality as EMBEDDING_DIM (src/config/settings.py) (the Milvus schema is
fixed to that dimension). Mixing embedding models with different output
dimensions will corrupt retrieval — this service does not validate that for
you. Switching dimensions requires recreating the Milvus collection (see the
"Milvus migration" section in the README) and setting EMBEDDING_DIM to match.
"""
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from src.logger import get_logger
from src.config.settings import settings
from src.services.provider_chain import parse_provider_chain

logger = get_logger(__name__)


def _build_embeddings(provider: str, model: str) -> Embeddings:
    if provider == "openai":
        return OpenAIEmbeddings(model=model, api_key=settings.require_openai_api_key())
    raise ValueError(f"Unknown embedding provider: {provider!r}. Supported: openai.")


class EmbeddingService(Embeddings):
    """Holds an ordered list of embedding clients and tries each in turn,
    falling back to the next on failure. Drop-in replacement anywhere a
    plain LangChain Embeddings instance was used before."""

    def __init__(self, chain: list[tuple[str, str]]):
        if not chain:
            raise ValueError("EmbeddingService requires at least one (provider, model) entry")
        self.chain = chain
        self._clients = [_build_embeddings(provider, model) for provider, model in chain]
        logger.info(
            "EmbeddingService initialized: chain=%s",
            [f"{provider}:{model}" for provider, model in chain],
        )

    async def _try_each(self, method_name: str, *args):
        last_exc: Exception | None = None
        for (provider, model), client in zip(self.chain, self._clients):
            try:
                return await getattr(client, method_name)(*args)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Embedding provider %s:%s failed on %s, trying next: %s",
                    provider,
                    model,
                    method_name,
                    exc,
                )
        raise RuntimeError(
            f"All embedding providers failed for {method_name}: {self.chain}"
        ) from last_exc

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._try_each("aembed_documents", texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await self._try_each("aembed_query", text)

    # Sync interface required by the Embeddings ABC. Not used by this async
    # codebase (aembed_* above is what's called), implemented only so this
    # class satisfies langchain_core.embeddings.Embeddings.
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._clients[0].embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._clients[0].embed_query(text)

    @classmethod
    def from_env(cls) -> "EmbeddingService":
        raw = settings.embedding_fallback_chain or f"openai:{settings.embedding_model}"
        chain = parse_provider_chain(raw, "EMBEDDING_FALLBACK_CHAIN")
        return cls(chain)


# Module-level instance — stateless, safe to share across the ingestion and
# retrieval code paths (previously each instantiated its own OpenAIEmbeddings
# client independently, risking silent drift between the two).
embedding_service = EmbeddingService.from_env()
