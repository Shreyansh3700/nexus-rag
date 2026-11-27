"""Chat-model service with OpenAI-only model fallback.

Configure the fallback chain with LLM_FALLBACK_CHAIN, a comma-separated
"provider:model" list tried in order, e.g.:

    LLM_FALLBACK_CHAIN=openai:gpt-4o-mini,openai:gpt-4o

Only the first entry is required (defaults to the previous single-model
behavior via LLM_MODEL). Fallback is implemented via LangChain's
Runnable.with_fallbacks(), so it applies to .ainvoke(), .astream_events(),
etc. the same way a single model would. Note that for streaming, a mid-stream
failure after tokens have already been yielded to the client cannot
retroactively fall back — only a failure before the first token is emitted
triggers the next model in the chain.
"""
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.logger import get_logger
from src.config.settings import settings
from src.services.provider_chain import parse_provider_chain

logger = get_logger(__name__)


def _build_chat_model(provider: str, model: str) -> BaseChatModel:
    if provider == "openai":
        return ChatOpenAI(
            model=model,
            api_key=settings.require_openai_api_key(),
            temperature=settings.llm_temperature,
            max_retries=3,
            max_tokens=settings.llm_max_tokens,
        )
    raise ValueError(f"Unknown chat model provider: {provider!r}. Supported: openai.")


class ModelService:
    """Holds an ordered list of chat models and exposes a single Runnable
    (``.runnable``) that tries each in turn, falling back to the next on
    failure. Use ``.runnable`` anywhere a plain LangChain chat model was
    used before (e.g. piped into a prompt with ``|``)."""

    def __init__(self, chain: list[tuple[str, str]]):
        if not chain:
            raise ValueError("ModelService requires at least one (provider, model) entry")
        self.chain = chain
        self.primary_model = chain[0][1]
        models = [_build_chat_model(provider, model) for provider, model in chain]
        primary, *fallbacks = models
        self.runnable = primary.with_fallbacks(fallbacks) if fallbacks else primary
        logger.info(
            "ModelService initialized: chain=%s",
            [f"{provider}:{model}" for provider, model in chain],
        )

    @classmethod
    def from_env(cls) -> "ModelService":
        raw = settings.llm_fallback_chain or f"openai:{settings.llm_model}"
        chain = parse_provider_chain(raw, "LLM_FALLBACK_CHAIN")
        return cls(chain)
