"""Non-streaming chat: one chain.ainvoke call per turn.

Shared turn logic (history, ownership, title, retrieval, citations, saving)
lives in src/chatbot/turn.py.
"""
from src.chatbot.turn import prepare_turn, save_turn, select_cited_sources
from src.logger import get_logger

logger = get_logger(__name__)


async def get_answer(
    session_id: str,
    user_id: str,
    user_query: str,
    db,
    chain,
    title_chain,
    milvus_client,
    session_context: dict | None = None,
) -> dict:
    """Answer one user message and persist the turn.

    Raises:
        SessionAccessError: The session belongs to another user.
        RuntimeError: The turn could not be saved.
    """
    turn = await prepare_turn(
        session_id=session_id,
        user_id=user_id,
        user_query=user_query,
        db=db,
        title_chain=title_chain,
        milvus_client=milvus_client,
        session_context=session_context,
    )

    response = await chain.ainvoke(turn.chain_input())

    metadata = response.response_metadata
    # Prefer the provider-specific usage shape (OpenAI's "token_usage"), but
    # fall back to LangChain's cross-provider usage_metadata field — needed
    # when a fallback model from a different provider (see ModelService)
    # served this response and doesn't populate "token_usage".
    token_usage = metadata.get("token_usage") or getattr(response, "usage_metadata", None) or {}
    answer = str(response.content)
    model_used = str(metadata.get("model_name") or metadata.get("model") or "unknown")
    finish_reason = str(metadata.get("finish_reason", "unknown"))
    total_tokens = int(token_usage.get("total_tokens", 0))
    latency = float(token_usage.get("total_time", 0.0))

    sources = select_cited_sources(turn, answer)

    # The response was not sent yet, so a failed save fails the request.
    if not await save_turn(turn, response, sources, db):
        raise RuntimeError("Failed to persist conversation history")

    logger.info(
        "Non-stream chat completed: session_id=%s user_id=%s model=%s tokens=%s finish_reason=%s",
        session_id,
        user_id,
        model_used,
        total_tokens,
        finish_reason,
    )

    return {
        "answer": answer,
        "model_used": model_used,
        "tokens": total_tokens,
        "latency_time": latency,
        "sources": sources,
    }
