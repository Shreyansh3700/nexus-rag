"""Streaming chat: chain.astream_events, sent to the client as SSE events.

Events:
    event: token   data: {"token": "..."}            — one per non-empty chunk
    event: done    data: {model, tokens, latency, answer, sources}

Shared turn logic (history, ownership, title, retrieval, citations, saving)
lives in src/chatbot/turn.py.
"""
import json
import time
from typing import Any, AsyncGenerator

from langchain_core.messages import AIMessage

from src.chatbot.turn import prepare_turn, save_turn, select_cited_sources
from src.logger import get_logger

logger = get_logger(__name__)


def _extract_text(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "content"):
        value = getattr(value, "content")
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("text", item)))
            elif hasattr(item, "content"):
                parts.append(str(getattr(item, "content")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(value)


def _preview(text: str, limit: int = 200) -> str:
    clean = text.replace("\n", " ").strip()
    return clean[:limit] + "..." if len(clean) > limit else clean


async def stream_answer(
    session_id: str,
    user_id: str,
    user_query: str,
    db,
    chain,
    title_chain,
    milvus_client,
    session_context: dict | None = None,
) -> AsyncGenerator[str, None]:
    turn = await prepare_turn(
        session_id=session_id,
        user_id=user_id,
        user_query=user_query,
        db=db,
        title_chain=title_chain,
        milvus_client=milvus_client,
        session_context=session_context,
    )

    # ------------------------------------------------------------------
    # Stream the chain response
    # ------------------------------------------------------------------
    final_answer = ""
    saw_nonempty_chunk = False
    model_name = "unknown"
    total_tokens = 0
    latency = 0.0
    start_time = time.perf_counter()

    def _get_token_usage(source: dict | object | None) -> dict:
        if source is None:
            return {}
        if isinstance(source, dict):
            return source
        if hasattr(source, "usage_metadata") and source.usage_metadata:
            return dict(source.usage_metadata)
        if hasattr(source, "response_metadata") and source.response_metadata:
            return dict(source.response_metadata.get("token_usage", {}))
        return {}

    def _get_model_metadata(source: dict | object | None) -> dict:
        if source is None:
            return {}
        if isinstance(source, dict):
            return source
        if hasattr(source, "response_metadata") and source.response_metadata:
            return dict(source.response_metadata)
        return {}

    async for event in chain.astream_events(turn.chain_input(), version="v2"):
        event_type = event["event"]

        if event_type == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            chunk_text = _extract_text(getattr(chunk, "content", None))
            logger.debug(
                "Token chunk: session=%s chunk_len=%s preview=%r",
                session_id,
                len(chunk_text),
                _preview(chunk_text),
            )
            if chunk_text:
                saw_nonempty_chunk = True
                final_answer += chunk_text
                yield (
                    f"event: token\n"
                    f"data: {json.dumps({'token': chunk_text})}\n\n"
                )

        elif event_type in {"on_chat_model_end", "on_llm_end", "on_chain_end"}:
            output = event["data"].get("output")
            event_metadata = event.get("metadata") or {}
            usage_metadata = _get_token_usage(output)
            if not usage_metadata:
                usage_metadata = _get_token_usage(event_metadata)
            if not usage_metadata and isinstance(output, dict):
                usage_metadata = _get_token_usage(output.get("llm_output"))

            if not final_answer:
                final_answer = _extract_text(output).strip()

            model_metadata = _get_model_metadata(output)
            model_metadata = {**event_metadata, **model_metadata}

            total_tokens = usage_metadata.get("total_tokens")
            if total_tokens is None:
                total_tokens = int(
                    usage_metadata.get("output_tokens", 0)
                    + usage_metadata.get("input_tokens", 0)
                )
            latency = usage_metadata.get("total_time")
            if latency is None:
                latency = usage_metadata.get("completion_time")
            if latency is None:
                latency = time.perf_counter() - start_time
            latency = float(latency)
            model_name = str(
                model_metadata.get("model_name")
                or model_metadata.get("model")
                or "unknown"
            )

    final_answer = final_answer.strip()
    sources = select_cited_sources(turn, final_answer)

    if not final_answer:
        logger.warning(
            "Empty streamed answer: session=%s user_id=%s model=%s tokens=%s",
            session_id,
            user_id,
            model_name,
            total_tokens,
        )

    if final_answer and not saw_nonempty_chunk:
        yield (
            "event: token\n"
            f"data: {json.dumps({'token': final_answer})}\n\n"
        )

    # ------------------------------------------------------------------
    # Persist conversation history
    # ------------------------------------------------------------------
    final_ai_message = AIMessage(
        content=final_answer,
        response_metadata={
            "token_usage": {
                "total_tokens": total_tokens,
                "total_time": latency,
            },
            "model_name": model_name,
        },
    )

    # Tokens were already sent, so a failed save cannot fail the request:
    # log it and still send the final "done" event.
    try:
        saved = await save_turn(turn, final_ai_message, sources, db)
    except Exception:
        saved = False
        logger.exception(
            "Failed to persist streamed chat history: session=%s user_id=%s",
            session_id,
            user_id,
        )
    if saved:
        logger.info(
            "Persisted streamed answer: session=%s user_id=%s model=%s tokens=%s latency=%s",
            session_id,
            user_id,
            model_name,
            total_tokens,
            latency,
        )

    yield (
        "event: done\n"
        f"data: {json.dumps({'model': model_name, 'tokens': total_tokens, 'latency': latency, 'answer': final_answer, 'sources': sources})}\n\n"
    )
