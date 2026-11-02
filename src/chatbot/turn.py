"""Shared steps of one chat turn, used by both the non-stream and stream paths.

A turn has three phases:

    prepare_turn()          load history, check ownership, generate a title for
                            a new session, retrieve RAG context
    <model call>            chain.ainvoke (chat.py) or chain.astream_events
                            (stream.py) — the only part that differs
    select_cited_sources()  keep only the sources the answer actually cites
    save_turn()             persist the user + AI messages and sources

The title call and retrieval are independent, so prepare_turn runs them
concurrently. This removes one LLM round trip from the time to first token
of every new session.
"""
import asyncio
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

from src.database.exceptions import SessionAccessError
from src.database.fetch_data import get_session_context_from_db
from src.database.update_data import update_session_history
from src.documents.retriever import cited_sources, retrieve_context
from src.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TITLE = "New Chat"


@dataclass
class PreparedTurn:
    """Everything the model call and the save step need for one turn."""

    session_id: str
    user_id: str
    user_query: str
    is_new_session: bool
    history: list[BaseMessage]
    title: str | None
    context: str
    sources: list[dict] = field(default_factory=list)

    def chain_input(self) -> dict[str, Any]:
        """Input for the QA prompt (see set_environment in src/config/config.py)."""
        return {
            "chat_history": self.history,
            "query": self.user_query,
            "context": self.context,
        }


async def _generate_title(title_chain, session_id: str, user_query: str) -> str | None:
    """Generate a session title. Returns None on failure, so the save step
    falls back to a title built from the user's message."""
    try:
        result = await title_chain.ainvoke({"query": user_query})
    except Exception:
        logger.warning(
            "Title generation failed; using fallback title: session_id=%s",
            session_id,
            exc_info=True,
        )
        return None
    title = str(getattr(result, "content", result)).strip() or DEFAULT_TITLE
    logger.info("Generated title: session_id=%s title=%r", session_id, title)
    return title


async def _retrieve(
    milvus_client, session_id: str, user_id: str, user_query: str
) -> tuple[str, list[dict]]:
    """Retrieve RAG context. Any failure degrades to "no context"."""
    try:
        retrieval = await retrieve_context(
            query=user_query,
            session_id=session_id,
            user_id=user_id,
            milvus_client=milvus_client,
        )
    except Exception:
        logger.warning(
            "SOURCES-TRACE: retrieval failed — proceeding without RAG context: session_id=%s",
            session_id,
            exc_info=True,
        )
        return "", []

    logger.info(
        "SOURCES-TRACE: after retrieve_context: session_id=%s context_chars=%s raw_sources_count=%s",
        session_id,
        len(retrieval.context),
        len(retrieval.sources),
    )
    return retrieval.context, retrieval.sources


async def prepare_turn(
    session_id: str,
    user_id: str,
    user_query: str,
    db,
    title_chain,
    milvus_client,
    session_context: dict | None = None,
) -> PreparedTurn:
    """Load the session, check ownership, and gather title + RAG context.

    Raises:
        SessionAccessError: The session belongs to another user.
    """
    if session_context is None:
        session_context = await get_session_context_from_db(
            session_id=session_id,
            user_id=user_id,
            db=db,
        )

    if session_context["foreign"]:
        raise SessionAccessError("Session does not belong to the current user")

    is_new_session = not session_context["exists"]
    history = session_context["history"]
    logger.info(
        "Preparing chat turn: session_id=%s user_id=%s new_session=%s history_len=%s query_len=%s",
        session_id,
        user_id,
        is_new_session,
        len(history),
        len(user_query),
    )

    retrieval_task = _retrieve(milvus_client, session_id, user_id, user_query)
    if is_new_session:
        title, (context, sources) = await asyncio.gather(
            _generate_title(title_chain, session_id, user_query),
            retrieval_task,
        )
    else:
        title = None
        context, sources = await retrieval_task

    return PreparedTurn(
        session_id=session_id,
        user_id=user_id,
        user_query=user_query,
        is_new_session=is_new_session,
        history=history,
        title=title,
        context=context,
        sources=sources,
    )


def select_cited_sources(turn: PreparedTurn, answer: str) -> list[dict]:
    """Keep only the retrieved sources the answer cites (see cited_sources)."""
    sources = cited_sources(answer, turn.sources)
    logger.info(
        "SOURCES-TRACE: after cited_sources filter: session_id=%s raw_sources_count=%s "
        "cited_sources_count=%s",
        turn.session_id,
        len(turn.sources),
        len(sources),
    )
    return sources


async def save_turn(
    turn: PreparedTurn,
    ai_message: AIMessage,
    sources: list[dict],
    db,
) -> bool:
    """Persist the user message, the AI message, and its cited sources.

    Returns:
        True if saved, False if the database write failed.

    Raises:
        SessionAccessError: The session was claimed by another user meanwhile.
    """
    saved = await update_session_history(
        session_id=turn.session_id,
        user_id=turn.user_id,
        user_message=turn.user_query,
        ai_message=ai_message,
        db=db,
        title=turn.title,
        sources=sources,
    )
    if not saved:
        logger.error(
            "Failed to persist conversation history: session_id=%s user_id=%s",
            turn.session_id,
            turn.user_id,
        )
    return saved
