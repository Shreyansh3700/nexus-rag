import json
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("JWT_SECRET", "test-jwt-secret")

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from src.chatbot import turn as turn_module
from src.chatbot.chat import get_answer
from src.chatbot.stream import stream_answer
from src.database.exceptions import SessionAccessError
from src.documents.retriever import RetrievalResult

SOURCES = [
    {"citation": 1, "filename": "one.pdf", "page": 1, "chunk": 0},
    {"citation": 2, "filename": "two.pdf", "page": 5, "chunk": 3},
]
NEW_SESSION = {"exists": False, "foreign": False, "title": None, "history": []}
EXISTING_SESSION = {
    "exists": True,
    "foreign": False,
    "title": "Old chat",
    "history": [HumanMessage(content="hi"), AIMessage(content="hello")],
}
FOREIGN_SESSION = {"exists": False, "foreign": True, "title": None, "history": []}


def title_chain(title="Quarterly report questions"):
    chain = MagicMock()
    chain.ainvoke = AsyncMock(return_value=AIMessage(content=title))
    return chain


def answer_chain(answer):
    chain = MagicMock()
    chain.ainvoke = AsyncMock(
        return_value=AIMessage(
            content=answer,
            response_metadata={
                "model_name": "gpt-test",
                "finish_reason": "stop",
                "token_usage": {"total_tokens": 42},
            },
        )
    )
    return chain


def streaming_chain(tokens, model="gpt-test", total_tokens=42):
    async def astream_events(_input, version):
        for token in tokens:
            yield {"event": "on_chat_model_stream", "data": {"chunk": AIMessageChunk(content=token)}}
        yield {
            "event": "on_chat_model_end",
            "data": {
                "output": AIMessage(
                    content="".join(tokens),
                    response_metadata={"model_name": model},
                    usage_metadata={"input_tokens": 30, "output_tokens": 12, "total_tokens": total_tokens},
                )
            },
            "metadata": {},
        }

    chain = MagicMock()
    chain.astream_events = astream_events
    return chain


def parse_sse(frames):
    events = []
    for frame in frames:
        lines = frame.strip().split("\n")
        events.append((lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))))
    return events


class TurnTestCase(unittest.IsolatedAsyncioTestCase):
    """Patches the DB and retrieval boundaries used by src/chatbot/turn.py."""

    def setUp(self):
        self.retrieve = AsyncMock(return_value=RetrievalResult(context="ctx", sources=SOURCES))
        self.save = AsyncMock(return_value=True)
        self.load = AsyncMock(return_value=EXISTING_SESSION)
        for name, mock in (
            ("retrieve_context", self.retrieve),
            ("update_session_history", self.save),
            ("get_session_context_from_db", self.load),
        ):
            patcher = patch.object(turn_module, name, mock)
            patcher.start()
            self.addCleanup(patcher.stop)

    def saved_kwargs(self):
        self.save.assert_awaited_once()
        return self.save.await_args.kwargs


class PrepareTurnTests(TurnTestCase):
    async def test_new_session_gets_title_and_context(self):
        turn = await turn_module.prepare_turn(
            "s1", "u1", "What is in the report?", db=None,
            title_chain=title_chain(), milvus_client=None, session_context=NEW_SESSION,
        )

        self.assertTrue(turn.is_new_session)
        self.assertEqual(turn.title, "Quarterly report questions")
        self.assertEqual(turn.context, "ctx")
        self.assertEqual(turn.sources, SOURCES)
        self.assertEqual(
            turn.chain_input(),
            {"chat_history": [], "query": "What is in the report?", "context": "ctx"},
        )

    async def test_existing_session_skips_title(self):
        chain = title_chain()
        turn = await turn_module.prepare_turn(
            "s1", "u1", "next question", db=None,
            title_chain=chain, milvus_client=None, session_context=EXISTING_SESSION,
        )

        chain.ainvoke.assert_not_awaited()
        self.assertIsNone(turn.title)
        self.assertEqual(len(turn.history), 2)

    async def test_loads_session_context_when_not_given(self):
        await turn_module.prepare_turn(
            "s1", "u1", "q", db="pool", title_chain=title_chain(), milvus_client=None,
        )

        self.load.assert_awaited_once_with(session_id="s1", user_id="u1", db="pool")

    async def test_foreign_session_is_rejected(self):
        with self.assertRaises(SessionAccessError):
            await turn_module.prepare_turn(
                "s1", "u1", "q", db=None, title_chain=title_chain(),
                milvus_client=None, session_context=FOREIGN_SESSION,
            )
        self.retrieve.assert_not_awaited()

    async def test_retrieval_failure_degrades_to_no_context(self):
        self.retrieve.side_effect = RuntimeError("milvus down")

        turn = await turn_module.prepare_turn(
            "s1", "u1", "q", db=None, title_chain=title_chain(),
            milvus_client=None, session_context=EXISTING_SESSION,
        )

        self.assertEqual((turn.context, turn.sources), ("", []))

    async def test_title_failure_falls_back_to_none(self):
        chain = MagicMock()
        chain.ainvoke = AsyncMock(side_effect=TimeoutError("llm"))

        turn = await turn_module.prepare_turn(
            "s1", "u1", "q", db=None, title_chain=chain,
            milvus_client=None, session_context=NEW_SESSION,
        )

        self.assertIsNone(turn.title)
        self.assertEqual(turn.context, "ctx")

    async def test_blank_title_uses_default(self):
        turn = await turn_module.prepare_turn(
            "s1", "u1", "q", db=None, title_chain=title_chain("   "),
            milvus_client=None, session_context=NEW_SESSION,
        )

        self.assertEqual(turn.title, turn_module.DEFAULT_TITLE)


class GetAnswerTests(TurnTestCase):
    async def test_returns_answer_and_saves_only_cited_sources(self):
        result = await get_answer(
            "s1", "u1", "q", db=None, chain=answer_chain("Revenue grew [2]."),
            title_chain=title_chain(), milvus_client=None, session_context=NEW_SESSION,
        )

        self.assertEqual(result["answer"], "Revenue grew [2].")
        self.assertEqual(result["model_used"], "gpt-test")
        self.assertEqual(result["tokens"], 42)
        self.assertEqual(result["sources"], [SOURCES[1]])
        saved = self.saved_kwargs()
        self.assertEqual(saved["sources"], [SOURCES[1]])
        self.assertEqual(saved["title"], "Quarterly report questions")
        self.assertEqual(saved["user_message"], "q")

    async def test_failed_save_raises(self):
        self.save.return_value = False

        with self.assertRaises(RuntimeError):
            await get_answer(
                "s1", "u1", "q", db=None, chain=answer_chain("ok"),
                title_chain=title_chain(), milvus_client=None, session_context=EXISTING_SESSION,
            )


class StreamAnswerTests(TurnTestCase):
    async def collect(self, chain, session_context=NEW_SESSION):
        frames = [
            frame
            async for frame in stream_answer(
                "s1", "u1", "q", db=None, chain=chain, title_chain=title_chain(),
                milvus_client=None, session_context=session_context,
            )
        ]
        return parse_sse(frames)

    async def test_streams_tokens_then_done_with_cited_sources(self):
        events = await self.collect(streaming_chain(["Revenue ", "grew [1]."]))

        self.assertEqual(
            events[:2],
            [("token", {"token": "Revenue "}), ("token", {"token": "grew [1]."})],
        )
        name, done = events[-1]
        self.assertEqual(name, "done")
        self.assertEqual(done["answer"], "Revenue grew [1].")
        self.assertEqual(done["model"], "gpt-test")
        self.assertEqual(done["tokens"], 42)
        self.assertEqual(done["sources"], [SOURCES[0]])
        saved = self.saved_kwargs()
        self.assertEqual(saved["ai_message"].content, "Revenue grew [1].")
        self.assertEqual(saved["sources"], [SOURCES[0]])

    async def test_save_error_still_sends_done(self):
        self.save.side_effect = ConnectionError("db down")

        events = await self.collect(streaming_chain(["ok"]), session_context=EXISTING_SESSION)

        self.assertEqual(events[-1][0], "done")
        self.assertEqual(events[-1][1]["answer"], "ok")

    async def test_answer_without_token_events_is_sent_once(self):
        async def astream_events(_input, version):
            yield {
                "event": "on_chat_model_end",
                "data": {"output": AIMessage(content="Full answer", response_metadata={"model_name": "m"})},
                "metadata": {},
            }

        chain = MagicMock()
        chain.astream_events = astream_events

        events = await self.collect(chain, session_context=EXISTING_SESSION)

        self.assertEqual(events[0], ("token", {"token": "Full answer"}))
        self.assertEqual(events[-1][1]["answer"], "Full answer")


if __name__ == "__main__":
    unittest.main()
