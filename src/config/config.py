"""FastAPI lifespan: builds the chains and connects Postgres, Milvus, Redis,
MinIO and the reranker client once per process.

All settings come from src/config/settings.py.
"""
import asyncio
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pymilvus import DataType, Function, FunctionType, MilvusClient

from src.config.prompts import system_prompt, title_prompt
from src.config.settings import settings
from src.database.schema import ensure_postgres_schema
from src.logger import get_logger
from src.services.model_service import ModelService

logger = get_logger(__name__)

# ModelService holds the primary chat model plus any configured fallbacks
# (see src/services/model_service.py). chat_model is kept as the primary
# model's name for callers that only need it for things like tiktoken
# encoding lookups (src/database/fetch_data.py).
model_service = ModelService.from_env()
chat_model = model_service.primary_model


def create_milvus_client() -> MilvusClient:
    """Milvus client from settings. Shared by the API and the worker."""
    return MilvusClient(
        uri=settings.milvus_uri,
        token=settings.milvus_token.get_secret_value(),
    )


def _bootstrap_milvus_collection(client: MilvusClient, collection_name: str) -> None:
    """Create the doc_chunks collection with all required fields if it does not exist."""
    if client.has_collection(collection_name):
        description = client.describe_collection(collection_name)
        field_names = {
            field.get("name") or field.get("field_name")
            for field in description.get("fields", [])
        }
        if "page" not in field_names:
            raise RuntimeError(
                f"Milvus collection '{collection_name}' uses the pre-citation schema. "
                "Milvus 2.5 does not support adding fields to an existing collection. "
                "Drop and recreate the collection, clear existing document records, "
                "then re-upload documents so page metadata can be indexed."
            )
        logger.info("Milvus collection already exists: %s", collection_name)
        return

    logger.info("Creating Milvus collection: %s", collection_name)
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("document_id", DataType.VARCHAR, max_length=64)
    schema.add_field("user_id", DataType.VARCHAR, max_length=128)
    schema.add_field("session_id", DataType.VARCHAR, max_length=128)
    schema.add_field("chunk_index", DataType.INT32)
    schema.add_field("page", DataType.INT32, nullable=True)
    schema.add_field("filename", DataType.VARCHAR, max_length=512)
    schema.add_field("text", DataType.VARCHAR, max_length=4096, enable_analyzer=True)
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=settings.embedding_dim)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(
        Function(
            name="text_bm25",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse_vector"],
        )
    )

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 128},
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
    )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )
    logger.info("Milvus collection created: %s", collection_name)


@asynccontextmanager
async def set_environment(app: FastAPI):
    logger.info("Initializing application environment")

    # ------------------------------------------------------------------ #
    # LangChain prompt + chain                                            #
    # {context} is populated by the retriever before each chain call.    #
    # When no documents exist for the session, context="" and the        #
    # prompt renders cleanly with just the existing chat history.        #
    # ------------------------------------------------------------------ #
    app.state.qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),   # system_prompt contains {context}
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{query}"),
        ]
    )
    app.state.title_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", title_prompt),
            ("human", "{query}"),
        ]
    )
    app.state.chain = app.state.qa_prompt | model_service.runnable
    app.state.title_chain = app.state.title_prompt | model_service.runnable

    # ------------------------------------------------------------------ #
    # PostgreSQL pool + schema                                            #
    # ------------------------------------------------------------------ #
    logger.debug("Creating database pool")
    app.state.db = await asyncpg.create_pool(**settings.postgres_pool_kwargs())

    try:
        async with app.state.db.acquire() as connection:
            logger.debug("Ensuring database schema and indexes")
            await ensure_postgres_schema(connection)

        logger.info("PostgreSQL schema ready")

        # ------------------------------------------------------------------ #
        # Milvus connection + collection bootstrap                           #
        # ------------------------------------------------------------------ #
        logger.debug("Connecting to Milvus: uri=%s", settings.milvus_uri)
        milvus_client = create_milvus_client()
        await asyncio.to_thread(
            _bootstrap_milvus_collection, milvus_client, settings.milvus_collection_name
        )
        app.state.milvus = milvus_client
        logger.info("Milvus ready: collection=%s", settings.milvus_collection_name)

        # ------------------------------------------------------------------ #
        # Redis pool (ARQ job queue) — created once, shared by all requests  #
        # ------------------------------------------------------------------ #
        # Imported here: arq_config imports this module at load time.
        from src.config.arq_config import get_redis_pool
        from src.storage.minio_client import get_minio_client

        app.state.redis = await get_redis_pool()

        # ------------------------------------------------------------------ #
        # MinIO bucket — ensured once instead of on every upload             #
        # ------------------------------------------------------------------ #
        await asyncio.to_thread(get_minio_client().ensure_bucket, settings.minio_bucket)
        logger.info("MinIO bucket ready: %s", settings.minio_bucket)

        if not settings.reranker_enabled:
            logger.info("Reranker disabled")
        elif settings.reranker_url:
            logger.info("Reranker service: url=%s", settings.reranker_url)
        else:
            logger.warning(
                "RERANKER_URL is not set: the reranker model will load in the API "
                "process. Use this for local development only."
            )

        logger.info("Application environment initialized")
        yield

    finally:
        # Imported here for the same reason as get_redis_pool above.
        from src.documents.reranker import close_reranker_client

        await close_reranker_client()
        redis_pool = getattr(app.state, "redis", None)
        if redis_pool is not None:
            logger.info("Closing Redis pool")
            await redis_pool.aclose()
        logger.info("Closing database pool")
        await app.state.db.close()
        logger.info("Application shutdown complete")
