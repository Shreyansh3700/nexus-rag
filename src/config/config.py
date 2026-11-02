import asyncio
import os
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pymilvus import DataType, Function, FunctionType, MilvusClient

from src.config.prompts import system_prompt, title_prompt
from src.logger import get_logger
from src.services.model_service import ModelService

MAX_CHAT_TOKENS = int(os.getenv("MAX_CHAT_TOKENS", "2500"))
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60")
)
PASSWORD_HASH_ITERATIONS = int(os.getenv("PASSWORD_HASH_ITERATIONS", "210000"))

# Milvus settings
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "doc_chunks")
MILVUS_TOP_K = int(os.getenv("MILVUS_TOP_K", "5"))
MILVUS_HYBRID_CANDIDATE_K = int(os.getenv("MILVUS_HYBRID_CANDIDATE_K", "20"))
MILVUS_RRF_K = int(os.getenv("MILVUS_RRF_K", "60"))
# Dense-leg relevance floor: chunks whose cosine distance (1 - cosine
# similarity) to the query exceeds this are dropped before ranking/reranking.
# 0.6 is a starting point for text-embedding-3-small, not a measured value —
# tune it against real queries/documents (see SOURCES-TRACE retrieval logs).
RETRIEVAL_MAX_COSINE_DISTANCE = float(
    os.getenv("RETRIEVAL_MAX_COSINE_DISTANCE", "0.6")
)
# token format: "username:password" — default Milvus root credentials are root:Milvus
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "root:Milvus")

# Cross-encoder reranking settings. The model is loaded lazily on the first
# retrieval so application startup remains fast.
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() in {
    "1", "true", "yes", "on"
}
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2"
)
RERANKER_CANDIDATE_K = int(os.getenv("RERANKER_CANDIDATE_K", "30"))
RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", str(MILVUS_TOP_K)))
RERANKER_MAX_LENGTH = int(os.getenv("RERANKER_MAX_LENGTH", "512"))

# MinIO settings (object storage for uploaded documents)
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "documents")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() in {"true", "1", "yes"}

# Largest accepted upload, in bytes (default 25 MB).
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))

# Redis settings (ARQ task queue broker)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# ARQ worker settings
ARQ_MAX_JOBS = int(os.getenv("ARQ_MAX_JOBS", "5"))
ARQ_JOB_TIMEOUT = int(os.getenv("ARQ_JOB_TIMEOUT", "600"))
# Total attempts per ingestion job (first try + retries) for transient errors.
ARQ_MAX_TRIES = int(os.getenv("ARQ_MAX_TRIES", "3"))

# Embedding vector dimension — must match whatever EMBEDDING_FALLBACK_CHAIN's
# first entry actually produces. Defaults to 1536 for OpenAI's
# text-embedding-3-small. Common alternatives if you switch embedding models:
# 384 for BAAI/bge-small-en-v1.5, 768 for bge-base-en-v1.5 / nomic-embed-text.
# Changing this requires recreating the Milvus collection (see README).
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))


def required_setting(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} must be configured")
    return value


logger = get_logger(__name__)

# ModelService holds the primary chat model plus any configured fallbacks
# (see src/services/model_service.py). chat_model is kept as the primary
# model's name for callers that only need it for things like tiktoken
# encoding lookups (src/database/fetch_data.py).
model_service = ModelService.from_env()
chat_model = model_service.primary_model


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
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
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
    app.state.db = await asyncpg.create_pool(
        host=required_setting("PG_HOST"),
        port=int(required_setting("PG_PORT")),
        database=required_setting("PG_DATABASE"),
        user=required_setting("PG_USER"),
        password=required_setting("PG_PASSWORD"),
        min_size=1,
        max_size=int(os.getenv("DB_POOL_SIZE", "10")),
    )

    try:
        async with app.state.db.acquire() as connection:
            logger.debug("Ensuring database schema and indexes")
            # --- existing tables ---
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    title TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id BIGSERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('Human', 'AI')),
                    content TEXT NOT NULL,
                    sources JSONB NOT NULL DEFAULT '[]'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS messages_session_id_id_idx
                    ON messages (session_id, id);
                CREATE INDEX IF NOT EXISTS sessions_user_id_idx
                    ON sessions (user_id);
                """
            )
            await connection.execute(
                "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS user_id TEXT"
            )
            await connection.execute(
                "ALTER TABLE messages ADD COLUMN IF NOT EXISTS sources JSONB NOT NULL DEFAULT '[]'::jsonb"
            )

            # --- RAG: document metadata table ---
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    chunk_count INT NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'processing'
                );
                CREATE INDEX IF NOT EXISTS documents_user_session_idx
                    ON documents (user_id, session_id);
                """
            )
            # Add new columns for queue-based architecture
            await connection.execute(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS object_name TEXT"
            )
            await connection.execute(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS failure_reason TEXT"
            )
            # SHA-256 of the file bytes, used to skip duplicate uploads.
            await connection.execute(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_hash TEXT"
            )
            await connection.execute(
                """
                CREATE INDEX IF NOT EXISTS documents_user_session_hash_idx
                    ON documents (user_id, session_id, content_hash)
                """
            )

            # --- RAG: chunk persistence table (Milvus fallback) ---
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    chunk_index INT NOT NULL,
                    chunk_text TEXT NOT NULL,
                    page INT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS document_chunks_session_id_idx
                    ON document_chunks (session_id);
                CREATE INDEX IF NOT EXISTS document_chunks_document_id_idx
                    ON document_chunks (document_id);
                """
            )
            await connection.execute(
                "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS page INT"
            )

        logger.info("PostgreSQL schema ready")

        # ------------------------------------------------------------------ #
        # Milvus connection + collection bootstrap                           #
        # ------------------------------------------------------------------ #
        logger.debug(
            "Connecting to Milvus: host=%s port=%s", MILVUS_HOST, MILVUS_PORT
        )
        milvus_client = MilvusClient(
            uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}",
            token=MILVUS_TOKEN,
        )
        _bootstrap_milvus_collection(milvus_client, MILVUS_COLLECTION_NAME)
        app.state.milvus = milvus_client
        logger.info("Milvus ready: collection=%s", MILVUS_COLLECTION_NAME)

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
        await asyncio.to_thread(get_minio_client().ensure_bucket, MINIO_BUCKET)
        logger.info("MinIO bucket ready: %s", MINIO_BUCKET)

        logger.info("Application environment initialized")
        yield

    finally:
        redis_pool = getattr(app.state, "redis", None)
        if redis_pool is not None:
            logger.info("Closing Redis pool")
            await redis_pool.aclose()
        logger.info("Closing database pool")
        await app.state.db.close()
        logger.info("Application shutdown complete")
