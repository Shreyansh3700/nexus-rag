"""Typed application settings, read once from environment variables.

Every setting that the API, the ARQ worker and the reranker service use is
declared here. Do not call os.getenv anywhere else.

Field names map to environment variables without regard to case
(``milvus_top_k`` <- ``MILVUS_TOP_K``). Empty variables count as unset, so an
``ENV X=`` line with no value falls back to the default below instead of
failing an int() conversion.

Invalid values (for example ``MILVUS_PORT=abc``) fail at import time with a
pydantic ValidationError that names the variable. Settings that only some
processes need (Postgres, the OpenAI key, the JWT secret) are optional here
and checked by the require_* helpers when a process actually uses them.
"""
from typing import Any

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_ignore_empty=True,
        extra="ignore",
    )

    # --- Logging / HTTP ---------------------------------------------------
    log_level: str = "INFO"
    frontend_origin: str = "http://localhost:5173"

    # --- Auth -------------------------------------------------------------
    jwt_secret: SecretStr | None = None
    # Legacy name for jwt_secret, still accepted.
    auth_api_key: SecretStr | None = None
    jwt_access_token_expire_minutes: int = Field(60, gt=0)
    password_hash_iterations: int = Field(210_000, gt=0)

    # --- Chat model -------------------------------------------------------
    openai_api_key: SecretStr | None = None
    llm_model: str = "gpt-4o-mini"
    # Comma-separated "provider:model" list; defaults to "openai:<llm_model>".
    llm_fallback_chain: str | None = None
    llm_temperature: float = Field(0.6, ge=0, le=2)
    llm_max_tokens: int = Field(6000, gt=0)
    # Token budget for the chat history sent to the model.
    max_chat_tokens: int = Field(2500, gt=0)

    # --- Embeddings -------------------------------------------------------
    embedding_model: str = "text-embedding-3-small"
    embedding_fallback_chain: str | None = None
    # Must match the output size of every model in the embedding chain.
    # Changing it requires recreating the Milvus collection (see README).
    embedding_dim: int = Field(1536, gt=0)

    # --- PostgreSQL (required by the API and the worker) ------------------
    pg_host: str | None = None
    pg_port: int | None = None
    pg_database: str | None = None
    pg_user: str | None = None
    pg_password: SecretStr | None = None
    db_pool_size: int = Field(10, gt=0)

    # --- Milvus -----------------------------------------------------------
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_name: str = "doc_chunks"
    milvus_top_k: int = Field(5, gt=0)
    milvus_hybrid_candidate_k: int = Field(20, gt=0)
    milvus_rrf_k: int = Field(60, gt=0)
    # Format "username:password". Default Milvus root credentials.
    milvus_token: SecretStr = SecretStr("root:Milvus")
    # Dense-leg relevance floor: chunks whose cosine distance to the query is
    # larger than this are dropped. 0.6 is a starting point for
    # text-embedding-3-small, not a measured value.
    retrieval_max_cosine_distance: float = Field(0.6, ge=0, le=2)

    # --- Cross-encoder reranker -------------------------------------------
    reranker_enabled: bool = True
    # Base URL of the reranker service (src/reranker_service.py). When unset,
    # the model runs in the API process (local development only; needs
    # requirements-reranker.txt).
    reranker_url: str | None = None
    reranker_timeout_seconds: float = Field(10.0, gt=0)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_candidate_k: int = Field(30, gt=0)
    # Defaults to milvus_top_k.
    reranker_top_k: int | None = Field(None, gt=0)
    reranker_max_length: int = Field(512, gt=0)
    # Parallel inference calls in the reranker service. More than 1 on a
    # small CPU makes every request slower.
    reranker_max_concurrency: int = Field(1, gt=0)

    # --- MinIO ------------------------------------------------------------
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: SecretStr = SecretStr("minioadmin")
    minio_bucket: str = "documents"
    minio_secure: bool = False

    # --- Uploads ----------------------------------------------------------
    max_upload_bytes: int = Field(25 * 1024 * 1024, gt=0)

    # --- Redis / ARQ ------------------------------------------------------
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = Field(0, ge=0)
    arq_max_jobs: int = Field(5, gt=0)
    arq_job_timeout: int = Field(600, gt=0)
    # Total attempts per ingestion job (first try + retries).
    arq_max_tries: int = Field(3, gt=0)

    @model_validator(mode="after")
    def _apply_derived_defaults(self) -> "Settings":
        if self.reranker_top_k is None:
            self.reranker_top_k = self.milvus_top_k
        if self.reranker_url:
            self.reranker_url = self.reranker_url.rstrip("/")
        return self

    # --- Derived values ---------------------------------------------------
    @property
    def milvus_uri(self) -> str:
        return f"http://{self.milvus_host}:{self.milvus_port}"

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # --- Required-setting checks ------------------------------------------
    def _require(self, *names: str) -> None:
        missing = [name.upper() for name in names if getattr(self, name) in (None, "")]
        if missing:
            raise RuntimeError(f"{', '.join(missing)} must be configured")

    def postgres_pool_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for asyncpg.create_pool. Raises if any PG_* is missing."""
        self._require("pg_host", "pg_port", "pg_database", "pg_user", "pg_password")
        assert self.pg_password is not None
        return {
            "host": self.pg_host,
            "port": self.pg_port,
            "database": self.pg_database,
            "user": self.pg_user,
            "password": self.pg_password.get_secret_value(),
            "min_size": 1,
            "max_size": self.db_pool_size,
        }

    def require_openai_api_key(self) -> str:
        self._require("openai_api_key")
        assert self.openai_api_key is not None
        return self.openai_api_key.get_secret_value()

    def require_jwt_secret(self) -> str:
        secret = self.jwt_secret or self.auth_api_key
        if secret is None or not secret.get_secret_value():
            raise RuntimeError("JWT_SECRET must be configured")
        return secret.get_secret_value()


settings = Settings()
