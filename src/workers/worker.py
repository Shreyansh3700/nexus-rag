"""ARQ worker entrypoint.

This script starts the ARQ worker process that consumes document processing
jobs from Redis and executes them using the shared Postgres and Milvus resources.

Run with:
    python -m src.workers.worker

The worker will:
- Connect to Redis using settings from src.config.arq_config
- Connect to Postgres and Milvus on startup
- Process jobs with the configured concurrency (ARQ_MAX_JOBS)
- Gracefully shut down on SIGTERM/SIGINT
"""

import asyncpg
from arq import run_worker

from src.config.arq_config import WorkerSettings
from src.config.config import create_milvus_client
from src.config.settings import settings
from src.logger import configure_logging, get_logger
from src.storage.minio_client import get_minio_client

configure_logging()
logger = get_logger(__name__)


async def startup(ctx: dict) -> None:
    """Worker startup hook: initialize shared resources.

    Creates:
    - Postgres connection pool (shared across all tasks)
    - Milvus client (shared across all tasks)

    These are stored in the ARQ context dict and passed to each task.

    Args:
        ctx: ARQ worker context dictionary
    """
    logger.info("Worker starting up: initializing shared resources")

    # ------------------------------------------------------------------ #
    # Postgres connection pool
    # ------------------------------------------------------------------ #
    logger.debug("Creating Postgres connection pool")
    # Same required settings as the API, so a missing value fails the worker
    # at startup instead of connecting somewhere else.
    ctx["db"] = await asyncpg.create_pool(**settings.postgres_pool_kwargs())
    logger.info("Postgres pool created")

    # ------------------------------------------------------------------ #
    # Milvus client
    # ------------------------------------------------------------------ #
    logger.debug("Connecting to Milvus: uri=%s", settings.milvus_uri)
    ctx["milvus"] = create_milvus_client()
    logger.info("Milvus client initialized: collection=%s", settings.milvus_collection_name)

    ctx["minio"] = get_minio_client()

    logger.info("Worker startup complete")


async def shutdown(ctx: dict) -> None:
    """Worker shutdown hook: clean up shared resources.

    Closes:
    - Postgres connection pool
    - Milvus client (if needed)

    Args:
        ctx: ARQ worker context dictionary
    """
    logger.info("Worker shutting down: closing shared resources")

    # ------------------------------------------------------------------ #
    # Close Postgres pool
    # ------------------------------------------------------------------ #
    if "db" in ctx:
        logger.debug("Closing Postgres connection pool")
        await ctx["db"].close()
        logger.info("Postgres pool closed")

    # Milvus client doesn't need explicit cleanup in current SDK version
    logger.info("Worker shutdown complete")


# arq.worker.get_kwargs() reads settings_cls.__dict__ directly, ignoring
# inherited attributes — so subclassing WorkerSettings to add hooks silently
# drops `functions`, `redis_settings`, etc. Attach the hooks directly instead.
WorkerSettings.on_startup = startup
WorkerSettings.on_shutdown = shutdown


def main() -> None:
    """Run the ARQ worker with configured settings."""
    logger.info("Starting ARQ document ingestion worker")
    logger.info("Press Ctrl+C to stop")

    # Run worker (blocks until SIGTERM/SIGINT)
    run_worker(WorkerSettings)


if __name__ == "__main__":
    main()
