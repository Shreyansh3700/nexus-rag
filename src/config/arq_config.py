"""ARQ worker configuration and Redis connection setup.

This module defines the ARQ worker settings (Redis connection, concurrency,
timeouts) and provides a Redis pool factory for the upload API to enqueue jobs.
"""
from arq import create_pool
from arq.connections import RedisSettings, ArqRedis

from src.config.settings import settings
from src.logger import get_logger

logger = get_logger(__name__)


def get_redis_settings() -> RedisSettings:
    """Build RedisSettings from environment configuration.

    Returns:
        Configured RedisSettings instance for ARQ
    """
    return RedisSettings(
        host=settings.redis_host,
        port=settings.redis_port,
        database=settings.redis_db,
    )


async def get_redis_pool() -> ArqRedis:
    """Create an ARQ Redis connection pool for enqueueing jobs.

    This is used by the upload API to enqueue document processing jobs.
    The pool should be created once and reused across requests.

    Returns:
        Connected ArqRedis pool

    Raises:
        ConnectionError: If Redis is unreachable
    """
    logger.debug(
        "Creating Redis pool: host=%s port=%s db=%s",
        settings.redis_host,
        settings.redis_port,
        settings.redis_db,
    )
    try:
        pool = await create_pool(get_redis_settings())
        logger.info("Redis pool created successfully")
        return pool
    except Exception as exc:
        logger.exception("Failed to create Redis pool")
        raise ConnectionError(f"Redis connection failed: {exc}") from exc


class WorkerSettings:
    """ARQ worker configuration.

    This class is consumed by `arq.run_worker()` when starting the worker
    process. It defines:
    - Which Redis instance to connect to
    - Which task functions are available
    - Concurrency and timeout settings
    """

    # Import task functions here to register them with the worker.
    # This import is intentionally inside the class to avoid circular imports
    # when the worker module imports this config.
    from src.workers.tasks import process_document

    functions = [process_document]

    redis_settings = get_redis_settings()

    # Worker concurrency: number of jobs processed simultaneously
    max_jobs = settings.arq_max_jobs

    # Job timeout: seconds before a job is killed (embedding can be slow)
    job_timeout = settings.arq_job_timeout

    # Total attempts per job. process_document raises arq.Retry with an
    # exponential defer for transient errors and stops retrying itself on the
    # last attempt (see src/workers/tasks.py).
    max_tries = settings.arq_max_tries

    # Worker health check interval
    health_check_interval = 60

    # Worker name (shows in logs)
    worker_name = "document-ingestion-worker"

    # Log all job results (useful for debugging, disable in prod for performance)
    log_results = True


logger.info(
    "ARQ worker settings loaded: redis=%s:%s max_jobs=%s timeout=%s",
    settings.redis_host,
    settings.redis_port,
    settings.arq_max_jobs,
    settings.arq_job_timeout,
)
