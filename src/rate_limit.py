"""Redis-backed rate limiting shared across routers.

Uses the same Redis instance as the ARQ task queue (REDIS_HOST/PORT/DB), so
limits are enforced correctly across multiple uvicorn workers/replicas
instead of resetting per-process.
"""
from fastapi import HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.auth import _extract_bearer_token, decode_access_token
from src.config.settings import settings

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=settings.redis_url,
)


def get_user_or_ip(request: Request) -> str:
    """Rate-limit key: authenticated user id if a valid bearer token is
    present, otherwise the client IP.

    slowapi calls this synchronously with only the raw Request, so it can't
    go through the async, DB-backed get_current_user dependency. Falling
    back to IP (rather than raising) keeps a missing/malformed/expired
    token from breaking the request here instead of at the real auth check.
    """
    try:
        payload = decode_access_token(
            _extract_bearer_token(request.headers.get("authorization"))
        )
        user_id = payload.get("sub")
        if isinstance(user_id, str) and user_id:
            return f"user:{user_id}"
    except HTTPException:
        pass
    return get_remote_address(request)
