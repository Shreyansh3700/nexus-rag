from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone, timedelta

from src.config.settings import settings

IST = timezone(timedelta(hours=5, minutes=30))
_user_id: ContextVar[str] = ContextVar("user_id", default="-")
_configured = False


class ISTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):  # noqa: N802
        timestamp = datetime.fromtimestamp(record.created, tz=IST)
        return timestamp.strftime(datefmt or "%Y-%m-%d %H:%M:%S IST")


class UserIdFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "user_id") or not record.user_id:
            record.user_id = _user_id.get()
        return True


def configure_logging(level: str | int | None = None) -> None:
    global _configured
    if _configured:
        return

    resolved_level = level or settings.log_level.upper()
    handler = logging.StreamHandler()
    handler.setFormatter(ISTFormatter("%(asctime)s | %(levelname)s | %(user_id)s | %(message)s"))
    handler.addFilter(UserIdFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved_level)
    root.propagate = False

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True

    _configured = True


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name)


def set_user_id(user_id: str | None) -> None:
    _user_id.set(str(user_id) if user_id else "-")


@contextmanager
def bind_user_id(user_id: str | None):
    token = _user_id.set(str(user_id) if user_id else "-")
    try:
        yield
    finally:
        _user_id.reset(token)
