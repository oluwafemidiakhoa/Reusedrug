from __future__ import annotations

import logging
import os
from logging.config import dictConfig
from typing import Any, Dict

from opentelemetry import trace

LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "INFO")

_BASE_RECORD_FACTORY = logging.getLogRecordFactory()


def _log_record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
    record = _BASE_RECORD_FACTORY(*args, **kwargs)
    span = trace.get_current_span()
    if span and span.get_span_context().is_valid:
        span_context = span.get_span_context()
        record.trace_id = format(span_context.trace_id, "032x")
        record.span_id = format(span_context.span_id, "016x")
    else:
        record.trace_id = ""
        record.span_id = ""
    return record


def configure_logging() -> None:
    """Configure structured JSON logging for the application."""
    logging.setLogRecordFactory(_log_record_factory)
    config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "fmt": "%(asctime)s %(levelname)s %(name)s %(message)s %(trace_id)s %(span_id)s",
            },
        },
        "handlers": {
            "default": {
                "level": LOG_LEVEL,
                "class": "logging.StreamHandler",
                "formatter": "json",
            }
        },
        "loggers": {
            "": {"handlers": ["default"], "level": LOG_LEVEL},
            "uvicorn": {"handlers": ["default"], "level": LOG_LEVEL, "propagate": False},
            "uvicorn.error": {"handlers": ["default"], "level": LOG_LEVEL, "propagate": False},
            "uvicorn.access": {"handlers": ["default"], "level": LOG_LEVEL, "propagate": False},
        },
    }
    dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
