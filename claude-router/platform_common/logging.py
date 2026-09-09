"""Structured JSON logging for CloudWatch Logs Insights / Fluent Bit."""
from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)

# Optional LogRecord attributes services may attach via logger.info(..., extra={...})
_OPTIONAL_ATTRS = (
    "steps",
    "latency_ms",
    "input_tokens",
    "output_tokens",
    "chunk_ids",
    "mcp_operation",
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        rid = request_id_ctx.get()
        if rid:
            log_obj["request_id"] = rid
        for attr in _OPTIONAL_ATTRS:
            if hasattr(record, attr):
                val = getattr(record, attr)
                if val is not None:
                    log_obj[attr] = val
        if record.exc_info:
            log_obj["error"] = self.formatException(record.exc_info)
        return json.dumps(log_obj, default=str)


def configure_json_logging(log_level: str = "INFO") -> None:
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(log_level)


def bind_request_id(request_id: str):
    """Set request_id for current context. Returns the ContextVar token."""
    return request_id_ctx.set(request_id)
