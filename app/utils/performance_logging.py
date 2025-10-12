from __future__ import annotations

from typing import Any, Dict, Optional

import structlog

from app.core.config import settings

_logger = structlog.get_logger("app.performance")


def log_stage_timing(
    stage: str,
    duration_ms: Optional[float] = None,
    duration_ns: Optional[int] = None,
    *,
    request_id: Optional[str] = None,
    page_index: Optional[int] = None,
    start_ns: Optional[int] = None,
    end_ns: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Emit a debug log with timing information for profiling hot stages.

    Only logs when the application runs in debug mode to avoid noise in production.
    """
    if not settings.debug:
        return

    resolved_duration_ns: Optional[int] = duration_ns
    if resolved_duration_ns is None:
        if duration_ms is not None:
            resolved_duration_ns = int(duration_ms * 1_000_000)
        elif start_ns is not None and end_ns is not None:
            resolved_duration_ns = int(end_ns - start_ns)

    if resolved_duration_ns is None:
        return

    resolved_duration_ms = resolved_duration_ns / 1_000_000.0

    payload: Dict[str, Any] = {
        "stage": stage,
        "duration_ms": round(resolved_duration_ms, 6),
        "duration_ns": resolved_duration_ns,
    }

    if request_id is not None:
        payload["request_id"] = request_id
    if page_index is not None:
        payload["page_index"] = page_index
    if start_ns is not None:
        payload["started_monotonic_ns"] = start_ns
    if end_ns is not None:
        payload["ended_monotonic_ns"] = end_ns
    if extra:
        for key, value in extra.items():
            if value is not None:
                payload[key] = value

    _logger.debug("performance.stage_timing", **payload)
