# app/core/logging_config.py

import logging
from logging.handlers import RotatingFileHandler
import sys
from contextvars import ContextVar
from typing import Optional

import structlog
from structlog.types import EventDict, WrappedLogger
from app.core.config import settings # برای دسترسی به مسیر فایل لاگ

# --- Context variable for request correlation ---
correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)

def add_correlation_id(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """A structlog processor to add the correlation ID from the context variable."""
    if (correlation_id := correlation_id_var.get()):
        event_dict["correlation_id"] = correlation_id
    return event_dict

class InfoErrorCriticalFilter(logging.Filter):
    """Filter that only allows INFO, WARNING, ERROR, and CRITICAL level logs."""
    def filter(self, record):
        return record.levelno >= logging.INFO

class DebugOnlyFilter(logging.Filter):
    """Filter that only allows DEBUG level logs."""
    def filter(self, record):
        return record.levelno == logging.DEBUG

def configure_logging():
    """
    Configures a sophisticated logging system using structlog and standard logging.
    - Logs are structured as JSON for machine-readability in files.
    - Logs are also sent to the console for development visibility.
    - Application and Celery logs are separated into different rotating files.
    - DEBUG logs are separated from INFO/WARNING/ERROR/CRITICAL logs.
    """
    # 1. Define shared processors for structlog
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        add_correlation_id,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # 2. Configure structlog to wrap the standard logging library
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # 3. Define Handlers for different outputs

    # --- Handler for Console Logs (human-readable) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(colors=True),
    )
    console_handler.setFormatter(console_formatter)

    # --- JSON Formatter for file logs ---
    json_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
    )

    # --- Handler for Application File Logs (INFO/WARNING/ERROR/CRITICAL) ---
    app_log_handler = RotatingFileHandler(
        filename=settings.LOG_FILE_PATH,
        maxBytes=100 * 1024 * 1024,  # 100 MB
        backupCount=7,   # Keep 7 old log files
        encoding="utf-8"
    )
    app_log_handler.setFormatter(json_formatter)
    app_log_handler.addFilter(InfoErrorCriticalFilter())

    # --- Handler for Application Debug Logs (DEBUG only) ---
    debug_log_path = settings.LOG_FILE_PATH.parent / "debug.log"
    debug_log_handler = RotatingFileHandler(
        filename=debug_log_path,
        maxBytes=100 * 1024 * 1024,  # 100 MB
        backupCount=7,
        encoding="utf-8"
    )
    debug_log_handler.setFormatter(json_formatter)
    debug_log_handler.addFilter(DebugOnlyFilter())

    # --- Handler for Celery File Logs (INFO/WARNING/ERROR/CRITICAL) ---
    celery_log_path = settings.LOG_FILE_PATH.parent / "celery.log"
    celery_log_handler = RotatingFileHandler(
        filename=celery_log_path,
        maxBytes=100 * 1024 * 1024,  # 100 MB
        backupCount=7,
        encoding="utf-8"
    )
    celery_log_handler.setFormatter(json_formatter)
    celery_log_handler.addFilter(InfoErrorCriticalFilter())

    # --- Handler for Celery Debug Logs (DEBUG only) ---
    celery_debug_log_path = settings.LOG_FILE_PATH.parent / "celery_debug.log"
    celery_debug_log_handler = RotatingFileHandler(
        filename=celery_debug_log_path,
        maxBytes=100 * 1024 * 1024,  # 100 MB
        backupCount=7,
        encoding="utf-8"
    )
    celery_debug_log_handler.setFormatter(json_formatter)
    celery_debug_log_handler.addFilter(DebugOnlyFilter())
    
    # 4. Configure specific loggers to use the handlers

    # --- Root Logger (for FastAPI app and general logs) ---
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)      # Log to console
    root_logger.addHandler(app_log_handler)      # Log INFO+ to app.log
    root_logger.addHandler(debug_log_handler)    # Log DEBUG to debug.log
    root_logger.setLevel(logging.DEBUG)          # Changed to DEBUG for more detailed logging

    # --- Celery Logger ---
    celery_logger = logging.getLogger("celery")
    celery_logger.addHandler(console_handler)           # Log to console
    celery_logger.addHandler(celery_log_handler)        # Log INFO+ to celery.log
    celery_logger.addHandler(celery_debug_log_handler)  # Log DEBUG to celery_debug.log
    celery_logger.setLevel(logging.DEBUG)               # Changed to DEBUG for more detailed logging
    celery_logger.propagate = False  # IMPORTANT: Prevents Celery logs from being duplicated in the root logger (app.log)

    # --- Application-specific loggers with detailed logging ---
    app_loggers = [
        "app.services",
        "app.worker",
        "app.main",
        "app.core"
    ]

    for logger_name in app_loggers:
        app_logger = logging.getLogger(logger_name)
        app_logger.addHandler(console_handler)
        app_logger.addHandler(app_log_handler)      # Log INFO+ to app.log
        app_logger.addHandler(debug_log_handler)    # Log DEBUG to debug.log
        app_logger.setLevel(logging.DEBUG)
        app_logger.propagate = False

    # --- Third-party library loggers (reduce noise but keep important info) ---
    third_party_loggers = {
        "uvicorn": logging.INFO,
        "uvicorn.access": logging.WARNING,  # Reduce access log noise
        "uvicorn.error": logging.INFO,
        "gunicorn": logging.INFO,
        "torch": logging.WARNING,  # Reduce PyTorch noise
        "transformers": logging.WARNING,  # Reduce transformers noise
        "ultralytics": logging.WARNING,  # Reduce ultralytics noise
        "paddleocr": logging.WARNING,  # Reduce PaddleOCR noise
    }
    
    for logger_name, level in third_party_loggers.items():
        logging.getLogger(logger_name).setLevel(level)

    # --- Silence Uvicorn's default loggers to prevent duplicate outputs ---
    for name in ["uvicorn.access", "uvicorn.error"]:
        logging.getLogger(name).handlers.clear()
        logging.getLogger(name).propagate = True

    logger = structlog.get_logger(__name__)
    logger.info(
        "logging.configured",
        description="Advanced logging configured successfully with separate debug logs.",
        app_log_path=str(settings.LOG_FILE_PATH),
        debug_log_path=str(debug_log_path),
        celery_log_path=str(celery_log_path),
        celery_debug_log_path=str(celery_debug_log_path)
    )