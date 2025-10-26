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

def add_short_context_ids(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Add shortened IDs for better readability while preserving full IDs."""
    # Shorten correlation_id for console display
    if "correlation_id" in event_dict:
        full_id = event_dict["correlation_id"]
        event_dict["req_id"] = full_id[:8] if len(full_id) > 8 else full_id
    
    # Shorten GUID
    if "guid" in event_dict:
        full_guid = event_dict["guid"]
        if isinstance(full_guid, str) and len(full_guid) > 8:
            event_dict["file_id"] = full_guid[:8]
    
    # Shorten task_id
    if "task_id" in event_dict:
        full_task = event_dict["task_id"]
        if isinstance(full_task, str) and len(full_task) > 8:
            event_dict["task"] = full_task[:8]
    
    return event_dict

class InfoOnlyFilter(logging.Filter):
    """Filter that only allows INFO level logs."""
    def filter(self, record):
        return record.levelno == logging.INFO

class WarningOnlyFilter(logging.Filter):
    """Filter that only allows WARNING level logs."""
    def filter(self, record):
        return record.levelno == logging.WARNING

class DebugOnlyFilter(logging.Filter):
    """Filter that only allows DEBUG level logs."""
    def filter(self, record):
        return record.levelno == logging.DEBUG

class ErrorCriticalFilter(logging.Filter):
    """Filter that only allows ERROR and CRITICAL level logs."""
    def filter(self, record):
        return record.levelno >= logging.ERROR

class ThirdPartyDebugFilter(logging.Filter):
    """Filter out debug logs from noisy third-party libraries."""
    NOISY_LOGGERS = [
        "matplotlib",
        "PIL",
        "urllib3",
        "asyncio",
        "multipart",
        "celery.worker.strategy",
        "celery.app.trace",
        "celery.worker.consumer",
        "kombu",
        "amqp",
    ]
    
    def filter(self, record):
        # Allow all non-debug logs
        if record.levelno > logging.DEBUG:
            return True
        # For debug logs, filter out noisy third-party libraries
        return not any(record.name.startswith(noisy) for noisy in self.NOISY_LOGGERS)

class NoCeleryFilter(logging.Filter):
    """Filter out ALL Celery-related logs from file handlers."""
    CELERY_LOGGERS = [
        "celery",
        "kombu",
        "amqp",
        "billiard",
        "vine",
    ]
    
    def filter(self, record):
        # Block all Celery-related logs
        return not any(record.name.startswith(celery_logger) for celery_logger in self.CELERY_LOGGERS)

def configure_logging():
    """
    Configures a sophisticated logging system using structlog and standard logging.
    - Logs are structured as JSON for machine-readability in files.
    - Logs are also sent to the console for development visibility.
    - Each log level (DEBUG, INFO, WARNING, ERROR/CRITICAL) is written to its own dedicated file.
    - No log duplication between files - each level goes to exactly one file.
    - Celery logs appear ONLY on console, not in log files.
    - Third-party library logs are filtered to reduce noise.
    """
    # 1. Define shared processors for structlog
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.CallsiteParameterAdder(
            parameters=(
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
            )
        ),
        add_correlation_id,
        add_short_context_ids,  # Add shortened IDs for readability
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
        processor=structlog.dev.ConsoleRenderer(
            colors=True,
            pad_event=30,  # Pad event names for better alignment
            exception_formatter=structlog.dev.plain_traceback,
        ),
    )
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(ThirdPartyDebugFilter())  # Filter noisy third-party debug logs from console

    # --- JSON Formatter for file logs ---
    json_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
    )

    # --- Handler for Application Info Logs (INFO only) ---
    info_log_path = settings.LOG_FILE_PATH.parent / "info.log"
    info_log_handler = RotatingFileHandler(
        filename=info_log_path,
        maxBytes=100 * 1024 * 1024,  # 100 MB
        backupCount=7,
        encoding="utf-8"
    )
    info_log_handler.setFormatter(json_formatter)
    info_log_handler.addFilter(InfoOnlyFilter())
    info_log_handler.addFilter(NoCeleryFilter())  # Exclude Celery logs from file

    # --- Handler for Application Warning Logs (WARNING only) ---
    warning_log_path = settings.LOG_FILE_PATH.parent / "warnings.log"
    warning_log_handler = RotatingFileHandler(
        filename=warning_log_path,
        maxBytes=100 * 1024 * 1024,  # 100 MB
        backupCount=7,
        encoding="utf-8"
    )
    warning_log_handler.setFormatter(json_formatter)
    warning_log_handler.addFilter(WarningOnlyFilter())
    warning_log_handler.addFilter(NoCeleryFilter())  # Exclude Celery logs from file

    # --- Handler for Application Error Logs (ERROR/CRITICAL only) ---
    error_log_path = settings.LOG_FILE_PATH.parent / "errors.log"
    error_log_handler = RotatingFileHandler(
        filename=error_log_path,
        maxBytes=100 * 1024 * 1024,  # 100 MB
        backupCount=7,
        encoding="utf-8"
    )
    error_log_handler.setFormatter(json_formatter)
    error_log_handler.addFilter(ErrorCriticalFilter())
    error_log_handler.addFilter(NoCeleryFilter())  # Exclude Celery logs from file

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
    debug_log_handler.addFilter(ThirdPartyDebugFilter())  # Filter noisy third-party debug logs
    debug_log_handler.addFilter(NoCeleryFilter())  # Exclude Celery logs from file

    
    # 4. Configure specific loggers to use the handlers

    # --- Root Logger (for FastAPI app and general logs) ---
    root_logger = logging.getLogger()
    # Clear any existing handlers to prevent duplication
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)      # Log to console
    root_logger.addHandler(info_log_handler)     # Log INFO to info.log
    root_logger.addHandler(warning_log_handler)  # Log WARNING to warnings.log
    root_logger.addHandler(error_log_handler)    # Log ERROR/CRITICAL to errors.log
    root_logger.addHandler(debug_log_handler)    # Log DEBUG to debug.log
    root_logger.setLevel(logging.DEBUG)          # Changed to DEBUG for more detailed logging

    # --- Celery Logger (console only - no files) ---
    celery_logger = logging.getLogger("celery")
    celery_logger.handlers.clear()  # Clear existing handlers
    celery_logger.addHandler(console_handler)  # Log to console only
    celery_logger.setLevel(logging.INFO)  # Show INFO and above
    celery_logger.propagate = False  # Prevent propagation to root logger

    performance_log_path = None
    if settings.debug:
        performance_log_path = settings.LOG_FILE_PATH.parent / "performance_debug.log"
        performance_handler = RotatingFileHandler(
            filename=performance_log_path,
            maxBytes=50 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8"
        )
        performance_handler.setFormatter(json_formatter)
        performance_handler.addFilter(DebugOnlyFilter())

        performance_logger = logging.getLogger("app.performance")
        performance_logger.handlers.clear()  # Clear existing handlers
        performance_logger.addHandler(performance_handler)
        performance_logger.setLevel(logging.DEBUG)
        performance_logger.propagate = False

    # --- Application-specific loggers with detailed logging ---
    app_loggers = [
        "app.services",
        "app.worker",
        "app.main",
        "app.core"
    ]

    for logger_name in app_loggers:
        app_logger = logging.getLogger(logger_name)
        app_logger.handlers.clear()  # Clear existing handlers to prevent duplication
        app_logger.addHandler(console_handler)
        app_logger.addHandler(info_log_handler)     # Log INFO to info.log
        app_logger.addHandler(warning_log_handler)  # Log WARNING to warnings.log
        app_logger.addHandler(error_log_handler)    # Log ERROR/CRITICAL to errors.log
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
        "multipart": logging.ERROR,  # Silence multipart form parser logging
        "python_multipart": logging.ERROR,  # Silence multipart form parser logging
        "starlette": logging.WARNING,  # Reduce Starlette noise
        # Celery-related loggers (console only - files filtered via NoCeleryFilter)
        "kombu": logging.WARNING,  # Reduce Kombu (Celery messaging) noise
        "amqp": logging.WARNING,  # Reduce AMQP noise
        "billiard": logging.WARNING,  # Reduce Billiard (Celery worker pool) noise
        "vine": logging.WARNING,  # Reduce Vine (Celery promises) noise
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
        description="Advanced logging configured successfully with separate files per log level. Celery logs: console only.",
        info_log_path=str(info_log_path),
        warning_log_path=str(warning_log_path),
        error_log_path=str(error_log_path),
        debug_log_path=str(debug_log_path),
        performance_log_path=str(performance_log_path) if performance_log_path else None,
        celery_logs_console_only=True
    )
