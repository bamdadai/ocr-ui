import time
import structlog
from celery import Celery
from celery.signals import task_prerun, task_postrun

from app.core.config import settings
from app.core.logging import configure_logging, correlation_id_var

# Configure logging for the worker process as soon as the module is imported.
# This ensures that all logs, including Celery's internal ones, are structured.
configure_logging()

logger = structlog.get_logger(__name__)

# Initialize Celery
app = Celery(
    'ocr_app',
    broker=settings.CELERY_BROKER_URL,
    broker_connection_timeout=30,
    broker_transport_options={'visibility_timeout': 3600},
    backend=settings.CELERY_BACKEND_URL,
    include=['app.worker.tasks.ocr_tasks']
)

# --- Configure Celery for robustness and specialized routing ---
app.conf.broker_connection_retry_on_startup = True
app.conf.task_track_started = True

# Set prefetch multiplier to 1 for better load balancing of long-running tasks.
# This prevents a worker from hoarding tasks it can't process yet.
app.conf.worker_prefetch_multiplier = 1

# --- Define Task Routes for Specialized Queues ---
# This is the core of specialized scaling. We can now run different workers
# that listen to different queues (e.g., a GPU worker for 'recognition').
app.conf.task_routes = {
    # OCR pipeline stages get distinct queues so workers can be scaled per stage.
    'ocr.pipeline.detect_lines': {'queue': 'ocr_lines'},
    'ocr.pipeline.detect_words': {'queue': 'ocr_words'},
    'ocr.pipeline.recognize_page': {'queue': 'ocr_recognize'},
    'ocr.pipeline.finalize_and_notify': {'queue': 'ocr_recognize'},

    # Dispatch/orchestration tasks stay on their existing queues.
    'app.worker.tasks.process_ocr_task': {'queue': 'dispatch'},
    'app.worker.tasks.send_webhook_result': {'queue': 'webhooks'},
}


@task_prerun.connect
def on_task_prerun(task_id=None, task=None, args=None, kwargs=None, **extras):
    """
    Runs before each task starts. It sets up the logging context by extracting
    the correlation_id, ensuring all logs for a request can be traced.
    """
    # Propagate the correlation_id from the parent task if available.
    correlation_id = kwargs.get("correlation_id")
    if correlation_id:
        correlation_id_var.set(correlation_id)

    # Store the start time on the task object to calculate duration later.
    task.start_time = time.time()

    # Log at DEBUG level for OCR pipeline tasks to reduce noise
    task_name = task.name
    if task_name in ('ocr.pipeline.detect_lines', 'ocr.pipeline.detect_words', 'ocr.pipeline.recognize_page'):
        logger.debug(
            "task_started",
            task_name=task_name,
            task_id=task_id
        )
    else:
        logger.info(
            "task_started",
            task_name=task_name,
            task_id=task_id
        )


@task_postrun.connect
def on_task_postrun(task_id=None, task=None, state=None, **kwargs):
    """
    Runs after each task finishes. It logs the completion event with its
    duration and status, then clears the logging context.
    """
    start_time = getattr(task, 'start_time', None)
    duration = time.time() - start_time if start_time else -1

    # Log at DEBUG level for OCR pipeline tasks to reduce noise
    task_name = task.name
    if task_name in ('ocr.pipeline.detect_lines', 'ocr.pipeline.detect_words', 'ocr.pipeline.recognize_page'):
        logger.debug(
            "task_finished",
            task_name=task_name,
            task_id=task_id,
            status=state,
            duration_seconds=round(duration, 4)
        )
    else:
        logger.info(
            "task_finished",
            task_name=task_name,
            task_id=task_id,
            status=state,
            duration_seconds=round(duration, 4)
        )

    # Clear the correlation ID to prevent it from leaking to other tasks
    # that might be processed by the same worker process.
    correlation_id_var.set(None)
