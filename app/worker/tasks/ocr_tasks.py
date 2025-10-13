# app/worker/tasks/ocr_tasks.py

import base64
import uuid
import cv2
import numpy as np
import requests
import structlog
import re
from time import perf_counter_ns
from pathlib import Path
from celery import chain, group
from requests.exceptions import RequestException
from typing import TYPE_CHECKING
import sys
from app.core.config import settings
# DEFER heavy import to runtime to avoid ImportError in lightweight workers
# from app.services.pipeline_service import PipelineService
from app.utils.file_io import pdf_to_images
from app.worker.celery_app import app
from app.worker.state_manager import StateManager
from app.core.logging import correlation_id_var
from app.utils.orientation import correct_images  # NEW
from app.utils.performance_logging import log_stage_timing  # NEW

if TYPE_CHECKING:
    # Only for type checkers; not executed at runtime
    from ...services.pipeline_service import PipelineService

logger = structlog.get_logger(__name__)

# --- Lazy Loading and Micro-tasks (These remain unchanged) ---

pipeline_singleton: 'PipelineService | None' = None
ALLOWED_FILE_EXTENSIONS = {".jpeg", ".png", ".pdf", ".jpg", ".tif", ".tiff"}

def get_pipeline_service() -> 'PipelineService':
    global pipeline_singleton
    if pipeline_singleton is None:
        logger.debug("lazy_loading.pipeline_service.initializing")
        try:
            # Prefer absolute import first
            from app.services.pipeline_service import PipelineService  # type: ignore
            logger.debug("lazy_loading.pipeline_service.import_ok", origin=str(PipelineService.__module__))
        except (ModuleNotFoundError, ImportError) as e_abs:
            # Fallback to package-relative import in case of sys.path/module shadowing issues OR name import errors
            logger.warning(
                "lazy_loading.pipeline_service.abs_import_failed",
                error=str(e_abs), sys_path=list(sys.path)
            )
            try:
                from ...services.pipeline_service import PipelineService  # type: ignore
                logger.debug("lazy_loading.pipeline_service.import_ok", origin=str(PipelineService.__module__))
            except Exception as e_rel:
                logger.critical(
                    "lazy_loading.pipeline_service.failed",
                    error=str(e_rel), sys_path=list(sys.path), exc_info=True
                )
                raise
        except Exception as e:
            logger.critical("lazy_loading.pipeline_service.failed", error=str(e), sys_path=list(sys.path), exc_info=True)
            raise
        pipeline_singleton = PipelineService(settings.model_dump())
        logger.debug("lazy_loading.pipeline_service.success")
    return pipeline_singleton

@app.task(name="ocr.pipeline.detect_lines", acks_late=True, bind=True)
def detect_lines_task(self, request_id: str, page_index: int) -> int:
    pipeline = get_pipeline_service()
    state = StateManager(request_id)
    image = state.load_page_image(page_index)
    start_ns = perf_counter_ns()
    telemetry = {"request_id": request_id, "page_index": page_index}
    line_boxes = pipeline.detect_lines(image, telemetry=telemetry)
    end_ns = perf_counter_ns()
    duration_ns = end_ns - start_ns
    state.save_line_boxes(page_index, line_boxes)
    logger.debug("detect_lines.success", page=page_index + 1, lines_found=len(line_boxes))
    log_stage_timing(
        "detect_lines",
        duration_ns=duration_ns,
        request_id=request_id,
        page_index=page_index,
        start_ns=start_ns,
        end_ns=end_ns,
        extra={
            "line_count": len(line_boxes),
            "height": int(image.shape[0]),
            "width": int(image.shape[1]),
        },
    )
    return page_index

@app.task(name="ocr.pipeline.detect_words", acks_late=True, bind=True)
def detect_words_task(self, request_id: str, page_index: int) -> int:
    pipeline = get_pipeline_service()
    state = StateManager(request_id)
    image = state.load_page_image(page_index)
    line_boxes = state.load_line_boxes(page_index)
    start_ns = perf_counter_ns()
    telemetry = {"request_id": request_id, "page_index": page_index}
    word_polygons = pipeline.detect_words(image, line_boxes, telemetry=telemetry)
    end_ns = perf_counter_ns()
    duration_ns = end_ns - start_ns
    state.save_word_polygons(page_index, word_polygons)
    logger.debug("detect_words.success", page=page_index + 1)
    total_words = 0
    if isinstance(word_polygons, list):
        for group in word_polygons:
            if isinstance(group, list):
                total_words += len(group)
    log_stage_timing(
        "detect_words",
        duration_ns=duration_ns,
        request_id=request_id,
        page_index=page_index,
        start_ns=start_ns,
        end_ns=end_ns,
        extra={
            "line_count": len(line_boxes),
            "word_group_count": len(word_polygons) if isinstance(word_polygons, list) else None,
            "word_count": total_words,
        },
    )
    return page_index

@app.task(name="ocr.pipeline.recognize_page", acks_late=True, bind=True)
def recognize_page_task(self, request_id: str, page_index: int) -> dict:
    pipeline = get_pipeline_service()
    state = StateManager(request_id)
    image = state.load_page_image(page_index)
    # Create page_id for debug output
    page_id = f"page{page_index}"
    line_boxes = state.load_line_boxes(page_index)
    word_polygons = state.load_word_polygons(page_index)
    start_ns = perf_counter_ns()
    telemetry = {"request_id": request_id, "page_index": page_index}
    full_text, confidence = pipeline.recognize_page(
        image,
        line_boxes,
        word_polygons,
        page_id=page_id,
        telemetry=telemetry
    )
    end_ns = perf_counter_ns()
    duration_ns = end_ns - start_ns
    state.save_page_result(page_index, full_text, confidence)
    logger.debug("recognize_page.success", page=page_index + 1, confidence=confidence)
    log_stage_timing(
        "recognize_page",
        duration_ns=duration_ns,
        request_id=request_id,
        page_index=page_index,
        start_ns=start_ns,
        end_ns=end_ns,
        extra={
            "char_count": len(full_text) if isinstance(full_text, str) else None,
            "confidence": confidence,
        },
    )
    return {"page_index": page_index}

@app.task(name='app.worker.tasks.send_webhook_result', bind=True, autoretry_for=(RequestException,), retry_kwargs={"max_retries": 3, "countdown": 10})
def send_webhook_result(self, webhook_url: str, payload: dict, **kwargs):
    guid = payload.get('guid')
    task_id = payload.get('task_id')
    retry_count = self.request.retries

    logger.debug(
        "webhook.attempt.start",
        guid=guid,
        task_id=task_id,
        webhook_url=webhook_url,
        retry_count=retry_count,
        max_retries=3,
        ssl_verify=not settings.ALLOW_INSECURE_WEBHOOKS
    )

    try:
        response = requests.post(
            webhook_url,
            json=payload,
            verify=not settings.ALLOW_INSECURE_WEBHOOKS,
            timeout=60
        )
        response.raise_for_status()

        logger.debug(
            "webhook.send.success",
            guid=guid,
            task_id=task_id,
            webhook_url=webhook_url,
            status_code=response.status_code,
            response_time_ms=response.elapsed.total_seconds() * 1000,
            retry_count=retry_count
        )

    except requests.exceptions.Timeout as exc:
        logger.error(
            "webhook.send.timeout",
            guid=guid,
            task_id=task_id,
            webhook_url=webhook_url,
            retry_count=retry_count,
            error=str(exc)
        )
        raise self.retry(exc=exc)

    except requests.exceptions.ConnectionError as exc:
        logger.error(
            "webhook.send.connection_error",
            guid=guid,
            task_id=task_id,
            webhook_url=webhook_url,
            retry_count=retry_count,
            error=str(exc)
        )
        raise self.retry(exc=exc)

    except requests.exceptions.HTTPError as exc:
        logger.error(
            "webhook.send.http_error",
            guid=guid,
            task_id=task_id,
            webhook_url=webhook_url,
            status_code=exc.response.status_code if exc.response else None,
            response_body=exc.response.text[:500] if exc.response else None,
            retry_count=retry_count,
            error=str(exc)
        )
        raise self.retry(exc=exc)

    except RequestException as exc:
        logger.error(
            "webhook.send.failed",
            guid=guid,
            task_id=task_id,
            webhook_url=webhook_url,
            retry_count=retry_count,
            error=str(exc),
            error_type=type(exc).__name__
        )
        raise self.retry(exc=exc)

    except Exception as exc:
        # Non-retryable exception
        logger.critical(
            "webhook.send.critical_error",
            guid=guid,
            task_id=task_id,
            webhook_url=webhook_url,
            retry_count=retry_count,
            error=str(exc),
            error_type=type(exc).__name__,
            exc_info=True
        )
        raise


# --- Postprocessing Functions ---

def postprocess_ocr_text(text: str, custom_replacements: dict = None, replacements_json_path: str = None) -> str:
    if not text:
        return text

    # Import here to avoid circular imports
    from app.utils.text_processing import (
        join_spaced_numbers,
        fix_dash_positioning,
        fix_period_positioning,
        fix_dash_comma_spacing,
        convert_mixed_digit_sequences,
        apply_custom_replacements,
        remove_parentheses,
    )

    # Remove parentheses first (before other processing)
    text = remove_parentheses(text)

    # Join spaced numbers first
    text = join_spaced_numbers(text)

    # Ensure digits adjacent to Persian digits use the Persian glyphs
    text = convert_mixed_digit_sequences(text)

    # Fix dash positioning (move dashes from before numbers to after)
    text = fix_dash_positioning(text)

    # Fix period positioning (move periods from before Persian numbers to after)
    text = fix_period_positioning(text)

    # Fix dash and comma spacing (remove spaces around dashes and Persian commas)
    text = fix_dash_comma_spacing(text)

    # Apply custom replacements if provided (either dict or JSON file)
    if custom_replacements or replacements_json_path:
        text = apply_custom_replacements(text, custom_replacements, replacements_json_path)

    # Normalize spacing: replace multiple consecutive spaces with at most 2 spaces
    text = re.sub(r' {2,}', '  ', text)

    return text.strip()

# --- Orchestrator and Finalizer Tasks (MODIFIED) ---

@app.task(name='app.worker.tasks.process_ocr_task', acks_late=True, bind=True)
def process_ocr_task(
    self,
    file_content: bytes,
    metadata: dict,
    webhook_url: str | None = None,
    correlation_id: str | None = None,
    workflow_id: str | None = None,
):
    """
    The main entry point task. It dispatches the parallel OCR workflow
    and returns the ID of the main chord workflow for polling.
    """
    request_id = workflow_id or str(uuid.uuid4())
    correlation_context = correlation_id or request_id
    guid = metadata.get('guid', request_id)

    # Normalise and validate the declared file format against the allow-list.
    declared_format = (metadata.get('format') or metadata.get('file_extension') or "").lower()
    if declared_format and not declared_format.startswith("."):
        declared_format = f".{declared_format}"

    if not declared_format:
        filename = metadata.get("filename") or ""
        declared_format = Path(filename).suffix.lower()

    if declared_format not in ALLOWED_FILE_EXTENSIONS:
        logger.error(
            "ocr_task.unsupported_file_format",
            guid=guid,
            declared_format=declared_format,
            allowed_extensions=list(ALLOWED_FILE_EXTENSIONS)
        )
        raise ValueError(f"Unsupported file format: {declared_format or 'unknown'}")

    correlation_id_var.set(correlation_context)
    logger.debug("ocr_task.received", guid=guid, task_id=request_id, file_format=declared_format)

    # Debug: Log the type and size of file_content
    logger.debug(
        "ocr_task.file_content_debug",
        guid=guid,
        content_type=type(file_content).__name__,
        content_size=len(file_content) if hasattr(file_content, '__len__') else 'unknown'
    )

    try:
        # Ensure file_content is bytes (handle Celery serialization quirks)
        if isinstance(file_content, str):
            # If it's a string, it's likely base64-encoded due to JSON serialization
            logger.debug("ocr_task.decoding_base64_string", guid=guid)
            file_bytes = base64.b64decode(file_content)
        elif isinstance(file_content, bytes):
            file_bytes = file_content
        else:
            logger.error("ocr_task.unexpected_type", guid=guid, type=type(file_content).__name__)
            raise ValueError(f"Unexpected file_content type: {type(file_content).__name__}")

        is_pdf = declared_format == ".pdf"

        if is_pdf:
            images = pdf_to_images(file_bytes)
        else:
            image_array = np.frombuffer(file_bytes, np.uint8)
            decoded_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if decoded_image is None:
                logger.error(
                    "ocr_task.image_decode_failed",
                    guid=guid,
                    declared_format=declared_format
                )
                raise ValueError("Failed to decode file as an image. The file may be corrupt or in an unsupported format.")
            images = [decoded_image]

        # NEW: Rotate images upright ONCE at the ingestion stage, so all downstream coordinates match
        orientation_start_ns = perf_counter_ns()
        images = correct_images(images)
        orientation_end_ns = perf_counter_ns()
        log_stage_timing(
            "orientation_correct",
            duration_ns=orientation_end_ns - orientation_start_ns,
            request_id=request_id,
            start_ns=orientation_start_ns,
            end_ns=orientation_end_ns,
            extra={"page_count": len(images)}
        )

        state = StateManager(request_id)
        state.save_initial_images(images)

        page_indices = list(range(len(images)))
        if not page_indices:
            logger.warning("ocr_task.no_pages_found", guid=guid)
            return None

        line_stage = group(
            detect_lines_task.si(request_id, page_index)
            for page_index in page_indices
        )

        word_stage = group(
            detect_words_task.si(request_id, page_index)
            for page_index in page_indices
        )

        recognition_stage = group(
            recognize_page_task.si(request_id, page_index)
            for page_index in page_indices
        )

        workflow = chain(
            line_stage,
            word_stage,
            recognition_stage,
            finalize_and_notify_task.s(request_id=request_id, guid=guid, webhook_url=webhook_url)
        )
        async_result = workflow.apply_async(task_id=request_id, correlation_id=correlation_context)
        logger.debug("ocr_task.workflow_dispatched", guid=guid, workflow_task_id=async_result.id)
        return async_result.id
    except Exception as e:
        logger.exception("ocr_task.initialization_failed", guid=guid, error=str(e))
        # This makes the task fail, so the frontend polling will receive a 'FAILURE' status.
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise

@app.task(name="ocr.pipeline.finalize_and_notify", acks_late=True)
def finalize_and_notify_task(page_results: list, request_id: str, guid: str, webhook_url: str | None = None):
    """
    Assembles the final document, forcefully corrects text encoding,
    and returns it for the polling mechanism.
    """
    state = StateManager(request_id)
    page_indices: list[int] = []
    if isinstance(page_results, (list, tuple)):
        for res in page_results:
            if isinstance(res, dict) and 'page_index' in res:
                page_indices.append(int(res['page_index']))
    else:
        logger.warning(
            "finalize.unexpected_page_results_type",
            guid=guid,
            result_type=type(page_results).__name__
        )

    if not page_indices:
        try:
            page_indices = state.load_page_indices()
            logger.debug("finalize.page_indices_loaded_from_state", guid=guid, count=len(page_indices))
        except KeyError:
            logger.error("finalize.missing_page_indices", guid=guid)
            page_indices = []

    all_pages = state.load_all_page_results(page_indices)
 
    # Join the text parts from all pages
    original_text = "\n\nصفحه\n\n".join([p['text'] for p in all_pages])

    # --- FINAL ENCODING FIX ---
    # This standard pattern corrects text that was decoded incorrectly as Latin-1
    # when it should have been UTF-8. We apply it directly.
    try:
        full_text = original_text.encode('latin-1').decode('utf-8')
    except Exception as e:
        logger.warning(
            "finalize.encoding_fix_failed",
            guid=guid,
            error=str(e),
            fallback_used=True
        )
        # Fallback in the rare case the text is already correct
        full_text = original_text
    # ---------------------------

    # --- POSTPROCESSING ---
    full_text = postprocess_ocr_text(full_text)
    # ---------------------------

    avg_confidence = sum(p['confidence'] for p in all_pages) / len(all_pages) if all_pages else 0.0

    final_payload = {
        "task_id": request_id,  # Add task_id to match documentation
        "guid": guid,
        "text": base64.b64encode(full_text.encode('utf-8')).decode('utf-8'),
        "confidence": avg_confidence,
        "status": "completed",
        "error": ""
    }

    logger.debug("finalize.success", guid=guid, total_pages=len(all_pages), confidence=avg_confidence)

    if webhook_url:
        logger.debug(
            "webhook.dispatch",
            guid=guid,
            task_id=request_id,
            webhook_url=webhook_url,
            payload_size=len(final_payload.get('text', '')),
            status=final_payload.get('status')
        )
        send_webhook_result.delay(webhook_url, final_payload, correlation_id=correlation_id_var.get())
    else:
        logger.debug(
            "webhook.skipped",
            guid=guid,
            task_id=request_id,
            reason="no_webhook_url_provided"
        )

    return final_payload
