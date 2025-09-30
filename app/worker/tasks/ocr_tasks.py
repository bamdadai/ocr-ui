# app/worker/tasks/ocr_tasks.py

import base64
import uuid
import cv2
import numpy as np
import requests
import structlog
import re
from celery import chain, group, chord
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

if TYPE_CHECKING:
    # Only for type checkers; not executed at runtime
    from ...services.pipeline_service import PipelineService

logger = structlog.get_logger(__name__)

# --- Lazy Loading and Micro-tasks (These remain unchanged) ---

pipeline_singleton: 'PipelineService | None' = None

def get_pipeline_service() -> 'PipelineService':
    global pipeline_singleton
    if pipeline_singleton is None:
        logger.info("lazy_loading.pipeline_service.initializing")
        try:
            # Prefer absolute import first
            from app.services.pipeline_service import PipelineService  # type: ignore
            logger.info("lazy_loading.pipeline_service.import_ok", origin=str(PipelineService.__module__))
        except (ModuleNotFoundError, ImportError) as e_abs:
            # Fallback to package-relative import in case of sys.path/module shadowing issues OR name import errors
            logger.warning(
                "lazy_loading.pipeline_service.abs_import_failed",
                error=str(e_abs), sys_path=list(sys.path)
            )
            try:
                from ...services.pipeline_service import PipelineService  # type: ignore
                logger.info("lazy_loading.pipeline_service.import_ok", origin=str(PipelineService.__module__))
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
        logger.info("lazy_loading.pipeline_service.success")
    return pipeline_singleton

@app.task(name="ocr.pipeline.detect_lines", acks_late=True, bind=True)
def detect_lines_task(self, context: dict) -> dict:
    pipeline = get_pipeline_service()
    state = StateManager(context["request_id"])
    image = state.load_page_image(context["page_index"])
    line_boxes = pipeline.detect_lines(image)
    context["line_boxes"] = line_boxes
    logger.info("detect_lines.success", page=context["page_index"] + 1, lines_found=len(line_boxes))
    return context

@app.task(name="ocr.pipeline.detect_words", acks_late=True, bind=True)
def detect_words_task(self, context: dict) -> dict:
    pipeline = get_pipeline_service()
    state = StateManager(context["request_id"])
    image = state.load_page_image(context["page_index"])
    word_polygons = pipeline.detect_words(image, context["line_boxes"])
    context["word_polygons"] = word_polygons
    logger.info("detect_words.success", page=context["page_index"] + 1)
    return context

@app.task(name="ocr.pipeline.recognize_page", acks_late=True, bind=True)
def recognize_page_task(self, context: dict) -> dict:
    pipeline = get_pipeline_service()
    state = StateManager(context["request_id"])
    image = state.load_page_image(context["page_index"])
    full_text, confidence = pipeline.recognize_page(image, context["line_boxes"], context["word_polygons"])
    state.save_page_result(context["page_index"], full_text, confidence)
    logger.info("recognize_page.success", page=context["page_index"] + 1, confidence=confidence)
    return {"page_index": context["page_index"]}

@app.task(name='app.worker.tasks.send_webhook_result', bind=True, autoretry_for=(RequestException,), retry_kwargs={"max_retries": 3, "countdown": 10})
def send_webhook_result(self, webhook_url: str, payload: dict, **kwargs):
    guid = payload.get('guid')
    try:
        logger.info("webhook.sending", guid=guid, url=webhook_url)
        response = requests.post(webhook_url, json=payload, verify=not settings.ALLOW_INSECURE_WEBHOOKS, timeout=10)
        response.raise_for_status()
        logger.info("webhook.send.success", guid=guid, status_code=response.status_code)
    except RequestException as exc:
        logger.error("webhook.send.failed", guid=guid, error=str(exc))
        raise self.retry(exc=exc)


# --- Postprocessing Functions ---

def postprocess_ocr_text(text: str) -> str:
    """
    Remove standalone 'x' or 'X' characters from OCR text.
    Removes single 'x'/'X' characters and sequences of two or more 'x'/'X' characters.
    Only removes 'x'/'X' that appear as separate characters, not as part of words.
    """
    if not text:
        return text

    # Remove sequences of two or more x/X characters (xx, xxx, XX, xX, etc.)
    text = re.sub(r'[xX]{2,}', '', text)

    # Remove standalone single x/X characters surrounded by whitespace
    text = re.sub(r'\s+[xX]\s+', '', text)

    return text.strip()

# --- Orchestrator and Finalizer Tasks (MODIFIED) ---

@app.task(name='app.worker.tasks.process_ocr_task', acks_late=True, bind=True)
def process_ocr_task(self, file_content: bytes, metadata: dict, webhook_url: str | None = None, correlation_id: str | None = None):
    """
    The main entry point task. It dispatches the parallel OCR workflow
    and returns the ID of the main chord workflow for polling.
    """
    request_id = correlation_id or str(uuid.uuid4())
    guid = metadata.get('guid', request_id)
    correlation_id_var.set(request_id)
    logger.info("ocr_task.received", guid=guid)
    try:
        # THE MORE LOGICAL WAY: Check the file's magic numbers to determine its type.
        # PDF files start with the bytes '%PDF'.
        is_pdf = file_content.startswith(b'%PDF')
        
        if is_pdf:
            images = pdf_to_images(file_content)
        else:
            image_array = np.frombuffer(file_content, np.uint8)
            decoded_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if decoded_image is None:
                raise ValueError("Failed to decode file as an image. The file may be corrupt or in an unsupported format.")
            images = [decoded_image]

        # NEW: Rotate images upright ONCE at the ingestion stage, so all downstream coordinates match
        images = correct_images(images)

        state = StateManager(request_id)
        state.save_initial_images(images)
        page_workflows = [chain(detect_lines_task.s(context={"request_id": request_id, "page_index": i}), detect_words_task.s(), recognize_page_task.s()) for i in range(len(images))]

        if not page_workflows:
            logger.warning("ocr_task.no_pages_found", guid=guid)
            return None

        workflow = chord(
            header=group(page_workflows),
            body=finalize_and_notify_task.s(request_id=request_id, guid=guid, webhook_url=webhook_url)
        )
        async_result = workflow.apply_async(correlation_id=request_id)
        logger.info("ocr_task.workflow_dispatched", guid=guid, chord_task_id=async_result.id)
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
    page_indices = [res['page_index'] for res in page_results]
    all_pages = state.load_all_page_results(page_indices)

    # Join the text parts from all pages
    original_text = "\n\n--- PAGE BREAK ---\n\n".join([p['text'] for p in all_pages])

    # --- FINAL ENCODING FIX ---
    # This standard pattern corrects text that was decoded incorrectly as Latin-1
    # when it should have been UTF-8. We apply it directly.
    try:
        full_text = original_text.encode('latin-1').decode('utf-8')
    except Exception:
        # Fallback in the rare case the text is already correct
        full_text = original_text
    # ---------------------------

    # --- POSTPROCESSING ---
    # Remove standalone 'x' or 'X' characters from OCR output
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

    logger.info("finalize.success", guid=guid, total_pages=len(all_pages))

    if webhook_url:
        send_webhook_result.delay(webhook_url, final_payload, correlation_id=correlation_id_var.get())

    return final_payload
