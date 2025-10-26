# app/api/v1/endpoints/ocr.py

import json
import uuid
from pathlib import Path
from typing import List

import structlog
from celery.result import AsyncResult
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.logging import correlation_id_var
from app.core.log_events import LogEvent
from app.worker.celery_app import app as celery_app
from app.worker.state_manager import StateManager
from app.schemas.ocr import (
    TaskQueueItem,
    TaskStatusResponse,
    FinalOCRResult,
    TaskErrorResult,
)

# Define the router with a prefix and tags for organization
router = APIRouter(tags=["OCR Processing"])

ALLOWED_FILE_EXTENSIONS = {".jpeg", ".png", ".pdf", ".jpg", ".tif", ".tiff"}
logger = structlog.get_logger(__name__)

@router.post("/ocr", response_model=List[TaskQueueItem])
async def create_ocr_task(
    files: List[UploadFile] = File(..., description="List of files to be processed."),
    metadata: str = Form(None, description="JSON-encoded string containing file metadata (must match the number of uploaded files). For UI requests, this can be omitted."),
    webhook_url: str = Form(None, description="URL where OCR results will be sent upon completion. For UI requests, results are polled directly."),
):
    """
    Uploads multiple files for OCR processing and returns per-file queue results.
    The results will be sent to the provided webhook URL asynchronously, or polled directly for UI requests.
    """
    # Normalize webhook URL: strip whitespace and convert empty strings to None
    normalized_webhook = None
    if webhook_url:
        stripped = webhook_url.strip()
        normalized_webhook = stripped if stripped else None

    logger.info(
        LogEvent.API_REQUEST_RECEIVED,
        file_count=len(files),
        webhook_url=normalized_webhook,
        has_metadata=metadata is not None
    )

    # Determine if webhook should be used (based on webhook_url presence)
    use_webhook = bool(normalized_webhook)

    logger.debug(
        "ocr.webhook.mode",
        use_webhook=use_webhook,
        webhook_url=normalized_webhook,
        has_metadata=metadata is not None
    )

    if metadata is None:
        # UI request or simple request without metadata - generate metadata automatically
        metadata_list = []
        for file in files:
            metadata_list.append(
                {
                    "guid": str(uuid.uuid4()),
                    "format": (Path(file.filename or "").suffix or "").lower(),
                }
            )
    else:
        try:
            metadata_list = json.loads(metadata) or []
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in metadata parameter.")

        if len(files) != len(metadata_list):
            raise HTTPException(status_code=400, detail="Number of files must match number of metadata entries.")

        if not all(isinstance(item, dict) for item in metadata_list):
            raise HTTPException(status_code=400, detail="Each metadata entry must be a JSON object.")

    queue_results: List[TaskQueueItem] = []
    correlation_id = correlation_id_var.get()

    for upload_file, file_metadata in zip(files, metadata_list):
        normalized_metadata = dict(file_metadata or {})
        guid = normalized_metadata.get("guid") or str(uuid.uuid4())
        normalized_metadata["guid"] = guid

        filename_suffix = (Path(upload_file.filename or "").suffix or "").lower()
        metadata_format = (normalized_metadata.get("format") or filename_suffix or "").lower()
        if metadata_format and not metadata_format.startswith("."):
            metadata_format = f".{metadata_format}"
        normalized_metadata["format"] = metadata_format

        if metadata_format not in ALLOWED_FILE_EXTENSIONS:
            logger.warning(LogEvent.API_UNSUPPORTED_FORMAT, guid=guid, format=metadata_format)
            queue_results.append(TaskQueueItem(task_id=None, status="error", guid=guid))
            continue

        try:
            task_name = "app.worker.tasks.process_ocr_task"
            workflow_task_id = str(uuid.uuid4())
            state = StateManager(workflow_task_id)

            file_bytes = await upload_file.read()
            full_metadata = {
                **normalized_metadata,
                "file_type": upload_file.content_type,
                "filename": upload_file.filename,
            }

            original_size = len(file_bytes)
            staged_metadata = state.save_upload_blob(file_bytes)
            staged_size = staged_metadata.get("size", original_size)
            staged_chunks = staged_metadata.get("chunks", 0)
            del file_bytes

            logger.debug(
                LogEvent.API_TASK_DISPATCHING,
                guid=guid,
                task_id=workflow_task_id,
                filename=upload_file.filename,
                file_format=metadata_format,
                webhook_url=normalized_webhook if use_webhook else None,
                use_webhook=use_webhook,
                staged_upload_size=staged_size,
                staged_upload_chunks=staged_chunks,
            )

            async_result = celery_app.send_task(
                name=task_name,
                kwargs={
                    "metadata": full_metadata,
                    "webhook_url": normalized_webhook if use_webhook else None,
                    "correlation_id": correlation_id,
                    "workflow_id": workflow_task_id,
                    "staged_upload": True,
                },
            )

            logger.info(
                LogEvent.API_TASK_DISPATCHED,
                guid=guid,
                task_id=workflow_task_id,
                celery_task_id=async_result.id,
                webhook_url=normalized_webhook if use_webhook else None
            )

        except HTTPException:
            raise
        except Exception as exc:
            logger.error(LogEvent.API_TASK_DISPATCH_FAILED, guid=guid, error=str(exc), exc_info=True)
            queue_results.append(TaskQueueItem(task_id=None, status="error", guid=guid))
            continue

        queue_results.append(TaskQueueItem(task_id=workflow_task_id, status="queued", guid=guid))

    logger.info(
        LogEvent.API_REQUEST_COMPLETED,
        total_files=len(files),
        queued_count=sum(1 for r in queue_results if r.status == "queued"),
        error_count=sum(1 for r in queue_results if r.status == "error"),
        webhook_url=normalized_webhook
    )

    return queue_results


@router.get("/ocr/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Retrieves the current status and final result of a task using its ID.
    """
    task_result = AsyncResult(task_id, app=celery_app)
    
    status = task_result.status
    result_payload = None

    if task_result.ready():
        if task_result.successful():
            # On success, populate with the FinalOCRResult model
            success_data = task_result.get()
            result_payload = FinalOCRResult(**success_data)
        else:
            # On failure, populate with the TaskErrorResult model
            error_info = str(task_result.info) # .info contains the exception
            result_payload = TaskErrorResult(error=error_info)
            
    return TaskStatusResponse(
        task_id=task_id,
        status=status,
        result=result_payload
    )
