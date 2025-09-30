# app/api/v1/endpoints/ocr.py

import json
from typing import Optional, Any, List
from celery.result import AsyncResult
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Depends

from app.core.logging import correlation_id_var
from app.worker.celery_app import app as celery_app
from app.schemas.ocr import TaskQueueResponse, TaskStatusResponse, FinalOCRResult, TaskErrorResult

# Define the router with a prefix and tags for organization
router = APIRouter(tags=["OCR Processing"])

@router.post("/ocr", response_model=TaskQueueResponse)
async def create_ocr_task(
    files: List[UploadFile] = File(..., description="List of files to be processed."),
    metadata: str = Form(None, description="JSON-encoded string containing file metadata (must match the number of uploaded files). For UI requests, this can be omitted."),
    webhook_url: str = Form(None, description="URL where OCR results will be sent upon completion. For UI requests, results are polled directly."),
):
    """
    Uploads multiple files for OCR processing and returns task IDs along with file metadata.
    The results will be sent to the provided webhook URL asynchronously, or polled directly for UI requests.
    """
    try:
        # Handle UI requests vs API requests
        if metadata is None:
            # UI request - generate metadata automatically
            import uuid
            metadata_list = [{"guid": str(uuid.uuid4()), "file_type": file.content_type} for file in files]
            use_webhook = False
        else:
            # API request - parse metadata JSON
            try:
                metadata_list = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON in metadata parameter.")

            if len(files) != len(metadata_list):
                raise HTTPException(status_code=400, detail="Number of files must match number of metadata entries.")

            use_webhook = webhook_url is not None

        task_ids = []

        for file, file_metadata in zip(files, metadata_list):
            file_bytes = await file.read()

            # Merge file metadata with file type info
            full_metadata = {**file_metadata, "file_type": file.content_type}

            # Call the Celery task by its string name to keep the API and worker decoupled.
            task_name = 'app.worker.tasks.process_ocr_task'
            async_result = celery_app.send_task(
                name=task_name,
                kwargs={
                    'file_content': file_bytes,
                    'metadata': full_metadata,
                    'webhook_url': webhook_url if use_webhook else None,
                    'correlation_id': correlation_id_var.get()
                }
            )

            # Block briefly to get the main workflow ID returned by the orchestrator task.
            main_workflow_id = async_result.get(timeout=10)

            if main_workflow_id:
                task_ids.append(main_workflow_id)
            else:
                raise HTTPException(status_code=500, detail=f"Failed to dispatch OCR workflow for file {file.filename}.")

        return TaskQueueResponse(
            task_ids=task_ids,
            status="queued"
        )
    except HTTPException:
        raise
    except Exception as e:
        # Catch potential timeouts or other exceptions
        raise HTTPException(status_code=500, detail=f"Failed to queue tasks: {str(e)}")


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