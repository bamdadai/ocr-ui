# app/schemas/ocr.py

from pydantic import BaseModel, Field
from typing import Optional, List, Any, Union

# --- Output Schemas ---

class TaskQueueResponse(BaseModel):
    """The response returned after successfully queuing tasks."""
    task_ids: List[str] = Field(..., description="List of task IDs for the queued OCR tasks.")
    status: str = Field(..., description="The initial status of the tasks, always 'queued'.")


# --- Task Status Schemas ---

class FinalOCRResult(BaseModel):
    """The structure of the final successful OCR result data."""
    guid: str
    text: str  # This is a Base64 encoded string
    confidence: float
    status: str
    error: str = ""

class TaskErrorResult(BaseModel):
    """The structure for returning an error from a failed task."""
    error: str

class TaskStatusResponse(BaseModel):
    """The response structure for the task status polling endpoint."""
    task_id: str = Field(..., description="The ID of the task being polled.")
    status: str = Field(..., description="The current status of the task (e.g., PENDING, SUCCESS, FAILURE).")
    result: Optional[Union[FinalOCRResult, TaskErrorResult]] = Field(None, description="The final result, present on success or failure.")