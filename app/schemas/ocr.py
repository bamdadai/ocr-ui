# app/schemas/ocr.py

from pydantic import BaseModel, Field
from typing import Optional, List, Any, Union, Literal

# --- Output Schemas ---

class TaskQueueItem(BaseModel):
    """Per-file submission status returned after queuing."""
    task_id: Optional[str] = Field(None, description="Identifier for the OCR task. Present when the file was queued.")
    status: Literal["queued", "error"] = Field(..., description="Submission status for this file.")
    guid: str = Field(..., description="Client-provided or auto-generated GUID for the file.")


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
