"""
Main application file for the Unified OCR Service.

This file initializes the FastAPI application, configures middleware,
mounts static file directories, sets up HTML templates, and defines
the primary API and UI endpoints.
"""

# --- 1. Core Imports ---
import uuid
import time
import json
import logging
import structlog

# --- 2. Third-party Imports ---
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse

# --- 3. Local Application Imports ---
from app.api.v1.endpoints import ocr as ocr_endpoint
from app.core.logging import configure_logging, correlation_id_var

# --- Application Initialization ---

# Configure structured logging at startup.
configure_logging()
logger = structlog.get_logger(__name__)
access_logger = logging.getLogger("app.access")

# Create the main FastAPI application instance.
app = FastAPI(
    title="Unified OCR Service",
    description="An integrated service for text detection, recognition, and document processing.",
    version="1.0.0"
)

# --- Middleware Configuration ---

# Best practice: For production, restrict origins to your specific frontend domain.
# Using ["*"] is suitable for development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_correlation_id_middleware(request: Request, call_next):
    """
    Inject a correlation ID into every incoming request for traceability.

    Checks for an 'X-Correlation-ID' header. If not found, a new UUID
    is generated. The ID is added to logs and the response headers.
    """
    correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
    token = correlation_id_var.set(correlation_id)

    logger.info("request.started", method=request.method, url=str(request.url))
    query_params = dict(request.query_params)
    path_params = dict(request.scope.get("path_params") or {})
    body_preview = None
    if request.method.upper() in {"POST", "PUT", "PATCH", "DELETE"}:
        content_type = request.headers.get("content-type", "")
        if content_type and "multipart/form-data" not in content_type:
            try:
                body_bytes = await request.body()
                if body_bytes:
                    request._body = body_bytes  # Allow downstream handlers to re-read the body
                    body_sent = False
                    async def receive() -> dict:
                        nonlocal body_sent
                        if not body_sent:
                            body_sent = True
                            return {"type": "http.request", "body": body_bytes, "more_body": False}
                        return {"type": "http.request", "body": b"", "more_body": False}
                    request._receive = receive  # type: ignore[attr-defined]
                    text_body = body_bytes.decode("utf-8", errors="replace")
                    if "application/json" in content_type:
                        try:
                            parsed_json = json.loads(text_body)
                            body_preview = json.dumps(parsed_json)
                        except json.JSONDecodeError:
                            body_preview = text_body
                    elif "application/x-www-form-urlencoded" in content_type:
                        body_preview = text_body
                    else:
                        body_preview = f"<{len(body_bytes)} bytes not logged>"
                    if body_preview and len(body_preview) > 500:
                        body_preview = body_preview[:500] + "... [truncated]"
            except Exception as exc:  # pragma: no cover - defensive logging
                body_preview = f"<unavailable: {exc}>"
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else "unknown"
    query_string = request.url.query
    path = request.url.path if not query_string else f"{request.url.path}?{query_string}"
    parts = [
        f"ip={client_ip}",
        request.method,
        path,
        f"-> {response.status_code}",
        f"({duration_ms:.2f} ms)",
        f"req_id={correlation_id}",
    ]
    if query_params:
        parts.append(f"query={query_params}")
    if path_params:
        parts.append(f"path={path_params}")
    file_count = getattr(request.state, "ocr_file_count", None)
    if file_count is not None:
        parts.append(f"files={file_count}")
    extra_fields = getattr(request.state, "access_extra_fields", None)
    if extra_fields:
        parts.append(f"fields={extra_fields}")
    if body_preview is not None:
        parts.append(f"body={body_preview}")
    access_logger.info(" ".join(parts))
    logger.info("request.finished", status_code=response.status_code)

    response.headers["X-Correlation-ID"] = correlation_id
    correlation_id_var.reset(token)
    return response


# --- Static Files and Templates ---

# Mount the 'static' directory to serve CSS, JS, and image files.
app.mount("/static", StaticFiles(directory="static"), name="static")

# Point to the 'templates' directory for HTML files.
templates = Jinja2Templates(directory="templates")


# --- API Router Inclusion ---

# Include the OCR endpoint router with a versioned prefix.
app.include_router(ocr_endpoint.router, prefix="/v3", tags=["V3 - OCR"])


# --- UI and Health Check Endpoints ---

@app.get("/", tags=["Health Check"])
async def root_endpoint(request: Request):
    """
    Serve the health check for API clients or the frontend UI for browsers.

    This endpoint serves both purposes:
    - Returns JSON health check for API clients (Accept: application/json)
    - Returns HTML frontend for browser requests
    """
    # Check if client accepts JSON (API client) or prefers HTML (browser)
    accept_header = request.headers.get("accept", "").lower()

    if "application/json" in accept_header or request.headers.get("content-type") == "application/json":
        # API client requesting health check
        logger.info("health_check.called", status="healthy")
        return {"message": "OCR Service is running"}
    else:
        # Browser requesting frontend UI
        return templates.TemplateResponse("index.html", {"request": request})
