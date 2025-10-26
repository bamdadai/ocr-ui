"""
Centralized log event name constants.

This module provides standardized event names following the pattern: category.action.status
Using constants ensures consistency, prevents typos, and enables IDE autocomplete.

Usage:
    from app.core.log_events import LogEvent
    
    logger.info(LogEvent.API_TASK_DISPATCHED, guid=guid, task_id=task_id)
"""

class LogEvent:
    """Standard log event names following consistent pattern: category.action.status"""
    
    # ============================================================================
    # API Events (Request/Response Handling)
    # ============================================================================
    API_REQUEST_RECEIVED = "api.request.received"
    API_REQUEST_COMPLETED = "api.request.completed"
    API_TASK_DISPATCHING = "api.task.dispatching"
    API_TASK_DISPATCHED = "api.task.dispatched"
    API_TASK_DISPATCH_FAILED = "api.task.dispatch_failed"
    API_UNSUPPORTED_FORMAT = "api.validation.unsupported_format"
    
    # ============================================================================
    # Task Lifecycle Events
    # ============================================================================
    TASK_RECEIVED = "task.ocr.received"
    TASK_PROCESSING = "task.ocr.processing"
    TASK_COMPLETED = "task.ocr.completed"
    TASK_FAILED = "task.ocr.failed"
    TASK_INITIALIZATION_FAILED = "task.ocr.initialization_failed"
    
    # Task-specific errors
    TASK_IMAGE_DECODE_FAILED = "task.image.decode_failed"
    TASK_PDF_CONVERSION_FAILED = "task.pdf.conversion_failed"
    TASK_IMAGE_LOAD_FAILED = "task.image.load_failed"
    TASK_UNSUPPORTED_FORMAT = "task.validation.unsupported_format"
    
    # Staged upload events
    TASK_STAGED_PAYLOAD_MISSING = "task.storage.staged_payload_missing"
    TASK_STAGED_PAYLOAD_RETRIEVED = "task.storage.staged_payload_retrieved"
    TASK_STAGED_PAYLOAD_CLEANUP_FAILED = "task.storage.cleanup_failed"
    
    # ============================================================================
    # Detection Service Events (Word/Line Detection)
    # ============================================================================
    DETECT_SERVICE_INIT = "detect.service.initialized"
    DETECT_SERVICE_INIT_FAILED = "detect.service.init_failed"
    
    # Model loading
    DETECT_MODEL_LOADING = "detect.model.loading"
    DETECT_MODEL_LOADED = "detect.model.loaded"
    DETECT_MODEL_LOAD_FAILED = "detect.model.load_failed"
    
    # Warmup
    DETECT_WARMUP_START = "detect.warmup.started"
    DETECT_WARMUP_COMPLETE = "detect.warmup.completed"
    DETECT_WARMUP_FAILED = "detect.warmup.failed"
    
    # Processing
    DETECT_WORDS_STARTED = "detect.words.started"
    DETECT_WORDS_COMPLETED = "detect.words.completed"
    DETECT_WORDS_FAILED = "detect.words.failed"
    
    DETECT_LINES_STARTED = "detect.lines.started"
    DETECT_LINES_COMPLETED = "detect.lines.completed"
    DETECT_LINES_FAILED = "detect.lines.failed"
    
    # Processing details
    DETECT_NO_RESULTS = "detect.processing.no_results"
    DETECT_PREDICTION_FAILED = "detect.processing.prediction_failed"
    DETECT_COMPATIBILITY_ERROR = "detect.processing.compatibility_error"
    
    # Batch processing
    DETECT_BATCH_PROCESSING = "detect.batch.processing"
    DETECT_BATCH_MEMORY_HIGH = "detect.batch.memory_high"
    DETECT_BATCH_MEMORY_LOW = "detect.batch.memory_low"
    DETECT_PARALLEL_FAILURE = "detect.batch.parallel_failure"
    
    # ============================================================================
    # Recognition Service Events (Text Recognition)
    # ============================================================================
    RECOGNIZE_SERVICE_INIT = "recognize.service.initialized"
    RECOGNIZE_SERVICE_INIT_FAILED = "recognize.service.init_failed"
    
    RECOGNIZE_MODEL_LOADING = "recognize.model.loading"
    RECOGNIZE_WARMUP_START = "recognize.warmup.started"
    RECOGNIZE_WARMUP_COMPLETE = "recognize.warmup.completed"
    RECOGNIZE_WARMUP_FAILED = "recognize.warmup.failed"
    
    RECOGNIZE_PAGE_STARTED = "recognize.page.started"
    RECOGNIZE_PAGE_COMPLETED = "recognize.page.completed"
    RECOGNIZE_PAGE_FAILED = "recognize.page.failed"
    
    RECOGNIZE_NOT_LOADED = "recognize.service.not_loaded"
    
    # ============================================================================
    # Pipeline Service Events (OCR Orchestration)
    # ============================================================================
    PIPELINE_INIT = "pipeline.service.initialized"
    PIPELINE_DETECTION_INIT_FAILED = "pipeline.detection.init_failed"
    PIPELINE_RECOGNITION_INIT_FAILED = "pipeline.recognition.init_failed"
    PIPELINE_RECOGNITION_LOADED = "pipeline.recognition.loaded_eagerly"
    
    # ============================================================================
    # Orientation Service Events (Image Rotation)
    # ============================================================================
    ORIENTATION_MODEL_LOADING = "orientation.model.loading"
    ORIENTATION_MODEL_LOADED = "orientation.model.loaded"
    ORIENTATION_MODEL_LOAD_FAILED = "orientation.model.load_failed"
    ORIENTATION_WARMUP_START = "orientation.warmup.started"
    ORIENTATION_WARMUP_COMPLETE = "orientation.warmup.completed"
    ORIENTATION_WARMUP_FAILED = "orientation.warmup.failed"
    ORIENTATION_PREDICT_FAILED = "orientation.prediction.failed"
    ORIENTATION_BATCH_PREDICT_FAILED = "orientation.batch.prediction_failed"
    ORIENTATION_MISSING_MODEL_PATH = "orientation.config.missing_model_path"
    
    # ============================================================================
    # Finalization Events (Result Assembly)
    # ============================================================================
    FINALIZE_STARTED = "finalize.process.started"
    FINALIZE_COMPLETED = "finalize.process.completed"
    FINALIZE_FAILED = "finalize.process.failed"
    
    FINALIZE_PAGE_INDICES_LOADED = "finalize.state.page_indices_loaded"
    FINALIZE_MISSING_PAGE_INDICES = "finalize.state.missing_page_indices"
    
    FINALIZE_UNEXPECTED_RESULTS_TYPE = "finalize.data.unexpected_page_results_type"
    FINALIZE_ENCODING_FIX_FAILED = "finalize.encoding.fix_failed"
    
    # ============================================================================
    # Webhook Events (Callback Delivery)
    # ============================================================================
    WEBHOOK_SENDING = "webhook.delivery.sending"
    WEBHOOK_SUCCESS = "webhook.delivery.success"
    WEBHOOK_FAILED = "webhook.delivery.failed"
    WEBHOOK_TIMEOUT = "webhook.delivery.timeout"
    WEBHOOK_RETRY = "webhook.delivery.retry"
    
    # ============================================================================
    # State Management Events (Redis Storage)
    # ============================================================================
    STATE_UPLOAD_SAVED = "state.upload_blob.saved"
    STATE_UPLOAD_RETRIEVED = "state.upload_blob.retrieved"
    STATE_UPLOAD_CLEANUP_FAILED = "state.upload_blob.cleanup_failed"
    STATE_POLICY_ADJUSTED = "state.upload_storage.policy_adjusted"
    
    # ============================================================================
    # System Events (Application Lifecycle)
    # ============================================================================
    SYSTEM_STARTUP = "system.init.completed"
    SYSTEM_SHUTDOWN = "system.shutdown.started"
    SYSTEM_LOGGING_CONFIGURED = "system.logging.configured"
    SYSTEM_HEALTH_CHECK = "system.health.check"
    
    # ============================================================================
    # Performance/Debug Events
    # ============================================================================
    PERF_METHOD_ENTRY = "perf.method.entry"
    PERF_METHOD_TIMING = "perf.method.timing"
    PERF_STAGE_TIMING = "perf.stage.timing"
    
    DEBUG_SAVE_STARTED = "debug.visualization.saving"
    DEBUG_SAVE_COMPLETED = "debug.visualization.saved"
    DEBUG_SAVE_FAILED = "debug.visualization.save_failed"
    
    # ============================================================================
    # Line Splitting Events
    # ============================================================================
    LINE_SPLIT_PARTS_CREATED = "line_split.parts.created"
    LINE_SPLIT_NO_PARTS = "line_split.parts.none_created"
    LINE_SPLIT_PART_CLOSED = "line_split.part.closed"
    
    # ============================================================================
    # Celery Worker Events
    # ============================================================================
    CELERY_TASK_SUCCESS_AFTER_RETRY = "celery.task.success_after_retry"
    CELERY_TASK_REJECTED = "celery.task.rejected_before_retry"
    CELERY_CHORD_DISPATCH_FAILED = "celery.chord.dispatch_failed"
    
    # ============================================================================
    # Lazy Loading Events
    # ============================================================================
    LAZY_LOADING_STARTED = "lazy_loading.pipeline_service.initializing"
    LAZY_LOADING_IMPORT_SUCCESS = "lazy_loading.pipeline_service.import_ok"
    LAZY_LOADING_IMPORT_FAILED = "lazy_loading.pipeline_service.abs_import_failed"
    LAZY_LOADING_SUCCESS = "lazy_loading.pipeline_service.success"
    LAZY_LOADING_FAILED = "lazy_loading.pipeline_service.failed"


class LogCategory:
    """Log categories for grouping and filtering"""
    API = "api"
    TASK = "task"
    DETECT = "detect"
    RECOGNIZE = "recognize"
    PIPELINE = "pipeline"
    ORIENTATION = "orientation"
    FINALIZE = "finalize"
    WEBHOOK = "webhook"
    STATE = "state"
    SYSTEM = "system"
    PERF = "perf"
    DEBUG = "debug"
    CELERY = "celery"


class LogAction:
    """Common action verbs for events"""
    INIT = "init"
    LOADING = "loading"
    LOADED = "loaded"
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    PROCESSING = "processing"
    SENDING = "sending"
    RECEIVED = "received"
    SAVED = "saved"
    RETRIEVED = "retrieved"

