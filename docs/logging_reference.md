# OCR Service Logging Reference

## Overview

The OCR service uses **structlog** for structured JSON logging with separate log files for each severity level. All logs include timestamps, correlation IDs, and contextual metadata for debugging and monitoring.

## Log File Structure

### Application & Celery Unified Logs

| File | Level | Purpose |
|------|-------|---------|
| `logs/info.log` | INFO | Operational events, workflow progress, successful operations |
| `logs/warnings.log` | WARNING | Non-critical issues, fallbacks, degraded functionality |
| `logs/errors.log` | ERROR/CRITICAL | Failures, exceptions, critical system errors |
| `logs/debug.log` | DEBUG | Detailed diagnostic information, method entry/exit, timing |
| `logs/performance_debug.log` | DEBUG | Performance metrics (when debug mode enabled) |

### Log Rotation
- **Max file size**: 100 MB per file
- **Backup count**: 7 files retained
- **Format**: JSON (structured)
- **Encoding**: UTF-8

---

## Log Levels & Event Types

### 🔵 DEBUG Level

Debug logs provide detailed diagnostic information for development and troubleshooting.

#### Core System
| Event | Context | Description |
|-------|---------|-------------|
| `method.entry` | All services | Method invocation with argument count |
| `method.timing` | All services | Method execution duration (success) |
| `logging.configured` | System startup | Logging system initialization complete |

#### Detection Service
| Event | Context | Description |
|-------|---------|-------------|
| `detection_service.init` | Initialization | Service initialization with config |
| `detection_service.paddle_warmup_start` | Startup | PaddleOCR model warmup beginning |
| `detection_service.paddle_warmup_complete` | Startup | PaddleOCR warmup successful |
| `detection_service.yolo_model_loading` | Initialization | YOLO model loading started |
| `detection_service.yolo_model_loaded` | Initialization | YOLO model loaded successfully |
| `detection_service.yolo_warmup_start` | Startup | YOLO model warmup beginning |
| `detection_service.yolo_warmup_complete` | Startup | YOLO model warmup successful |
| `detection_service.initialized` | Startup | Service fully initialized |
| `detection_service.loading_paddle_model` | Initialization | PaddleOCR model loading with kwargs |
| `detection_service.predicting` | Processing | Prediction started with image count |
| `detection_service.line_paddle_debug_saved` | Debug mode | Debug image saved successfully |
| `detection_service.class_filter` | Processing | Class filtering results (kept/total) |
| `detection_service.debug_saving` | Debug mode | Saving debug visualization |
| `detection_service.debug_saved` | Debug mode | Debug image saved successfully |
| `detection_service.single_image_done` | Processing | Single image processing complete |
| `detection_service.processing_masks` | Processing | Processing YOLO segmentation masks |
| `detection_service.word_polygons_extracted` | Processing | Polygon extraction count |
| `detection_service.word_bbox_fallback` | Processing | Fallback to bounding boxes (count) |
| `detection_service.removed_nested_box` | Processing | Nested box removal details |
| `detection_service.nested_boxes_removed` | Processing | Summary of nested box removal |
| `detection_service.batch.processing` | Batch processing | Processing batch with size |
| `detection_service.batch.memory_low` | Batch processing | Memory usage low, increasing batch |

#### Recognition Service
| Event | Context | Description |
|-------|---------|-------------|
| `recognition_service.loading_model` | Initialization | Loading recognition model checkpoint |
| `recognition_service.warmup_start` | Startup | Model warmup beginning |
| `recognition_service.warmup_complete` | Startup | Model warmup successful |
| `recognition_service.initialized` | Startup | Service fully initialized |

#### Pipeline Service
| Event | Context | Description |
|-------|---------|-------------|
| `pipeline_service.recognition_loaded_eagerly` | Initialization | Recognition loaded at startup |
| `pipeline_service.word_polygons_on_page_saved` | Debug mode | Debug polygons saved (count) |
| `pipeline_service.parts_visualization_saved` | Debug mode | Parts visualization saved |
| `pipeline_service.word_crops_debug_saved` | Debug mode | Word crop images saved |

#### OCR Tasks
| Event | Context | Description |
|-------|---------|-------------|
| `lazy_loading.pipeline_service.initializing` | Lazy loading | Starting lazy load of pipeline |
| `lazy_loading.pipeline_service.import_ok` | Lazy loading | Import successful with module origin |
| `lazy_loading.pipeline_service.success` | Lazy loading | Lazy loading complete |
| `detect_lines.success` | Processing | Line detection complete with count |
| `detect_words.success` | Processing | Word detection complete for page |
| `recognize_page.success` | Processing | Page recognition complete with confidence |
| `ocr_task.received` | Task start | Task received with GUID and format |
| `ocr_task.decoding_base64_string` | Processing | Decoding base64 file content |
| `ocr_task.workflow_dispatched` | Processing | Celery workflow task dispatched |
| `finalize.page_indices_loaded_from_state` | Finalization | Page indices retrieved from state |
| `finalize.success` | Completion | OCR finalization successful |

#### Line Splitting
| Event | Context | Description |
|-------|---------|-------------|
| `line_splitting.word_in_multiple_parts` | Processing | Word spans multiple line parts |
| `line_splitting.part_created` | Processing | New line part created |
| `line_splitting.part_closed` | Processing | Line part finalized |
| `line_splitting.parts_created` | Processing | Summary of parts created |
| `line_splitting.sorted` | Processing | Parts sorted by position |

#### State Manager
| Event | Context | Description |
|-------|---------|-------------|
| `state.upload_blob.saved` | Upload handling | File staged in Redis (chunks) |
| `state.upload_blob.retrieved` | Upload handling | File retrieved from Redis |

#### Performance
| Event | Context | Description |
|-------|---------|-------------|
| `performance.stage_timing` | Performance tracking | Stage execution timing metrics |

---

### 🟢 INFO Level

Info logs track normal operational events and workflow progress.

#### API Endpoints
| Event | Context | Description |
|-------|---------|-------------|
| `ocr.request.received` | Request handling | OCR request received with file count |
| `ocr.webhook.mode` | Request handling | Webhook mode determination |
| `ocr.task.dispatching` | Task creation | Preparing to dispatch task |
| `ocr.task.dispatched` | Task creation | Task dispatched to Celery |
| `ocr.request.completed` | Request completion | Request summary (queued/errors) |
| `request.started` | Middleware | HTTP request started |
| `request.finished` | Middleware | HTTP request completed |
| `health_check.called` | Health check | Health endpoint called |

#### Celery Worker
| Event | Context | Description |
|-------|---------|-------------|
| `celery.task_success_after_retry` | Task management | Task succeeded after retries |
| `celery.task_rejected_before_retry` | Task management | Task rejected, will retry |

#### Orientation Service
| Event | Context | Description |
|-------|---------|-------------|
| `orientation.warmup_start` | Startup | Orientation model warmup beginning |
| `orientation.model_loaded_and_warmed` | Startup | Model loaded and warmed successfully |
| `orientation.model_loaded` | Startup | Model loaded (warmup failed) |

#### State Manager
| Event | Context | Description |
|-------|---------|-------------|
| `state.upload_storage.policy_adjusted` | Upload handling | Storage policy adjusted for size |

---

### 🟡 WARNING Level

Warning logs indicate non-critical issues, fallbacks, or degraded performance.

#### Detection Service
| Event | Context | Description |
|-------|---------|-------------|
| `method.timing` | Performance | Method execution slower than expected |
| `detection_service.paddle_warmup_failed` | Startup | PaddleOCR warmup failed (continuing) |
| `detection_service.yolo_model_structure_unexpected` | Initialization | Unexpected YOLO model structure |
| `detection_service.yolo_warmup_failed` | Startup | YOLO warmup failed (continuing) |
| `detection_service.paddle_no_results` | Processing | PaddleOCR returned no results |
| `detection_service.line_paddle_debug_save_failed` | Debug mode | Failed to save debug image |
| `detection_service.class_filter_failed` | Processing | Class filtering failed, using all |
| `detection_service.model_layer_compatibility_issue` | Processing | Model layer compatibility issue |
| `detection_service.debug_save_failed` | Debug mode | Debug image save failed |
| `detection_service.no_polygons_extracted_from_masks` | Processing | Mask processing yielded no polygons |
| `detection_service.word_masks_postprocess_failed` | Processing | Mask post-processing failed |
| `detection_service.word_bbox_fallback_failed` | Processing | Bounding box fallback failed |
| `detection_service.batch.memory_high` | Performance | High memory, reducing batch size |

#### API Endpoints
| Event | Context | Description |
|-------|---------|-------------|
| `queue.unsupported_format` | Validation | File format not supported |

#### OCR Tasks
| Event | Context | Description |
|-------|---------|-------------|
| `lazy_loading.pipeline_service.abs_import_failed` | Lazy loading | Absolute import failed, trying relative |
| `ocr_task.no_pages_found` | Processing | No pages extracted from document |
| `ocr_task.staged_payload_cleanup_failed` | Cleanup | Failed to cleanup staged upload |
| `finalize.unexpected_page_results_type` | Finalization | Unexpected page results data type |
| `finalize.encoding_fix_failed` | Finalization | Text encoding fix failed |

#### Orientation Service
| Event | Context | Description |
|-------|---------|-------------|
| `orientation.missing_model_path` | Configuration | Model path not configured |
| `orientation.warmup_failed` | Startup | Model warmup failed |
| `orientation.predict_failed` | Processing | Orientation prediction failed |
| `orientation.batch_predict_failed` | Processing | Batch orientation prediction failed |

#### Recognition Service
| Event | Context | Description |
|-------|---------|-------------|
| `recognition_service.warmup_failed` | Startup | Recognition warmup failed |

#### Pipeline Service
| Event | Context | Description |
|-------|---------|-------------|
| `pipeline_service.recognition_not_loaded` | Processing | Recognition service unavailable |
| `pipeline_service.word_polygons_debug_failed` | Debug mode | Debug polygon save failed |
| `pipeline_service.parts_visualization_failed` | Debug mode | Parts visualization save failed |
| `pipeline_service.word_crops_debug_failed` | Debug mode | Word crops debug save failed |

#### Line Splitting
| Event | Context | Description |
|-------|---------|-------------|
| `line_splitting.no_parts_created` | Processing | No line parts created from words |

---

### 🔴 ERROR Level

Error logs indicate failures requiring attention or causing degraded functionality.

#### Detection Service
| Event | Context | Description |
|-------|---------|-------------|
| `detection_service.yolo_model_load_failed` | Initialization | YOLO model failed to load |
| `detection_service.model_compatibility_error` | Processing | Model compatibility issue detected |
| `detection_service.prediction_failed` | Processing | Prediction operation failed |
| `detection_service.parallel.failure` | Processing | Parallel batch processing failed |
| `detection_service.sequential.failure` | Processing | Sequential processing failed |

#### API Endpoints
| Event | Context | Description |
|-------|---------|-------------|
| `queue.dispatch_failed` | Task creation | Failed to dispatch task to Celery |

#### OCR Tasks
| Event | Context | Description |
|-------|---------|-------------|
| `ocr_task.staged_payload_missing` | Processing | Staged file not found in Redis |
| `ocr_task.unexpected_type` | Processing | File content has unexpected type |
| `ocr_task.image_conversion_failed` | Processing | Failed to convert file to images |
| `finalize.missing_page_indices` | Finalization | Page indices not found in state |

#### Orientation Service
| Event | Context | Description |
|-------|---------|-------------|
| `orientation.model_load_failed` | Initialization | Orientation model load failed |

---

### 🔴 CRITICAL Level

Critical logs indicate severe failures that prevent core functionality.

#### Pipeline Service
| Event | Context | Description |
|-------|---------|-------------|
| `pipeline_service.detection_init_failed` | Initialization | Detection service failed to initialize |
| `pipeline_service.recognition_init_failed` | Initialization | Recognition service failed to initialize |

#### OCR Tasks
| Event | Context | Description |
|-------|---------|-------------|
| `lazy_loading.pipeline_service.failed` | Lazy loading | Pipeline service failed to load entirely |
| `ocr_task.pdf_conversion_failed` | Processing | PDF to image conversion failed |
| `ocr_task.image_load_failed` | Processing | Image loading failed |
| `ocr_task.unsupported_format` | Validation | File format completely unsupported |
| `ocr_task.chord_dispatch_failed` | Task orchestration | Celery chord failed to dispatch |

---

## Event Metadata

### Common Fields

All log events include these standard fields:

| Field | Type | Description |
|-------|------|-------------|
| `event` | string | Event name (e.g., `"ocr.task.dispatched"`) |
| `timestamp` | string | ISO 8601 UTC timestamp |
| `level` | string | Log level (DEBUG/INFO/WARNING/ERROR/CRITICAL) |
| `logger` | string | Logger name (module path) |
| `filename` | string | Source file name |
| `lineno` | integer | Source line number |
| `correlation_id` | string | Request correlation ID (when available) |

### Context-Specific Fields

#### Task Processing
- `guid`: File/task GUID
- `task_id`: Celery task ID
- `workflow_task_id`: Workflow task ID
- `page_index`: Page number (0-based)
- `file_format`: File extension

#### Performance
- `duration_ms`: Duration in milliseconds
- `duration_ns`: Duration in nanoseconds
- `memory_mb`: Memory usage in megabytes
- `batch_size`: Batch size for processing

#### Detection/Recognition
- `image_count`: Number of images processed
- `model_type`: Model type (word/line)
- `confidence`: Recognition confidence score
- `count`: Count of detected items

#### Errors
- `error`: Error message
- `exc_info`: Exception traceback (when available)

---

## Correlation IDs

Every HTTP request receives an `X-Correlation-ID` header. This ID is propagated through:
- All log entries related to that request
- Celery tasks spawned from the request
- Webhook callbacks (if configured)

Use correlation IDs to trace a complete request lifecycle across logs.

---

## Monitoring Recommendations

### Critical Alerts
Monitor `logs/errors.log` for:
- `*.model_load_failed` - Model loading failures
- `*.init_failed` - Service initialization failures
- `ocr_task.chord_dispatch_failed` - Task orchestration failures

### Performance Monitoring
Monitor `logs/warnings.log` for:
- `method.timing` with `success=False` - Slow operations
- `detection_service.batch.memory_high` - Memory pressure
- `*.warmup_failed` - Model warmup issues

### Operational Metrics
Monitor `logs/info.log` for:
- `ocr.request.received` / `ocr.request.completed` - Request volume
- `ocr.task.dispatched` - Task throughput
- Request/response patterns

---

## Debug Mode

When `debug: true` in `master_config.json`:

1. **Debug images saved** to:
   - `debug/word_detections/`
   - `debug/line_detections/`
   - `debug/word_polygons/`
   - `debug/parts/`
   - `debug/word_crops/`

2. **Additional logs**:
   - `logs/performance_debug.log` - Detailed performance metrics
   - Method entry/exit logging
   - Detailed batch processing information

3. **Performance impact**: Debug mode significantly increases I/O and storage usage. Use only for troubleshooting.

---

## Log Analysis Examples

### Find all errors for a specific GUID
```bash
jq 'select(.guid == "invoice-001" and .level == "error")' logs/errors.log
```

### Track request lifecycle by correlation ID
```bash
grep '"correlation_id": "abc-123"' logs/info.log logs/warnings.log logs/errors.log
```

### Calculate average confidence scores
```bash
jq -s 'map(select(.event == "recognize_page.success")) | map(.confidence) | add / length' logs/debug.log
```

### Monitor task throughput
```bash
jq -r 'select(.event == "ocr.task.dispatched") | .timestamp' logs/info.log | wc -l
```

---

## Log Retention

Current configuration:
- **Active log**: Current file until 100 MB
- **Backup files**: 7 previous rotations kept
- **Total storage**: ~800 MB per log file (100 MB × 8)
- **Total system**: ~3.2 GB (4 files × 800 MB)

To adjust retention, modify `maxBytes` and `backupCount` in `/home/web/projects/ocr_ui/app/core/logging.py`.

---

## Related Documentation

- [API Specification](ocr_minimal_doc.md) - REST API endpoints
- [Configuration Guide](../README.md) - Application configuration
- `app/core/logging.py` - Logging implementation

---

**Last Updated**: October 2025  
**Version**: 1.0

