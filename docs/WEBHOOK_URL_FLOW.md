# Webhook URL Flow - Complete Trace

This document traces how the `webhook_url` parameter flows through the entire OCR system.

## Overview

The webhook URL is passed through these components:

```
Client Request → API Endpoint → Celery Task → Worker Task → Webhook Sender
```

## Detailed Flow

### 1. Client Submits Request

**File:** `scripts/ocr_client.py:86-87`

```python
if webhook_url:
    data['webhook_url'] = webhook_url
```

**Request Format:**
```
POST /v3/ocr
Content-Type: multipart/form-data

files: [file1.png]
webhook_url: http://localhost:8080/webhook/ocr
metadata: [{"guid": "abc-123", "format": ".png"}]
```

---

### 2. API Receives and Validates

**File:** `app/api/v1/endpoints/ocr.py:31-41`

```python
webhook_url: str = Form(None, ...)

# Normalize webhook URL: strip whitespace and convert empty strings to None
normalized_webhook = None
if webhook_url:
    stripped = webhook_url.strip()
    normalized_webhook = stripped if stripped else None
```

**Normalization Rules:**
- `None` → `None` (parameter not provided)
- `""` → `None` (empty string)
- `"   "` → `None` (whitespace only)
- `"  http://example.com  "` → `"http://example.com"` (trimmed)

**Logged:** `ocr.request.received` with `webhook_url=normalized_webhook`

---

### 3. API Determines Webhook Usage

**File:** `app/api/v1/endpoints/ocr.py:69`

```python
use_webhook = bool(normalized_webhook)
```

**Logic:**
- If `metadata` is provided AND `normalized_webhook` is not None → `use_webhook = True`
- Otherwise → `use_webhook = False`

---

### 4. API Dispatches to Celery

**File:** `app/api/v1/endpoints/ocr.py:111-128`

```python
logger.info(
    "ocr.task.dispatching",
    webhook_url=normalized_webhook if use_webhook else None,
    ...
)

async_result = celery_app.send_task(
    name="app.worker.tasks.process_ocr_task",
    kwargs={
        "webhook_url": normalized_webhook if use_webhook else None,
        ...
    },
)

logger.info(
    "ocr.task.dispatched",
    webhook_url=normalized_webhook if use_webhook else None,
    ...
)
```

**Key Point:** Webhook URL is only passed if `use_webhook = True`

**Logged:**
- `ocr.task.dispatching` with `webhook_url`
- `ocr.task.dispatched` with `webhook_url`

---

### 5. Worker Receives Task

**File:** `app/worker/tasks/ocr_tasks.py:241`

```python
def process_ocr_task(
    self,
    file_content: bytes,
    metadata: dict,
    webhook_url: str | None = None,
    ...
):
```

**Logged:** `ocr_task.received` with `guid` and `task_id`

---

### 6. Worker Creates Workflow

**File:** `app/worker/tasks/ocr_tasks.py:292`

```python
workflow = chord(
    header=group(page_workflows),
    body=finalize_and_notify_task.s(
        request_id=request_id,
        guid=guid,
        webhook_url=webhook_url  # ← Passed to finalize task
    )
)
```

**Key Point:** Webhook URL is passed to the chord's body (finalize task)

---

### 7. Finalize Task Receives Results

**File:** `app/worker/tasks/ocr_tasks.py:304`

```python
def finalize_and_notify_task(
    page_results: list,
    request_id: str,
    guid: str,
    webhook_url: str | None = None
):
```

**Logged:** `finalize.success` with `guid`, `total_pages`, `confidence`

---

### 8. Finalize Task Dispatches Webhook

**File:** `app/worker/tasks/ocr_tasks.py:343-359`

```python
if webhook_url:
    logger.info(
        "webhook.dispatch",
        webhook_url=webhook_url,
        ...
    )
    send_webhook_result.delay(webhook_url, final_payload, ...)
else:
    logger.info(
        "webhook.skipped",
        reason="no_webhook_url_provided"
    )
```

**Logged:**
- `webhook.dispatch` with `webhook_url` (if provided)
- `webhook.skipped` (if not provided)

---

### 9. Webhook Sender Executes

**File:** `app/worker/tasks/ocr_tasks.py:98-191`

```python
def send_webhook_result(self, webhook_url: str, payload: dict, **kwargs):
    logger.info(
        "webhook.attempt.start",
        webhook_url=webhook_url,
        retry_count=self.request.retries,
        ...
    )

    response = requests.post(webhook_url, json=payload, ...)

    logger.info(
        "webhook.send.success",
        webhook_url=webhook_url,
        status_code=response.status_code,
        ...
    )
```

**Logged:**
- `webhook.attempt.start` with `webhook_url`
- `webhook.send.success` with `webhook_url` (on success)
- `webhook.send.timeout` / `webhook.send.connection_error` / `webhook.send.http_error` (on error)

---

## Complete Log Sequence

When a file is submitted with `webhook_url=http://localhost:8080/webhook/ocr`:

```
[API]
ocr.request.received         webhook_url=http://localhost:8080/webhook/ocr
ocr.task.dispatching         webhook_url=http://localhost:8080/webhook/ocr
ocr.task.dispatched          webhook_url=http://localhost:8080/webhook/ocr
ocr.request.completed        webhook_url=http://localhost:8080/webhook/ocr

[Worker]
ocr_task.received            guid=abc-123 task_id=...
detect_lines.success         page=1
detect_words.success         page=1
recognize_page.success       page=1 confidence=0.95
finalize.success             guid=abc-123 total_pages=1

[Webhook]
webhook.dispatch             webhook_url=http://localhost:8080/webhook/ocr
webhook.attempt.start        webhook_url=http://localhost:8080/webhook/ocr retry_count=0
webhook.send.success         webhook_url=http://localhost:8080/webhook/ocr status_code=200
```

## Verification Commands

### Check webhook URL in API logs
```bash
grep "ocr.request.received\|ocr.task.dispatching" app.log | grep webhook_url
```

### Check webhook URL in worker logs
```bash
grep "webhook.dispatch\|webhook.attempt.start" app.log
```

### Trace webhook URL for specific GUID
```bash
grep "guid.*abc-123" app.log | grep webhook
```

### Verify webhook was sent
```bash
grep "webhook.send.success" app.log
```

## Testing

### Test with webhook URL
```bash
python scripts/ocr_client.py submit test.png \
    --webhook http://localhost:8080/webhook/ocr
```

### Test without webhook URL (polling)
```bash
python scripts/ocr_client.py submit test.png --poll
```

### Run complete flow test
```bash
python scripts/test_webhook_flow.py \
    --api-url http://localhost:8000 \
    --webhook-url http://localhost:8080/webhook/ocr
```

## Common Issues

### Issue: Webhook URL not appearing in logs

**Possible Causes:**
1. Empty or whitespace-only webhook URL sent
2. Metadata not provided (webhook disabled for UI requests)
3. Webhook URL normalized to None

**Debug:**
```bash
# Check what was received
grep "ocr.request.received" app.log | tail -5

# Check normalization
grep "ocr.task.dispatching" app.log | grep "webhook_url=None"
```

### Issue: Webhook skipped despite providing URL

**Cause:** Metadata was not provided, so `use_webhook = False`

**Solution:** Always provide metadata when using webhooks:
```python
metadata = [{"guid": "my-guid", "format": ".png"}]
```

### Issue: Webhook URL has extra whitespace

**Cause:** Client didn't trim the URL

**Solution:** The API now automatically trims whitespace (as of this update)

## Code Changes for Better Handling

The webhook URL handling was improved to:

1. **Normalize empty/whitespace strings to None**
   ```python
   if webhook_url:
       stripped = webhook_url.strip()
       normalized_webhook = stripped if stripped else None
   ```

2. **Log webhook URL at every stage**
   - API: received, dispatching, dispatched, completed
   - Worker: dispatch, attempt.start, send.success/error

3. **Track webhook even when skipped**
   ```python
   logger.info("webhook.skipped", reason="no_webhook_url_provided")
   ```

This ensures you can always trace what happened to the webhook URL!
