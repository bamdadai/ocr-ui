# Webhook Logging Reference

This document describes all the webhook-related log messages in the OCR service.

## Log Flow

When a file is submitted with a webhook URL, you'll see logs in this sequence:

```
1. ocr.request.received       → API receives request
2. ocr.task.dispatching       → Task is being queued
3. ocr.task.dispatched        → Task queued to Celery
4. ocr.request.completed      → API response sent
5. ocr_task.received          → Worker starts processing
6. finalize.success           → OCR processing complete
7. webhook.dispatch           → Webhook task queued
8. webhook.attempt.start      → Webhook HTTP request starting
9. webhook.send.success       → Webhook delivered successfully
```

## Log Messages

### 1. API Endpoint Logs

#### `ocr.request.received`
Logged when the API receives an OCR request.

```json
{
  "event": "ocr.request.received",
  "file_count": 1,
  "webhook_url": "http://localhost:8080/webhook/ocr",
  "has_metadata": true
}
```

#### `ocr.task.dispatching`
Logged before each file is queued to Celery.

```json
{
  "event": "ocr.task.dispatching",
  "guid": "abc-123",
  "task_id": "uuid-task-id",
  "filename": "document.pdf",
  "file_format": ".pdf",
  "webhook_url": "http://localhost:8080/webhook/ocr",
  "use_webhook": true
}
```

#### `ocr.task.dispatched`
Logged after task is successfully queued.

```json
{
  "event": "ocr.task.dispatched",
  "guid": "abc-123",
  "task_id": "uuid-task-id",
  "celery_task_id": "celery-uuid",
  "webhook_url": "http://localhost:8080/webhook/ocr"
}
```

#### `ocr.request.completed`
Logged when API completes processing the request.

```json
{
  "event": "ocr.request.completed",
  "total_files": 2,
  "queued_count": 2,
  "error_count": 0,
  "webhook_url": "http://localhost:8080/webhook/ocr"
}
```

### 2. Worker Task Logs

#### `finalize.success`
Logged when OCR processing completes.

```json
{
  "event": "finalize.success",
  "guid": "abc-123",
  "total_pages": 5,
  "confidence": 0.95
}
```

#### `webhook.dispatch`
Logged when webhook task is queued.

```json
{
  "event": "webhook.dispatch",
  "guid": "abc-123",
  "task_id": "uuid-task-id",
  "webhook_url": "http://localhost:8080/webhook/ocr",
  "payload_size": 1024,
  "status": "completed"
}
```

#### `webhook.skipped`
Logged when no webhook URL was provided.

```json
{
  "event": "webhook.skipped",
  "guid": "abc-123",
  "task_id": "uuid-task-id",
  "reason": "no_webhook_url_provided"
}
```

### 3. Webhook Delivery Logs

#### `webhook.attempt.start`
Logged at the start of each webhook delivery attempt.

```json
{
  "event": "webhook.attempt.start",
  "guid": "abc-123",
  "task_id": "uuid-task-id",
  "webhook_url": "http://localhost:8080/webhook/ocr",
  "retry_count": 0,
  "max_retries": 3,
  "ssl_verify": true
}
```

**Fields:**
- `retry_count`: Current retry attempt (0 = first attempt)
- `max_retries`: Maximum retry attempts configured
- `ssl_verify`: Whether SSL certificate verification is enabled

#### `webhook.send.success`
Logged when webhook is successfully delivered.

```json
{
  "event": "webhook.send.success",
  "guid": "abc-123",
  "task_id": "uuid-task-id",
  "webhook_url": "http://localhost:8080/webhook/ocr",
  "status_code": 200,
  "response_time_ms": 45.2,
  "retry_count": 0
}
```

**Fields:**
- `status_code`: HTTP response code from webhook endpoint
- `response_time_ms`: Time taken for the HTTP request in milliseconds
- `retry_count`: Which attempt succeeded

### 4. Webhook Error Logs

#### `webhook.send.timeout`
Logged when webhook request times out (10 second timeout).

```json
{
  "event": "webhook.send.timeout",
  "level": "error",
  "guid": "abc-123",
  "task_id": "uuid-task-id",
  "webhook_url": "http://localhost:8080/webhook/ocr",
  "retry_count": 0,
  "error": "HTTPSConnectionPool(...): Read timed out."
}
```

**Action:** Task will be retried (up to 3 times with 10 second delay).

#### `webhook.send.connection_error`
Logged when webhook endpoint is unreachable.

```json
{
  "event": "webhook.send.connection_error",
  "level": "error",
  "guid": "abc-123",
  "task_id": "uuid-task-id",
  "webhook_url": "http://localhost:8080/webhook/ocr",
  "retry_count": 0,
  "error": "Failed to establish a new connection: [Errno 111] Connection refused"
}
```

**Action:** Task will be retried (up to 3 times with 10 second delay).

#### `webhook.send.http_error`
Logged when webhook returns HTTP error status (4xx, 5xx).

```json
{
  "event": "webhook.send.http_error",
  "level": "error",
  "guid": "abc-123",
  "task_id": "uuid-task-id",
  "webhook_url": "http://localhost:8080/webhook/ocr",
  "status_code": 500,
  "response_body": "Internal Server Error",
  "retry_count": 0,
  "error": "500 Server Error: Internal Server Error for url: ..."
}
```

**Fields:**
- `status_code`: HTTP error status code
- `response_body`: First 500 characters of error response

**Action:** Task will be retried (up to 3 times with 10 second delay).

#### `webhook.send.failed`
Logged for other request exceptions.

```json
{
  "event": "webhook.send.failed",
  "level": "error",
  "guid": "abc-123",
  "task_id": "uuid-task-id",
  "webhook_url": "http://localhost:8080/webhook/ocr",
  "retry_count": 0,
  "error": "...",
  "error_type": "RequestException"
}
```

**Action:** Task will be retried (up to 3 times with 10 second delay).

#### `webhook.send.critical_error`
Logged for non-retryable exceptions.

```json
{
  "event": "webhook.send.critical_error",
  "level": "critical",
  "guid": "abc-123",
  "task_id": "uuid-task-id",
  "webhook_url": "http://localhost:8080/webhook/ocr",
  "retry_count": 0,
  "error": "...",
  "error_type": "ValueError"
}
```

**Action:** Task fails permanently (no retry).

## Monitoring Tips

### 1. Track Webhook Success Rate

```bash
# Count successful webhook deliveries
grep "webhook.send.success" app.log | wc -l

# Count failed webhook deliveries
grep "webhook.send.failed\|webhook.send.timeout\|webhook.send.connection_error\|webhook.send.http_error" app.log | wc -l
```

### 2. Find Failed Webhooks by URL

```bash
# Find all errors for a specific webhook URL
grep "webhook_url.*your-webhook.com" app.log | grep "error"
```

### 3. Monitor Retry Attempts

```bash
# Find webhooks that required retries
grep "webhook.send.success" app.log | grep -v "retry_count.*0"
```

### 4. Track Response Times

```bash
# Extract response times from successful webhooks
grep "webhook.send.success" app.log | grep -o "response_time_ms\":[0-9.]*"
```

### 5. Find Webhooks by GUID

```bash
# Track all webhook logs for a specific GUID
grep "guid.*abc-123" app.log | grep webhook
```

## Example Log Sequence

### Successful Webhook Delivery

```
2025-10-07 10:00:00 - ocr.request.received - file_count=1 webhook_url=http://localhost:8080/webhook/ocr
2025-10-07 10:00:00 - ocr.task.dispatching - guid=test-123 webhook_url=http://localhost:8080/webhook/ocr
2025-10-07 10:00:00 - ocr.task.dispatched - guid=test-123 task_id=abc-uuid webhook_url=http://localhost:8080/webhook/ocr
2025-10-07 10:00:00 - ocr.request.completed - total_files=1 queued_count=1 webhook_url=http://localhost:8080/webhook/ocr
2025-10-07 10:00:05 - finalize.success - guid=test-123 total_pages=1 confidence=0.95
2025-10-07 10:00:05 - webhook.dispatch - guid=test-123 webhook_url=http://localhost:8080/webhook/ocr
2025-10-07 10:00:05 - webhook.attempt.start - guid=test-123 webhook_url=http://localhost:8080/webhook/ocr retry_count=0
2025-10-07 10:00:05 - webhook.send.success - guid=test-123 webhook_url=http://localhost:8080/webhook/ocr status_code=200 response_time_ms=45.2
```

### Failed Webhook with Retries

```
2025-10-07 10:00:05 - webhook.attempt.start - guid=test-123 webhook_url=http://down.example.com/webhook retry_count=0
2025-10-07 10:00:15 - webhook.send.connection_error - guid=test-123 webhook_url=http://down.example.com/webhook retry_count=0
2025-10-07 10:00:25 - webhook.attempt.start - guid=test-123 webhook_url=http://down.example.com/webhook retry_count=1
2025-10-07 10:00:35 - webhook.send.connection_error - guid=test-123 webhook_url=http://down.example.com/webhook retry_count=1
2025-10-07 10:00:45 - webhook.attempt.start - guid=test-123 webhook_url=http://down.example.com/webhook retry_count=2
2025-10-07 10:00:55 - webhook.send.connection_error - guid=test-123 webhook_url=http://down.example.com/webhook retry_count=2
2025-10-07 10:01:05 - webhook.attempt.start - guid=test-123 webhook_url=http://down.example.com/webhook retry_count=3
2025-10-07 10:01:15 - webhook.send.connection_error - guid=test-123 webhook_url=http://down.example.com/webhook retry_count=3
```

## Configuration

Webhook behavior is controlled by these settings:

| Setting | Default | Description |
|---------|---------|-------------|
| Max Retries | 3 | Maximum retry attempts |
| Retry Delay | 10 seconds | Delay between retries |
| Timeout | 10 seconds | HTTP request timeout |
| SSL Verify | True | Verify SSL certificates (disable with `ALLOW_INSECURE_WEBHOOKS=true`) |

## Troubleshooting

### Issue: Webhook not being called

1. Check if webhook URL was provided:
   ```bash
   grep "webhook.dispatch\|webhook.skipped" app.log
   ```

2. Verify task completed successfully:
   ```bash
   grep "finalize.success" app.log
   ```

### Issue: Webhook timing out

- Ensure your webhook endpoint responds within 10 seconds
- Check network connectivity
- Look for `webhook.send.timeout` in logs

### Issue: Connection refused

- Verify webhook server is running
- Check firewall rules
- Ensure URL is correct (check `webhook_url` in logs)

### Issue: SSL certificate errors

- For development, set `ALLOW_INSECURE_WEBHOOKS=true`
- For production, ensure valid SSL certificate on webhook endpoint
