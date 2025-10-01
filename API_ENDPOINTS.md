# OCR API Documentation

## Quick Start

Base URL: `http://localhost:8000`

## 1. Submit OCR Task

**Endpoint**: `POST /v3/ocr`

Upload files for OCR processing.

### Request (multipart/form-data)
- `files`: PDF or image files
- `metadata`: JSON array of file info (optional)
- `webhook_url`: URL for results (optional)

### Example
```bash
curl -X POST http://localhost:8000/v3/ocr \
  -F "files=@document.pdf" \
  -F "webhook_url=https://your-server.com/webhook"
```

### Response
```json
{
  "task_ids": ["550e8400-e29b-41d4-a716-446655440000"],
  "status": "queued"
}
```

## 2. Check Task Status

**Endpoint**: `GET /v3/ocr/tasks/{task_id}`

Get processing status and results.

### Example
```bash
curl http://localhost:8000/v3/ocr/tasks/550e8400-e29b-41d4-a716-446655440000
```

### Response (Success)
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "SUCCESS",
  "result": {
    "guid": "doc-123",
    "text": "Base64-encoded text",
    "confidence": 0.95,
    "status": "completed",
    "error": ""
  }
}
```

### Response (Processing)
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "PENDING",
  "result": null
}
```

## 3. Health Check

**Endpoint**: `GET /`

Check if service is running.

### Example
```bash
curl http://localhost:8000/
```

### Response
```json
{
  "message": "OCR Service is running"
}
```

## Supported Files

- **PDF**: Multi-page documents
- **Images**: JPG, PNG, JPEG

## Text Output

Text is Base64-encoded. Decode with:
```python
import base64
decoded_text = base64.b64decode(encoded_text).decode('utf-8')
```

## Webhook Integration

Provide `webhook_url` in upload request. Results sent automatically to your endpoint.

**Webhook Payload**:
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "guid": "doc-123",
  "text": "Base64-encoded text",
  "confidence": 0.95,
  "status": "completed",
  "error": ""
}
```

## Error Codes

- `200` - Success
- `400` - Bad Request
- `404` - Task Not Found
- `500` - Server Error

## Web UI

Visit `http://localhost:8000` for a web interface to upload files and view results.

---

*For setup instructions, see README.md*
