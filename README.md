# Unified Persian OCR System

[![Python-Version](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/Framework-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Celery](https://img.shields.io/badge/Async-Celery-orange.svg)](https://docs.celeryq.dev/en/stable/)

## ✨ Key Features

* **Asynchronous & Scalable**: FastAPI handles requests, Celery with RabbitMQ processes tasks in the background.
* **Multi-Format Support**: PDF (multi-page) and images (`.jpg`, `.png`, `.jpeg`).
* **Unified Pipeline**: Single optimized service for detection (YOLO) and recognition (Parseq).
* **High Accuracy**: Advanced deep-learning models for precise line/word detection and text extraction.
* **Webhook Notifications**: Results pushed to client-provided endpoint with retry logic.
* **Horizontal Scaling**: Increase Celery workers to match load.

---

## 🏗️ Architecture Overview

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI
    participant Queue as RabbitMQ
    participant Worker as Celery Worker
    participant Webhook

    Client->>+API: POST /api/v1/ocr (files + metadata + webhook_url)
    API->>API: Validate & queue tasks
    API->>Queue: enqueue process_task
    API-->>-Client: 200 OK (queued)

    Worker->>+Queue: dequeued process_task
    Worker->>Worker: OCR Pipeline Service
    Worker->>Queue: enqueue webhook_task

    Worker->>+Queue: dequeued webhook_task
    Worker->>+Webhook: POST results
    Webhook-->>-Worker: 200 OK
```

```mermaid
flowchart LR
    A[Client]
    B[FastAPI]
    C[RabbitMQ]
    D[Celery Worker]
    E[Webhook Endpoint]

    A -->|POST /api/v1/ocr| B
    B -->|enqueue| C
    C -->|dequeue| D
    D -->|process OCR| D
    D -->|enqueue| C
    C -->|dequeue| D
    D -->|POST results| E
```

---

## 🚀 Setup & Installation

1. **Clone** the repository:

   ```bash
   git clone <repo_url>
   cd <project_dir>
   ```

2. **Install** dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure** environment:

   * Create a `.env` in project root:

     ```env
     CELERY_BROKER_URL="amqp://guest:guest@localhost:5672//"
     CELERY_BACKEND_URL="redis://localhost:6379/0"
     ALLOW_INSECURE_WEBHOOKS=True
     ```
   * Place `master_config.json` alongside with model paths:

     ```json
     {
       "debug": false,
       "detection": { ... },
       "recognition": { ... },
       "pipeline": { ... },
       "valid_ocr_formats": [ ... ]
     }
     ```

4. **Start** services:

   * **Celery Workers**:

     ```bash
     celery -A app.worker.celery_app worker --loglevel=info -c 4
     ```
   * **FastAPI Server**:

     ```bash
     uvicorn app.main:app --host 0.0.0.0 --port 8000
     ```

---

## 📦 API Usage

**Endpoint**: `POST /api/v1/ocr`

**Parameters** (multipart/form-data):

* `files`: PDF/image files (one or more).
* `metadata`: JSON array of objects `{ guid: string, file_type: string }`.
* `webhook_url`: URL to receive results.

**Example**:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ocr \
  -F "files=@/path/doc.pdf" \
  -F 'metadata=[{"guid":"uuid-123","file_type":"application/pdf"}]' \
  -F "webhook_url=http://your-server.com/receive"
```

**Webhook Payload** (`application/json`):

```json
{
  "task_id": "...",
  "guid": "uuid-123",
  "text": "Base64-encoded text",
  "status": "completed",
  "confidence": 0.95,
  "error": ""
}
```

---

## 🧪 Stress Testing

Use `stress_test.py` to simulate load:

1. Place test files next to `stress_test.py`.
2. Install extras:

   ```bash
   pip install aiohttp aiofiles
   ```
3. Run:

   ```bash
   python stress_test.py
   ```

## 6. Performance & Memory Optimization

### 6.1 Image Compression (Redis)
**Implemented:** PNG + Zlib lossless compression

All page images stored in Redis are now automatically:
1. Encoded to PNG (lossless compression)
2. Compressed with zlib level 6 (lossless)

**Benefits:**
- **Memory saved:** 50-60% reduction in Redis memory usage
- **Quality impact:** ZERO (both PNG and zlib are lossless)
- **AI accuracy:** Unchanged (same pixel-perfect images)

**Example:**
```
10MB uncompressed PNG → 4-5MB after zlib compression
100-page PDF: 1000MB → 400-500MB in Redis
```

**Monitoring compression:**
```bash
# Check logs for compression stats
docker logs ocr_worker_ocr | grep "compression_applied"

# Output example:
# state.image.compression_applied original_size_bytes=10485760 
# compressed_size_bytes=4194304 compression_ratio_percent=60.0%
```

### 6.2 Backward Compatibility
The system automatically handles both:
- New format: `png_zlib` (compressed) - automatic on new images
- Legacy format: `png` (uncompressed) - for images uploaded before compression was enabled
- Raw numpy arrays - for very old legacy data

No migration needed - old images work automatically!

### 6.3 Celery Worker Optimization
- **Dispatch worker:** 1 concurrent task (fast, CPU-light)
- **OCR worker:** 1 concurrent task (heavy, memory-intensive)
- **Webhook worker:** 8 concurrent tasks (I/O bound, parallelizable)

### 6.4 Retry Mechanism
Webhook delivery uses exponential backoff:
- Attempt 1: 5 seconds delay
- Attempt 2: 10 seconds delay
- Max retries: 2 (total ~15 seconds)

This prevents queue starvation from failed webhooks.

### 6.5 PDF Processing
Adaptive DPI fallback handles large/corrupted PDFs:
- Attempt 1: DPI 300 (best quality)
- Attempt 2: DPI 200 (if first fails)
- Attempt 3: DPI 150 (if second fails)

Prevents OOM crashes on large PDFs.
