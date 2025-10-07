# OCR Client & Webhook Server (v3 API)

This directory contains tools for testing and interacting with the OCR service:

- **`ocr_client.py`**: Client for submitting files and retrieving OCR results (uses v3 API)
- **`webhook_server.py`**: Webhook server for receiving OCR results asynchronously

## Quick Start

### 1. Start the Webhook Server

```bash
# Start on default port (8080)
python scripts/webhook_server.py

# Start on custom port
python scripts/webhook_server.py --port 9000

# Start without saving to disk
python scripts/webhook_server.py --no-save
```

The webhook server will:
- Listen on `http://0.0.0.0:8080/webhook/ocr` for OCR results
- Save results to `webhook_results/` directory (JSON + text files)
- Provide endpoints to view received results

### 2. Submit OCR Tasks

#### Option A: Webhook-based (Async)

Submit files and receive results via webhook:

```bash
# Submit single file
python scripts/ocr_client.py submit image.png \
    --webhook http://localhost:8080/webhook/ocr

# Submit multiple files
python scripts/ocr_client.py submit file1.png file2.pdf \
    --webhook http://localhost:8080/webhook/ocr

# With custom GUID
python scripts/ocr_client.py submit image.png \
    --webhook http://localhost:8080/webhook/ocr \
    --guid my-custom-guid-123
```

#### Option B: Polling-based (Sync)

Submit files and wait for results by polling:

```bash
# Submit and poll until complete
python scripts/ocr_client.py submit image.png --poll

# Save result to file
python scripts/ocr_client.py submit image.png --poll --output result.txt

# Custom polling interval and timeout
python scripts/ocr_client.py submit image.png --poll \
    --poll-interval 1.0 --timeout 600
```

#### Option C: Manual Status Check

Submit first, then check status manually:

```bash
# Submit
python scripts/ocr_client.py submit image.png

# Check status (use task_id from submit response)
python scripts/ocr_client.py status <task-id>
```

## Installation Requirements

Install the required dependencies:

```bash
pip install flask requests
```

## Webhook Server Endpoints

The webhook server provides these endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/webhook/ocr` | POST | Receive OCR results |
| `/webhook/health` | GET | Health check |
| `/webhook/results` | GET | List all received results |
| `/webhook/results/<guid>` | GET | Get specific result by GUID |

### Examples

```bash
# Check webhook server health
curl http://localhost:8080/webhook/health

# View all received results
curl http://localhost:8080/webhook/results | jq

# Get specific result by GUID
curl http://localhost:8080/webhook/results/<guid> | jq
```

## Complete Workflow Example

### Terminal 1: Start Webhook Server
```bash
python scripts/webhook_server.py
```

### Terminal 2: Submit OCR Task
```bash
python scripts/ocr_client.py submit test_image.png \
    --webhook http://localhost:8080/webhook/ocr \
    --guid test-123
```

### Terminal 3: Monitor Results
```bash
# Watch for results
watch -n 1 'curl -s http://localhost:8080/webhook/results | jq'

# Or view saved files
ls -l webhook_results/
cat webhook_results/test-123_*.txt
```

## OCR Result Format

The webhook receives results in this format:

```json
{
  "task_id": "uuid-task-id",
  "guid": "client-provided-guid",
  "text": "base64-encoded-utf8-text",
  "confidence": 0.95,
  "status": "completed",
  "error": ""
}
```

The webhook server automatically:
- Decodes the base64 text
- Logs the result
- Saves to disk (JSON + plain text)
- Responds with confirmation

## Advanced Usage

### Using with Remote OCR Service

```bash
# Point client to remote API
python scripts/ocr_client.py submit image.png \
    --base-url https://ocr-api.example.com \
    --webhook https://my-webhook-server.com/webhook/ocr
```

### Using ngrok for Public Webhook

If the OCR service can't reach your local webhook server:

```bash
# Terminal 1: Start ngrok
ngrok http 8080

# Terminal 2: Start webhook server
python scripts/webhook_server.py

# Terminal 3: Submit with ngrok URL
python scripts/ocr_client.py submit image.png \
    --webhook https://abc123.ngrok.io/webhook/ocr
```

### Custom Metadata

The client supports custom metadata for each file:

```python
import json

metadata = [
    {
        "guid": "doc-001",
        "format": ".png",
        "custom_field": "custom_value"
    }
]

# Save to file and use
with open('metadata.json', 'w') as f:
    json.dump(metadata, f)
```

### Production Deployment

For production use of the webhook server:

1. **Use a production WSGI server:**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:8080 scripts.webhook_server:app
   ```

2. **Add authentication:**
   - Implement API key verification
   - Use HTTPS/TLS
   - Validate webhook signatures

3. **Add persistence:**
   - Store results in a database
   - Use message queue for reliability
   - Implement result retention policy

4. **Monitor and alert:**
   - Add metrics collection
   - Set up logging aggregation
   - Configure failure alerts

## Troubleshooting

### Webhook not receiving results

1. Check OCR service can reach webhook:
   ```bash
   # From OCR service host
   curl http://webhook-host:8080/webhook/health
   ```

2. Check firewall/network settings

3. Verify webhook URL in request:
   ```bash
   python scripts/ocr_client.py submit image.png \
       --webhook http://localhost:8080/webhook/ocr
   ```

4. Check webhook server logs

### Connection refused errors

- Ensure the OCR API is running: `http://localhost:8000`
- Check the `--base-url` parameter matches your API URL
- Verify network connectivity

### Timeout errors

- Increase timeout: `--timeout 600`
- Check OCR service logs for processing errors
- Verify file format is supported

### Decoding errors

- The text is base64-encoded UTF-8
- Client automatically decodes it
- Check for encoding issues in source files

## Testing the Webhook Integration

To verify webhook delivery works correctly:

```bash
# Terminal 1: Start webhook with verbose output
python scripts/webhook_server.py

# Terminal 2: Submit test file
python scripts/ocr_client.py submit tests/samples/test_image.png \
    --webhook http://localhost:8080/webhook/ocr \
    --guid webhook-test-001

# Terminal 3: Verify result was received
curl http://localhost:8080/webhook/results/webhook-test-001 | jq

# Check saved files
cat webhook_results/webhook-test-001_*.txt
```

## License

See main project LICENSE file.
