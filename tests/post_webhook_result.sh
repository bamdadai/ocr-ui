#!/usr/bin/env bash
set -euo pipefail

# Sends a sample OCR webhook payload to the callback endpoint.

PAYLOAD_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --payload-file)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --payload-file" >&2
        exit 1
      fi
      PAYLOAD_FILE=$2
      shift 2
      ;;
    --help|-h)
      cat <<'USAGE'
Usage: post_webhook_result.sh [--payload-file <path>]

When --payload-file is provided, its JSON body is sent verbatim.
Otherwise a payload is generated from WEBHOOK_* environment variables:
  WEBHOOK_TASK_ID       Override task_id (default demo-task-<timestamp>)
  WEBHOOK_GUID          Override guid (default demo-guid)
  WEBHOOK_TEXT          Plaintext to encode as Base64 (default sample text)
  WEBHOOK_CONFIDENCE    Confidence score (default 0.99)
  WEBHOOK_STATUS        Result status (default completed)
  WEBHOOK_ERROR         Error message (default empty)
  WEBHOOK_BASE_URL      Callback base URL (default http://127.0.0.1:8080)
USAGE
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

BASE_URL=${WEBHOOK_BASE_URL:-http://127.0.0.1:8080}
ENDPOINT="${BASE_URL%/}/webhook/ocr"

if [[ -n "$PAYLOAD_FILE" ]]; then
  if [[ ! -f "$PAYLOAD_FILE" ]]; then
    echo "Payload file not found: $PAYLOAD_FILE" >&2
    exit 1
  fi
  CURL_DATA=(--data-binary @"$PAYLOAD_FILE")
else
  PAYLOAD=$(python - <<'PY'
import base64
import json
import os
import sys
import time

def float_or_default(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

payload = {
    "task_id": os.environ.get("WEBHOOK_TASK_ID") or f"demo-task-{int(time.time())}",
    "guid": os.environ.get("WEBHOOK_GUID") or "demo-guid",
    "confidence": float_or_default(os.environ.get("WEBHOOK_CONFIDENCE"), 0.99),
    "status": os.environ.get("WEBHOOK_STATUS") or "completed",
    "error": os.environ.get("WEBHOOK_ERROR") or "",
}

text = os.environ.get("WEBHOOK_TEXT") or "Sample OCR text delivered via webhook."
payload["text"] = base64.b64encode(text.encode("utf-8")).decode("ascii")

json.dump(payload, sys.stdout, separators=(",", ":"))
PY
)
  CURL_DATA=(--data "$PAYLOAD")
fi

curl \
  -sS \
  -w "\nHTTP %{http_code}\n" \
  -H "Content-Type: application/json" \
  -X POST \
  "$ENDPOINT" \
  "${CURL_DATA[@]}"
