#!/usr/bin/env bash
set -euo pipefail

# Polls the task-status endpoint using a provided task ID.
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <task_id>" >&2
  exit 1
fi

TASK_ID=$1
BASE_URL=${BASE_URL:-http://127.0.0.1:8000}
ENDPOINT="${BASE_URL%/}/v3/ocr/tasks/${TASK_ID}"

curl -sS -w "\nHTTP %{http_code}\n" "$ENDPOINT"
