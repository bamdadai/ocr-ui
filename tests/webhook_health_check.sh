#!/usr/bin/env bash
set -euo pipefail

# Health check probe for the webhook callback server.

BASE_URL=${WEBHOOK_BASE_URL:-http://127.0.0.1:8080}
ENDPOINT="${BASE_URL%/}/webhook/health"

curl -sS -w "\nHTTP %{http_code}\n" "$ENDPOINT"
