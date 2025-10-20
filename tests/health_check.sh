#!/usr/bin/env bash
set -euo pipefail

# Simple smoke test for the root health-check endpoint.
BASE_URL=${BASE_URL:-http://127.0.0.1:8000}
ENDPOINT="${BASE_URL%/}/"

curl -sS -w "\nHTTP %{http_code}\n" \
  -H "Accept: application/json" \
  "$ENDPOINT"
