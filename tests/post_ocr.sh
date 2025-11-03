#!/usr/bin/env bash
set -euo pipefail

# Uploads a provided file to the OCR queue and prints the response.
usage() {
  cat >&2 <<'USAGE'
Usage: post_ocr.sh <path-to-file> [--webhook URL]

Environment overrides:
  BASE_URL       (default: http://127.0.0.1:8000)
  METADATA_GUID  (default: generated)
  METADATA_FORMAT (default: inferred from file extension)
  WEBHOOK_URL    (default: empty; overridden by --webhook)
USAGE
  exit 1
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_URL=${BASE_URL:-http://127.0.0.1:8000}
WEBHOOK_URL=${WEBHOOK_URL:-}
FILE_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --webhook)
      shift || usage
      WEBHOOK_URL=$1
      ;;
    -h|--help)
      usage
      ;;
    *)
      if [[ -z "$FILE_PATH" ]]; then
        FILE_PATH=$1
      else
        usage
      fi
      ;;
  esac
  shift || true
done

if [[ -z "$FILE_PATH" ]]; then
  usage
fi

if [[ ! -f "$FILE_PATH" ]]; then
  echo "File not found: $FILE_PATH" >&2
  exit 1
fi

# Auto-detect the file extension unless METADATA_FORMAT is provided.
FILE_EXT=${FILE_PATH##*.}
if [[ "$FILE_EXT" != "$FILE_PATH" ]]; then
  DEFAULT_FORMAT=".${FILE_EXT,,}"
else
  DEFAULT_FORMAT=""
fi

METADATA_GUID=${METADATA_GUID:-"test-guid-$(date +%s)"}
METADATA_FORMAT=${METADATA_FORMAT:-$DEFAULT_FORMAT}
ENDPOINT="${BASE_URL%/}/v3/ocr"
METADATA_PAYLOAD=$(printf '[{"guid":"%s","format":"%s"}]' "$METADATA_GUID" "$METADATA_FORMAT")

curl_args=(
  -sS
  -w "\nHTTP %{http_code}\n"
  -X POST "$ENDPOINT"
  -F "files=@${FILE_PATH}"
  -F "metadata=${METADATA_PAYLOAD}"
)

if [[ -n "$WEBHOOK_URL" ]]; then
  curl_args+=(-F "webhook_url=${WEBHOOK_URL}")
fi

curl "${curl_args[@]}"
