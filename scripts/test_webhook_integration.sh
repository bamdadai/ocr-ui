#!/bin/bash
#
# Test script for webhook integration
#
# This script tests the complete webhook flow:
# 1. Starts the webhook server
# 2. Submits a test OCR task
# 3. Verifies the result is received
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== OCR Webhook Integration Test ===${NC}\n"

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo -e "${RED}Error: Flask is not installed${NC}"
    echo "Install with: pip install flask"
    exit 1
fi

# Check if requests is installed
if ! python -c "import requests" 2>/dev/null; then
    echo -e "${RED}Error: requests library is not installed${NC}"
    echo "Install with: pip install requests"
    exit 1
fi

# Check if test file exists
TEST_FILE="${1:-tests/samples/test_image.png}"
if [ ! -f "$TEST_FILE" ]; then
    echo -e "${RED}Error: Test file not found: $TEST_FILE${NC}"
    echo "Usage: $0 [path/to/test/image.png]"
    exit 1
fi

# Start webhook server in background
echo -e "${GREEN}1. Starting webhook server on port 8080...${NC}"
python "$SCRIPT_DIR/webhook_server.py" --port 8080 > /tmp/webhook_server.log 2>&1 &
WEBHOOK_PID=$!

# Wait for webhook server to start
sleep 2

# Check if webhook server is running
if ! kill -0 $WEBHOOK_PID 2>/dev/null; then
    echo -e "${RED}Error: Webhook server failed to start${NC}"
    cat /tmp/webhook_server.log
    exit 1
fi

echo -e "${GREEN}✓ Webhook server started (PID: $WEBHOOK_PID)${NC}\n"

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    if [ ! -z "$WEBHOOK_PID" ]; then
        kill $WEBHOOK_PID 2>/dev/null || true
        echo -e "${GREEN}✓ Webhook server stopped${NC}"
    fi
}
trap cleanup EXIT

# Check webhook server health
echo -e "${GREEN}2. Checking webhook server health...${NC}"
if curl -s http://localhost:8080/webhook/health | grep -q "healthy"; then
    echo -e "${GREEN}✓ Webhook server is healthy${NC}\n"
else
    echo -e "${RED}Error: Webhook server health check failed${NC}"
    exit 1
fi

# Submit OCR task
TEST_GUID="test-$(date +%s)"
echo -e "${GREEN}3. Submitting OCR task (GUID: $TEST_GUID)...${NC}"
echo "   File: $TEST_FILE"
echo "   Webhook: http://localhost:8080/webhook/ocr"

SUBMIT_RESULT=$(python "$SCRIPT_DIR/ocr_client.py" submit "$TEST_FILE" \
    --webhook http://localhost:8080/webhook/ocr \
    --guid "$TEST_GUID" 2>/dev/null)

echo "$SUBMIT_RESULT" | python -m json.tool

TASK_ID=$(echo "$SUBMIT_RESULT" | python -c "import sys, json; data=json.load(sys.stdin); print(data[0]['task_id'])" 2>/dev/null)

if [ -z "$TASK_ID" ]; then
    echo -e "${RED}Error: Failed to get task ID from submission${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Task submitted (Task ID: $TASK_ID)${NC}\n"

# Wait for webhook to receive result
echo -e "${GREEN}4. Waiting for webhook result (max 60 seconds)...${NC}"
COUNTER=0
MAX_WAIT=60

while [ $COUNTER -lt $MAX_WAIT ]; do
    if curl -s "http://localhost:8080/webhook/results/$TEST_GUID" | grep -q "received_at"; then
        echo -e "${GREEN}✓ Webhook received result!${NC}\n"
        break
    fi
    sleep 2
    COUNTER=$((COUNTER + 2))
    echo -n "."
done

if [ $COUNTER -ge $MAX_WAIT ]; then
    echo -e "\n${RED}Error: Webhook did not receive result within $MAX_WAIT seconds${NC}"
    echo -e "${YELLOW}Checking task status directly...${NC}"
    python "$SCRIPT_DIR/ocr_client.py" status "$TASK_ID"
    exit 1
fi

# Retrieve and display result
echo -e "${GREEN}5. Retrieving result from webhook server...${NC}"
RESULT=$(curl -s "http://localhost:8080/webhook/results/$TEST_GUID")
echo "$RESULT" | python -m json.tool

# Check if text was decoded
if echo "$RESULT" | grep -q '"text"'; then
    TEXT_LENGTH=$(echo "$RESULT" | python -c "import sys, json; data=json.load(sys.stdin); print(len(data.get('text', '')))")
    echo -e "\n${GREEN}✓ OCR text received (length: $TEXT_LENGTH characters)${NC}"
else
    echo -e "\n${RED}Error: No text in result${NC}"
    exit 1
fi

# Check saved files
if [ -d "webhook_results" ]; then
    SAVED_FILES=$(ls -1 webhook_results/${TEST_GUID}_* 2>/dev/null | wc -l)
    if [ "$SAVED_FILES" -gt 0 ]; then
        echo -e "${GREEN}✓ Result saved to disk ($SAVED_FILES files)${NC}"
        ls -lh webhook_results/${TEST_GUID}_*
    fi
fi

echo -e "\n${GREEN}=== All tests passed! ===${NC}"
echo -e "${YELLOW}Webhook server log:${NC}"
tail -20 /tmp/webhook_server.log
