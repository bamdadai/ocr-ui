#!/bin/bash
#
# Test script to verify webhook URL is now correctly passed
#
# This tests the fix for the bug where webhook_url was not being sent
# when metadata was not provided.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}=== Webhook URL Fix Test ===${NC}\n"

# Test 1: With metadata (should have worked before)
echo -e "${GREEN}Test 1: Webhook with metadata (previously working)${NC}"
python "$SCRIPT_DIR/ocr_client.py" submit tests/samples/*.png \
    --webhook http://localhost:8080/webhook/ocr \
    --guid test-with-metadata 2>&1 | head -20

echo ""
read -p "Check logs - do you see 'webhook.dispatch'? (y/n) " answer1

# Test 2: Without metadata (the bug we fixed)
echo -e "\n${GREEN}Test 2: Webhook WITHOUT metadata (the fix)${NC}"
python "$SCRIPT_DIR/ocr_client.py" submit tests/samples/*.png \
    --webhook http://localhost:8080/webhook/ocr \
    --no-metadata 2>&1 | head -20

echo ""
read -p "Check logs - do you see 'webhook.dispatch' now? (y/n) " answer2

# Test 3: No webhook (should skip)
echo -e "\n${GREEN}Test 3: No webhook URL (should skip)${NC}"
python "$SCRIPT_DIR/ocr_client.py" submit tests/samples/*.png 2>&1 | head -20

echo ""
read -p "Check logs - do you see 'webhook.skipped'? (y/n) " answer3

# Summary
echo -e "\n${YELLOW}=== Summary ===${NC}"

if [[ "$answer1" == "y" ]]; then
    echo -e "${GREEN}✅ Test 1 PASSED${NC} - Webhook with metadata works"
else
    echo -e "${RED}❌ Test 1 FAILED${NC} - Webhook with metadata doesn't work"
fi

if [[ "$answer2" == "y" ]]; then
    echo -e "${GREEN}✅ Test 2 PASSED${NC} - Webhook WITHOUT metadata works (FIX CONFIRMED)"
else
    echo -e "${RED}❌ Test 2 FAILED${NC} - Webhook without metadata still broken"
fi

if [[ "$answer3" == "y" ]]; then
    echo -e "${GREEN}✅ Test 3 PASSED${NC} - No webhook correctly skips"
else
    echo -e "${RED}❌ Test 3 FAILED${NC} - No webhook behavior unexpected"
fi

echo -e "\n${YELLOW}Grep commands to check logs:${NC}"
echo "  grep 'ocr.webhook.mode' app.log | tail -5"
echo "  grep 'webhook.dispatch\\|webhook.skipped' app.log | tail -5"
echo "  grep 'webhook.send.success' app.log | tail -5"
