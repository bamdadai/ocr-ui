#!/bin/bash
# Log Analysis Helper Scripts
# Makes JSON logs easier to read and query

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

LOG_DIR="${LOG_DIR:-./logs}"

# Pretty-print logs in real-time
function tail_logs() {
    local log_file="${1:-info}"
    echo -e "${GREEN}Tailing ${log_file}.log (Ctrl+C to stop)${NC}"
    tail -f "${LOG_DIR}/${log_file}.log" | jq -r --unbuffered '
        "\(.timestamp | split("T")[1] | split(".")[0]) [\(.level | ascii_upcase | .[0:4])] \(.event | .[0:35]) \(if .guid then "guid=\(.guid[0:8])" else "" end) \(if .correlation_id then "req=\(.correlation_id[0:8])" else "" end) \(if .error then "error=\(.error[0:50])" else "" end)"
    ' 2>/dev/null | while IFS= read -r line; do
        if [[ $line == *"ERROR"* ]] || [[ $line == *"CRIT"* ]]; then
            echo -e "${RED}${line}${NC}"
        elif [[ $line == *"WARN"* ]]; then
            echo -e "${YELLOW}${line}${NC}"
        elif [[ $line == *"INFO"* ]]; then
            echo -e "${GREEN}${line}${NC}"
        else
            echo -e "${BLUE}${line}${NC}"
        fi
    done
}

# Show all errors for a specific GUID
function errors_by_guid() {
    local guid="$1"
    if [ -z "$guid" ]; then
        echo "Usage: errors_by_guid <guid>"
        return 1
    fi
    echo -e "${RED}Errors for GUID: ${guid}${NC}"
    jq -r "select(.guid == \"$guid\") | \"\(.timestamp) [\(.level)] \(.event): \(.error // .message // \"no message\")\"" "${LOG_DIR}/errors.log" 2>/dev/null
}

# Show complete request timeline by correlation ID
function request_timeline() {
    local correlation_id="$1"
    if [ -z "$correlation_id" ]; then
        echo "Usage: request_timeline <correlation_id>"
        return 1
    fi
    echo -e "${BLUE}Timeline for request: ${correlation_id}${NC}"
    grep -h "\"correlation_id\": \"$correlation_id\"" "${LOG_DIR}"/*.log 2>/dev/null | \
    jq -s 'sort_by(.timestamp) | .[] | "\(.timestamp | split("T")[1] | split(".")[0]) [\(.level | ascii_upcase | .[0:4])] \(.event) \(if .guid then "guid=\(.guid[0:8])" else "" end)"' -r
}

# Count events by type
function event_summary() {
    local log_file="${1:-info}"
    echo -e "${GREEN}Event summary for ${log_file}.log${NC}"
    jq -r '.event' "${LOG_DIR}/${log_file}.log" 2>/dev/null | sort | uniq -c | sort -rn | head -20
}

# Find all warnings for a specific event type
function warnings_by_event() {
    local event_pattern="$1"
    if [ -z "$event_pattern" ]; then
        echo "Usage: warnings_by_event <event_pattern>"
        echo "Example: warnings_by_event 'encoding'"
        return 1
    fi
    echo -e "${YELLOW}Warnings matching: ${event_pattern}${NC}"
    jq -r "select(.event | contains(\"$event_pattern\")) | \"\(.timestamp | split(\"T\")[1] | split(\".\")[0]) \(.event) guid=\(.guid // \"N/A\") error=\(.error // \"N/A\")\"" "${LOG_DIR}/warnings.log" 2>/dev/null
}

# Show error rate over time (last N lines)
function error_rate() {
    local lines="${1:-100}"
    echo -e "${RED}Error distribution (last ${lines} errors)${NC}"
    tail -n "$lines" "${LOG_DIR}/errors.log" 2>/dev/null | \
    jq -r '.event' | sort | uniq -c | sort -rn
}

# Show recent failed tasks
function failed_tasks() {
    local count="${1:-10}"
    echo -e "${RED}Last ${count} failed tasks${NC}"
    jq -r "select(.event | contains(\"failed\") or contains(\"error\")) | \"\(.timestamp | split(\"T\")[1] | split(\".\")[0]) \(.event) guid=\(.guid // \"N/A\")[0:8] error=\(.error // \"N/A\")[0:60]\"" "${LOG_DIR}/errors.log" 2>/dev/null | tail -n "$count"
}

# Show webhook delivery status
function webhook_status() {
    echo -e "${BLUE}Webhook delivery summary${NC}"
    echo "Successes:"
    grep -c "webhook.*success" "${LOG_DIR}/info.log" 2>/dev/null || echo "0"
    echo "Failures:"
    grep -c "webhook.*failed\|webhook.*timeout" "${LOG_DIR}/errors.log" 2>/dev/null || echo "0"
}

# Search logs by GUID across all log files
function search_guid() {
    local guid="$1"
    if [ -z "$guid" ]; then
        echo "Usage: search_guid <guid>"
        return 1
    fi
    echo -e "${BLUE}All logs for GUID: ${guid}${NC}"
    for log in "${LOG_DIR}"/{info,warnings,errors,debug}.log; do
        if [ -f "$log" ]; then
            echo -e "\n${GREEN}=== $(basename $log) ===${NC}"
            jq -r "select(.guid == \"$guid\") | \"\(.timestamp | split(\"T\")[1] | split(\".\")[0]) [\(.level | ascii_upcase)] \(.event)\"" "$log" 2>/dev/null
        fi
    done
}

# Show processing stats
function processing_stats() {
    echo -e "${GREEN}Processing Statistics${NC}"
    echo -e "\nTotal tasks received:"
    jq -s 'map(select(.event == "ocr.task.dispatched")) | length' "${LOG_DIR}/info.log" 2>/dev/null || echo "0"
    
    echo -e "\nCompleted tasks:"
    jq -s 'map(select(.event == "finalize.success")) | length' "${LOG_DIR}/debug.log" 2>/dev/null || echo "0"
    
    echo -e "\nFailed tasks:"
    jq -s 'map(select(.event | contains("failed"))) | length' "${LOG_DIR}/errors.log" 2>/dev/null || echo "0"
    
    echo -e "\nAverage confidence (last 100 tasks):"
    jq -s 'map(select(.event == "finalize.success")) | .[-100:] | map(.confidence) | add / length' "${LOG_DIR}/debug.log" 2>/dev/null || echo "N/A"
}

# Show help
function log_help() {
    echo -e "${GREEN}Log Analysis Helper Commands${NC}"
    echo ""
    echo "Real-time monitoring:"
    echo "  tail_logs [info|warnings|errors|debug]  - Pretty-print logs in real-time"
    echo ""
    echo "Search and filter:"
    echo "  search_guid <guid>                      - Show all logs for a GUID"
    echo "  errors_by_guid <guid>                   - Show errors for a GUID"
    echo "  request_timeline <correlation_id>       - Show request lifecycle"
    echo "  warnings_by_event <pattern>             - Find warnings matching pattern"
    echo ""
    echo "Statistics:"
    echo "  event_summary [info|warnings|errors]    - Count events by type"
    echo "  error_rate [lines]                      - Show error distribution"
    echo "  processing_stats                        - Show processing statistics"
    echo "  webhook_status                          - Show webhook delivery status"
    echo ""
    echo "Recent activity:"
    echo "  failed_tasks [count]                    - Show recent failed tasks"
    echo ""
    echo "Examples:"
    echo "  tail_logs info"
    echo "  search_guid abc123"
    echo "  request_timeline d9dbb059-e15a-45e9-bfc0-4d7e2fe39cc5"
    echo "  warnings_by_event encoding"
}

# If script is sourced, make functions available
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    # Script is executed, show help
    log_help
fi

