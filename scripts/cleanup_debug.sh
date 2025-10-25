#!/bin/bash
# ============================================================================
# Debug Directory Cleanup Script
# ============================================================================
# This script cleans up accumulated debug images and old logs to prevent
# disk space issues in production OCR workers.
#
# Usage:
#   ./scripts/cleanup_debug.sh              # Dry run (shows what would be deleted)
#   ./scripts/cleanup_debug.sh --execute    # Actually delete files
#   ./scripts/cleanup_debug.sh --keep-days 7  # Keep only last 7 days
# ============================================================================

set -euo pipefail

# Default configuration
DRY_RUN=true
KEEP_DAYS=3
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --execute)
            DRY_RUN=false
            shift
            ;;
        --keep-days)
            KEEP_DAYS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--execute] [--keep-days N]"
            echo "  --execute      Actually delete files (default: dry run)"
            echo "  --keep-days N  Keep files from last N days (default: 3)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== OCR Debug Cleanup Script ===${NC}"
echo "Project root: $PROJECT_ROOT"
echo "Keep days: $KEEP_DAYS"
echo "Mode: $([ "$DRY_RUN" = true ] && echo "DRY RUN" || echo "EXECUTE")"
echo ""

# Function to get directory size
get_dir_size() {
    local dir="$1"
    if [ -d "$dir" ]; then
        du -sh "$dir" 2>/dev/null | cut -f1
    else
        echo "0"
    fi
}

# Function to count files
count_files() {
    local dir="$1"
    if [ -d "$dir" ]; then
        find "$dir" -type f 2>/dev/null | wc -l
    else
        echo "0"
    fi
}

# Directories to clean
DIRS=(
    "$PROJECT_ROOT/debug/word_detections"
    "$PROJECT_ROOT/debug/line_detections"
    "$PROJECT_ROOT/debug/word_polygons"
    "$PROJECT_ROOT/debug/parts"
    "$PROJECT_ROOT/debug/word_crops"
)

# Show current status
echo -e "${YELLOW}Current Status:${NC}"
for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        size=$(get_dir_size "$dir")
        count=$(count_files "$dir")
        echo "  $dir: $size ($count files)"
    fi
done

# Calculate total size before
TOTAL_BEFORE=$(get_dir_size "$PROJECT_ROOT/debug")
echo -e "\n${YELLOW}Total debug directory size: $TOTAL_BEFORE${NC}\n"

# Clean up old files
echo -e "${GREEN}Cleaning files older than $KEEP_DAYS days...${NC}"

TOTAL_DELETED=0
for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        if [ "$DRY_RUN" = true ]; then
            echo "  [DRY RUN] Would delete from: $dir"
            count=$(find "$dir" -type f -mtime +$KEEP_DAYS 2>/dev/null | wc -l)
            echo "    Files to delete: $count"
            TOTAL_DELETED=$((TOTAL_DELETED + count))
        else
            echo "  Cleaning: $dir"
            count=$(find "$dir" -type f -mtime +$KEEP_DAYS 2>/dev/null | wc -l)
            find "$dir" -type f -mtime +$KEEP_DAYS -delete 2>/dev/null || true
            echo "    Deleted: $count files"
            TOTAL_DELETED=$((TOTAL_DELETED + count))
        fi
    fi
done

# Remove empty directories
if [ "$DRY_RUN" = false ]; then
    for dir in "${DIRS[@]}"; do
        if [ -d "$dir" ]; then
            find "$dir" -type d -empty -delete 2>/dev/null || true
        fi
    done
fi

# Show final status
echo ""
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}=== DRY RUN SUMMARY ===${NC}"
    echo "Would delete: $TOTAL_DELETED files"
    echo ""
    echo -e "${GREEN}To actually delete these files, run:${NC}"
    echo "  $0 --execute --keep-days $KEEP_DAYS"
else
    TOTAL_AFTER=$(get_dir_size "$PROJECT_ROOT/debug")
    echo -e "${GREEN}=== CLEANUP COMPLETE ===${NC}"
    echo "Deleted: $TOTAL_DELETED files"
    echo "Size before: $TOTAL_BEFORE"
    echo "Size after:  $TOTAL_AFTER"
fi

echo ""

