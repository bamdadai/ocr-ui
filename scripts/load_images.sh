#!/bin/bash
# Load Docker images on offline machine
# Supports both .tar and .tar.gz files
# Run this on the offline machine after transferring image files

# Don't exit on error - we want to continue loading other images
# set -e  # Commented out to allow partial success

IMAGE_DIR="${1:-images}"

if [ ! -d "$IMAGE_DIR" ]; then
    echo "Error: Directory '$IMAGE_DIR' not found"
    echo "Usage: $0 [image_directory]"
    exit 1
fi

echo "=== Loading Docker Images from $IMAGE_DIR ==="
echo ""

# Function to load a single image file (tar or tar.gz)
load_image() {
    local file="$1"
    if [ ! -f "$file" ]; then
        return 1
    fi
    
    local filename=$(basename "$file")
    echo "Loading: $filename..."
    
    # Check if file is readable and has content
    if [ ! -s "$file" ]; then
        echo "  Warning: File is empty, skipping: $filename"
        return 1
    fi
    
    if [[ "$file" == *.tar.gz ]]; then
        # Compressed file - verify integrity first, then decompress and load
        if ! gzip -t "$file" 2>/dev/null; then
            echo "  Error: $filename is corrupted (gzip test failed)"
            echo "  Attempting to load anyway (might be incomplete)..."
            # Try alternative: decompress to temp file first
            local temp_tar=$(mktemp)
            if gunzip -c "$file" > "$temp_tar" 2>/dev/null; then
                docker load -i "$temp_tar"
                rm -f "$temp_tar"
            else
                echo "  Failed to decompress $filename"
                rm -f "$temp_tar"
                return 1
            fi
        else
            # File is valid, load via pipe
            gunzip -c "$file" | docker load || {
                echo "  Error loading $filename, trying alternative method..."
                # Fallback: decompress to temp file
                local temp_tar=$(mktemp)
                gunzip -c "$file" > "$temp_tar" 2>/dev/null && docker load -i "$temp_tar"
                rm -f "$temp_tar"
            }
        fi
    elif [[ "$file" == *.tar ]]; then
        # Uncompressed file - load directly
        docker load -i "$file"
    else
        echo "  Warning: Skipping unknown file type: $filename"
        return 1
    fi
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Successfully loaded: $filename"
    else
        echo "  ✗ Failed to load: $filename"
        return 1
    fi
}

# Load application images
for img in ocr-api ocr-api-gpu ocr-worker ocr-worker-gpu; do
    # Try .tar.gz first (compressed), then .tar (uncompressed)
    if [ -f "$IMAGE_DIR/${img}-latest.tar.gz" ]; then
        load_image "$IMAGE_DIR/${img}-latest.tar.gz"
    elif [ -f "$IMAGE_DIR/${img}.tar.gz" ]; then
        load_image "$IMAGE_DIR/${img}.tar.gz"
    elif [ -f "$IMAGE_DIR/${img}-latest.tar" ]; then
        load_image "$IMAGE_DIR/${img}-latest.tar"
    elif [ -f "$IMAGE_DIR/${img}.tar" ]; then
        load_image "$IMAGE_DIR/${img}.tar"
    fi
done

# Load base images (common patterns)
for img in redis rabbitmq minio python; do
    for file in "$IMAGE_DIR/${img}"*.tar.gz "$IMAGE_DIR/${img}"*.tar; do
        if [ -f "$file" ]; then
            load_image "$file"
            break  # Only load first match
        fi
    done
done

echo ""
echo "=== Verifying Loaded Images ==="
docker images | grep -E "(ocr-api|ocr-worker|redis|rabbitmq|minio|python)" || true

echo ""
echo "=== Load Complete ==="
if docker images | grep -q "ocr-worker-gpu"; then
    echo "GPU images detected. Available compose files:"
    echo "  CPU:  docker-compose -f docker-compose.cpu.yml up -d"
    echo "  GPU:  docker-compose -f docker-compose.gpu.yml up -d"
else
    echo "Images ready. Use:"
    echo "  docker-compose -f docker-compose.cpu.yml up -d"
fi

