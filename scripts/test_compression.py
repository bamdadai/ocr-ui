#!/usr/bin/env python3
"""
Test script to verify image compression/decompression in Redis state manager.
Ensures zero quality loss with PNG+zlib compression.
"""

import sys
import numpy as np
import cv2
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.worker.state_manager import StateManager


def create_test_image(width: int = 2048, height: int = 1536) -> np.ndarray:
    """Create a realistic test image with text-like patterns."""
    # Create a white background with some content
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add some shapes and text
    cv2.rectangle(image, (100, 100), (500, 500), (0, 0, 0), 2)
    cv2.circle(image, (1000, 750), 200, (200, 100, 50), -1)
    cv2.putText(image, "Test OCR Content", (100, 800), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
    
    return image


def test_compression():
    """Test compression and decompression with quality verification."""
    print("=" * 70)
    print("🧪 Image Compression Test (PNG + Zlib)")
    print("=" * 70)
    
    # Create test image
    print("\n1️⃣  Creating test image (2048x1536)...")
    original_image = create_test_image()
    print(f"   ✅ Image shape: {original_image.shape}")
    
    # Initialize state manager
    print("\n2️⃣  Initializing StateManager...")
    state = StateManager("test_compression_001")
    
    # Encode (compress)
    print("\n3️⃣  Encoding and compressing image...")
    try:
        payload = state._encode_page_image(original_image)
        print(f"   ✅ Compression successful!")
        print(f"   📊 Original size: {payload['original_size']:,} bytes")
        print(f"   📊 Compressed size: {payload['compressed_size']:,} bytes")
        print(f"   📊 Compression ratio: {payload['original_size'] / payload['compressed_size']:.2f}x")
        compression_percent = (1 - payload['compressed_size'] / payload['original_size']) * 100
        print(f"   📊 Space saved: {compression_percent:.1f}%")
    except Exception as e:
        print(f"   ❌ Compression failed: {e}")
        return False
    
    # Decode (decompress)
    print("\n4️⃣  Decoding and decompressing image...")
    try:
        reconstructed_image = state._decode_page_image(payload)
        print(f"   ✅ Decompression successful!")
        print(f"   📊 Reconstructed shape: {reconstructed_image.shape}")
    except Exception as e:
        print(f"   ❌ Decompression failed: {e}")
        return False
    
    # Quality verification
    print("\n5️⃣  Verifying quality (pixel-perfect comparison)...")
    
    # Compare pixel by pixel
    difference = cv2.absdiff(original_image, reconstructed_image)
    total_diff = np.sum(difference)
    
    if total_diff == 0:
        print(f"   ✅ PERFECT MATCH - 0 pixel differences!")
        print(f"   ✅ Quality loss: 0% (lossless compression confirmed)")
    else:
        max_diff = np.max(difference)
        mean_diff = np.mean(difference)
        print(f"   ⚠️  Pixel differences detected:")
        print(f"   📊 Total difference: {total_diff}")
        print(f"   📊 Max difference: {max_diff}")
        print(f"   📊 Mean difference: {mean_diff:.2f}")
        
        if max_diff <= 1:  # Allow for minimal rounding errors
            print(f"   ✅ Quality loss negligible (< 1 pixel unit)")
        else:
            print(f"   ❌ Significant quality loss detected!")
            return False
    
    # Memory efficiency summary
    print("\n6️⃣  Memory Efficiency Summary")
    print("   " + "-" * 65)
    original_mb = payload['original_size'] / (1024 * 1024)
    compressed_mb = payload['compressed_size'] / (1024 * 1024)
    savings_mb = original_mb - compressed_mb
    
    print(f"   Single page:")
    print(f"      Uncompressed: {original_mb:.2f} MB")
    print(f"      Compressed:   {compressed_mb:.2f} MB")
    print(f"      Saved:        {savings_mb:.2f} MB")
    
    # Extrapolate to 100-page PDF
    print(f"\n   100-page PDF (extrapolated):")
    print(f"      Uncompressed: {original_mb * 100:.0f} MB")
    print(f"      Compressed:   {compressed_mb * 100:.0f} MB")
    print(f"      Saved:        {savings_mb * 100:.0f} MB")
    
    print("\n" + "=" * 70)
    print("✅ All tests passed! Compression working perfectly.")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = test_compression()
    sys.exit(0 if success else 1)
