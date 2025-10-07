#!/usr/bin/env python3
"""
Test Masks Object Compatibility
Tests the new Ultralytics Masks object handling
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.utils.image_processing import get_polygons_from_masks

def test_masks_compatibility():
    """Test different mask formats to ensure compatibility."""
    
    print("🧪 Testing Masks Object Compatibility")
    print("=" * 50)
    
    # Test 1: Legacy list of numpy arrays
    print("\n1️⃣ Testing legacy List[np.ndarray] format...")
    legacy_masks = [
        np.random.randint(0, 2, (100, 100), dtype=np.uint8),
        np.random.randint(0, 2, (100, 100), dtype=np.uint8)
    ]
    
    try:
        polygons = get_polygons_from_masks(legacy_masks)
        print(f"   ✅ Legacy format: {len(polygons)} polygons extracted")
    except Exception as e:
        print(f"   ❌ Legacy format failed: {e}")
    
    # Test 2: Single numpy array
    print("\n2️⃣ Testing single numpy array...")
    single_mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
    
    try:
        polygons = get_polygons_from_masks(single_mask)
        print(f"   ✅ Single array: {len(polygons)} polygons extracted")
    except Exception as e:
        print(f"   ❌ Single array failed: {e}")
    
    # Test 3: 3D numpy array
    print("\n3️⃣ Testing 3D numpy array...")
    masks_3d = np.random.randint(0, 2, (2, 100, 100), dtype=np.uint8)
    
    try:
        polygons = get_polygons_from_masks(masks_3d)
        print(f"   ✅ 3D array: {len(polygons)} polygons extracted")
    except Exception as e:
        print(f"   ❌ 3D array failed: {e}")
    
    # Test 4: Mock Masks object (simulating new Ultralytics structure)
    print("\n4️⃣ Testing mock Masks object...")
    
    class MockMasks:
        def __init__(self, data):
            self.data = data
        
        def cpu(self):
            return self
        
        def numpy(self):
            return self.data
    
    mock_masks = MockMasks(np.random.randint(0, 2, (2, 100, 100), dtype=np.uint8))
    
    try:
        polygons = get_polygons_from_masks(mock_masks)
        print(f"   ✅ Mock Masks object: {len(polygons)} polygons extracted")
    except Exception as e:
        print(f"   ❌ Mock Masks object failed: {e}")
    
    # Test 5: Test with real YOLO model (if available)
    print("\n5️⃣ Testing with real YOLO model...")
    model_path = project_root / "weights" / "word_detection.pt"
    
    if model_path.exists():
        try:
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            
            # Create a dummy image
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # Run prediction
            results = model(dummy_image, task='segment', retina_masks=True)
            result = results[0]
            
            if result.masks is not None:
                print(f"   📊 Real model masks type: {type(result.masks).__name__}")
                print(f"   📊 Has data attribute: {hasattr(result.masks, 'data')}")
                
                polygons = get_polygons_from_masks(result.masks)
                print(f"   ✅ Real model: {len(polygons)} polygons extracted")
            else:
                print("   ⚠️  No masks in prediction results")
                
        except Exception as e:
            print(f"   ❌ Real model test failed: {e}")
    else:
        print("   ⚠️  Model file not found, skipping real model test")
    
    print("\n" + "=" * 50)
    print("🎉 Masks compatibility test completed!")

if __name__ == "__main__":
    test_masks_compatibility()
