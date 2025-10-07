#!/usr/bin/env python3
"""
Model Compatibility Checker
Checks YOLO model compatibility with current Ultralytics version
"""

import sys
import os
from pathlib import Path
import structlog

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO
import torch

logger = structlog.get_logger(__name__)

def check_model_compatibility(model_path: str):
    """Check if a YOLO model is compatible with current Ultralytics version."""
    
    print(f"🔍 Checking model compatibility: {model_path}")
    print(f"📦 Ultralytics version: {YOLO.__version__ if hasattr(YOLO, '__version__') else 'Unknown'}")
    print(f"🔥 PyTorch version: {torch.__version__}")
    print("-" * 50)
    
    try:
        # Load model
        model = YOLO(model_path)
        print("✅ Model loaded successfully")
        
        # Check model structure
        if hasattr(model, 'model'):
            print("✅ Model has 'model' attribute")
            
            if hasattr(model.model, 'model'):
                print("✅ Model has nested 'model' attribute")
                
                # Check layer structure
                model_layers = model.model.model
                if hasattr(model_layers, '__iter__'):
                    layer_count = 0
                    conv_layers = 0
                    bn_issues = 0
                    
                    for i, layer in enumerate(model_layers):
                        layer_count += 1
                        layer_type = type(layer).__name__
                        
                        if 'Conv' in layer_type:
                            conv_layers += 1
                            # Check for bn attribute
                            if hasattr(layer, 'bn'):
                                print(f"  ✅ Conv layer {i}: has 'bn' attribute")
                            else:
                                bn_issues += 1
                                print(f"  ❌ Conv layer {i}: missing 'bn' attribute")
                        
                        if layer_count > 20:  # Limit output
                            print(f"  ... (showing first 20 layers)")
                            break
                    
                    print(f"\n📊 Model Analysis:")
                    print(f"  - Total layers checked: {min(layer_count, 20)}")
                    print(f"  - Conv layers: {conv_layers}")
                    print(f"  - BN issues: {bn_issues}")
                    
                    if bn_issues > 0:
                        print(f"\n❌ COMPATIBILITY ISSUE DETECTED!")
                        print(f"   {bn_issues} Conv layers are missing 'bn' attribute")
                        print(f"   This suggests the model was trained with a different Ultralytics version")
                        return False
                    else:
                        print(f"\n✅ Model appears compatible")
                        return True
                else:
                    print("⚠️  Model layers are not iterable")
                    return False
            else:
                print("❌ Model missing nested 'model' attribute")
                return False
        else:
            print("❌ Model missing 'model' attribute")
            return False
            
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

def main():
    """Main function to check model compatibility."""
    
    # Default model path from config
    default_model_path = project_root / "weights" / "word_detection.pt"
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = str(default_model_path)
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print(f"💡 Usage: python {sys.argv[0]} [model_path]")
        sys.exit(1)
    
    print("🚀 YOLO Model Compatibility Checker")
    print("=" * 50)
    
    is_compatible = check_model_compatibility(model_path)
    
    print("\n" + "=" * 50)
    if is_compatible:
        print("🎉 Model is compatible with current Ultralytics version!")
    else:
        print("⚠️  Model compatibility issues detected!")
        print("\n💡 Solutions:")
        print("   1. Retrain the model with current Ultralytics version")
        print("   2. Use a compatible model weights file")
        print("   3. Downgrade Ultralytics to match model version")
        print("   4. Check if model file is corrupted")
    
    sys.exit(0 if is_compatible else 1)

if __name__ == "__main__":
    main()
