#!/usr/bin/env python3
"""
OCR Inference Module for Arabic Text Recognition
Returns text and confidence scores from images.

Usage:
    from ocr_inference_module import ArabicOCR

    ocr = ArabicOCR()
    text, confidence = ocr.predict("image.jpg")
"""

import os
import sys
import numpy as np
from typing import Tuple, Optional

# Add PaddleOCR to path
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, ".")))

os.environ["FLAGS_allocator_strategy"] = "auto_growth"

import paddle
from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model


class ArabicOCR:
    """Arabic OCR text recognition model."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        model_path: Optional[str] = None,
        use_gpu: bool = True
    ):
        """
        Initialize the Arabic OCR model.

        Args:
            config_path: Path to model config YAML file. If None, uses default.
            model_path: Path to model weights (without extension). If None, uses best_accuracy.
            use_gpu: Whether to use GPU for inference.
        """
        # Set default paths
        if config_path is None:
            config_path = os.path.join(
                __dir__,
                "configs/rec/PP-OCRv5/multi_language/arabic_PP-OCRv5_mobile_rec.yaml"
            )

        if model_path is None:
            model_path = os.path.join(
                __dir__,
                "output/arabic_rec_ppocr_v5_real_naft/best_accuracy"
            )

        self.config_path = config_path
        self.model_path = model_path

        # Set device
        if use_gpu and paddle.is_compiled_with_cuda():
            paddle.set_device('gpu:0')
        else:
            paddle.set_device('cpu')

        # Load config
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Update config with model path
        self.config['Global']['pretrained_model'] = model_path
        self.config['Global']['infer_mode'] = True

        # Build post process
        self.post_process_class = build_post_process(
            self.config['PostProcess'],
            self.config['Global']
        )

        # Build model
        if hasattr(self.post_process_class, 'character'):
            char_num = len(getattr(self.post_process_class, 'character'))
            if self.config['Architecture']['Head']['name'] == 'MultiHead':
                out_channels_list = {}
                out_channels_list['CTCLabelDecode'] = char_num
                out_channels_list['SARLabelDecode'] = char_num + 2
                out_channels_list['NRTRLabelDecode'] = char_num + 3
                self.config['Architecture']['Head']['out_channels_list'] = out_channels_list
            else:
                self.config['Architecture']['Head']['out_channels'] = char_num

        self.model = build_model(self.config['Architecture'])
        load_model(self.config, self.model)
        self.model.eval()

        # Create data ops
        transforms = []
        for op in self.config['Eval']['dataset']['transforms']:
            op_name = list(op)[0]
            if 'Label' in op_name:
                continue
            elif op_name in ['RecResizeImg']:
                op[op_name]['infer_mode'] = True
            elif op_name == 'KeepKeys':
                op[op_name]['keep_keys'] = ['image']
            transforms.append(op)

        self.ops = create_operators(transforms, self.config['Global'])

    def predict(self, image_path: str) -> Tuple[str, float]:
        """
        Perform OCR on an image and return text with confidence score.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (text, confidence) where:
                - text (str): Recognized text from the image
                - confidence (float): Confidence score between 0 and 1

        Raises:
            FileNotFoundError: If image file does not exist
            Exception: If inference fails
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            # Load and preprocess image
            with open(image_path, 'rb') as f:
                img = f.read()
                data = {'image': img}

            batch = transform(data, self.ops)
            images = np.expand_dims(batch[0], axis=0)
            images = paddle.to_tensor(images)

            # Run inference
            preds = self.model(images)
            post_result = self.post_process_class(preds)

            # Extract text and confidence
            text = ""
            confidence = 0.0

            if isinstance(post_result, dict):
                for key in post_result:
                    if len(post_result[key][0]) >= 2:
                        text = post_result[key][0][0]
                        confidence = float(post_result[key][0][1])
                        break
            else:
                if len(post_result[0]) >= 2:
                    text = post_result[0][0]
                    confidence = float(post_result[0][1])

            # Reverse text for correct RTL display (Arabic/Persian)
            # PaddleOCR outputs text in visual LTR order, so we reverse
            # the entire string to get the correct RTL logical order
            text = text[::-1]

            return (text, confidence)

        except Exception as e:
            raise Exception(f"OCR inference failed for {image_path}: {str(e)}")

    def predict_batch(self, image_paths: list) -> list:
        """
        Perform OCR on multiple images.

        Args:
            image_paths: List of paths to image files

        Returns:
            List of tuples (text, confidence) for each image
        """
        results = []
        for img_path in image_paths:
            try:
                result = self.predict(img_path)
                results.append(result)
            except Exception as e:
                print(f"Warning: Failed to process {img_path}: {e}")
                results.append(("", 0.0))
        return results


def main():
    """Test the OCR module with sample images."""
    import argparse

    parser = argparse.ArgumentParser(description='Arabic OCR Inference')
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to the image file'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config YAML file (optional)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to model weights (optional)'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Use CPU instead of GPU'
    )

    args = parser.parse_args()

    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"ERROR: Image not found: {args.image_path}")
        sys.exit(1)

    print("=" * 70)
    print("Arabic OCR - Text Recognition")
    print("=" * 70)
    print(f"Image:  {args.image_path}")
    print(f"Model:  {args.model or 'output/arabic_rec_ppocr_v5_real_naft/best_accuracy'}")
    print(f"Config: {args.config or 'configs/rec/PP-OCRv5/multi_language/arabic_PP-OCRv5_mobile_rec.yaml'}")
    print(f"Device: {'CPU' if args.cpu else 'GPU (if available)'}")
    print("=" * 70)
    print()

    # Initialize OCR model
    print("Loading model...")
    ocr = ArabicOCR(
        config_path=args.config,
        model_path=args.model,
        use_gpu=not args.cpu
    )
    print("Model loaded successfully!")
    print()

    # Perform inference
    print("Running inference...")
    try:
        text, confidence = ocr.predict(args.image_path)

        print("=" * 70)
        print("RESULT:")
        print("=" * 70)
        print(f"Text:       {text}")
        print(f"Confidence: {confidence:.6f} ({confidence*100:.2f}%)")
        print("=" * 70)

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
