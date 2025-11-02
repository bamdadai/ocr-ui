from typing import List, Tuple
import os
import sys
import numpy as np
import structlog
import re
import tempfile
import cv2

# Add project root to path to access ppocr module
__dir__ = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(__dir__, "..", ".."))
sys.path.insert(0, project_root)

os.environ["FLAGS_allocator_strategy"] = "auto_growth"

try:
    from paddleocr import TextRecognition
    TEXT_RECOGNITION_AVAILABLE = True
except ImportError:
    TEXT_RECOGNITION_AVAILABLE = False

# Low-level PaddleOCR API (fallback for custom models)
import paddle
from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model

# --- CHANGE: Updated import path for batchify from the new common module ---
from app.utils.common import batchify

# --- CHANGE: Use structlog for structured logging ---
logger = structlog.get_logger(__name__)


def is_persian_digit(char):
    """Check if character is a Persian/Arabic digit (۰-۹ or 0-9)."""
    return char in '۰۱۲۳۴۵۶۷۸۹0123456789'


def is_persian_arabic_char(char):
    """Check if character is Persian/Arabic letter."""
    # Persian/Arabic Unicode ranges
    return ('\u0600' <= char <= '\u06FF' or  # Arabic
            '\u0750' <= char <= '\u077F' or  # Arabic Supplement
            '\uFB50' <= char <= '\uFDFF' or  # Arabic Presentation Forms-A
            '\uFE70' <= char <= '\uFEFF')    # Arabic Presentation Forms-B


def reverse_rtl_with_numbers(text):
    """
    Reverse RTL text while preserving number sequences in their original order.
    For example:
    - Input (from model):  "1.234.567 نارهت"
    - After full reverse:  "تهران 765.432.1"  (WRONG - numbers reversed)
    - Correct output:      "تهران 1.234.567"  (numbers preserved)

    Args:
        text: Input text from OCR model

    Returns:
        Text with RTL portions reversed but numbers preserved
    """
    if not text:
        return text

    # Check if text contains English letters (then don't reverse at all)
    has_english = any(c.isascii() and c.isalpha() for c in text)
    if has_english:
        return text

    # Tokenize text into segments: numbers (with separators) and non-numbers
    # Pattern to match: sequences of digits and common number separators (.,،٫٬/-%)
    pattern = r'[۰-۹0-9.,،٫٬/\-٪%]+'
    segments = []
    last_end = 0

    for match in re.finditer(pattern, text):
        # Add text before this number
        if match.start() > last_end:
            segments.append(('text', text[last_end:match.start()]))
        # Add the number
        segments.append(('number', match.group()))
        last_end = match.end()

    # Add remaining text
    if last_end < len(text):
        segments.append(('text', text[last_end:]))

    # Reverse the order of segments, but keep number content as-is
    reversed_segments = []
    for seg_type, seg_value in reversed(segments):
        if seg_type == 'number':
            # Keep numbers in original order (don't reverse the digits)
            reversed_segments.append(seg_value)
        else:
            # Reverse the text portion
            reversed_segments.append(seg_value[::-1])

    return ''.join(reversed_segments)


class RecognitionService:
    """
    A specialized service for recognizing text from cropped word images using PaddleOCR TextRecognition API.
    This service has no knowledge of debug modes or how to visualize results.
    """
    def __init__(self, config: dict):
        self.device = config['device']
        self.batch_size = config['batch_size']
        self.min_conf = config['min_conf']

        # Get paddle_rec configuration - three required parameters
        paddle_config = config.get('paddle_rec', {})
        
        # 1. model_dir - directory containing inference files
        model_dir = paddle_config.get('model_dir')
        if not model_dir:
            # Fallback: derive from model_path if provided
            self.model_path = paddle_config.get('model_path')
            if self.model_path:
                # Resolve path relative to project root if needed
                potential_dir = os.path.join(project_root, self.model_path) if not os.path.isabs(self.model_path) else self.model_path
                # Check if it's a directory or file
                if os.path.isdir(potential_dir):
                    model_dir = self.model_path
                elif os.path.isfile(potential_dir):
                    model_dir = os.path.dirname(self.model_path)
                else:
                    # Path doesn't exist yet, assume it's a directory
                    model_dir = self.model_path
            else:
                # Fallback: derive from config_path
                self.config_path = paddle_config.get('config_path')
                if self.config_path:
                    model_dir = os.path.dirname(self.config_path)
        
        # 2. model_name - default is arabic_PP-OCRv5_mobile_rec
        model_name = paddle_config.get('model_name', 'arabic_PP-OCRv5_mobile_rec')
        
        # 3. use_gpu - boolean flag
        use_gpu = paddle_config.get('use_gpu', True)
        
        # Store config_path for reference
        self.config_path = paddle_config.get('config_path')

        # Convert model_dir to absolute path if relative
        if model_dir and not os.path.isabs(model_dir):
            model_dir = os.path.join(project_root, model_dir)
        
        # Verify model_dir exists and contains inference files
        if model_dir:
            abs_model_dir = model_dir if os.path.isabs(model_dir) else os.path.join(project_root, model_dir)
            inference_files = ['inference.yml', 'inference.pdiparams']
            missing_files = []
            for inf_file in inference_files:
                inf_path = os.path.join(abs_model_dir, inf_file)
                if not os.path.exists(inf_path):
                    missing_files.append(inf_file)
            if missing_files:
                logger.warning(
                    "recognition_service.missing_inference_files",
                    model_dir=model_dir,
                    missing=missing_files
                )
            else:
                logger.debug(
                    "recognition_service.inference_files_verified",
                    model_dir=model_dir
                )

        # Log that we're using PaddleOCR TextRecognition API with three configs
        logger.info(
            "recognition_service.using_textrecognition_api",
            model_dir=model_dir,
            model_name=model_name,
            use_gpu=use_gpu,
            config_path=self.config_path
        )

        # Initialize TextRecognition model with three required parameters
        device_str = "gpu:0" if use_gpu else "cpu"
        try:
            # Always pass model_name, model_dir, and device
            self.model = TextRecognition(
                model_name=model_name,
                model_dir=model_dir,
                device=device_str
            )
            logger.debug(
                "recognition_service.model_loaded",
                device=device_str,
                model_name=model_name,
                model_dir=model_dir
            )
        except Exception as e:
            error_log = {
                "error": str(e),
                "model_dir": model_dir,
                "model_name": model_name,
                "use_gpu": use_gpu
            }
            logger.error("recognition_service.model_load_failed", **error_log)
            raise

        # Warmup: force model to fully load by running a dummy prediction
        logger.debug("recognition_service.warmup_start")
        try:
            dummy_img = np.zeros((32, 100, 3), dtype=np.uint8)
            _ = self._predict_single(dummy_img)
            logger.debug("recognition_service.warmup_complete")
        except Exception as e:
            logger.warning("recognition_service.warmup_failed", error=str(e))

        logger.debug("recognition_service.initialized")
        
        # Track which input method works (set on first successful prediction)
        self._input_method_used = None

    def _predict_single(self, img: np.ndarray) -> Tuple[str, float]:
        """
        Predict text from a single image crop.

        Args:
            img: NumPy array of the image (H, W, C)

        Returns:
            Tuple of (text, confidence)
        """
        try:
            # Ensure image is in BGR format (OpenCV default)
            if len(img.shape) == 2:
                # Grayscale to BGR
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                # RGBA to BGR
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif img.shape[2] == 3:
                # Assume it's already BGR, but ensure it's uint8
                img_bgr = img.copy()
                if img_bgr.dtype != np.uint8:
                    img_bgr = img_bgr.astype(np.uint8)
            else:
                img_bgr = img.copy()

            # Try passing numpy array directly first (most efficient)
            input_method = None
            try:
                output = self.model.predict(input=img_bgr, batch_size=1)
                input_method = "numpy_array"
                logger.debug("recognition_service.input_method_success", method="numpy_array")
            except (TypeError, AttributeError) as e:
                logger.debug(
                    "recognition_service.input_method_failed",
                    method="numpy_array",
                    error_type=type(e).__name__,
                    error=str(e)
                )
                # If numpy array doesn't work, try encoding to bytes
                try:
                    # Encode image to PNG bytes (lossless)
                    _, img_bytes = cv2.imencode('.png', img_bgr)
                    img_bytes = img_bytes.tobytes()
                    output = self.model.predict(input=img_bytes, batch_size=1)
                    input_method = "png_bytes"
                    logger.debug("recognition_service.input_method_success", method="png_bytes")
                except (TypeError, AttributeError) as e2:
                    logger.debug(
                        "recognition_service.input_method_failed",
                        method="png_bytes",
                        error_type=type(e2).__name__,
                        error=str(e2)
                    )
                    # Fallback: save to temporary file (last resort)
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                        tmp_path = tmp_file.name
                        cv2.imwrite(tmp_path, img_bgr)
                    try:
                        output = self.model.predict(input=tmp_path, batch_size=1)
                        input_method = "temp_file"
                        logger.debug(
                            "recognition_service.input_method_success",
                            method="temp_file",
                            temp_path=tmp_path
                        )
                    finally:
                        # Clean up temporary file
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass
            
            # Log which method was used for this prediction
            if input_method:
                # Log at info level on first use to make it visible
                if self._input_method_used is None:
                    self._input_method_used = input_method
                    logger.info(
                        "recognition_service.input_method_determined",
                        method=input_method,
                        note="This method will be used for all subsequent predictions"
                    )
                else:
                    logger.debug(
                        "recognition_service.prediction_input_method",
                        method=input_method
                    )

            # Extract text and confidence from result
            text = ""
            confidence = 0.0

            for res in output:
                # Extract text and confidence from result
                if hasattr(res, 'rec_text'):
                    text = res.rec_text
                    confidence = res.rec_score if hasattr(res, 'rec_score') else 0.0
                else:
                    # Try to get from dict representation
                    if isinstance(res, dict):
                        text = res.get('rec_text', '')
                        confidence = res.get('rec_score', 0.0)
                    else:
                        # Fallback: try to extract from any iterable structure
                        if hasattr(res, '__iter__') and not isinstance(res, str):
                            res_list = list(res)
                            if len(res_list) >= 2:
                                text = str(res_list[0])
                                confidence = float(res_list[1])
                break  # Only process first result

            # Reverse RTL text while preserving numbers
            text = reverse_rtl_with_numbers(text)
            
            # Fix Persian parentheses: replace )( with ()
            # Browsers don't always handle bidi correctly, so we fix it at the source
            from app.utils.text_processing import fix_persian_parentheses
            text = fix_persian_parentheses(text)

            return (text, float(confidence))

        except Exception as e:
            logger.warning("recognition_service.prediction_failed", error=str(e))
            return ("", 0.0)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocesses a single image.
        For TextRecognition API, we just return the image as-is.
        """
        return img

    def __call__(self, crops: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Main entry point for the service. Takes a list of word images
        and returns a list of (text, confidence_score) tuples.
        """
        if not crops:
            return []

        logger.info(
            "recognition_service.textrecognition_api_start",
            num_crops=len(crops),
            device=self.device
        )

        all_results = []

        # Process each crop individually
        # Note: TextRecognition API processes one image at a time in the current implementation
        for idx, crop in enumerate(crops):
            if idx % self.batch_size == 0:
                logger.debug(
                    "recognition_service.processing_batch",
                    batch_number=(idx // self.batch_size) + 1,
                    progress=f"{idx}/{len(crops)}"
                )

            text, confidence = self._predict_single(crop)

            # If confidence is below the threshold, discard the label
            if confidence < self.min_conf:
                text = ""

            all_results.append((text, confidence))

        logger.info(
            "recognition_service.textrecognition_api_complete",
            num_crops=len(crops),
            num_results=len(all_results),
            input_method=self._input_method_used
        )

        return all_results
