from typing import List, Tuple
import os
import sys
import numpy as np
import structlog
import re

# Add project root to path to access ppocr module
__dir__ = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(__dir__, "..", ".."))
sys.path.insert(0, project_root)

os.environ["FLAGS_allocator_strategy"] = "auto_growth"

import paddle
from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model

# --- CHANGE: Updated import path for batchify from the new common module ---
from app.utils.common import batchify

# --- CHANGE: Use structlog for structured logging ---
logger = structlog.get_logger(__name__)


class RecognitionService:
    """
    A specialized service for recognizing text from cropped word images using PaddleOCR.
    This service has no knowledge of debug modes or how to visualize results.
    """
    def __init__(self, config: dict):
        self.device = config['device']
        self.batch_size = config['batch_size']
        self.min_conf = config['min_conf']

        # Get paddle_rec configuration
        paddle_config = config.get('paddle_rec', {})
        self.config_path = paddle_config.get('config_path')
        self.model_path = paddle_config.get('model_path')
        use_gpu = paddle_config.get('use_gpu', True)

        # Set device
        if use_gpu and paddle.is_compiled_with_cuda():
            paddle.set_device('gpu:0')
            logger.debug("recognition_service.device_set", device="gpu:0")
        else:
            paddle.set_device('cpu')
            logger.debug("recognition_service.device_set", device="cpu")

        # Load config
        logger.debug("recognition_service.loading_config", config_path=self.config_path)
        import yaml
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Update config with model path
        self.config['Global']['pretrained_model'] = self.model_path
        self.config['Global']['infer_mode'] = True

        # Build post process
        logger.debug("recognition_service.building_post_process")
        self.post_process_class = build_post_process(
            self.config['PostProcess'],
            self.config['Global']
        )

        # Build model
        logger.debug("recognition_service.building_model")
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
        logger.debug("recognition_service.creating_data_ops")
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

        # Warmup: force model to fully load by running a dummy prediction
        logger.debug("recognition_service.warmup_start")
        try:
            dummy_img = np.zeros((32, 100, 3), dtype=np.uint8)
            _ = self._predict_single(dummy_img)
            logger.debug("recognition_service.warmup_complete")
        except Exception as e:
            logger.warning("recognition_service.warmup_failed", error=str(e))

        logger.debug("recognition_service.initialized")

    def _predict_single(self, img: np.ndarray) -> Tuple[str, float]:
        """
        Predict text from a single image crop.

        Args:
            img: NumPy array of the image (H, W, C)

        Returns:
            Tuple of (text, confidence)
        """
        try:
            # Convert to bytes for PaddleOCR pipeline
            import cv2
            _, img_bytes = cv2.imencode('.jpg', img)
            img_bytes = img_bytes.tobytes()

            data = {'image': img_bytes}
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
            logger.warning("recognition_service.prediction_failed", error=str(e))
            return ("", 0.0)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocesses a single image.
        For PaddleOCR, we just return the image as-is since preprocessing
        is handled by the ops pipeline.
        """
        return img

    def __call__(self, crops: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Main entry point for the service. Takes a list of word images
        and returns a list of (text, confidence_score) tuples.
        """
        if not crops:
            return []

        all_results = []

        # Process each crop individually
        # Note: PaddleOCR's recognition model processes one image at a time
        # in the current implementation
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

        return all_results