import uuid
import cv2
import numpy as np
import torch
import structlog
import time
import os
from typing import List, Callable, Dict, Any, Literal
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from ultralytics import YOLO
from ultralytics.engine.results import Results
from pathlib import Path

# --- Centralized imports from utility modules ---
from app.utils.visualization import draw_polygons, draw_boxes
from app.utils.image_processing import get_polygons_from_masks, merge_overlapping_masks
from app.utils.common import get_current_memory_usage_mb

logger = structlog.get_logger(__name__)
ModelType = Literal['word', 'line']

class DetectionService:
    """
    Manages object detection models and orchestrates prediction logic using
    clean, explicit function signatures and structured logging.
    """
    def __init__(self, config: Dict[str, Any]):
        self.device = config['device']
        self.debug = config['debug']

        self.models = {
            'word': self._load_yolo_model(config['word_detect']['path']),
            'line': self._load_yolo_model(config['line_detect']['path'])
        }
        self.model_params = {
            'word': {'iou': config['word_detect']['iou'], 'conf': config['word_detect']['conf'], 'task': 'segment', 'retina_masks': True},
            'line': {'iou': config['line_detect']['iou'], 'conf': config['line_detect']['conf']}
        }
        self.post_process_funcs = {
            'word': self._post_process_word_results,
            'line': self._post_process_line_results
        }
        self.debug_info = {
            'word': {'path': config['debug_word_path'], 'draw_func': lambda im, res: draw_polygons(im, res, color=(255, 0, 0))},
            'line': {'path': config['debug_line_path'], 'draw_func': lambda im, res: draw_boxes(im, res, color=(0, 0, 255))}
        }
        self.merging_iou = config['word_detect']['merging_iou']

        parallel_config = config['parallel_processing']
        self.parallel_enabled = parallel_config['enabled']
        self.max_workers = parallel_config['max_workers']
        self.max_batch_size = parallel_config['max_batch_size']
        self.memory_limit_mb = parallel_config['memory_limit_mb']

        logger.info("detection_service.initialized")

    def _load_yolo_model(self, model_path: str) -> YOLO:
        """
        Loads, validates, and prepares a YOLO model for thread-safe inference.
        """
        model_path_obj = Path(model_path)
        if not model_path_obj.is_file():
            logger.error("detection_service.model_not_found", path=model_path)
            raise FileNotFoundError(f"YOLO model file not found at: {model_path}")

        model = YOLO(model_path)
        
        if getattr(model, 'model', None) is None:
            logger.error("detection_service.model_load_failed", path=model_path)
            raise ValueError(f"YOLO model loading failed for path: {model_path}. File might be corrupt.")

        if not str(model_path).endswith('.onnx'):
            model.to(self.device)
            model.fuse()
            
        model.eval()
        logger.info("detection_service.model_loaded", path=model_path, device=self.device)
        return model

    def predict_word_polygons(self, images: List[np.ndarray]) -> Dict[str, List]:
        logger.info("detection_service.predicting", model_type="word", image_count=len(images))
        return {'word_polygons': self._predict(images, model_type='word')}

    def predict_line_boxes(self, images: List[np.ndarray]) -> Dict[str, List]:
        logger.info("detection_service.predicting", model_type="line", image_count=len(images))
        return {'line_boxes': self._predict(images, model_type='line')}

    def _predict(self, images: List[np.ndarray], model_type: ModelType) -> List:
        """A generic prediction orchestrator."""
        processor_function = lambda img: self._execute_single_image_prediction(
            image=img,
            model=self.models[model_type],
            model_params=self.model_params[model_type],
            post_process_func=self.post_process_funcs[model_type],
            debug_path=self.debug_info[model_type]['path'],
            debug_draw_func=self.debug_info[model_type]['draw_func']
        )
        return self._process_in_optimal_batches(images, processor_function)

    def _execute_single_image_prediction(
        self, 
        image: np.ndarray, 
        model: YOLO, 
        model_params: Dict, 
        post_process_func: Callable,
        debug_path: str,
        debug_draw_func: Callable
    ) -> List:
        """
        Executes prediction on a single image with explicit parameters for clarity.
        """
        with torch.no_grad():
            results = model([image], verbose=False, device=self.device, batch=1, **model_params)[0]
            results = results[results.boxes.cls == 0]
            processed_output = post_process_func(results)
            
            if self.debug and processed_output:
                debug_image = debug_draw_func(image.copy(), processed_output)
                # Ensure debug path exists before writing
                Path(debug_path).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(os.path.join(debug_path, f'{uuid.uuid4()}.jpg'), debug_image)
                
        return processed_output

    def _post_process_word_results(self, word_result: Results) -> List[List[int]]:
        """Post-processes word detection results."""
        if word_result.masks is None: return []
        orig_h, orig_w = word_result.orig_shape
        cpu_masks = word_result.masks.data.cpu().numpy()
        resized_masks = [cv2.resize(m.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST) for m in cpu_masks]
        merged_masks = merge_overlapping_masks(resized_masks, dice_threshold=self.merging_iou)
        return get_polygons_from_masks(merged_masks)

    def _post_process_line_results(self, line_result: Results) -> List[List[float]]:
        """Post-processes line detection results."""
        if line_result.boxes is None: return []
        return [box.xyxy[0].cpu().numpy().tolist() for box in line_result.boxes]

    def _process_in_optimal_batches(self, images: List[np.ndarray], processor_function: Callable) -> List:
        """Processes images in dynamically adjusted batches based on memory."""
        if not self.parallel_enabled or len(images) <= 1:
            return self._process_batch_sequentially(images, processor_function)
        
        results, batch_size = [], min(self.max_batch_size, len(images))
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            logger.info("detection_service.batch.processing", batch_size=len(batch))
            results.extend(self._process_batch_parallel(batch, processor_function))
            
            current_memory = get_current_memory_usage_mb()
            if current_memory > self.memory_limit_mb * 0.8:
                batch_size = max(1, batch_size // 2)
                logger.warning("detection_service.batch.memory_high", new_batch_size=batch_size, memory_mb=current_memory)
            elif current_memory < self.memory_limit_mb * 0.4 and batch_size < self.max_batch_size:
                batch_size = min(self.max_batch_size, batch_size * 2)
                logger.info("detection_service.batch.memory_low", new_batch_size=batch_size, memory_mb=current_memory)
        return results

    def _process_batch_parallel(self, images: List[np.ndarray], processor_function: Callable) -> List:
        """Processes a batch of images in parallel."""
        temp_results = [None] * len(images)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {executor.submit(processor_function, img): i for i, img in enumerate(images)}
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    temp_results[idx] = future.result()
                except Exception as e:
                    logger.error("detection_service.batch.error", image_index=idx, error=str(e), exc_info=True)
                    temp_results[idx] = []
        return temp_results

    def _process_batch_sequentially(self, images: List[np.ndarray], processor_function: Callable) -> List:
        """Processes a batch of images sequentially."""
        results = []
        for i, img in enumerate(images):
            try:
                results.append(processor_function(img))
            except Exception as e:
                logger.error("detection_service.batch.error", image_index=i, error=str(e), exc_info=True)
                results.append([])
        return results
