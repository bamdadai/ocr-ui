import uuid
import cv2
import numpy as np
import torch
import structlog
import time
import os
import functools
from typing import List, Callable, Dict, Any, Literal, Optional, Sequence, Tuple, Iterable
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from ultralytics import YOLO
from ultralytics.engine.results import Results
from pathlib import Path
from torchvision.ops import nms
from paddleocr import TextDetection

# --- Centralized imports from utility modules ---
from app.utils.visualization import draw_polygons, draw_boxes
from app.utils.image_processing import get_polygons_from_masks
from app.utils.common import get_current_memory_usage_mb

logger = structlog.get_logger(__name__)
ModelType = Literal['word', 'line']


def time_method(func: Callable) -> Callable:
    """
    Decorator to measure and log execution time of methods.

    Best practices:
    - Uses time.perf_counter() for high-precision wall-clock timing
    - Logs method entry, execution time, and exit
    - Handles exceptions without interfering with error handling
    - Includes method name and basic parameter info in logs
    - Minimal performance overhead when disabled
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Check if timing is enabled (configurable)
        if not getattr(self, 'timing_enabled', True):
            return func(self, *args, **kwargs)

        method_name = f"{self.__class__.__name__}.{func.__name__}"
        start_time = time.perf_counter()

        # Log method entry with parameter count (avoid logging large objects)
        arg_count = len(args) + len(kwargs)
        logger.debug("method.entry", method=method_name, arg_count=arg_count)

        try:
            result = func(self, *args, **kwargs)
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            # Log successful completion with timing
            logger.debug("method.timing",
                        method=method_name,
                        duration_ms=round(duration_ms, 2),
                        success=True)

            return result

        except Exception as e:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            # Log exception with timing
            logger.warning("method.timing",
                          method=method_name,
                          duration_ms=round(duration_ms, 2),
                          success=False,
                          error=str(e))

            # Re-raise the exception
            raise

    return wrapper

class DetectionService:
    """
    Manages object detection models and orchestrates prediction logic using
    clean, explicit function signatures and structured logging.
    """
    def __init__(self, config: Dict[str, Any]):
        self.device = config['device']
        self.debug = config['debug']
        self.timing_enabled = config.get('timing_enabled', True)  # Enable method timing by default
        logger.debug("detection_service.init", debug=self.debug, debug_word_path=config.get('debug_word_path'), debug_line_path=config.get('debug_line_path'))

        # Initialize YOLO models for word detection
        self.models = {
            'word': self._load_yolo_model(config['word_detect']['path']),
        }

        # Initialize PaddleOCR for line detection
        self.paddle_model = self._load_paddle_model(config.get('paddle_ocr', {}))

        # Warmup: force model to fully load by running a dummy prediction
        logger.debug("detection_service.paddle_warmup_start")
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        try:
            _ = list(self.paddle_model.predict(dummy_img, batch_size=1))
            logger.debug("detection_service.paddle_warmup_complete")
        except Exception as e:
            logger.warning("detection_service.paddle_warmup_failed", error=str(e))
        self.model_params = {
            'word': {
                'iou': config['word_detect']['iou'],
                'conf': config['word_detect']['conf'],
                'nms': config['word_detect'].get('nms', True),
                'task': 'segment',
                'retina_masks': True
            }
        }
        # Optional target classes for filtering; can be int ids or class names
        # If omitted, we do NOT filter by class to avoid dropping valid detections
        self.target_classes: Dict[ModelType, Optional[Sequence[int | str]]] = {
            'word': config['word_detect'].get('classes'),
        }

        # PaddleOCR configuration
        paddle_config = config.get('paddle_ocr', {})
        self.reading_direction: Literal["ltr", "rtl"] = paddle_config.get('reading_direction', 'rtl')
        self.paddle_pad: int = paddle_config.get('pad', 2)
        self.paddle_sort_reading_order: bool = paddle_config.get('sort_reading_order', True)
        self.post_process_funcs = {
            'word': self._post_process_word_results,
        }
        self.debug_info = {
            'word': {'path': config['debug_word_path'], 'draw_func': lambda im, res: draw_polygons(im, res, color=(255, 0, 0))},
            'line': {'path': config['debug_line_path'], 'draw_func': lambda im, res: draw_boxes(im, res, color=(0, 0, 255))}
        }
        self.remove_nested_boxes = config['word_detect'].get('remove_nested', True)

        parallel_config = config['parallel_processing']
        self.parallel_enabled = parallel_config['enabled']
        self.max_workers = parallel_config['max_workers']
        self.max_batch_size = parallel_config['max_batch_size']
        self.memory_limit_mb = parallel_config['memory_limit_mb']


        # --- Word detection configuration ---
        word_cfg = config.get('word_detect', {})

        logger.debug("detection_service.initialized")

    @time_method
    def _load_yolo_model(self, model_path: str) -> YOLO:
        """Load YOLO model with compatibility checks and error handling."""
        try:
            logger.debug("detection_service.yolo_model_loading", model_path=model_path, device=self.device)

            # Load model with compatibility mode
            model = YOLO(model_path)

            # Check model compatibility
            if hasattr(model.model, 'model'):
                logger.debug("detection_service.yolo_model_loaded",
                            model_path=model_path,
                            device=self.device,
                            model_type=type(model.model).__name__)
            else:
                logger.warning("detection_service.yolo_model_structure_unexpected", 
                              model_path=model_path)
            
            return model.to(self.device)
            
        except Exception as e:
            logger.error("detection_service.yolo_model_load_failed", 
                        model_path=model_path, 
                        device=self.device,
                        error=str(e), 
                        exc_info=True)
            raise RuntimeError(f"Failed to load YOLO model from {model_path}: {str(e)}")

    @time_method
    def _load_paddle_model(self, paddle_config: Dict[str, Any]) -> TextDetection:
        """Load PaddleOCR TextDetection model with configuration."""
        model_name = paddle_config.get('model_name', 'PP-OCRv5_server_det')
        model_kwargs = paddle_config.get('model_kwargs', {
        "model_dir": "/app/weights/PP-OCRv5_server_det_infer"})
        logger.debug("detection_service.loading_paddle_model", model_kwargs=model_kwargs)
        return TextDetection(**model_kwargs)

    @time_method
    def predict_word_polygons(self, images: List[np.ndarray]) -> Dict[str, List]:
        logger.debug("detection_service.predicting", model_type="word", image_count=len(images))
        return {'word_polygons': self._predict(images, model_type='word')}

    @time_method
    def predict_line_boxes(self, images: List[np.ndarray]) -> Dict[str, List]:
        logger.debug("detection_service.predicting", model_type="line", image_count=len(images))
        # Use PaddleOCR for line detection
        processor_function = lambda img: self._predict_lines_paddle(img)
        results = self._process_in_optimal_batches(images, processor_function)
        return {'line_boxes': results}

    @time_method
    def _predict_lines_paddle(self, image: np.ndarray) -> List[List[float]]:
        """Predict line boxes using PaddleOCR TextDetection."""
        if image is None:
            return []

        # Ensure image is in BGR format as expected by PaddleOCR
        img = self._ensure_bgr(image)
        H, W = img.shape[:2]

        # Run PaddleOCR detection
        output = self.paddle_model.predict(img, batch_size=1)
        if not isinstance(output, Iterable) or len(output) == 0:
            logger.warning("detection_service.paddle_no_results")
            return []

        res = list(output)[0]
        polys = self._extract_polys_from_result_obj(res)

        # Clean polygons: ensure shape (4,2)
        cleaned_polys = []
        for p in polys:
            p = np.array(p, dtype=np.float32).reshape(-1, 2)
            if p.shape[0] >= 4:
                if p.shape[0] > 4:
                    rect = cv2.minAreaRect(p.astype(np.float32))
                    box = cv2.boxPoints(rect)
                    p = box.astype(np.float32)
                cleaned_polys.append(p[:4])

        # Sort by reading order if enabled
        indices = list(range(len(cleaned_polys)))
        if self.paddle_sort_reading_order and len(cleaned_polys) > 1:
            indices = self._sort_reading_order(cleaned_polys, self.reading_direction)

        # Convert polygons to bounding boxes
        line_boxes = []
        for idx in indices:
            poly = cleaned_polys[idx]
            x1, y1, x2, y2 = self._poly_to_bbox(poly, self.paddle_pad, W, H)
            line_boxes.append([float(x1), float(y1), float(x2), float(y2)])

        # Debug visualization
        if self.debug and line_boxes:
            try:
                debug_path_str = str(self.debug_info['line']['path'])
                os.makedirs(debug_path_str, exist_ok=True)
                debug_img = image.copy()
                debug_img = draw_boxes(debug_img, line_boxes, color=(0, 0, 255))
                out_path = os.path.join(debug_path_str, f"line_paddle_{uuid.uuid4().hex[:8]}.jpg")
                cv2.imwrite(out_path, debug_img)
                logger.debug("detection_service.line_paddle_debug_saved", path=out_path)
            except Exception as e:
                logger.warning("detection_service.line_paddle_debug_save_failed", error=str(e), exc_info=True)

        return line_boxes

    @time_method
    def _predict(self, images: List[np.ndarray], model_type: ModelType) -> List:
        """A generic prediction orchestrator."""
        processor_function = lambda img: self._execute_single_image_prediction(
            image=img,
            model=self.models[model_type],
            model_params=self.model_params[model_type],
            post_process_func=self.post_process_funcs[model_type],
            debug_path=self.debug_info[model_type]['path'],
            debug_draw_func=self.debug_info[model_type]['draw_func'],
            model_type=model_type
        )
        return self._process_in_optimal_batches(images, processor_function)

    @time_method
    def _resolve_allowed_class_ids(self, results: Results, model_type: ModelType) -> Optional[List[int]]:
        """Optionally filter detections by configured classes. If not configured, return unchanged."""
        try:
            if results.boxes is None or results.boxes.cls is None:
                return None
            allowed = self.target_classes.get(model_type)
            if not allowed:
                return None
            # Map class names to IDs if necessary
            ids: List[int] = []
            for c in allowed:
                if isinstance(c, (int, np.integer)):
                    ids.append(int(c))
                else:
                    # Attempt to map class name to id via model names
                    names = self.models[model_type].names
                    if isinstance(names, dict):
                        for k, v in names.items():
                            if str(v) == str(c):
                                ids.append(int(k))
                                break
            return ids
        except Exception:
            return None

    @time_method
    def _filter_by_classes(self, results: Results, model_type: ModelType) -> Results:
        """Optionally filter detections by configured classes. If not configured, return unchanged."""
        try:
            if results.boxes is None or results.boxes.cls is None:
                return results
            allowed = self._resolve_allowed_class_ids(results, model_type)
            if not allowed:
                # No filtering configured; keep all detections
                return results
            cls_tensor = results.boxes.cls
            mask = torch.zeros_like(cls_tensor, dtype=torch.bool)
            for cid in allowed:
                mask |= (cls_tensor == cid)
            filtered = results[mask]
            logger.debug("detection_service.class_filter", model_type=model_type, kept=int(mask.sum().item()), total=len(cls_tensor))
            return filtered
        except Exception as e:
            logger.warning("detection_service.class_filter_failed", model_type=model_type, error=str(e))
            return results

    @time_method
    def _execute_single_image_prediction(
        self,
        image: np.ndarray,
        model: YOLO,
        model_params: Dict,
        post_process_func: Callable,
        debug_path: str,
        debug_draw_func: Callable,
        model_type: ModelType
    ) -> List:
        start = time.time()

        try:
            # Add compatibility check for model structure
            if not hasattr(model, 'model'):
                raise AttributeError("YOLO model structure is incompatible - missing 'model' attribute")
            
            # Check if model has the expected Conv layer structure
            if hasattr(model.model, 'model'):
                model_layers = model.model.model
                if hasattr(model_layers, '__iter__'):
                    for layer in model_layers:
                        if hasattr(layer, 'bn') and not hasattr(layer, 'bn'):
                            logger.warning("detection_service.model_layer_compatibility_issue", 
                                         layer_type=type(layer).__name__,
                                         model_type=model_type)
                            break

            results: List[Results] = model(image, **model_params)
            result = results[0]

            # Optional class filter
            filtered = self._filter_by_classes(result, model_type)
            post_processed = post_process_func(filtered)
            
        except AttributeError as e:
            if "'Conv' object has no attribute 'bn'" in str(e):
                logger.error("detection_service.model_compatibility_error", 
                           error=str(e),
                           model_type=model_type,
                           suggestion="Model was trained with different Ultralytics version")
                raise RuntimeError(f"Model compatibility error: {str(e)}. "
                                 f"Please retrain the model with current Ultralytics version or use compatible model weights.")
            else:
                raise
        except Exception as e:
            logger.error("detection_service.prediction_failed", 
                        error=str(e), 
                        model_type=model_type,
                        exc_info=True)
            raise
            
        duration = time.time() - start
        if self.debug:
            try:
                fname = f"{model_type}_debug_{uuid.uuid4().hex[:8]}.jpg"
                debug_path_str = str(self.debug_info[model_type]['path'])
                out_path = os.path.join(debug_path_str, fname)
                logger.debug("detection_service.debug_saving", model_type=model_type, path=out_path)
                os.makedirs(debug_path_str, exist_ok=True)
                debug_image = image.copy()
                debug_image = debug_draw_func(debug_image, post_processed)
                success = cv2.imwrite(out_path, debug_image)
                if success:
                    logger.debug("detection_service.debug_saved", model_type=model_type, path=out_path)
                else:
                    logger.warning("detection_service.debug_save_failed", model_type=model_type, path=out_path, reason="cv2.imwrite returned False")
            except Exception as e:
                logger.warning("detection_service.debug_save_failed", model_type=model_type, error=str(e), exc_info=True)
        logger.debug("detection_service.single_image_done", model_type=model_type, duration_ms=int(duration * 1000))
        return post_processed

    @time_method
    def _post_process_word_results(self, word_result: Results) -> List[List[int]]:
        """Post-processes word detection results.
        If segmentation masks are unavailable (e.g., using a bbox-only model),
        fall back to rectangle polygons derived from xyxy boxes.
        """
        # Preferred path: segmentation masks -> polygons
        try:
            if word_result.masks is not None:
                logger.debug("detection_service.processing_masks",
                            masks_type=type(word_result.masks).__name__,
                            has_data=hasattr(word_result.masks, 'data'))

                polygons = get_polygons_from_masks(word_result.masks)
                logger.debug("detection_service.word_polygons_extracted",
                            count=len(polygons))

                if polygons:
                    return polygons
                else:
                    logger.warning("detection_service.no_polygons_extracted_from_masks")
                    
        except Exception as e:
            logger.warning("detection_service.word_masks_postprocess_failed", 
                          error=str(e), 
                          masks_type=type(word_result.masks).__name__ if word_result.masks else None,
                          exc_info=True)
            # continue to bbox fallback
        # Fallback: use bounding boxes if masks are missing
        try:
            if word_result.boxes is None or word_result.boxes.data is None or word_result.boxes.data.numel() == 0:
                return []
            boxes_xyxy = word_result.boxes.xyxy.cpu().numpy()
            confidences = word_result.boxes.conf.cpu().numpy()
            
            # Remove nested boxes (small boxes inside large ones)
            if len(boxes_xyxy) > 1 and self.remove_nested_boxes:
                boxes_xyxy, confidences = self._remove_nested_boxes(boxes_xyxy, confidences)
            
            rect_polys: List[List[int]] = []
            for x1, y1, x2, y2 in boxes_xyxy:
                x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                poly = np.array([[x1i, y1i], [x2i, y1i], [x2i, y2i], [x1i, y2i]], dtype=np.int32)
                rect_polys.append(poly.reshape(-1).tolist())
            if rect_polys:
                logger.debug("detection_service.word_bbox_fallback", count=len(rect_polys))
            return rect_polys
        except Exception as e:
            logger.warning("detection_service.word_bbox_fallback_failed", error=str(e))
            return []

   
    
    @time_method
    def _remove_nested_boxes(self, boxes: np.ndarray, confidences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Removes small boxes that are completely inside larger boxes.
        This handles cases where NMS fails due to very different box sizes.
        """
        if len(boxes) <= 1:
            return boxes, confidences
            
        # Sort by confidence (highest first)
        sorted_indices = np.argsort(confidences)[::-1]
        boxes_sorted = boxes[sorted_indices]
        confidences_sorted = confidences[sorted_indices]
        
        keep_mask = np.ones(len(boxes), dtype=bool)
        
        for i in range(len(boxes_sorted)):
            if not keep_mask[sorted_indices[i]]:
                continue
                
            box_i = boxes_sorted[i]
            area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
            
            for j in range(i + 1, len(boxes_sorted)):
                if not keep_mask[sorted_indices[j]]:
                    continue
                    
                box_j = boxes_sorted[j]
                area_j = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])
                
                # Check if box_j is completely inside box_i
                if (box_j[0] >= box_i[0] and box_j[1] >= box_i[1] and 
                    box_j[2] <= box_i[2] and box_j[3] <= box_i[3]):
                    
                    # Remove the smaller box (box_j)
                    keep_mask[sorted_indices[j]] = False
                    logger.debug("detection_service.removed_nested_box", 
                               outer_box=box_i.tolist(), 
                               inner_box=box_j.tolist(),
                               outer_area=area_i,
                               inner_area=area_j)
        
        final_boxes = boxes[keep_mask]
        final_confidences = confidences[keep_mask]
        
        if len(final_boxes) < len(boxes):
            logger.debug("detection_service.nested_boxes_removed",
                        original_count=len(boxes),
                        final_count=len(final_boxes))
        
        return final_boxes, final_confidences



    @time_method
    def _add_padding_all_sides(self, img: np.ndarray, pad_ratio: float = 0.1) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        h, w = img.shape[:2]
        pad_h = int(h * pad_ratio)
        pad_w = int(w * pad_ratio)
        padded = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return padded, (pad_h, pad_h, pad_w, pad_w)

    @time_method
    def _process_in_optimal_batches(self, images: List[np.ndarray], processor_function: Callable) -> List:
        """Processes images in dynamically adjusted batches based on memory."""
        if not self.parallel_enabled or len(images) <= 1:
            return self._process_batch_sequentially(images, processor_function)
        
        results, batch_size = [], min(self.max_batch_size, len(images))
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            logger.debug("detection_service.batch.processing", batch_size=len(batch))
            results.extend(self._process_batch_parallel(batch, processor_function))

            current_memory = get_current_memory_usage_mb()
            if current_memory > self.memory_limit_mb * 0.8:
                batch_size = max(1, batch_size // 2)
                logger.warning("detection_service.batch.memory_high", new_batch_size=batch_size, memory_mb=current_memory)
            elif current_memory < self.memory_limit_mb * 0.4 and batch_size < self.max_batch_size:
                batch_size = min(self.max_batch_size, batch_size * 2)
                logger.debug("detection_service.batch.memory_low", new_batch_size=batch_size, memory_mb=current_memory)
        return results

    @time_method
    def _process_batch_parallel(self, images: List[np.ndarray], processor_function: Callable) -> List:
        results: List = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {executor.submit(processor_function, img): idx for idx, img in enumerate(images)}
            for future in as_completed(future_to_idx):
                try:
                    res = future.result()
                except Exception as e:
                    logger.error("detection_service.parallel.failure", error=str(e))
                    res = []
                results.append(res)
        return results


    def _process_batch_sequentially(self, images: List[np.ndarray], processor_function: Callable) -> List:
        results: List = []
        for img in images:
            try:
                results.append(processor_function(img))
            except Exception as e:
                logger.error("detection_service.sequential.failure", error=str(e))
                results.append([])
        return results

    # --- PaddleOCR utility methods (adapted from line_det.py) ---

    def _ensure_bgr(self, img: np.ndarray) -> np.ndarray:
        """Ensure uint8 BGR image (OpenCV convention)."""
        if img is None:
            raise ValueError("Input image is None.")
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 3:
            return img
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def _extract_polys_from_result_obj(self, res: Any) -> List[np.ndarray]:
        """
        Try to extract Nx4x2 polygons from various possible result object formats.
        Supports PaddleOCR TextDetection result from the high-level API.
        """
        for attr in ("boxes", "polygons", "dt_polys"):
            if hasattr(res, attr):
                polys = getattr(res, attr)
                if polys is not None:
                    return [np.array(p, dtype=np.float32).reshape(-1, 2) for p in polys]

        for key in ("boxes", "polygons", "dt_polys", "poly", "points"):
            try:
                val = res[key]  # type: ignore[index]
                if val is not None:
                    return [np.array(p, dtype=np.float32).reshape(-1, 2) for p in val]
            except Exception:
                pass

        for method in ("to_dict", "as_dict"):
            if hasattr(res, method) and callable(getattr(res, method)):
                d = getattr(res, method)()
                for key in ("boxes", "polygons", "dt_polys", "points"):
                    if key in d and d[key] is not None:
                        return [np.array(p, dtype=np.float32).reshape(-1, 2) for p in d[key]]

        if hasattr(res, "__dict__"):
            d = res.__dict__
            for key in ("boxes", "polygons", "dt_polys", "points"):
                if key in d and d[key] is not None:
                    return [np.array(p, dtype=np.float32).reshape(-1, 2) for p in d[key]]

        raise RuntimeError(
            "Could not extract polygons from detection result. "
            "Check the result object's attributes/keys."
        )

    def _sort_reading_order(
        self,
        polys: List[np.ndarray],
        reading_direction: Literal["ltr", "rtl"] = "ltr",
    ) -> List[int]:
        """
        Sort polygons row-wise top->bottom, then within each row by reading direction.
        For Persian, use reading_direction="rtl".
        Returns indices in sorted order.
        """
        centers = np.array([self._centroid(p) for p in polys])  # (N,2): x, y
        ys = centers[:, 1]
        xs = centers[:, 0]

        # Estimate row tolerance from polygon heights
        heights = np.array([float(np.max(p[:, 1]) - np.min(p[:, 1])) for p in polys])
        row_tol = max(8.0, float(np.median(heights)) * 0.6)

        # Assign a row id to each polygon
        row_ids = np.round(ys / row_tol).astype(int)
        rows = {}
        for i, rid in enumerate(row_ids):
            rows.setdefault(rid, []).append(i)

        # Sort rows by their vertical position (increasing y)
        sorted_row_keys = sorted(rows.keys(), key=lambda rid: np.median(ys[rows[rid]]))

        ordered: List[int] = []
        for rid in sorted_row_keys:
            idxs = rows[rid]
            # Within a row: LTR -> ascending x; RTL -> descending x
            if reading_direction == "rtl":
                idxs.sort(key=lambda i: xs[i], reverse=True)
            else:
                idxs.sort(key=lambda i: xs[i])
            ordered.extend(idxs)
        return ordered

    def _poly_to_bbox(self, poly: np.ndarray, pad: int, W: int, H: int) -> Tuple[int, int, int, int]:
        """Compute clamped XYXY bbox from polygon with padding."""
        xs = poly[:, 0]
        ys = poly[:, 1]
        x1 = max(int(np.floor(xs.min())) - pad, 0)
        y1 = max(int(np.floor(ys.min())) - pad, 0)
        x2 = min(int(np.ceil(xs.max())) + pad, W - 1)
        y2 = min(int(np.ceil(ys.max())) + pad, H - 1)
        if x2 <= x1:
            x2 = min(x1 + 1, W - 1)
        if y2 <= y1:
            y2 = min(y1 + 1, H - 1)
        return x1, y1, x2, y2

    def _centroid(self, poly: np.ndarray) -> Tuple[float, float]:
        c = poly.mean(axis=0)
        return float(c[0]), float(c[1])
