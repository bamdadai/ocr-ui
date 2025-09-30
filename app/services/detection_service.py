import uuid
import cv2
import numpy as np
import torch
import structlog
import time
import os
import functools
from typing import List, Callable, Dict, Any, Literal, Optional, Sequence, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from ultralytics import YOLO
from ultralytics.engine.results import Results
from pathlib import Path
from torchvision.ops import nms

# --- Centralized imports from utility modules ---
from app.utils.visualization import draw_polygons, draw_boxes
from app.utils.image_processing import get_polygons_from_masks, merge_overlapping_masks
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
            logger.info("method.timing",
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

        self.models = {
            'word': self._load_yolo_model(config['word_detect']['path']),
            'line': self._load_yolo_model(config['line_detect']['path'])
        }
        self.model_params = {
            'word': {'iou': config['word_detect']['iou'], 'conf': config['word_detect']['conf'], 'task': 'segment', 'retina_masks': True},
            'line': {'iou': config['line_detect']['iou'], 'conf': config['line_detect']['conf'], 'retina_masks': True}
        }
        # Optional target classes for filtering; can be int ids or class names
        # If omitted, we do NOT filter by class to avoid dropping valid detections
        self.target_classes: Dict[ModelType, Optional[Sequence[int | str]]] = {
            'word': config['word_detect'].get('classes'),
            'line': config['line_detect'].get('classes')
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

        # --- Line detection tiling/merging configuration ---
        line_cfg = config.get('line_detect', {})
        tiling_cfg = line_cfg.get('tiling', {}) or {}
        merge_cfg = line_cfg.get('merge', {}) or {}
        self.line_tiling_enabled: bool = bool(tiling_cfg.get('enabled', False))
        self.line_padding_ratio: float = float(line_cfg.get('padding_ratio', 0.1))
        self.line_tiles: int = int(tiling_cfg.get('tiles', 3))
        self.line_tile_overlap_ratio: float = float(tiling_cfg.get('overlap_ratio', 0.1))
        self.line_tile_imgsz: int = int(tiling_cfg.get('imgsz', 1024))
        self.line_tile_iou: float = float(tiling_cfg.get('iou', 0.4))  # per-tile YOLO IoU
        self.line_nms_iou: float = float(tiling_cfg.get('nms_iou', 0.5))  # final NMS IoU
        self.line_merge_y_thresh: float = float(merge_cfg.get('y_thresh', 0.005))
        self.line_merge_iou_thresh: float = float(merge_cfg.get('iou_thresh', 0.03))

        logger.info("detection_service.initialized",
                    line_tiling_enabled=self.line_tiling_enabled,
                    line_tiles=self.line_tiles,
                    )

    @time_method
    def _load_yolo_model(self, model_path: str) -> YOLO:
        model = YOLO(model_path)
        return model.to(self.device)

    @time_method
    def predict_word_polygons(self, images: List[np.ndarray]) -> Dict[str, List]:
        logger.info("detection_service.predicting", model_type="word", image_count=len(images))
        return {'word_polygons': self._predict(images, model_type='word')}

    @time_method
    def predict_line_boxes(self, images: List[np.ndarray]) -> Dict[str, List]:
        logger.info("detection_service.predicting", model_type="line", image_count=len(images))
        # Use tiled predictor if enabled; otherwise fallback to single-shot prediction
        if self.line_tiling_enabled:
            processor_function = lambda img: self._predict_lines_tiled(img)
            results = self._process_in_optimal_batches(images, processor_function)
            return {'line_boxes': results}
        else:
            return {'line_boxes': self._predict(images, model_type='line')}

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
            logger.info("detection_service.class_filter", model_type=model_type, kept=int(mask.sum().item()), total=len(cls_tensor))
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
        results: List[Results] = model(image, **model_params)
        result = results[0]
        # Optional class filter
        filtered = self._filter_by_classes(result, model_type)
        post_processed = post_process_func(filtered)
        duration = time.time() - start
        if self.debug:
            try:
                fname = f"{model_type}_debug_{uuid.uuid4().hex[:8]}.jpg"
                out_path = os.path.join(str(self.debug_info[model_type]['path']), fname)
                os.makedirs(self.debug_info[model_type]['path'], exist_ok=True)
                debug_image = image.copy()
                debug_image = debug_draw_func(debug_image, post_processed)
                cv2.imwrite(out_path, debug_image)
            except Exception as e:
                logger.warning("detection_service.debug_save_failed", error=str(e))
        logger.info("detection_service.single_image_done", model_type=model_type, duration_ms=int(duration * 1000))
        return post_processed

    @time_method
    def _post_process_word_results(self, word_result: Results) -> List[List[int]]:
        """Post-processes word detection results including polygon merging.
        If segmentation masks are unavailable (e.g., using a bbox-only model),
        fall back to rectangle polygons derived from xyxy boxes.
        """
        # Preferred path: segmentation masks -> polygons -> merge
        try:
            if word_result.masks is not None:
                polygons = get_polygons_from_masks(word_result.masks)
                merged_polygons = merge_overlapping_masks(polygons, iou_threshold=self.merging_iou)
                return [poly.reshape(-1).astype(int).tolist() for poly in merged_polygons]
        except Exception as e:
            logger.warning("detection_service.word_masks_postprocess_failed", error=str(e))
            # continue to bbox fallback
        # Fallback: use bounding boxes if masks are missing
        try:
            if word_result.boxes is None or word_result.boxes.data is None or word_result.boxes.data.numel() == 0:
                return []
            boxes_xyxy = word_result.boxes.xyxy.cpu().numpy()
            rect_polys: List[List[int]] = []
            for x1, y1, x2, y2 in boxes_xyxy:
                x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                poly = np.array([[x1i, y1i], [x2i, y1i], [x2i, y2i], [x1i, y2i]], dtype=np.int32)
                rect_polys.append(poly.reshape(-1).tolist())
            if rect_polys:
                logger.info("detection_service.word_bbox_fallback", count=len(rect_polys))
            return rect_polys
        except Exception as e:
            logger.warning("detection_service.word_bbox_fallback_failed", error=str(e))
            return []

    @time_method
    def _post_process_line_results(self, line_result: Results) -> List[List[float]]:
        """Post-processes line detection results."""
        if line_result.boxes is None: return []
        return [box.xyxy[0].cpu().numpy().tolist() for box in line_result.boxes]

    # --- Tiled line detection inspired by tests/line_detection.py ---
    @time_method
    def _predict_lines_tiled(self, image: np.ndarray) -> List[List[float]]:
        if image is None:
            return []
        padded_img, (pad_top, pad_bottom, pad_left, pad_right) = self._add_padding_all_sides(image, self.line_padding_ratio)
        detections: List[Tuple[np.ndarray, Optional[np.ndarray]]] = self._horizontal_tile_and_collect(padded_img)
        if detections:
            # Merge connected lines based on vertical alignment and IoU
            detections = self._merge_all_connected_lines(
                detections,
                img_shape=padded_img.shape[:2],
                y_thresh=self.line_merge_y_thresh,
                iou_thresh=self.line_merge_iou_thresh
            )
            if detections:
                # Final NMS on merged boxes
                boxes_xyxy = torch.from_numpy(np.array([d[0][:4] for d in detections], dtype=np.float32))
                scores = torch.from_numpy(np.array([d[0][4] for d in detections], dtype=np.float32))
                keep = nms(boxes_xyxy, scores, iou_threshold=float(self.line_nms_iou))
                detections = [detections[i] for i in keep.tolist()]
        # Map back to original (unpadded) coordinates and clamp
        final_boxes: List[List[float]] = []
        if detections:
            for box, _ in detections:
                x1 = max(0.0, float(box[0] - pad_left))
                y1 = max(0.0, float(box[1] - pad_top))
                x2 = max(0.0, float(box[2] - pad_left))
                y2 = max(0.0, float(box[3] - pad_top))
                final_boxes.append([x1, y1, x2, y2])
        # Optional debug output on original image
        if self.debug:
            try:
                os.makedirs(self.debug_info['line']['path'], exist_ok=True)
                dbg = draw_boxes(image.copy(), final_boxes, color=(0, 0, 255))
                out_path = os.path.join(str(self.debug_info['line']['path']), f"line_tiled_{uuid.uuid4().hex[:8]}.jpg")
                cv2.imwrite(out_path, dbg)
            except Exception as e:
                logger.warning("detection_service.line_tiled_debug_save_failed", error=str(e))
        # Sort top-to-bottom similar to pipeline behavior
        final_boxes.sort(key=lambda b: b[1])
        return final_boxes

    @time_method
    def _horizontal_tile_and_collect(self, padded_img: np.ndarray) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        h, w = padded_img.shape[:2]
        overlap = int(w * self.line_tile_overlap_ratio)
        detections: List[Tuple[np.ndarray, Optional[np.ndarray]]] = []
        # Compute tile coordinates
        if self.line_tiles <= 1:
            coords = [(0, w)]
        elif self.line_tiles == 3:
            third = w // 3
            coords = [
                (0, third + overlap // 2),
                (third - overlap // 2, 2 * third + overlap // 2),
                (2 * third - overlap // 2, w)
            ]
        else:
            tile_w = w // self.line_tiles
            coords = []
            for i in range(self.line_tiles):
                start = max(0, i * tile_w - (overlap // 2 if i > 0 else 0))
                end = min(w, (i + 1) * tile_w + (overlap // 2 if i < self.line_tiles - 1 else 0))
                coords.append((start, end))
        # Run model over each tile
        for x_start, x_end in coords:
            tile = padded_img[:, x_start:x_end]
            try:
                results: List[Results] = self.models['line'](
                    tile,
                    imgsz=int(self.line_tile_imgsz),
                    conf=float(self.model_params['line']['conf']),
                    iou=float(self.line_tile_iou),
                    retina_masks=True,
                    mask_ratio=2,
                    nms=True,
                )
            except Exception as e:
                logger.warning("detection_service.line_tile_infer_failed", error=str(e))
                continue
            if not results:
                continue
            result = results[0]
            if not result.boxes or result.boxes.data.numel() == 0:
                continue
            boxes = result.boxes.data.clone().detach().cpu().numpy()  # (N, 6)
            masks_list: List[np.ndarray] = []
            if getattr(result, 'masks', None) is not None and getattr(result.masks, 'xyn', None) is not None:
                for poly in result.masks.xyn:
                    pts = (poly * np.array([tile.shape[1], tile.shape[0]])).astype(np.int32)
                    pts = pts.reshape(-1, 1, 2)
                    pts[:, :, 0] += int(x_start)
                    masks_list.append(pts)
            for i, box in enumerate(boxes):
                adj_box = box.copy()
                adj_box[0] += x_start
                adj_box[2] += x_start
                mask = masks_list[i] if i < len(masks_list) else None
                detections.append((adj_box, mask))
        return detections

    @time_method
    def _merge_all_connected_lines(
        self,
        detections: List[Tuple[np.ndarray, Optional[np.ndarray]]],
        img_shape: Tuple[int, int],
        y_thresh: float,
        iou_thresh: float
    ) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        h, w = img_shape
        if not detections:
            return []
        merged: List[Tuple[np.ndarray, Optional[np.ndarray]]] = []
        used = [False] * len(detections)
        for i in range(len(detections)):
            if used[i]:
                continue
            current_group: List[Tuple[np.ndarray, Optional[np.ndarray]]] = [detections[i]]
            used[i] = True
            while True:
                new_added = False
                for j in range(len(detections)):
                    if used[j]:
                        continue
                    boxB, _ = detections[j]
                    yB_bottom = boxB[3] / h
                    yB_center = (boxB[1] + boxB[3]) / 2 / h
                    yB_top = boxB[1] / h
                    is_close = False
                    for boxA, _ in current_group:
                        yA_bottom = boxA[3] / h
                        yA_center = (boxA[1] + boxA[3]) / 2 / h
                        yA_top = boxA[1] / h
                        y_bottom_diff = abs(yA_bottom - yB_bottom)
                        y_center_diff = abs(yA_center - yB_center)
                        y_top_diff = abs(yA_top - yB_top)
                        iou = self._compute_iou(boxA, boxB)
                        if iou > iou_thresh and (y_bottom_diff < y_thresh or y_center_diff < y_thresh or y_top_diff < y_thresh):
                            is_close = True
                            break
                    if is_close:
                        current_group.append(detections[j])
                        used[j] = True
                        new_added = True
                if not new_added:
                    break
            group_boxes = np.array([item[0] for item in current_group])
            group_masks = [item[1] for item in current_group if item[1] is not None]
            x1, y1 = group_boxes[:, 0].min(), group_boxes[:, 1].min()
            x2, y2 = group_boxes[:, 2].max(), group_boxes[:, 3].max()
            conf = group_boxes[:, 4].mean() if group_boxes.shape[1] > 4 else 1.0
            cls = np.median(group_boxes[:, 5]) if group_boxes.shape[1] > 5 else 0
            merged_box = np.array([x1, y1, x2, y2, conf, cls], dtype=np.float32)
            merged_mask: Optional[np.ndarray] = None
            if group_masks:
                all_pts = np.vstack(group_masks)
                if all_pts.size > 0:
                    merged_mask = cv2.convexHull(all_pts)
            merged.append((merged_box, merged_mask))
        return merged

    @time_method
    def _compute_iou(self, boxA: np.ndarray, boxB: np.ndarray) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0.0, xB - xA + 1.0) * max(0.0, yB - yA + 1.0)
        boxAArea = max(0.0, (boxA[2] - boxA[0] + 1.0)) * max(0.0, (boxA[3] - boxA[1] + 1.0))
        boxBArea = max(0.0, (boxB[2] - boxB[0] + 1.0)) * max(0.0, (boxB[3] - boxB[1] + 1.0))
        denom = (boxAArea + boxBArea - interArea + 1e-6)
        return float(interArea / denom) if denom > 0 else 0.0

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

