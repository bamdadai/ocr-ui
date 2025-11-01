from typing import List, Tuple, TYPE_CHECKING, Optional, Dict, Any
import numpy as np
import structlog
from pathlib import Path
from time import perf_counter_ns

# Import PROJECT_ROOT for consistent path resolution
from app.core.config import PROJECT_ROOT

# NOTE: Defer heavy service imports to runtime to avoid ImportError masking due to import-time failures
# (e.g., missing CUDA libs, model weights, or optional deps). Use TYPE_CHECKING for hints only.
if TYPE_CHECKING:  # pragma: no cover
    from app.services.detection_service import DetectionService  # noqa: F401
    from app.services.recognition_service import RecognitionService  # noqa: F401

from app.utils.image_processing import (
    crop_boxes_from_image,
    make_box_from_poly,
    crop_word_from_polygon,
    rebase_polygon,
    preprocess_recognition_image,
    enlarge_polygon,
    enlarge_polygon_by_percentage
)
from app.utils.text_processing import fix_mixed_text_order
from app.utils.visualization import save_word_polygons_on_page, save_line_parts_visualization, save_word_crops, save_rec_debug_images
from app.utils.performance_logging import log_stage_timing

logger = structlog.get_logger(__name__)

class PipelineService:
    """
    Acts as a 'toolbox' of core OCR functions.
    It no longer controls the flow, but provides the implementation logic
    that Celery tasks will call.
    """
    def __init__(self, config: dict):
        # Keep original config for possible lazy initializations
        self._config = config

        # Load detection service immediately (needed for detection tasks)
        try:
            from app.services.detection_service import DetectionService as _DetectionService  # local import
            self.detection_service = _DetectionService(config['detection'])
        except Exception as e:
            logger.critical("pipeline_service.detection_init_failed", error=str(e), exc_info=True)
            raise

        # Load recognition service immediately if enabled
        pipeline_config = config.get('pipeline', {})
        self.enable_recognition = pipeline_config.get('enable_recognition', True)
        self.recognition_service = None  # type: ignore[assignment]

        if self.enable_recognition:
            try:
                from app.services.recognition_service import RecognitionService as _RecognitionService
                self.recognition_service = _RecognitionService(config['recognition'])
                logger.debug("pipeline_service.recognition_loaded_eagerly")
            except Exception as e:
                logger.critical("pipeline_service.recognition_init_failed", error=str(e), exc_info=True)
                raise
        
        # Store debug configuration for recognition
        recognition_config = config.get('recognition', {})
        self.recognition_debug = recognition_config.get('debug', False)
        self.word_polygons_debug_path = Path(recognition_config.get('debug_word_polygons_path', 'debug/word_polygons'))
        self.parts_debug_path = Path(recognition_config.get('debug_parts_path', 'debug/parts'))
        self.word_crops_debug_path = Path(recognition_config.get('debug_word_crops_path', 'debug/word_crops'))
        self.rec_debug_path = PROJECT_ROOT / 'debug' / 'rec_debug'
        
        # Store image preprocessing configuration
        preprocessing_config = recognition_config.get('image_preprocessing')
        if preprocessing_config:
            self.preprocessing_enabled = True
            self.preprocessing_padding = preprocessing_config.get('padding', 0)
            self.preprocessing_erosion_kernel_size = preprocessing_config.get('erosion_kernel_size', 0)
            self.preprocessing_erosion_iterations = preprocessing_config.get('erosion_iterations', 1)
            self.preprocessing_contrast = preprocessing_config.get('contrast', 1.0)
            self.preprocessing_brightness = preprocessing_config.get('brightness', 0.0)
            self.preprocessing_sharpness = preprocessing_config.get('sharpness', 0.0)
        else:
            self.preprocessing_enabled = False
        
        # Store word boundary enlargement configuration
        enlargement_config = recognition_config.get('word_boundary_enlargement')
        if enlargement_config and enlargement_config.get('enabled', False):
            self.enlargement_enabled = True
            self.enlargement_pixels = enlargement_config.get('pixels', 0)
            self.enlargement_percentage = enlargement_config.get('percentage')
        else:
            self.enlargement_enabled = False
            self.enlargement_pixels = 0
            self.enlargement_percentage = None

        # Text rendering configuration for line grouping
        text_rendering_cfg = config.get('text_rendering', {})
        self.same_row_separator: str = text_rendering_cfg.get('same_row_separator', '\t')
        self.height_tolerance_pixels: int = text_rendering_cfg.get('height_tolerance_pixels', 5)

        # Line splitting configuration
        line_splitting_cfg = config.get('line_splitting', {})
        self.max_width_height_ratio: float = line_splitting_cfg.get('max_width_height_ratio', 4.0)
        self.overlap_threshold: float = line_splitting_cfg.get('overlap_threshold', 0.1)

        # Orientation is now handled once at ingestion by tasks.process_ocr_task
        # Keep flags for logging/telemetry only
        orientation_cfg = (
            config.get('orientation')
            or pipeline_config.get('orientation')
            or {}
        )
        self.orientation_enabled: bool = bool(
            orientation_cfg.get('enabled', pipeline_config.get('orientation_enabled', False))
        )
        self.orientation_device: str = str(
            orientation_cfg.get('device', config.get('detection', {}).get('device', 'cpu'))
        )
        self.orientation_imgsz: int = int(orientation_cfg.get('imgsz', 224))

        logger.debug(
            "pipeline_service.initialized",
            recognition_enabled=self.enable_recognition,
            recognition_debug=self.recognition_debug,
            orientation_enabled=self.orientation_enabled,
            orientation_device=self.orientation_device if self.orientation_enabled else None,
        )

    def _ensure_recognition_loaded(self):
        """Legacy method - recognition is now loaded eagerly in __init__."""
        if not self.enable_recognition:
            return
        if self.recognition_service is None:
            logger.warning("pipeline_service.recognition_not_loaded",
                         msg="Recognition should have been loaded in __init__")

    def detect_lines(self, image: np.ndarray, telemetry: Optional[Dict[str, Any]] = None) -> List[list]:
        """Detects all line bounding boxes in a single image."""
        # Image is already oriented at ingestion; don't rotate here
        start_ns = perf_counter_ns()
        line_boxes = self.detection_service.predict_line_boxes([image])['line_boxes'][0]
        end_ns = perf_counter_ns()
        log_stage_timing(
            "pipeline.detect_lines.predict",
            start_ns=start_ns,
            end_ns=end_ns,
            request_id=telemetry.get("request_id") if telemetry else None,
            page_index=telemetry.get("page_index") if telemetry else None,
            extra={
                "line_count": len(line_boxes),
                "height": int(image.shape[0]),
                "width": int(image.shape[1]),
                "device": getattr(self.detection_service, "device", None),
            },
        )
        # Sort lines top-to-bottom for correct reading order.
        return sorted(line_boxes, key=lambda box: box[1])

    def detect_words(self, image: np.ndarray, line_boxes: List[list], telemetry: Optional[Dict[str, Any]] = None) -> List[list]:
        """Detects all word polygons within the given line boxes for an image."""
        if not line_boxes:
            return []
        # Image is already oriented at ingestion; don't rotate here
        crop_start_ns = perf_counter_ns()
        line_crops = crop_boxes_from_image(line_boxes, image)
        crop_end_ns = perf_counter_ns()
        log_stage_timing(
            "pipeline.detect_words.crop_lines",
            start_ns=crop_start_ns,
            end_ns=crop_end_ns,
            request_id=telemetry.get("request_id") if telemetry else None,
            page_index=telemetry.get("page_index") if telemetry else None,
            extra={"line_count": len(line_boxes)},
        )
        detect_start_ns = perf_counter_ns()
        word_polygons = self.detection_service.predict_word_polygons(line_crops)['word_polygons']
        detect_end_ns = perf_counter_ns()
        total_words = sum(len(polys) for polys in word_polygons if isinstance(polys, list))
        log_stage_timing(
            "pipeline.detect_words.predict",
            start_ns=detect_start_ns,
            end_ns=detect_end_ns,
            request_id=telemetry.get("request_id") if telemetry else None,
            page_index=telemetry.get("page_index") if telemetry else None,
            extra={
                "line_count": len(line_boxes),
                "word_group_count": len(word_polygons),
                "word_count": total_words,
                "device": getattr(self.detection_service, "device", None),
            },
        )
        return word_polygons

    def recognize_page(self, image: np.ndarray, line_boxes: List[list], word_polygons_per_line: List[list], page_id: str = None, telemetry: Optional[Dict[str, Any]] = None) -> Tuple[str, float]:
        """
        Recognizes text for an entire page and returns the full text and average confidence.
        
        Args:
            image: The page image
            line_boxes: List of line bounding boxes
            word_polygons_per_line: List of word polygons for each line
            page_id: Optional page identifier for debug output
        """
        if not self.enable_recognition:
            return ("[RECOGNITION DISABLED]", 0.0)

        # Lazy-load recognition only when needed
        self._ensure_recognition_loaded()
        assert self.recognition_service is not None  # for type checkers

        # Image is already oriented at ingestion; don't rotate here
        crop_start_ns = perf_counter_ns()
        line_crops = crop_boxes_from_image(line_boxes, image)
        crop_end_ns = perf_counter_ns()
        log_stage_timing(
            "pipeline.recognize_page.crop_lines",
            start_ns=crop_start_ns,
            end_ns=crop_end_ns,
            request_id=telemetry.get("request_id") if telemetry else None,
            page_index=telemetry.get("page_index") if telemetry else None,
            extra={"line_count": len(line_boxes)},
        )
        text_of_lines = []
        
        # Collect all word data with page-level coordinates for debug output
        all_page_word_data = []

        for line_idx, word_polygons in enumerate(word_polygons_per_line):
            if not word_polygons:
                continue

            line_crop = line_crops[line_idx]
            line_box = line_boxes[line_idx]

            # Recognize the line and get word data
            line_id = f"{page_id}_line{line_idx}" if page_id else f"line{line_idx}"
            line_text, line_conf, word_data = self._recognize_line(
                line_crop,
                word_polygons,
                line_box,
                line_id,
                telemetry={
                    "request_id": telemetry.get("request_id") if telemetry else None,
                    "page_index": telemetry.get("page_index") if telemetry else None,
                    "line_index": line_idx,
                    "line_height": int(line_box[3]),
                    "line_width": int(line_box[2]),
                } if telemetry else None
            )
            text_of_lines.append((line_text, line_conf))
            
            # Convert word polygons to page-level coordinates for debug
            if self.recognition_debug:
                line_offset = (int(line_box[0]), int(line_box[1]))
                for word_info in word_data:
                    # Convert polygon from line-local to page-level coordinates
                    page_polygon = rebase_polygon(word_info['polygon'], line_offset)
                    all_page_word_data.append({
                        'polygon': page_polygon,
                        'text': word_info['text'],
                        'line_idx': line_idx
                    })
        
        # Group lines by height for natural reading order
        group_start_ns = perf_counter_ns()
        full_text = self._group_lines_by_height(text_of_lines, line_boxes)
        group_end_ns = perf_counter_ns()
        log_stage_timing(
            "pipeline.recognize_page.group_lines",
            start_ns=group_start_ns,
            end_ns=group_end_ns,
            request_id=telemetry.get("request_id") if telemetry else None,
            page_index=telemetry.get("page_index") if telemetry else None,
            extra={"line_count": len(text_of_lines)},
        )
        overall_conf = sum(conf for _, conf in text_of_lines) / len(text_of_lines) if text_of_lines else 0.0
        
        # GPU memory cleanup after processing all lines
        if hasattr(self.recognition_service, 'device') and 'cuda' in self.recognition_service.device:
            import torch
            torch.cuda.empty_cache()
        
        # Save word polygons visualization from full page
        if self.recognition_debug and all_page_word_data:
            try:
                # Extract polygons and texts from full page using page-level coordinates
                page_word_polygons = []
                page_word_texts = []

                for word_info in all_page_word_data:
                    page_word_polygons.append(word_info['polygon'])
                    page_word_texts.append(word_info['text'])

                # Save word polygons drawn on full page
                save_word_polygons_on_page(
                    image=image,
                    word_polygons=page_word_polygons,
                    word_texts=page_word_texts,
                    save_dir=self.word_polygons_debug_path,
                    page_id=page_id
                )
                logger.debug("pipeline_service.word_polygons_on_page_saved", count=len(page_word_polygons), path=str(self.word_polygons_debug_path))
            except Exception as e:
                logger.warning("pipeline_service.word_polygons_debug_failed", error=str(e), exc_info=True)
        
        return full_text, overall_conf

    def _group_lines_by_height(self, text_of_lines: List[Tuple[str, float]], line_boxes: List[list]) -> str:
        """
        Groups lines that are at the same height (e.g., table cells, columns) and joins them
        with a configurable separator instead of newlines.

        Args:
            text_of_lines: List of (text, confidence) tuples for each line
            line_boxes: List of bounding boxes for each line [x, y, w, h]

        Returns:
            Full text with natural reading order
        """
        if not text_of_lines or not line_boxes:
            return ""

        # Create list of (y_position, text, box) tuples
        lines_with_positions = []
        for i, ((text, _), box) in enumerate(zip(text_of_lines, line_boxes)):
            y_position = box[1]  # y coordinate of the line
            lines_with_positions.append((y_position, text, box))

        # Sort by y position (top to bottom)
        lines_with_positions.sort(key=lambda x: x[0])

        # Group lines by similar y positions
        grouped_rows = []
        current_row = []
        current_y = None

        for y_pos, text, box in lines_with_positions:
            if current_y is None:
                # First line
                current_y = y_pos
                current_row.append((box[0], text))  # Store (x_position, text)
            elif abs(y_pos - current_y) <= self.height_tolerance_pixels:
                # Same row - add to current row
                current_row.append((box[0], text))
            else:
                # New row - save current row and start new one
                if current_row:
                    grouped_rows.append(current_row)
                current_row = [(box[0], text)]
                current_y = y_pos

        # Don't forget the last row
        if current_row:
            grouped_rows.append(current_row)

        # Build final text: within each row, sort by x position (right to left for RTL)
        # then join rows with newlines
        result_lines = []
        for row in grouped_rows:
            # Sort by x position (right to left)
            row.sort(key=lambda x: x[0], reverse=True)
            # Join texts in the row with the configured separator
            row_text = self.same_row_separator.join(text for _, text in row)
            result_lines.append(row_text)

        return "\n".join(result_lines)

    def _recognize_line(self, line_crop: np.ndarray, word_polygons: list, line_box: list = None, line_id: str = None, telemetry: Optional[Dict[str, Any]] = None) -> Tuple[str, float, List[dict]]:
        """
        Helper to recognize text in a single line using part-based recognition.

        The line is split into horizontal parts based on word positions, and each part
        is sent to the recognition model (instead of individual words). Parts are split
        at word boundaries only, with a maximum width of 4× the line height.

        Args:
            line_crop: The cropped line image
            word_polygons: List of word polygons in line-local coordinates
            line_box: Line bounding box in page coordinates [x, y, w, h] (optional)
            line_id: Line identifier for debug output (optional)

        Returns:
            Tuple of (line_text, line_confidence, word_data_list)
            word_data_list contains dicts with 'polygon', 'text', 'prob', 'box' for each word
        """
        from app.utils.line_splitting import split_line_into_parts

        # If line_box is not provided, create a dummy one based on line_crop dimensions
        if line_box is None:
            line_box = [0, 0, line_crop.shape[1], line_crop.shape[0]]

        # Enlarge word polygons if enabled
        if self.enlargement_enabled:
            enlarged_word_polygons = []
            for poly in word_polygons:
                if self.enlargement_percentage is not None:
                    enlarged_poly = enlarge_polygon_by_percentage(poly, self.enlargement_percentage)
                else:
                    enlarged_poly = enlarge_polygon(poly, self.enlargement_pixels)
                enlarged_word_polygons.append(enlarged_poly)
            word_polygons = enlarged_word_polygons

        # Split line into parts
        split_start_ns = perf_counter_ns()
        parts = split_line_into_parts(
            line_crop=line_crop,
            line_box=line_box,
            word_polygons=word_polygons,
            max_width_height_ratio=self.max_width_height_ratio,
            overlap_threshold=self.overlap_threshold
        )
        split_end_ns = perf_counter_ns()

        log_stage_timing(
            "pipeline.recognize_line.split",
            start_ns=split_start_ns,
            end_ns=split_end_ns,
            request_id=telemetry.get("request_id") if telemetry else None,
            page_index=telemetry.get("page_index") if telemetry else None,
            extra={
                "line_id": line_id,
                "line_index": telemetry.get("line_index") if telemetry else None,
                "part_count": len(parts),
                "word_count": len(word_polygons),
                "line_height": telemetry.get("line_height") if telemetry else None,
                "line_width": telemetry.get("line_width") if telemetry else None,
            } if telemetry else {
                "line_id": line_id,
                "part_count": len(parts),
                "word_count": len(word_polygons),
            },
        )

        logger.debug(
            "pipeline_service.line_split_into_parts",
            num_parts=len(parts),
            num_words=len(word_polygons)
        )

        # Save parts visualization if debug is enabled
        if self.recognition_debug and parts:
            try:
                save_line_parts_visualization(
                    line_crop=line_crop,
                    parts=parts,
                    save_dir=self.parts_debug_path,
                    line_id=line_id
                )
                logger.debug("pipeline_service.parts_visualization_saved",
                           line_id=line_id,
                           num_parts=len(parts),
                           path=str(self.parts_debug_path))
            except Exception as e:
                logger.warning("pipeline_service.parts_visualization_failed",
                             line_id=line_id,
                             error=str(e),
                             exc_info=True)

        # Recognize each part
        part_crops = [part['crop'] for part in parts]
        
        # Apply image preprocessing if enabled
        if self.preprocessing_enabled:
            part_crops = [
                preprocess_recognition_image(
                    crop,
                    padding=self.preprocessing_padding,
                    erosion_kernel_size=self.preprocessing_erosion_kernel_size,
                    erosion_iterations=self.preprocessing_erosion_iterations,
                    contrast=self.preprocessing_contrast,
                    brightness=self.preprocessing_brightness,
                    sharpness=self.preprocessing_sharpness
                )
                for crop in part_crops
            ]
        
        recognize_start_ns = perf_counter_ns()
        part_texts_with_probs = self.recognition_service(part_crops)  # type: ignore[misc]
        recognize_end_ns = perf_counter_ns()
        log_stage_timing(
            "pipeline.recognize_line.recognize_parts",
            start_ns=recognize_start_ns,
            end_ns=recognize_end_ns,
            request_id=telemetry.get("request_id") if telemetry else None,
            page_index=telemetry.get("page_index") if telemetry else None,
            extra={
                "line_id": line_id,
                "line_index": telemetry.get("line_index") if telemetry else None,
                "part_count": len(part_crops),
                "device": getattr(self.recognition_service, "device", None),
            } if telemetry else {
                "line_id": line_id,
                "part_count": len(part_crops),
            },
        )

        # Save debug images if enabled
        if self.recognition_debug and part_crops:
            try:
                part_texts_list = [text for text, _ in part_texts_with_probs]
                part_confidences = [prob for _, prob in part_texts_with_probs]
                save_rec_debug_images(part_crops, part_texts_list, part_confidences, self.rec_debug_path)
            except Exception as e:
                logger.warning("pipeline_service.rec_debug_save_failed", error=str(e), exc_info=True)

        # Combine part texts (right to left, already ordered by split_line_into_parts)
        part_texts = []
        part_probs = []

        for i, (text, prob) in enumerate(part_texts_with_probs):
            part_texts.append(text)
            part_probs.append(prob)

            logger.debug(
                "pipeline_service.part_recognized",
                part_index=i,
                text=text,
                confidence=prob,
                word_count=parts[i]['word_count']
            )

        # Join parts with spaces (parts are already in RTL order)
        text_line = " ".join(part_texts)
        text_line = fix_mixed_text_order(text_line)

        # Calculate average confidence
        line_conf = sum(part_probs) / len(part_probs) if part_probs else 0.0

        # Create word_data for backward compatibility (for debug visualization)
        # We'll create synthetic word data based on word polygons but with part-level text
        word_data = []
        for poly in word_polygons:
            word_data.append({
                'crop': crop_word_from_polygon(line_crop, poly),
                'box': make_box_from_poly(poly),
                'polygon': poly,
                'text': '',  # We don't have individual word text anymore
                'prob': line_conf  # Use line confidence as approximation
            })

        if self.recognition_debug and word_data:
            try:
                save_word_crops(
                    word_data=word_data,
                    save_dir=self.word_crops_debug_path,
                    line_id=line_id
                )
                logger.debug(
                    "pipeline_service.word_crops_saved",
                    line_id=line_id,
                    count=len(word_data),
                    path=str(self.word_crops_debug_path)
                )
            except Exception as e:
                logger.warning(
                    "pipeline_service.word_crops_debug_failed",
                    line_id=line_id,
                    error=str(e),
                    exc_info=True
                )

        return text_line, line_conf, word_data
