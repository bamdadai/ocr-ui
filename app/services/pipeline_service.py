from typing import List, Tuple, TYPE_CHECKING
import numpy as np
import structlog
from pathlib import Path

# NOTE: Defer heavy service imports to runtime to avoid ImportError masking due to import-time failures
# (e.g., missing CUDA libs, model weights, or optional deps). Use TYPE_CHECKING for hints only.
if TYPE_CHECKING:  # pragma: no cover
    from app.services.detection_service import DetectionService  # noqa: F401
    from app.services.recognition_service import RecognitionService  # noqa: F401

from app.utils.image_processing import (
    crop_boxes_from_image,
    make_box_from_poly,
    crop_word_from_polygon,
    rebase_polygon
)
from app.utils.text_processing import fix_mixed_text_order
from app.utils.visualization import save_page_recognition_result, save_word_crops, save_word_polygons_on_page

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

        # Recognition can be heavy; initialize lazily on first use
        pipeline_config = config.get('pipeline', {})
        self.enable_recognition = pipeline_config.get('enable_recognition', True)
        self.recognition_service = None  # type: ignore[assignment]
        
        # Store debug configuration for recognition
        recognition_config = config.get('recognition', {})
        self.recognition_debug = recognition_config.get('debug', False)
        self.recognition_debug_path = Path(recognition_config.get('debug_recog_path', 'debug/recog'))
        self.recognition_font_path = Path(recognition_config.get('debug_font_path', 'assets/fonts/XB Niloofar.ttf'))
        self.word_crops_debug_path = Path(recognition_config.get('debug_word_crops_path', 'debug/word_crops'))
        self.word_polygons_debug_path = Path(recognition_config.get('debug_word_polygons_path', 'debug/word_polygons'))

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

        logger.info(
            "pipeline_service.initialized",
            recognition_enabled=self.enable_recognition,
            recognition_debug=self.recognition_debug,
            orientation_enabled=self.orientation_enabled,
            orientation_device=self.orientation_device if self.orientation_enabled else None,
        )

    def _ensure_recognition_loaded(self):
        if not self.enable_recognition:
            return
        if self.recognition_service is None:
            try:
                from app.services.recognition_service import RecognitionService as _RecognitionService  # local import
                self.recognition_service = _RecognitionService(self._config['recognition'])
                logger.info("pipeline_service.recognition_loaded")
            except Exception as e:
                logger.critical("pipeline_service.recognition_init_failed", error=str(e), exc_info=True)
                raise

    def detect_lines(self, image: np.ndarray) -> List[list]:
        """Detects all line bounding boxes in a single image."""
        # Image is already oriented at ingestion; don't rotate here
        line_boxes = self.detection_service.predict_line_boxes([image])['line_boxes'][0]
        # Sort lines top-to-bottom for correct reading order.
        return sorted(line_boxes, key=lambda box: box[1])

    def detect_words(self, image: np.ndarray, line_boxes: List[list]) -> List[list]:
        """Detects all word polygons within the given line boxes for an image."""
        if not line_boxes:
            return []
        # Image is already oriented at ingestion; don't rotate here
        line_crops = crop_boxes_from_image(line_boxes, image)
        return self.detection_service.predict_word_polygons(line_crops)['word_polygons']

    def recognize_page(self, image: np.ndarray, line_boxes: List[list], word_polygons_per_line: List[list], page_id: str = None) -> Tuple[str, float]:
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
        line_crops = crop_boxes_from_image(line_boxes, image)
        text_of_lines = []
        
        # Collect all word data with page-level coordinates for debug output
        all_page_word_data = []

        for line_idx, word_polygons in enumerate(word_polygons_per_line):
            if not word_polygons:
                continue
            
            line_crop = line_crops[line_idx]
            line_box = line_boxes[line_idx]
            
            # Recognize the line and get word data
            line_text, line_conf, word_data = self._recognize_line(line_crop, word_polygons)
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
        
        full_text = "\n".join(text for text, _ in text_of_lines)
        overall_conf = sum(conf for _, conf in text_of_lines) / len(text_of_lines) if text_of_lines else 0.0
        
        # Save debug image with transcribed text if enabled
        if self.recognition_debug and text_of_lines:
            try:
                save_page_recognition_result(
                    image=image,
                    line_boxes=line_boxes,
                    text_of_lines=text_of_lines,
                    save_dir=self.recognition_debug_path,
                    font_path=self.recognition_font_path
                )
                logger.info("pipeline_service.recognition_debug_saved", path=str(self.recognition_debug_path))
            except Exception as e:
                logger.warning("pipeline_service.recognition_debug_failed", error=str(e), exc_info=True)
        
        # Save word crops and word polygons visualization from full page
        if self.recognition_debug and all_page_word_data:
            try:
                # Extract crops from full page using page-level coordinates
                page_word_crops = []
                page_word_texts = []
                page_word_polygons = []
                
                for word_info in all_page_word_data:
                    crop = crop_word_from_polygon(image, word_info['polygon'])
                    page_word_crops.append(crop)
                    page_word_texts.append(word_info['text'])
                    page_word_polygons.append(word_info['polygon'])
                
                # Save individual word crops
                save_word_crops(
                    word_crops=page_word_crops,
                    word_texts=page_word_texts,
                    save_dir=self.word_crops_debug_path,
                    page_id=page_id
                )
                logger.info("pipeline_service.word_crops_debug_saved", count=len(page_word_crops), path=str(self.word_crops_debug_path))
                
                # Save word polygons drawn on full page
                save_word_polygons_on_page(
                    image=image,
                    word_polygons=page_word_polygons,
                    word_texts=page_word_texts,
                    save_dir=self.word_polygons_debug_path,
                    page_id=page_id
                )
                logger.info("pipeline_service.word_polygons_on_page_saved", count=len(page_word_polygons), path=str(self.word_polygons_debug_path))
            except Exception as e:
                logger.warning("pipeline_service.word_crops_debug_failed", error=str(e), exc_info=True)
        
        return full_text, overall_conf

    def _recognize_line(self, line_crop: np.ndarray, word_polygons: list) -> Tuple[str, float, List[dict]]:
        """
        Helper to recognize all words in a single line.
        
        Returns:
            Tuple of (line_text, line_confidence, word_data_list)
            word_data_list contains dicts with 'polygon', 'text', 'prob', 'box' for each word
        """
        word_data = [
            {
                'crop': crop_word_from_polygon(line_crop, poly),
                'box': make_box_from_poly(poly),
                'polygon': poly  # Keep original polygon for coordinate conversion
            }
            for poly in word_polygons
        ]
        
        all_word_crops = [item['crop'] for item in word_data]
        # Recognition is ensured to be loaded by recognize_page
        word_texts_with_probs = self.recognition_service(all_word_crops)  # type: ignore[misc]

        for i, item in enumerate(word_data):
            item['text'], item['prob'] = word_texts_with_probs[i]

        sorted_words = sorted(word_data, key=lambda x: x['box'][0], reverse=True)
        
        text_line = " ".join(item['text'] for item in sorted_words)
        text_line = fix_mixed_text_order(text_line)
        
        line_probs = [item['prob'] for item in sorted_words]
        line_conf = sum(line_probs) / len(line_probs) if line_probs else 0.0
        
        return text_line, line_conf, word_data
