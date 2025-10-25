# app/utils/orientation.py
from __future__ import annotations
from typing import List, Optional
import re
import threading
import cv2
import numpy as np
import structlog
from ultralytics import YOLO

from app.core.config import settings

logger = structlog.get_logger(__name__)


def _get_orientation_cfg() -> dict:
    """Resolve orientation configuration from settings.
    Prefer top-level settings.orientation, fallback to settings.pipeline.orientation.
    Returns an empty dict if not configured.
    """
    cfg = {}
    try:
        if getattr(settings, 'orientation', None):
            cfg = settings.orientation.model_dump()
        elif getattr(settings, 'pipeline', None) and getattr(settings.pipeline, 'orientation', None):
            cfg = settings.pipeline.orientation.model_dump()
    except Exception:
        pass
    return cfg or {}


def _parse_angle_from_result(result) -> int:
    """Parse the predicted angle from a YOLO classification result.
    Accepts class labels like '0', '90', '180', '270' or 'cls_90', etc.
    Returns one of {0, 90, 180, 270} or 0 on failure.
    """
    names = getattr(result, "names", None)
    try:
        top1 = int(result.probs.top1)
        label = str(names[top1]) if names is not None else str(top1)
        m = re.search(r"(-?\d+)(?:\.0)?$", label)
        if not m:
            return 0
        angle = int(m.group(1))
        if angle in (0, 90, 180, 270):
            return angle
        return 0
    except Exception:
        return 0


def _rotate_image_right_angle(image: np.ndarray, rot_deg: int) -> np.ndarray:
    """Rotate image by 0/90/180/270 clockwise degrees using cv2 fast ops."""
    rot = rot_deg % 360
    if rot == 0:
        return image
    if rot == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rot == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if rot == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), rot, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


class _OrientationCorrector:
    def __init__(self) -> None:
        cfg = _get_orientation_cfg()
        self.enabled: bool = bool(cfg.get('enabled', False))
        self.device: str = str(cfg.get('device') or getattr(settings.detection, 'device', 'cpu'))
        self.imgsz: int = int(cfg.get('imgsz', 224))
        self.model_path: Optional[str] = cfg.get('path') or cfg.get('model_path')
        self._model: Optional[YOLO] = None

        # Load model eagerly if enabled
        if self.enabled:
            self._ensure_loaded()

    def _ensure_loaded(self) -> bool:
        if not self.enabled:
            return False
        if not self.model_path:
            logger.warning("orientation.missing_model_path")
            self.enabled = False
            return False
        if self._model is not None:
            return True
        try:
            self._model = YOLO(self.model_path)
            
            # Warmup: force model to fully load by running a dummy prediction
            logger.info("orientation.warmup_start")
            try:
                dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
                _ = self._model.predict(source=[dummy], imgsz=self.imgsz, 
                                       device=self.device, verbose=False)
                logger.info("orientation.model_loaded_and_warmed", 
                           path=self.model_path, device=self.device)
            except Exception as warmup_error:
                logger.warning("orientation.warmup_failed", error=str(warmup_error))
                # Still mark as loaded even if warmup fails
                logger.info("orientation.model_loaded", path=self.model_path, device=self.device)
            
            return True
        except Exception as e:
            logger.error("orientation.model_load_failed", error=str(e), exc_info=True)
            self.enabled = False
            self._model = None
            return False

    def correct_image(self, image: np.ndarray) -> np.ndarray:
        if not self._ensure_loaded():
            return image
        try:
            results = self._model.predict(source=[image], imgsz=self.imgsz, device=self.device, verbose=False)
            angle = _parse_angle_from_result(results[0])
            rot = (-angle) % 360
            return _rotate_image_right_angle(image, rot)
        except Exception as e:
            logger.warning("orientation.predict_failed", error=str(e))
            return image

    def correct_images(self, images: List[np.ndarray]) -> List[np.ndarray]:
        if not self._ensure_loaded():
            return images
        if not images:
            return images
        try:
            results = self._model.predict(source=images, imgsz=self.imgsz, device=self.device, verbose=False)
            out: List[np.ndarray] = []
            for img, res in zip(images, results):
                angle = _parse_angle_from_result(res)
                rot = (-angle) % 360
                out.append(_rotate_image_right_angle(img, rot))
            return out
        except Exception as e:
            logger.warning("orientation.batch_predict_failed", error=str(e))
            return images


_singleton: Optional[_OrientationCorrector] = None
_singleton_lock = threading.Lock()


def correct_images(images: List[np.ndarray]) -> List[np.ndarray]:
    """
    Public API: rotate images upright once based on orientation model (if enabled).
    Thread-safe singleton initialization using double-check locking pattern.
    """
    global _singleton
    
    # First check (without lock) for performance
    if _singleton is None:
        with _singleton_lock:
            # Second check (with lock) to prevent race condition
            if _singleton is None:
                _singleton = _OrientationCorrector()
    
    return _singleton.correct_images(images)
