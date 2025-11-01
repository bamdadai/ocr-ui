# app/utils/image_processing.py
from typing import List, Tuple, Optional
import numpy as np
import cv2
from collections import defaultdict

def make_box_from_poly(poly: List[int]) -> Tuple[int, int, int, int]:
    """Creates a bounding box from a flat list of polygon points."""
    x_coords = poly[0::2]
    y_coords = poly[1::2]
    return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

def crop_boxes_from_image(boxes_list: List[List[int]], image: np.ndarray) -> List[np.ndarray]:
    """Crops multiple bounding box regions from an image."""
    return [image[int(y1):int(y2), int(x1):int(x2)] for x1, y1, x2, y2 in boxes_list]

def crop_word_from_polygon(image: np.ndarray, polygon_points: list) -> np.ndarray:
    """Crops a precise word shape from an image using its polygon coordinates."""
    try:
        points = np.array(polygon_points, dtype=np.int32).reshape((-1, 1, 2))
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        x, y, w, h = cv2.boundingRect(points)
        masked_word = cv2.bitwise_and(image[y:y+h, x:x+w], image[y:y+h, x:x+w], mask=mask[y:y+h, x:x+w])
        bg = np.ones_like(masked_word, np.uint8) * 255
        cv2.bitwise_not(bg, bg, mask=mask[y:y+h, x:x+w])
        return bg + masked_word
    except Exception:
        return np.zeros((10, 10, 3), dtype=np.uint8)

def rebase_polygon(polygon: List[int], offset: Tuple[int, int]) -> List[int]:
    """Translates a polygon's coordinates to a new coordinate system."""
    x_offset, y_offset = offset
    return [c + (x_offset if i % 2 == 0 else y_offset) for i, c in enumerate(polygon)]

def get_polygons_from_masks(masks) -> List[List[int]]:
    """Converts masks (ultralytics Masks object or list of arrays) into polygon coordinates."""
    polygons = []

    # Handle ultralytics Masks object
    if hasattr(masks, 'data'):
        # Convert to numpy and get individual masks
        mask_data = masks.data.cpu().numpy() if hasattr(masks.data, 'cpu') else masks.data
        mask_list = [mask_data[i] for i in range(len(mask_data))]
    else:
        # Already a list of arrays
        mask_list = masks

    for mask in mask_list:
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 10:  # Filter small artifacts
                polygons.append(cnt.flatten().tolist())
    return polygons


def apply_padding(image: np.ndarray, padding: int) -> np.ndarray:
    """Adds padding around an image."""
    if padding <= 0:
        return image
    return cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=255)


def apply_erosion(image: np.ndarray, kernel_size: int, iterations: int = 1) -> np.ndarray:
    """Applies morphological erosion to the image."""
    if kernel_size <= 0 or iterations <= 0:
        return image
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(image, kernel, iterations=iterations)


def apply_contrast(image: np.ndarray, alpha: float) -> np.ndarray:
    """Adjusts image contrast. alpha=1.0 means no change, >1.0 increases contrast."""
    if alpha == 1.0:
        return image
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)


def apply_brightness(image: np.ndarray, beta: float) -> np.ndarray:
    """Adjusts image brightness. beta=0 means no change, >0 increases brightness."""
    if beta == 0:
        return image
    return cv2.convertScaleAbs(image, alpha=1.0, beta=beta)


def apply_sharpness(image: np.ndarray, strength: float) -> np.ndarray:
    """Applies sharpening filter to the image. strength=0 means no sharpening."""
    if strength <= 0:
        return image
    # Create sharpening kernel - standard unsharp masking approach
    # Higher strength values increase sharpening effect
    kernel = np.array([[0, -strength, 0],
                       [-strength, 1 + 4*strength, -strength],
                       [0, -strength, 0]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)


def preprocess_recognition_image(
    image: np.ndarray,
    padding: int = 0,
    erosion_kernel_size: int = 0,
    erosion_iterations: int = 1,
    contrast: float = 1.0,
    brightness: float = 0.0,
    sharpness: float = 0.0
) -> np.ndarray:
    """
    Applies configurable image preprocessing steps before recognition.
    
    Processing order: padding -> erosion -> contrast -> brightness -> sharpness
    
    Args:
        image: Input image as numpy array
        padding: Pixels to add around image (0 = disabled)
        erosion_kernel_size: Erosion kernel size (0 = disabled)
        erosion_iterations: Number of erosion iterations
        contrast: Contrast multiplier (1.0 = no change)
        brightness: Brightness adjustment (-255 to 255, 0 = no change)
        sharpness: Sharpening strength (0 = disabled)
        
    Returns:
        Preprocessed image
    """
    processed = image.copy()
    
    # Apply preprocessing in order
    if padding > 0:
        processed = apply_padding(processed, padding)
    
    if erosion_kernel_size > 0:
        processed = apply_erosion(processed, erosion_kernel_size, erosion_iterations)
    
    if contrast != 1.0:
        processed = apply_contrast(processed, contrast)
    
    if brightness != 0.0:
        processed = apply_brightness(processed, brightness)
    
    if sharpness > 0:
        processed = apply_sharpness(processed, sharpness)
    
    return processed

