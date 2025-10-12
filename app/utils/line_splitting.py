"""
Utilities for splitting text lines into horizontal parts for recognition.

This module implements intelligent line splitting that:
- Filters words by area threshold
- Splits lines into parts with width/height constraints
- Extracts part images for recognition
"""

import numpy as np
from typing import List, Tuple, Dict
import structlog

logger = structlog.get_logger(__name__)


def filter_significant_words(
    word_polygons: List[list],
    line_box: List[int],
    area_threshold: float = 0.10
) -> List[list]:
    """
    Filter words whose area is at least area_threshold of the line's total area.

    Args:
        word_polygons: List of word polygons (each is a flat list of coords)
        line_box: Line bounding box [x, y, w, h]
        area_threshold: Minimum area ratio (default 0.10 = 10%)

    Returns:
        List of significant word polygons
    """
    line_area = line_box[2] * line_box[3]  # width * height
    min_area = line_area * area_threshold

    significant_words = []
    for poly in word_polygons:
        word_area = calculate_polygon_area(poly)
        if word_area >= min_area:
            significant_words.append(poly)

    logger.debug(
        "line_splitting.filter_words",
        total_words=len(word_polygons),
        significant_words=len(significant_words),
        area_threshold=area_threshold
    )

    return significant_words


def calculate_polygon_area(polygon: List[int]) -> float:
    """
    Calculate the area of a polygon using the Shoelace formula.

    Args:
        polygon: Flat list of coordinates [x1, y1, x2, y2, ...]

    Returns:
        Area of the polygon
    """
    # Reshape to (n, 2) array
    points = np.array(polygon).reshape(-1, 2)

    # Shoelace formula
    x = points[:, 0]
    y = points[:, 1]

    area = 0.5 * np.abs(
        np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
    )

    return float(area)


def get_polygon_bbox(polygon: List[int]) -> Tuple[int, int, int, int]:
    """
    Get the bounding box of a polygon.

    Args:
        polygon: Flat list of coordinates [x1, y1, x2, y2, ...]

    Returns:
        Bounding box as (x, y, w, h)
    """
    points = np.array(polygon).reshape(-1, 2)
    x_coords = points[:, 0]
    y_coords = points[:, 1]

    x_min = int(np.min(x_coords))
    x_max = int(np.max(x_coords))
    y_min = int(np.min(y_coords))
    y_max = int(np.max(y_coords))

    return (x_min, y_min, x_max - x_min, y_max - y_min)


def split_line_into_parts(
    line_crop: np.ndarray,
    line_box: List[int],
    word_polygons: List[list],
    max_width_height_ratio: float = 4.0,
    max_words_per_part: int = 4,
    area_threshold: float = 0.10
) -> List[Dict]:
    """
    Split a line into horizontal parts for recognition.

    The line is split from right to left (RTL) into parts where:
    - Each part's width is at most max_width_height_ratio × part_height
    - Each part contains at most max_words_per_part significant words
    - Only words with area >= area_threshold × line_area are counted

    Args:
        line_crop: The cropped line image
        line_box: Line bounding box in page coordinates [x, y, w, h]
        word_polygons: List of word polygons in line-local coordinates
        max_width_height_ratio: Maximum width/height ratio for each part
        max_words_per_part: Maximum number of significant words per part
        area_threshold: Minimum word area ratio to be counted

    Returns:
        List of part dictionaries, each containing:
            - 'crop': The part image (numpy array)
            - 'bbox': Bounding box in line-local coordinates (x, y, w, h)
            - 'word_count': Number of significant words in this part
    """
    if not word_polygons:
        # No words, return the entire line as one part
        return [{
            'crop': line_crop,
            'bbox': [0, 0, line_crop.shape[1], line_crop.shape[0]],
            'word_count': 0
        }]

    # Filter significant words
    significant_words = filter_significant_words(word_polygons, line_box, area_threshold)

    if not significant_words:
        # No significant words, return the entire line as one part
        return [{
            'crop': line_crop,
            'bbox': [0, 0, line_crop.shape[1], line_crop.shape[0]],
            'word_count': 0
        }]

    # Get bounding boxes for significant words in line-local coordinates
    word_bboxes = [get_polygon_bbox(poly) for poly in significant_words]

    # Sort words from right to left (descending x)
    word_bboxes_sorted = sorted(word_bboxes, key=lambda b: b[0], reverse=True)

    line_height = line_crop.shape[0]
    line_width = line_crop.shape[1]
    max_part_width = int(line_height * max_width_height_ratio)

    parts = []
    current_words = []
    current_x_min = line_width  # Start from right edge
    current_x_max = 0

    logger.debug(
        "line_splitting.start",
        line_height=line_height,
        line_width=line_width,
        max_part_width=max_part_width,
        significant_words=len(significant_words)
    )

    # Process words from right to left
    for word_bbox in word_bboxes_sorted:
        word_x, word_y, word_w, word_h = word_bbox
        word_x_max = word_x + word_w

        # Calculate the extent if we add this word to current part
        potential_x_min = min(current_x_min, word_x)
        potential_x_max = max(current_x_max, word_x_max)
        potential_width = potential_x_max - potential_x_min

        # Check if adding this word violates constraints
        would_exceed_width = potential_width > max_part_width
        would_exceed_word_count = len(current_words) >= max_words_per_part

        if current_words and (would_exceed_width or would_exceed_word_count):
            # Finalize current part
            part_bbox = [
                int(current_x_min),
                0,
                int(current_x_max - current_x_min),
                line_height
            ]
            part_crop = extract_part_crop(line_crop, part_bbox)

            parts.append({
                'crop': part_crop,
                'bbox': part_bbox,
                'word_count': len(current_words)
            })

            logger.debug(
                "line_splitting.part_created",
                part_index=len(parts) - 1,
                bbox=part_bbox,
                word_count=len(current_words)
            )

            # Start new part with current word
            current_words = [word_bbox]
            current_x_min = word_x
            current_x_max = word_x_max
        else:
            # Add word to current part
            current_words.append(word_bbox)
            current_x_min = potential_x_min
            current_x_max = potential_x_max

    # Don't forget the last part
    if current_words:
        part_bbox = [
            int(current_x_min),
            0,
            int(current_x_max - current_x_min),
            line_height
        ]
        part_crop = extract_part_crop(line_crop, part_bbox)

        parts.append({
            'crop': part_crop,
            'bbox': part_bbox,
            'word_count': len(current_words)
        })

        logger.debug(
            "line_splitting.part_created",
            part_index=len(parts) - 1,
            bbox=part_bbox,
            word_count=len(current_words)
        )

    logger.debug(
        "line_splitting.complete",
        total_parts=len(parts),
        total_significant_words=len(significant_words)
    )

    return parts


def extract_part_crop(line_crop: np.ndarray, bbox: List[int]) -> np.ndarray:
    """
    Extract a part crop from a line image.

    Args:
        line_crop: The line image
        bbox: Bounding box in line-local coordinates [x, y, w, h]

    Returns:
        Cropped part image
    """
    x, y, w, h = bbox

    # Ensure coordinates are within bounds
    x = max(0, x)
    y = max(0, y)
    x_end = min(line_crop.shape[1], x + w)
    y_end = min(line_crop.shape[0], y + h)

    # Extract the crop
    part_crop = line_crop[y:y_end, x:x_end]

    return part_crop
