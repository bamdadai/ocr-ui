"""
Utilities for splitting text lines into horizontal parts for recognition.

This module implements intelligent line splitting that:
- Splits lines into parts with width/height constraints (width ≤ 4× height)
- Parts are split at word boundaries only
- Extracts part images for recognition
"""

import numpy as np
from typing import List, Tuple, Dict
import structlog

logger = structlog.get_logger(__name__)


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
    overlap_threshold: float = 0.1
) -> List[Dict]:
    """
    Split a line into horizontal parts for recognition.

    The line is split from right to left (RTL) into parts where:
    - Each part's width is at most max_width_height_ratio × part_height
    - Parts are split at word boundaries only
    - All words are included (no filtering)
    - Guarantees at least one part per line (even if empty or exceeds width constraint)
    - If only one part exists, it covers the entire line width
    - When words overlap significantly, they stay together in the same part

    Args:
        line_crop: The cropped line image
        line_box: Line bounding box in page coordinates [x, y, w, h]
        word_polygons: List of word polygons in line-local coordinates
        max_width_height_ratio: Maximum width/height ratio for each part
        overlap_threshold: Minimum overlap ratio (0.0-1.0) to consider words as overlapping.
                          Overlap ratio = overlap_width / min(word1_width, word2_width)
                          Default: 0.1 (10% overlap)

    Returns:
        List of part dictionaries, each containing:
            - 'crop': The part image (numpy array)
            - 'bbox': Bounding box in line-local coordinates (x, y, w, h)
            - 'word_count': Number of words in this part

        Always returns at least one part per line. Single parts span the full line width.
    """
    if not word_polygons:
        # No words, return the entire line as one part
        return [{
            'crop': line_crop.copy(),
            'bbox': [0, 0, line_crop.shape[1], line_crop.shape[0]],
            'word_count': 0
        }]

    # Get bounding boxes for all words in line-local coordinates
    word_bboxes = [get_polygon_bbox(poly) for poly in word_polygons]

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
        max_width_height_ratio=max_width_height_ratio,
        max_part_width=max_part_width,
        total_words=len(word_polygons)
    )

    # Process words from right to left
    for word_bbox in word_bboxes_sorted:
        word_x, word_y, word_w, word_h = word_bbox
        word_x_max = word_x + word_w

        # Check if this word overlaps significantly with current part's extent
        # Overlap occurs if word's right edge (x_max) is greater than current part's left edge (x_min)
        has_significant_overlap = False
        if current_words and word_x_max > current_x_min:
            # Calculate overlap amount
            overlap_width = word_x_max - current_x_min

            # Get the last word in current part to calculate overlap ratio
            last_word_in_part = current_words[-1]
            last_word_width = last_word_in_part[2]

            # Calculate overlap ratio relative to the smaller word
            min_word_width = min(word_w, last_word_width)
            overlap_ratio = overlap_width / min_word_width if min_word_width > 0 else 0

            # Consider it significant overlap if it exceeds threshold
            has_significant_overlap = overlap_ratio >= overlap_threshold

            if has_significant_overlap:
                logger.debug(
                    "line_splitting.significant_overlap_detected",
                    overlap_width=overlap_width,
                    overlap_ratio=f"{overlap_ratio:.2f}",
                    threshold=overlap_threshold,
                    current_word_width=word_w,
                    last_word_width=last_word_width
                )

        # Calculate the extent if we add this word to current part
        potential_x_min = min(current_x_min, word_x)
        potential_x_max = max(current_x_max, word_x_max)
        potential_width = potential_x_max - potential_x_min

        # Check if adding this word violates width constraint
        would_exceed_width = potential_width > max_part_width

        if current_words and would_exceed_width:
            if has_significant_overlap:
                # Word overlaps significantly with current part but exceeds width limit
                # Solution: Include BOTH the overlapping word AND the last word from current part in new part

                # Finalize current part WITHOUT the last word
                last_word = current_words.pop()

                if current_words:  # Only create part if there are remaining words
                    # Recalculate part extent without the last word
                    remaining_x_coords = []
                    for w in current_words:
                        wx, wy, ww, wh = w
                        remaining_x_coords.extend([wx, wx + ww])

                    part_x_min = int(min(remaining_x_coords))
                    part_x_max = int(max(remaining_x_coords))

                    part_bbox = [
                        part_x_min,
                        0,
                        part_x_max - part_x_min,
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

                # Start new part with BOTH last word and current overlapping word
                current_words = [last_word, word_bbox]
                new_x_coords = [last_word[0], last_word[0] + last_word[2], word_x, word_x_max]
                current_x_min = int(min(new_x_coords))
                current_x_max = int(max(new_x_coords))

                logger.debug(
                    "line_splitting.new_part_with_overlap",
                    word_count=2,
                    new_x_min=current_x_min,
                    new_x_max=current_x_max
                )
            else:
                # No overlap - standard split at word boundary
                # Finalize current part at word boundary
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
            # Add word to current part (either no words yet, or fits within limit)
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

    # Safety check: ensure at least one part exists (should always be true if words exist)
    if not parts and word_polygons:
        logger.warning(
            "line_splitting.no_parts_created",
            total_words=len(word_polygons),
            message="No parts created despite having words, returning entire line"
        )
        return [{
            'crop': line_crop,
            'bbox': [0, 0, line_crop.shape[1], line_crop.shape[0]],
            'word_count': len(word_polygons)
        }]

    # If there's only one part, expand it to cover the entire line width
    if len(parts) == 1:
        parts[0]['bbox'] = [0, 0, line_width, line_height]
        parts[0]['crop'] = line_crop.copy()
        logger.debug(
            "line_splitting.single_part_expanded",
            original_width=parts[0]['bbox'][2],
            expanded_to_full_line_width=line_width
        )

    logger.debug(
        "line_splitting.complete",
        total_parts=len(parts),
        total_words=len(word_polygons)
    )

    return parts


def extract_part_crop(line_crop: np.ndarray, bbox: List[int]) -> np.ndarray:
    """
    Extract a part crop from a line image.

    Args:
        line_crop: The line image
        bbox: Bounding box in line-local coordinates [x, y, w, h]

    Returns:
        Cropped part image (independent copy, not a view)
    """
    x, y, w, h = bbox

    # Ensure coordinates are within bounds
    x = max(0, x)
    y = max(0, y)
    x_end = min(line_crop.shape[1], x + w)
    y_end = min(line_crop.shape[0], y + h)

    # Extract the crop and make an independent copy to avoid any reference issues
    # This ensures drawings on line_crop don't affect the crop
    part_crop = line_crop[y:y_end, x:x_end].copy()

    return part_crop
