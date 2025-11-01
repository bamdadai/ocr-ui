# app/utils/visualization.py

"""
Utility functions for all visualization and debugging tasks,
such as drawing on images or saving debug plots.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Sequence, Union

import cv2
import numpy as np
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from .common import batchify
from .text_processing import make_farsi_text_for_display


def draw_boxes(image: np.ndarray, boxes: Sequence[Sequence[Union[int, float]]], color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Draws multiple bounding boxes on an image.
    - Accepts boxes in [x1, y1, x2, y2] format
    - Also accepts flat polygons (even-length lists), for which a bounding rectangle is drawn
    - Silently skips invalid entries
    """
    if not boxes:
        return image

    for box in boxes:
        try:
            arr = np.asarray(box).reshape(-1)
            if arr.size < 4:
                continue
            if arr.size == 4:
                x1, y1, x2, y2 = map(int, arr.tolist())
            elif arr.size % 2 == 0:
                xs = arr[0::2]
                ys = arr[1::2]
                x1, y1, x2, y2 = int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys))
            else:
                # Unknown layout (e.g., rotated boxes with angle). Best-effort: take first 4 as xyxy
                x1, y1, x2, y2 = map(int, arr[:4].tolist())

            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        except Exception:
            # Skip malformed box without interrupting debug flow
            continue
    return image


def draw_polygons(image: np.ndarray, polygons: Sequence[Sequence[Union[int, float]]], color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Draws multiple polygons on an image.
    - Expects each polygon as a flat list of coordinates [x1, y1, x2, y2, ...]
    - Skips polygons with odd number of coordinates or fewer than 6 values
    """
    if not polygons:
        return image

    for poly in polygons:
        try:
            flat = np.asarray(poly).reshape(-1)
            if flat.size < 6 or (flat.size % 2) != 0:
                continue
            pts = flat.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)
        except Exception:
            continue
    return image


def save_recognition_debug_image(crops: List[np.ndarray], results: List[Tuple[str, float]], save_dir: Path, font_path: Path):
    """Saves recognition results as a visual grid for debugging purposes."""
    if not crops or not results:
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    debug_font = fm.FontProperties(fname=str(font_path), size=18)
    data_to_plot = list(zip(crops, results))
    
    n_row = n_col = 5  # Display up to 25 images per debug file
    image_batches = batchify(data_to_plot, lambda x: x, n_row * n_col)

    for batch_data in image_batches:
        fig, axes = plt.subplots(n_row, n_col, figsize=(15, 15))
        axes = axes.ravel()
        
        for i, (img, (label, conf)) in enumerate(batch_data):
            # Convert BGR (from OpenCV) to RGB (for Matplotlib)
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            display_label = make_farsi_text_for_display(label)
            axes[i].set_title(f'{display_label}\n(conf: {conf:.2f})', fontproperties=debug_font)
            axes[i].axis('off')
        
        # Turn off any unused axes in the grid
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        # Save the figure with a unique name
        unique_id = uuid.uuid4()
        fig.savefig(save_dir / f'recognition_debug_{unique_id}.jpg', bbox_inches='tight')
        plt.close(fig) # Close the figure to free up memory






def save_word_polygons_on_page(
    image: np.ndarray,
    word_polygons: List[list],
    word_texts: List[str],
    save_dir: Path,
    page_id: str = None
) -> None:
    """
    Draws word polygons on the full page image at their exact locations.

    Args:
        image: The full page image (BGR format from OpenCV)
        word_polygons: List of word polygons in page-level coordinates
        word_texts: List of recognized text for each word
        save_dir: Directory to save the debug image
        page_id: Optional page identifier for filename
    """
    if not word_polygons:
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    # Create a copy of the image to draw on
    debug_image = image.copy()

    # Draw all word polygons on the page
    debug_image = draw_polygons(debug_image, word_polygons, color=(255, 0, 0), thickness=2)

    # Save the visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    page_prefix = f"{page_id}_" if page_id else ""
    filename = f"{timestamp}_{page_prefix}words_on_page.jpg"
    output_path = save_dir / filename

    cv2.imwrite(str(output_path), debug_image)


def save_line_parts_visualization(
    line_crop: np.ndarray,
    parts: List[dict],
    save_dir: Path,
    line_id: str = None
) -> None:
    """
    Visualizes part boundaries on the full line image for debugging.

    Each part's bounding box is drawn on the line image with different colors.

    Args:
        line_crop: The full line image (BGR format from OpenCV)
        parts: List of part dictionaries with 'bbox' key
                bbox format: [x, y, w, h] in line-local coordinates
        save_dir: Directory to save the debug image
        line_id: Optional line identifier for filename
    """
    if not parts:
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    # Create a copy of the image to draw on
    debug_image = line_crop.copy()

    # Define colors for different parts (BGR format)
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
    ]

    # Draw each part boundary
    for i, part in enumerate(parts):
        bbox = part['bbox']  # [x, y, w, h]

        # Get color for this part (cycle through colors if more than 8 parts)
        color = colors[i % len(colors)]

        # Draw rectangle for part boundary
        x, y, w, h = bbox
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), color, thickness=3)

    # Save the visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    line_prefix = f"{line_id}_" if line_id else ""
    filename = f"{timestamp}_{line_prefix}parts.jpg"
    output_path = save_dir / filename

    cv2.imwrite(str(output_path), debug_image)


def save_word_crops(
    word_data: List[dict],
    save_dir: Path,
    line_id: str = None
) -> None:
    """
    Writes individual word crops to disk for debug inspection.

    Args:
        word_data: List of dicts each with a 'crop' numpy array and optional 'text'.
        save_dir: Root directory where crops will be saved.
        line_id: Optional line identifier used to create a subfolder for the crops.
    """
    if not word_data:
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    if line_id:
        safe_line_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in line_id)
        target_dir = save_dir / safe_line_id
    else:
        target_dir = save_dir / f"line_{uuid.uuid4().hex[:6]}"
    target_dir.mkdir(parents=True, exist_ok=True)

    for idx, word_info in enumerate(word_data):
        crop = word_info.get('crop')
        if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
            continue

        raw_text = word_info.get('text') or ''
        sanitized = "".join(ch for ch in raw_text if ch.isalnum())
        if not sanitized:
            sanitized = "word"
        sanitized = sanitized[:32]  # Avoid very long filenames

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{idx:03d}_{sanitized}.jpg"
        output_path = target_dir / filename

        cv2.imwrite(str(output_path), crop)


def save_rec_debug_images(crops: List[np.ndarray], transcriptions: List[str], confidences: List[float], save_dir: Path) -> None:
    """
    Saves recognition images to rec_debug folder with filename: timestamp_confidence_transcription.jpg
    
    Args:
        crops: List of image arrays
        transcriptions: List of transcription strings
        confidences: List of confidence scores (0.0-1.0)
        save_dir: Directory path (typically 'rec_debug')
    """
    if not crops or not transcriptions or len(crops) != len(transcriptions):
        return
    if len(confidences) != len(crops):
        confidences = [0.0] * len(crops)  # Default to 0.0 if missing
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for crop, transcription, confidence in zip(crops, transcriptions, confidences):
        if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
            continue
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Format confidence as 3 decimal places
        conf_str = f"{confidence:.3f}"
        
        # Sanitize transcription for filename
        sanitized = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in transcription).strip()
        if not sanitized:
            sanitized = "empty"
        sanitized = sanitized[:200]  # Limit length
        
        filename = f"{timestamp}_{conf_str}_{sanitized}.jpg"
        output_path = save_dir / filename
        cv2.imwrite(str(output_path), crop)
