# app/utils/visualization.py

"""
Utility functions for all visualization and debugging tasks,
such as drawing on images or saving debug plots.
"""

import uuid
from pathlib import Path
from typing import List, Tuple, Sequence, Union

import cv2
import numpy as np
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt

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
