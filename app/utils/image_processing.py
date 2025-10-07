# app/utils/image_processing.py
from typing import List, Tuple
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


def merge_overlapping_masks(masks: List[np.ndarray], dice_threshold: float = 0.5) -> List[np.ndarray]:
    """Merges masks that have a high degree of overlap using the Dice score and a Union-Find algorithm."""
    n = len(masks)
    if n < 2:
        return masks
        
    parent = list(range(n))
    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i, root_j = find(i), find(j)
        if root_i != root_j: parent[root_j] = root_i

    merges_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            # Note: `dice_score` is now called directly as it's in the same module
            dice_val = dice_score(masks[i], masks[j])
            if dice_val > dice_threshold:
                union(i, j)
                merges_count += 1
                print(f"DEBUG: Merged masks {i} and {j}, dice_score={dice_val:.3f}")
    
    print(f"DEBUG: Total merges performed: {merges_count}")
                
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
        
    merged_masks = []
    for indices in groups.values():
        merged_mask = np.zeros_like(masks[0], dtype=np.uint8)
        for idx in indices:
            merged_mask = np.logical_or(merged_mask, masks[idx])
        merged_masks.append(merged_mask.astype(np.uint8))
        
    return merged_masks


def dice_score(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculates the Dice similarity coefficient between two binary masks."""
    mask1_bool, mask2_bool = mask1.astype(bool), mask2.astype(bool)
    intersection = np.sum(mask1_bool & mask2_bool)
    total = np.sum(mask1_bool) + np.sum(mask2_bool)
    return (2.0 * intersection) / total if total > 0 else 1.0