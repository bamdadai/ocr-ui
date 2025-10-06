# textdet_cropper.py
# -*- coding: utf-8 -*-

"""
OCR detector -> cropper utility for PaddleOCR TextDetection.

Usage
-----
from textdet_cropper import DetectorCropper

cropper = DetectorCropper(model_name="PP-OCRv5_server_det", reading_direction="rtl")
crops = cropper.extract_crops(
    image="general_ocr_001.png",     # str path or np.ndarray (BGR or RGB)
    warp=True,                       # perspective warp crops (rotated rects)
    pad=2,                           # pixels of padding around each box
    sort_reading_order=True,         # row-wise order, then in-row per reading_direction
    debug_dir="./debug_crops"        # optional: saves crops + annotated image
)
# `crops` is List[np.ndarray] (BGR, uint8)

CLI (quick test)
----------------
python -m textdet_cropper --image general_ocr_001.png --out ./debug_crops --reading-direction rtl
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union, Literal

import cv2
import numpy as np
from paddleocr import TextDetection


Quad = np.ndarray  # shape (4, 2), float32


@dataclass
class CropMeta:
    index: int
    polygon: Quad
    bbox_xyxy: Tuple[int, int, int, int]
    width: int
    height: int
    saved_path: Optional[str] = None


def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    """Ensure uint8 BGR image (OpenCV convention)."""
    if img is None:
        raise ValueError("Input image is None.")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _order_quad(pts: Quad) -> Quad:
    """Order quad points as [top-left, top-right, bottom-right, bottom-left]."""
    pts = np.array(pts, dtype=np.float32).reshape(-1, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _quad_size(quad: Quad) -> Tuple[int, int]:
    """Estimate target (w, h) for a perspective crop from a quad."""
    tl, tr, br, bl = _order_quad(quad)
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    w = int(max(widthA, widthB))
    h = int(max(heightA, heightB))
    w = max(w, 1)
    h = max(h, 1)
    return w, h


def _warp_quad(img: np.ndarray, quad: Quad) -> np.ndarray:
    """Perspective warp a quadrilateral region into a straight rectangle."""
    tl, tr, br, bl = _order_quad(quad)
    w, h = _quad_size([tl, tr, br, bl])
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype=np.float32), dst)
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC)
    return warped


def _poly_to_bbox(poly: Quad, pad: int, W: int, H: int) -> Tuple[int, int, int, int]:
    """Compute clamped XYXY bbox from polygon with padding."""
    xs = poly[:, 0]
    ys = poly[:, 1]
    x1 = max(int(np.floor(xs.min())) - pad, 0)
    y1 = max(int(np.floor(ys.min())) - pad, 0)
    x2 = min(int(np.ceil(xs.max())) + pad, W - 1)
    y2 = min(int(np.ceil(ys.max())) + pad, H - 1)
    if x2 <= x1:
        x2 = min(x1 + 1, W - 1)
    if y2 <= y1:
        y2 = min(y1 + 1, H - 1)
    return x1, y1, x2, y2


def _centroid(poly: Quad) -> Tuple[float, float]:
    c = poly.mean(axis=0)
    return float(c[0]), float(c[1])


def _sort_reading_order(
    polys: List[Quad],
    reading_direction: Literal["ltr", "rtl"] = "ltr",
) -> List[int]:
    """
    Sort polygons row-wise top->bottom, then within each row by reading direction.
    For Persian, use reading_direction="rtl".
    Returns indices in sorted order.
    """
    centers = np.array([_centroid(p) for p in polys])  # (N,2): x, y
    ys = centers[:, 1]
    xs = centers[:, 0]

    # Estimate row tolerance from polygon heights
    heights = np.array([float(np.max(p[:, 1]) - np.min(p[:, 1])) for p in polys])
    row_tol = max(8.0, float(np.median(heights)) * 0.6)

    # Assign a row id to each polygon
    row_ids = np.round(ys / row_tol).astype(int)
    rows = {}
    for i, rid in enumerate(row_ids):
        rows.setdefault(rid, []).append(i)

    # Sort rows by their vertical position (increasing y)
    sorted_row_keys = sorted(rows.keys(), key=lambda rid: np.median(ys[rows[rid]]))

    ordered: List[int] = []
    for rid in sorted_row_keys:
        idxs = rows[rid]
        # Within a row: LTR -> ascending x; RTL -> descending x
        if reading_direction == "rtl":
            idxs.sort(key=lambda i: xs[i], reverse=True)
        else:
            idxs.sort(key=lambda i: xs[i])
        ordered.extend(idxs)
    return ordered


def _maybe_make_dir(d: Optional[str]) -> Optional[str]:
    if not d:
        return None
    os.makedirs(d, exist_ok=True)
    return d


def _extract_polys_from_result_obj(res: Any) -> List[Quad]:
    """
    Try to extract Nx4x2 polygons from various possible result object formats.
    Supports PaddleOCR TextDetection result from the high-level API.
    """
    for attr in ("boxes", "polygons", "dt_polys"):
        if hasattr(res, attr):
            polys = getattr(res, attr)
            if polys is not None:
                return [np.array(p, dtype=np.float32).reshape(-1, 2) for p in polys]

    for key in ("boxes", "polygons", "dt_polys", "poly", "points"):
        try:
            val = res[key]  # type: ignore[index]
            if val is not None:
                return [np.array(p, dtype=np.float32).reshape(-1, 2) for p in val]
        except Exception:
            pass

    for method in ("to_dict", "as_dict"):
        if hasattr(res, method) and callable(getattr(res, method)):
            d = getattr(res, method)()
            for key in ("boxes", "polygons", "dt_polys", "points"):
                if key in d and d[key] is not None:
                    return [np.array(p, dtype=np.float32).reshape(-1, 2) for p in d[key]]

    if hasattr(res, "__dict__"):
        d = res.__dict__
        for key in ("boxes", "polygons", "dt_polys", "points"):
            if key in d and d[key] is not None:
                return [np.array(p, dtype=np.float32).reshape(-1, 2) for p in d[key]]

    raise RuntimeError(
        "Could not extract polygons from detection result. "
        "Check the result object's attributes/keys."
    )


class DetectorCropper:
    """
    Wraps PaddleOCR TextDetection and returns cropped regions as NumPy arrays.

    Parameters
    ----------
    model_name : str
        PaddleOCR detection model name (e.g., 'PP-OCRv5_server_det').
    model_kwargs : dict
        Extra kwargs passed to TextDetection constructor.
    reading_direction : {'ltr','rtl'}
        In-row reading order. Persian should use 'rtl'. Default 'rtl'.
    """

    def __init__(
        self,
        model_name: str = "PP-OCRv5_server_det",
        reading_direction: Literal["ltr", "rtl"] = "rtl",
        **model_kwargs: Any,
    ) -> None:
        self.model = TextDetection(model_name=model_name, **model_kwargs)
        self.reading_direction: Literal["ltr", "rtl"] = reading_direction

    def _load_image(self, image: Union[str, np.ndarray]) -> Tuple[np.ndarray, str]:
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Could not read image from path: {image}")
            return img, os.path.basename(image)
        elif isinstance(image, np.ndarray):
            return image.copy(), f"ndarray_{uuid.uuid4().hex[:8]}.png"
        else:
            raise TypeError("`image` must be a file path (str) or a numpy.ndarray.")

    def detect(self, image: Union[str, np.ndarray]) -> List[Quad]:
        """
        Run detection and return a list of polygons (float32, shape (4,2)).
        """
        img, _ = self._load_image(image)
        output = self.model.predict(img, batch_size=1)
        if not isinstance(output, Iterable) or len(output) == 0:  # type: ignore[arg-type]
            raise RuntimeError("Detector returned no results.")
        res = list(output)[0]
        polys = _extract_polys_from_result_obj(res)
        # Clean polygons: ensure shape (4,2)
        cleaned: List[Quad] = []
        for p in polys:
            p = np.array(p, dtype=np.float32).reshape(-1, 2)
            if p.shape[0] >= 4:
                if p.shape[0] > 4:
                    rect = cv2.minAreaRect(p.astype(np.float32))
                    box = cv2.boxPoints(rect)  # 4x2
                    p = box.astype(np.float32)
                cleaned.append(p[:4])
        return cleaned

    def extract_crops(
        self,
        image: Union[str, np.ndarray],
        warp: bool = True,
        pad: int = 2,
        sort_reading_order: bool = True,
        debug_dir: Optional[str] = None,
        return_meta: bool = False,
    ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[CropMeta]]]:
        """
        Detect text regions and return list of cropped images as numpy arrays.

        Parameters
        ----------
        image : str | np.ndarray
            Path or array (BGR/RGB/GRAY). Returned crops are BGR uint8.
        warp : bool
            If True, uses perspective transform of the polygon; otherwise
            returns simple rectangular bbox crops.
        pad : int
            Padding (pixels) applied around the polygon bounding box.
        sort_reading_order : bool
            Sort crops row-wise top-to-bottom, then in-row per reading_direction.
        debug_dir : str | None
            If provided, saves:
              - annotated image with polygons
              - each crop as a separate file
              - metadata JSON (polygons, bboxes, saved paths)
        return_meta : bool
            If True, also returns metadata list (CropMeta per crop).

        Returns
        -------
        crops : List[np.ndarray]
            Cropped regions (BGR, uint8).
        meta : List[CropMeta] (optional)
            Per-crop metadata when return_meta=True.
        """
        img_raw, name = self._load_image(image)
        img = _ensure_bgr(img_raw)
        H, W = img.shape[:2]

        # Run detection
        polys = self.detect(img)

        indices = list(range(len(polys)))
        if sort_reading_order and len(polys) > 1:
            indices = _sort_reading_order(polys, reading_direction=self.reading_direction)

        debug_dir = _maybe_make_dir(debug_dir)
        crops: List[np.ndarray] = []
        metas: List[CropMeta] = []

        # For annotated overview
        overlay = img.copy()
        for idx_out, pi in enumerate(indices):
            poly = polys[pi].astype(np.float32)
            if warp:
                crop = _warp_quad(img, poly)
            else:
                x1, y1, x2, y2 = _poly_to_bbox(poly, pad=pad, W=W, H=H)
                crop = img[y1 : y2 + 1, x1 : x2 + 1].copy()

            h, w = crop.shape[:2]
            bbox = _poly_to_bbox(poly, pad=pad, W=W, H=H)
            meta = CropMeta(
                index=idx_out,
                polygon=_order_quad(poly),
                bbox_xyxy=bbox,
                width=w,
                height=h,
            )

            if debug_dir:
                crop_fn = os.path.join(debug_dir, f"{os.path.splitext(name)[0]}_crop_{idx_out:03d}.png")
                cv2.imwrite(crop_fn, crop)
                meta.saved_path = crop_fn

                poly_i = _order_quad(poly).astype(int)
                cv2.polylines(overlay, [poly_i], isClosed=True, color=(0, 255, 0), thickness=2)
                cx, cy = map(int, _centroid(poly))
                cv2.putText(overlay, f"{idx_out}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 220, 30), 2)

            crops.append(crop)
            metas.append(meta)

        if debug_dir:
            ann_path = os.path.join(debug_dir, f"{os.path.splitext(name)[0]}_annotated.png")
            cv2.imwrite(ann_path, overlay)

            meta_json = []
            for m in metas:
                meta_json.append(
                    dict(
                        index=m.index,
                        polygon=m.polygon.astype(float).tolist(),
                        bbox_xyxy=list(map(int, m.bbox_xyxy)),
                        width=m.width,
                        height=m.height,
                        saved_path=m.saved_path,
                    )
                )
            with open(os.path.join(debug_dir, f"{os.path.splitext(name)[0]}_meta.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "image_name": name,
                        "image_size": [int(W), int(H)],
                        "num_crops": len(crops),
                        "reading_direction": self.reading_direction,
                        "crops": meta_json,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        return (crops, metas) if return_meta else crops


# ----------------------------- CLI (optional) -----------------------------

def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="Detect and crop text regions (PaddleOCR TextDetection).")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--model-name", default="PP-OCRv5_server_det", help="PaddleOCR detection model name.")
    parser.add_argument("--no-warp", action="store_true", help="Disable perspective warp; use bbox crops.")
    parser.add_argument("--pad", type=int, default=2, help="Padding (pixels) around bbox.")
    parser.add_argument("--no-sort", action="store_true", help="Do not sort by reading order.")
    parser.add_argument("--reading-direction", choices=["ltr", "rtl"], default="rtl",
                        help="In-row order within each text line (default: rtl).")
    parser.add_argument("--out", default=None, help="Directory to save debug outputs (crops + annotated + meta.json).")

    args = parser.parse_args()

    cropper = DetectorCropper(model_name=args.model_name, reading_direction=args.reading_direction)
    crops, metas = cropper.extract_crops(
        image=args.image,
        warp=not args.no_warp,
        pad=args.pad,
        sort_reading_order=not args.no_sort,
        debug_dir=args.out,
        return_meta=True,
    )
    print(f"Extracted {len(crops)} crops.")
    if args.out:
        print(f"Debug written to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    _cli()
