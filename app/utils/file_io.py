# app/utils/file_io.py
import base64
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fpdf import FPDF
from pdf2image import convert_from_bytes
from PIL import Image, ImageOps, ImageFile
import io
import structlog

from app.core.config import settings
from .text_processing import make_farsi_text_for_pdf

logger = structlog.get_logger(__name__)

# Allow PIL to load truncated images (incomplete files from upload failures)
# This handles cases where network issues cause incomplete file transfers
ImageFile.LOAD_TRUNCATED_IMAGES = True

def bytes_to_image(image_bytes: bytes) -> np.ndarray:
    """
    Decodes image bytes into a NumPy array in BGR format.
    
    Uses PIL for robust format support including TIFF, JPEG, PNG, and others.
    Converts to BGR numpy array for compatibility with OpenCV-based downstream processing.
    
    PIL is preferred over cv2.imdecode() because:
    - Reliable TIFF support (OpenCV TIFF support varies by build)
    - Better format coverage (WebP, HEIC, etc.)
    - Consistent behavior across platforms
    - Proper handling of EXIF orientation metadata
    
    Handles truncated/incomplete images gracefully:
    - Sets ImageFile.LOAD_TRUNCATED_IMAGES to allow processing incomplete files
    - Common with network interruptions during upload
    
    Args:
        image_bytes: Raw image file bytes
        
    Returns:
        NumPy array in BGR format (H, W, 3), dtype uint8
        
    Raises:
        ValueError: If image cannot be decoded or is corrupt
    """
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Load image fully to validate it can be read completely
        # This catches truncation issues early and ensures image is valid
        pil_image.load()
        
        # Handle EXIF orientation automatically
        # ImageOps.exif_transpose handles orientation tags correctly
        pil_image = ImageOps.exif_transpose(pil_image)
        
        # Convert to RGB if necessary (handles grayscale, RGBA, etc.)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert PIL Image to numpy array and then to BGR for OpenCV compatibility
        # PIL images are RGB, OpenCV uses BGR, so we swap channels
        image_array = np.array(pil_image)
        bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return bgr_image
        
    except Exception as e:
        logger.error(
            "bytes_to_image.decode_failed",
            error=str(e),
            error_type=type(e).__name__
        )
        raise ValueError(f"Failed to decode image. The file may be corrupt or in an unsupported format: {str(e)}") from e


def pdf_to_images(pdf_bytes: bytes, dpi: int = 300) -> List[np.ndarray]:
    """
    Converts a PDF byte stream into a list of NumPy array images.
    
    Implements adaptive DPI fallback:
    - Tries initial DPI (300)
    - Falls back to 200 if initial fails (large/corrupted PDFs)
    - Falls back to 150 if 200 fails (very large PDFs)
    - Raises error if all attempts fail
    
    This prevents OOM crashes on large PDFs and handles corrupted files gracefully.
    """
    dpi_attempts = [dpi, 200, 150]  # Progressive DPI fallback
    last_error = None
    
    for attempt_dpi in dpi_attempts:
        try:
            logger.debug("pdf_to_images.attempt", dpi=attempt_dpi)
            pil_images = convert_from_bytes(pdf_bytes, dpi=attempt_dpi)
            
            if not pil_images:
                logger.warning("pdf_to_images.empty_result", dpi=attempt_dpi)
                continue
                
            images = [cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR) for pil_img in pil_images]
            
            if attempt_dpi != dpi:
                logger.info("pdf_to_images.fallback_succeeded", original_dpi=dpi, fallback_dpi=attempt_dpi, page_count=len(images))
            
            return images
            
        except Exception as e:
            last_error = e
            logger.warning("pdf_to_images.conversion_failed", dpi=attempt_dpi, error=str(e), error_type=type(e).__name__)
            continue
    
    # All DPI attempts failed
    logger.error("pdf_to_images.all_attempts_failed", attempted_dpis=dpi_attempts, final_error=str(last_error))
    raise ValueError(f"Failed to convert PDF. Tried DPIs {dpi_attempts}. Last error: {str(last_error)}")

def create_searchable_pdf(texts: List[str], output_path: str):
    """Creates a searchable PDF file from a list of text strings."""
    pdf = FPDF()
    pdf.add_font('DejaVu', '', settings.SEARCHABLE_PDF_FONT_PATH, uni=True)
    pdf.set_font('DejaVu', '', 11)
    
    for text in texts:
        pdf.add_page()
        bidi_text = make_farsi_text_for_pdf(text)
        pdf.multi_cell(0, 10, bidi_text, align='R')
    pdf.output(output_path)

def save_as_text(text_data: List[str], output_path: str) -> str:
    """Saves extracted text content to a .txt file."""
    txt_output_path = Path(output_path).with_suffix('.txt')
    with open(txt_output_path, 'w', encoding='utf-8') as f:
        for i, page_text in enumerate(text_data):
            if i > 0: f.write('\n\n' + '='*20 + '\n\n')
            f.write(page_text.strip())
    return str(txt_output_path)

def read_text_file(file_path: str) -> str:
    """Reads the content of a text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def get_file_as_base64(file_path: str) -> str:
    """Reads a file and returns its base64 encoded content."""
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')
